# ViT-S Sanity Check — Experiment Design

## Objective

Reproduce the **ViT-Small** model from the reference repository
[mateuszgrzyb-pl/searchless-chess](https://github.com/mateuszgrzyb-pl/searchless-chess)
using our PyTorch infrastructure, trained on the **same 50M positions** as BDH-Chess v3.
This provides a direct, controlled comparison between:

| Model    | Params | Architecture                        | Reference ELO |
|----------|--------|-------------------------------------|---------------|
| BDH v3   | ~2.5M  | Iterative value refinement (custom) | -2969*        |
| ViT-S    | ~2.64M | Vision Transformer                  | 1817†         |

\* Our BDH result on 50M positions.  
† Reference result on 316M positions.

---

## 1. Reference Repo Analysis

### 1.1 Architecture (ViT-Small, `model_05`)

```
Input (B, 8, 8, 12)
  └─ Reshape → (B, 64, 12)       # 64 "patches" of dim 12
  └─ Linear(12 → 256)            # patch projection
  └─ + Positional Embedding(64, 256)  # learnable
  └─ 5× Transformer Block:
       ├─ LayerNorm(ε=1e-6)
       ├─ MultiHeadAttention(4 heads, key_dim=64, dropout=0.1)
       ├─ + Residual
       ├─ LayerNorm(ε=1e-6)
       ├─ FFN: Dense(512, GELU) → Dropout(0.1) → Dense(256)
       └─ + Residual
  └─ LayerNorm
  └─ GlobalAveragePooling         # (B, 256)
  └─ Dropout(0.1)
  └─ Dense(1, tanh)              # output in [-1, 1]
```

- **FFN ratio**: 2× (not 4× as standard ViT)
- **No CLS token**: uses global avg pooling instead
- **Pre-norm** architecture (LayerNorm before attention/FFN)
- **Parameters**: ~2.64M trainable

### 1.2 Data Pipeline (Reference)

| Aspect | Reference | Our Project (BDH v3) |
|--------|-----------|---------------------|
| Dataset | `mateuszgrzyb/lichess-stockfish-normalized` | Same HuggingFace dataset |
| Positions | 316M | 50M |
| Board encoding | `fen_to_tensor()` → (8,8,12) uint8 | Same function → (8,8,12) float32 |
| White perspective | `board.mirror()` for black | Same |
| Score normalization | `tanh(cp/1000)` | `tanh(cp/10000)` ← **DIFFERENT** |
| Mate handling | `sign(mate)` → ±1.0 | `mate_to_cp()` → then tanh |
| Format | TFRecord (GZIP) | NumPy (.npy mmap) |

**CRITICAL DIFFERENCE**: Score normalization scale (1000 vs 10000).
- `tanh(cp/1000)`: cp=500 → 0.46, cp=2000 → 0.96 (aggressive compression)
- `tanh(cp/10000)`: cp=500 → 0.05, cp=2000 → 0.20 (nearly linear)

This dramatically changes the target distribution and MSE loss landscape.

### 1.3 Training Configuration (Reference)

| Parameter | Value (inferred) |
|-----------|-----------------|
| Batch size | 8192 |
| Learning rate | 2e-4 (from WarmUpCosineDecay docstring) |
| Min learning rate | 1e-6 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Gradient clipping | clipnorm=1.0 |
| Mixed precision | mixed_float16 |
| LR schedule | Linear warmup → Cosine decay |
| Loss | MSE |
| Metric | MAE |
| Train/Val split | 90%/10% (file-level) |
| "Virtual epoch" | 1/10 of full data pass |
| Warmup | 5 virtual epochs |
| LR schedule horizon | 100 virtual epochs |
| Total training epochs | 30 virtual epochs |

**Virtual epoch concept**: Reference divides `steps_per_epoch` by 10 so
Keras reports 10× more epochs for finer-grained checkpointing and logging.
Each "virtual epoch" processes 1/10 of the training data.

### 1.4 Results (Reference on 316M)

| Model | Params | ELO |
|-------|--------|-----|
| CNN | 1.18M | 1112 |
| ResNet-M | 2.3M | 1515 |
| ResNet-L | 12.9M | 1711 |
| ResNet-XL | 24.7M | 1719 |
| **ViT-S** | **2.64M** | **1817** |
| ViT-M | 9.5M | 1960 |

---

## 2. Our Experiment Plan

### 2.1 Data Preparation

**Re-normalize scores** to match reference's `tanh(cp/1000)`:

The existing 50M .npy score files use `tanh(cp/10000)`. Instead of
re-downloading and re-encoding, we apply a mathematical conversion:

```python
# Inverse: recover raw cp from old normalization
raw_cp = atanh(old_score) * 10000
# Re-normalize with reference scale
new_score = tanh(raw_cp / 1000)  = tanh(atanh(old_score) * 10)
```

Boards (8×8×12 tensors) remain unchanged — encoding is identical.

**Output directory**: `/net/tscratch/people/plgantoniczapski/bdh-chess-data-1k/`
- Symlink `{split}_boards.npy` → original boards
- New `{split}_scores.npy` with re-normalized values

### 2.2 Model Implementation

Faithful PyTorch reimplementation of reference `build_vit()`:

```python
ChessViT(
    projection_dim=256,    # ViT-S
    num_heads=4,
    transformer_layers=5,
    ffn_ratio=2,           # 2× not 4×, matching reference
    dropout_rate=0.1,
)
```

Key translation decisions:
- `nn.MultiheadAttention(embed_dim=256, num_heads=4, dropout=0.1, batch_first=True)`
- `nn.LayerNorm(256, eps=1e-6)` — matching reference epsilon
- Positional embedding: `nn.Parameter(zeros(1, 64, dim))` with `trunc_normal_` init
- Input: (B, 12, 8, 8) → permute to (B, 8, 8, 12) → reshape (B, 64, 12) internally
- Output: `tanh` applied explicitly (not inside Linear)

### 2.3 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 30 (real) | Match BDH-v3 for direct comparison |
| Batch size | 8192 | Match reference ViT training |
| Learning rate | 2e-4 | Reference docstring default |
| Min LR | 1e-6 | Reference default |
| Weight decay | 1e-4 | Common transformer value |
| Clipnorm | 1.0 | Match reference |
| AMP dtype | bf16 | A100 native, no overflow risk |
| Warmup | 5 epochs | Match reference (proportional) |
| Scheduler | LinearLR + CosineAnnealingLR | Equivalent to WarmUpCosineDecay |
| Loss | MSE | Match reference |
| Early stopping | patience=10, delta=1e-5 | Match BDH-v3 |
| Checkpoint | every 2 epochs + best | Match BDH-v3 |

**Step arithmetic** (30 real epochs):
- Train samples: 45,000,000
- Batch size: 8,192
- Steps per epoch: 5,493
- Warmup steps: 5 × 5,493 = 27,465
- Total steps: 30 × 5,493 = 164,790

### 2.4 Comparison Framework

Both models evaluated on the **same 1,200 puzzle benchmark** (12 tiers):

| Metric | BDH v3 (known) | ViT-S (expected) |
|--------|----------------|-----------------|
| val_loss (MSE) | 0.01212 | TBD |
| val_mae | 0.1101 (scale=10000) | TBD (scale=1000) |
| ELO | -2969 | TBD |
| Puzzle accuracy | 5.8% | TBD |

**Note**: MAE values are NOT directly comparable because of different
score scales. MSE is also affected. ELO and puzzle accuracy are the
fair comparison metrics.

### 2.5 Compute Budget

| Resource | Estimate |
|----------|---------|
| Data re-normalization | ~5 min |
| Training (30 epochs) | ~8-12h on A100 |
| Evaluation | ~10 min |
| **Total** | **~12h** (well within 48h cap) |

---

## 3. Implementation Checklist

- [ ] `src/models/vit.py` — PyTorch ViT matching reference
- [ ] Register "vit" in `src/models/registry.py`
- [ ] `scripts/renormalize_data.py` — score transformation script
- [ ] `configs/train_vit_s_a100.yaml` — training config
- [ ] `train_vit_a100.sbatch` — SLURM submission script
- [ ] Run data re-normalization
- [ ] Submit training job
- [ ] Evaluate and compare with BDH v3

---

*Generated: 2025-03-22*
