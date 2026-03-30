# BDH Searchless Chess — Experiment Report

**Timestamp:** 2026-03-10 22:49  
**Experiment:** `bdh-chess-v1`  
**Author:** Antoni Czapski

---

## 1. Objective

Train the BDH (Biological Dendritic Hebbian) architecture — a post-transformer
model from Pathway Technology — on the searchless chess position evaluation task.
Evaluate playing strength via the 1200-puzzle Lichess benchmark.

---

## 2. Model: BDH-Chess

The original BDH is a character-level language model with shared-weight recurrent
layers, linear attention with RoPE, and a Hebbian fast-weight gating mechanism.

**Adaptations for chess evaluation:**

| Component              | Original BDH               | BDH-Chess                          |
|------------------------|----------------------------|-------------------------------------|
| Input                  | Token embedding (vocab→D)  | Linear projection (12→D)           |
| Sequence               | Variable-length text       | Fixed 64 tokens (8×8 board squares)|
| Attention mask         | Causal (lower triangular)  | **Bidirectional** (full)            |
| Output head            | LM head (D→vocab logits)   | GAP + MLP → Tanh ∈ (-1,1)         |
| Loss                   | Cross-entropy              | MSE                                |

**Architecture summary:**

```
Input (B, 12, 8, 8) → reshape to (B, 64, 12) → Linear(12, 128) → LayerNorm
  ↓ × 4 shared-weight BDH layers:
    Encoder:   (B, 1, 64, 128) → (B, 4, 64, 512)  [sparse latent]
    ReLU activation
    Bidirectional linear attention with RoPE
    Hebbian gating: x_sparse * y_sparse
    Decoder:   (B, 1, 64, 2048) → (B, 1, 64, 128)
    Residual + LayerNorm
  ↓
Global average pooling over 64 squares → (B, 128)
Regression head: Linear(128, 64) → GELU → Linear(64, 1) → Tanh
```

---

## 3. Configuration

| Parameter                    | Value                  |
|------------------------------|------------------------|
| `architecture`               | `bdh`                  |
| `n_layer`                    | 4                      |
| `n_embd`                     | 128                    |
| `n_head`                     | 4                      |
| `mlp_internal_dim_multiplier`| 16                     |
| `dropout`                    | 0.1                    |
| `regression_hidden`          | 64                     |
| **Total parameters**         | **796,417**            |
| Optimizer                    | AdamW                  |
| Learning rate                | 5×10⁻⁴ (cosine decay) |
| Weight decay                 | 1×10⁻⁴                |
| Warmup epochs                | 2                      |
| Mixed precision              | AMP fp16               |
| Gradient clipping            | max_norm=1.0           |
| Batch size                   | 128                    |
| Early stopping patience      | 7                      |

---

## 4. Data

| Split   | Size    |
|---------|---------|
| Train   | 400,000 |
| Val     | 50,000  |
| Test    | 50,000  |
| **Total** | **500,000** |

Source: `mateuszgrzyb/lichess-stockfish-normalized` (HuggingFace)  
Encoding: FEN → 8×8×12 binary tensor (always-white-perspective)  
Target: `tanh(cp / 10000)` ∈ [-1, 1]

---

## 5. Hardware

| Component | Details              |
|-----------|----------------------|
| GPU       | NVIDIA GeForce RTX 2060 (6 GB VRAM) |
| Driver    | 551.61, CUDA 12.4    |
| PyTorch   | 2.5.1+cu121          |
| Peak VRAM | ~1.5 GB (with AMP)   |

---

## 6. Training Results

**Training stopped early at epoch 9** (patience=7, no improvement after epoch 2).

| Epoch | Train Loss | Train MAE | Val Loss   | Val MAE | LR       | Time (s) |
|------:|------------|-----------|------------|---------|----------|----------|
|     2 | 0.111090   | 0.1484    | **0.110530** | 0.1451  | 5.00e-04 | 278.9    |
|     3 | 0.110637   | 0.1515    | 0.110723   | 0.1484  | 4.96e-04 | 279.1    |
|     4 | 0.111112   | 0.1524    | 0.111881   | 0.1407  | 4.85e-04 | 281.3    |
|     5 | 0.113057   | 0.1573    | 0.112595   | 0.1588  | 4.67e-04 | 287.8    |
|     6 | 0.113407   | 0.1568    | 0.112360   | 0.1553  | 4.42e-04 | 281.6    |
|     7 | 0.112119   | 0.1487    | 0.112353   | 0.1497  | 4.11e-04 | 281.7    |
|     8 | 0.112521   | 0.1494    | 0.112524   | 0.1454  | 3.75e-04 | 283.4    |
|     9 | 0.112410   | 0.1486    | 0.112623   | 0.1551  | 3.36e-04 | 282.0    |

**Best model:** epoch 2, val_loss = **0.110530**  
**Total training time:** ~42 minutes  
**Steps per epoch:** 3,125 (~4.7 min/epoch)

---

## 7. ELO Evaluation

Benchmark: 1200 Lichess rated puzzles, 12 tiers × 100 puzzles each.

| Tier          | Accuracy | Count |
|---------------|----------|-------|
| (0, 500]      | 2.0%     | 100   |
| (500, 750]    | 0.0%     | 100   |
| (750, 1000]   | 0.0%     | 100   |
| (1000, 1250]  | 1.0%     | 100   |
| (1250, 1500]  | 1.0%     | 100   |
| (1500, 1750]  | 1.0%     | 100   |
| (1750, 2000]  | 0.0%     | 100   |
| (2000, 2250]  | 0.0%     | 100   |
| (2250, 2500]  | 0.0%     | 100   |
| (2500, 2750]  | 0.0%     | 100   |
| (2750, 3000]  | 1.0%     | 100   |
| (3000, 3250]  | 0.0%     | 100   |

**Overall puzzle accuracy:** 0.5% (6 / 1200)  
**Estimated ELO:** N/A (below measurable threshold — random-level play)

---

## 8. Analysis

### Why the model is not learning well

1. **Insufficient data volume.** The model was trained on 500k positions. Prior work
   (original searchless-chess repo) shows that meaningful chess evaluation requires
   **10–100M+ positions**. The current dataset is 200–600× too small.

2. **Architecture mismatch.** BDH was designed for sequential/causal language modeling.
   Chess board evaluation is fundamentally a **spatial reasoning** task where all 64
   squares are observed simultaneously. The sequential RoPE positional encoding
   treats the 8×8 grid as a 1D sequence, losing spatial structure.

3. **Model capacity constrained by VRAM.** The original BDH uses
   `mlp_internal_dim_multiplier=128` (N=8192 per head). We had to reduce to 16
   (N=512 per head) to fit in the RTX 2060's 6 GB VRAM, cutting model capacity ~8×.

4. **Loss plateau.** Val loss plateaued at ~0.110 from epoch 2, close to the
   prediction-by-mean baseline (~0.108 for this data distribution). The model is
   essentially predicting near-zero for most positions.

### Comparison with reference models

| Model          | Params    | Data    | Val MSE | ELO   |
|----------------|-----------|---------|---------|-------|
| ViT-Small      | 2.6M      | 316M    | —       | 1817  |
| ViT-Medium     | 9.5M      | 316M    | —       | 1960  |
| **BDH-Chess**  | **796k**  | **500k**| **0.111** | **N/A** |
| MLP baseline   | 263k      | 50k     | 0.110   | N/A   |

---

## 9. Next Steps

1. **Scale data to 10M+ positions** — the single most impactful improvement.
2. **Add 2D positional encoding** — replace sequential RoPE with learnable
   row/column embeddings or a 2D sinusoidal encoding for the 8×8 grid.
3. **Increase model capacity** — train on a GPU with ≥16 GB VRAM (e.g., RTX 3080/4080,
   A100) to use `mlp_internal_dim_multiplier=64+`.
4. **Replace bidirectional attention with spatial variant** — consider treating the
   board as a 2D grid with local attention patterns (like ViT patches).
5. **Hyperparameter sweep** — systematic search over learning rate, dropout, number
   of layers, embedding dimension.

---

## 10. Artifacts

| Artifact              | Path                                              |
|-----------------------|---------------------------------------------------|
| Config                | `configs/train_bdh.yaml`                          |
| Model code            | `src/models/bdh.py`                               |
| Best checkpoint       | `outputs/bdh-chess-v1/checkpoints/best_model.pt`  |
| Final checkpoint      | `outputs/bdh-chess-v1/checkpoints/final_model.pt` |
| Training data         | `data/processed/{train,val,test}.npz`             |
| Puzzle benchmark      | `data/puzzles/test_puzzles.feather`                |

---

## 11. Reproducibility

```bash
cd project/

# 1. Prepare data
python scripts/prepare_data.py --config configs/train_bdh.yaml

# 2. Train
python scripts/train.py --config configs/train_bdh.yaml

# 3. Evaluate
python scripts/evaluate.py --config configs/train_bdh.yaml \
    --checkpoint outputs/bdh-chess-v1/checkpoints/best_model.pt
```

**Environment:** Python 3.11.9, torch 2.5.1+cu121, NVIDIA RTX 2060, Windows 11
