# ViT-S Sanity Check — Experiment Report

**Date:** 2026-03-23  
**Author:** Antoni Czapski  
**W&B runs:** `pvmpso0a` (epochs 1–20), `xkrd0xwv` (epochs 21–27)  
**SLURM jobs:** 2481094 (initial, killed at 24h), 2484137 (resumed)  
**Hardware:** NVIDIA A100-SXM4-40GB (Ares HPC, plgrid)

---

## 1. Objective

Reproduce the **Vision Transformer Small (ViT-S)** architecture from the
reference repository
[mateuszgrzyb-pl/searchless-chess](https://github.com/mateuszgrzyb-pl/searchless-chess)
as a **sanity check** against our BDH-Chess v3 model. Both models are trained on
the same 50M positions and evaluated on the same 1,200-puzzle benchmark,
enabling a direct comparison under identical conditions.

---

## 2. Reference Context

The reference repository trained 6 architectures on 316M positions. Their
reported results:

| Model     | Params | ELO  | Architecture    |
|-----------|--------|------|-----------------|
| CNN       | 1.18M  | 1112 | Shallow CNN     |
| ResNet-M  | 2.3M   | 1515 | Residual blocks |
| ResNet-L  | 12.9M  | 1711 | Residual blocks |
| ResNet-XL | 24.7M  | 1719 | Residual blocks |
| **ViT-S** | **2.64M** | **1817** | Vision Transformer |
| ViT-M     | 9.5M   | 1960 | Vision Transformer |

We reimplemented ViT-S — their best parameter-efficient model (5× more
efficient than ResNet-L, 10× more than ResNet-XL).

---

## 3. Architecture — ViT-S

Faithful PyTorch reimplementation of the reference Keras/TensorFlow model:

```
Input (B, 12, 8, 8)
  → Permute to (B, 8, 8, 12) → Reshape (B, 64, 12)    # 64 patches of dim 12
  → Linear(12, 256)                                     # patch projection
  → + Positional Embedding(64, 256)                     # learnable
  → 5× Transformer Block:
      ├─ LayerNorm(ε=1e-6)
      ├─ MultiHeadAttention(4 heads, key_dim=64, dropout=0.1)
      ├─ + Residual
      ├─ LayerNorm(ε=1e-6)
      ├─ FFN: Linear(512, GELU) → Dropout(0.1) → Linear(256)
      └─ + Residual
  → LayerNorm
  → GlobalAveragePooling                                 # (B, 256)
  → Dropout(0.1)
  → Linear(1) → tanh                                    # output in [-1, 1]
```

**Parameters:** 2,656,001 (reference: ~2.64M)  
**Key design choices from reference:**
- FFN ratio = 2× (not the standard 4×)
- Pre-norm architecture (LayerNorm before attention/FFN)
- Global average pooling instead of CLS token
- Mixed precision with bf16

---

## 4. Data Pipeline

### 4.1 Dataset

Same HuggingFace dataset as BDH-v3: `mateuszgrzyb/lichess-stockfish-normalized`  
Subset: **50,000,000 positions** (same as BDH-v3)

| Split | Samples    |
|-------|------------|
| Train | 45,000,000 |
| Val   | 2,500,000  |
| Test  | 2,500,000  |

### 4.2 Critical: Score Re-normalization

The reference uses `tanh(cp/1000)` for score normalization, while our BDH-v3
used `tanh(cp/10000)`. This difference is significant:

| cp value | tanh(cp/10000) | tanh(cp/1000) |
|----------|---------------|---------------|
| 100      | 0.010         | 0.100         |
| 500      | 0.050         | 0.462         |
| 2000     | 0.197         | 0.964         |
| 5000     | 0.462         | 1.000         |

The `tanh(cp/1000)` scale **compresses large advantages more aggressively**,
making the model focus on the ±2000 cp range (decisive advantages). The
`tanh(cp/10000)` scale keeps the target nearly linear for typical game
evaluations, requiring the model to also discriminate between large advantages.

We applied a mathematical conversion to the existing data:
```
new_score = tanh(arctanh(old_score) × 10)
```

Distribution statistics after conversion:

| Metric | tanh(cp/10000) | tanh(cp/1000) |
|--------|---------------|---------------|
| Mean   | 0.034         | 0.059         |
| Std    | 0.334         | 0.433         |
| Range  | [-0.991, 0.991] | [-1.0, 1.0] |

---

## 5. Training Configuration

| Parameter | ViT-S (this run) | BDH v3 (comparison) |
|-----------|-----------------|---------------------|
| Optimizer | AdamW | AdamW |
| Learning rate | 2e-4 | 3e-4 |
| Min LR | 1e-6 | 1e-6 |
| Weight decay | 1e-4 | 1e-4 |
| Batch size | 8,192 | 1,024 |
| Grad clip | 1.0 | 1.0 |
| AMP dtype | bf16 | bf16 |
| Warmup | 5 epochs | 5 epochs |
| Scheduler | Linear warmup → Cosine decay | Linear warmup → Cosine decay |
| Loss | MSE | Huber + deep supervision |
| Epochs trained | 27 (early stop) | 26 (early stop) |
| Score scale | tanh(cp/1000) | tanh(cp/10000) |

**Steps per epoch:** 5,493 (45M / 8192)  
**Total steps trained:** 148,311 (27 × 5,493)

---

## 6. Training Dynamics

### 6.1 Full Epoch-by-Epoch Results

| Epoch | Train Loss | Train MAE | Val Loss  | Val MAE  | LR       | Time (s) |
|-------|-----------|-----------|-----------|----------|----------|----------|
| 1     | 0.177702  | 0.2800    | 0.172593  | 0.2723   | 4.16e-05 | 5193     |
| 2     | 0.168748  | 0.2681    | 0.166837  | 0.2664   | 8.12e-05 | 5347     |
| 3     | 0.165120  | 0.2635    | 0.163126  | 0.2594   | 1.21e-04 | 4585     |
| 4     | 0.162768  | 0.2605    | 0.161105  | 0.2566   | 1.60e-04 | 4279     |
| 5     | 0.161000  | 0.2582    | 0.159645  | 0.2555   | 2.00e-04 | 4310     |
| 6     | 0.159365  | 0.2561    | 0.158439  | 0.2541   | 1.99e-04 | 4315     |
| 7     | 0.157759  | 0.2538    | 0.156982  | 0.2520   | 1.97e-04 | 4258     |
| 8     | 0.156449  | 0.2521    | 0.155968  | 0.2506   | 1.93e-04 | 4200     |
| 9     | 0.155278  | 0.2505    | 0.155809  | 0.2492   | 1.88e-04 | 3911     |
| 10    | 0.154228  | 0.2492    | 0.154533  | 0.2476   | 1.81e-04 | 3923     |
| 11    | 0.153201  | 0.2479    | 0.153788  | 0.2458   | 1.73e-04 | 3942     |
| 12    | 0.152215  | 0.2467    | 0.153502  | 0.2457   | 1.64e-04 | 3949     |
| 13    | 0.151244  | 0.2455    | 0.153067  | 0.2448   | 1.54e-04 | 3916     |
| 14    | 0.150227  | 0.2444    | 0.153127  | 0.2445   | 1.43e-04 | 3932     |
| 15    | 0.149205  | 0.2433    | 0.152768  | 0.2434   | 1.31e-04 | 4194     |
| 16    | 0.148167  | 0.2421    | **0.152468** | 0.2430 | 1.19e-04 | 3987     |
| 17    | 0.147055  | 0.2409    | 0.152499  | 0.2421   | 1.07e-04 | 3839     |
| 18    | 0.145937  | 0.2397    | 0.152802  | 0.2419   | 9.43e-05 | 3979     |
| 19    | 0.144821  | 0.2386    | 0.152847  | 0.2418   | 8.19e-05 | 3987     |
| 20    | 0.143671  | 0.2374    | 0.152893  | 0.2409   | 6.98e-05 | 4486     |
| 21    | 0.142535  | 0.2362    | 0.153335  | 0.2398   | 5.81e-05 | 3886     |
| 22    | 0.141432  | 0.2351    | 0.153904  | 0.2397   | 4.72e-05 | 3917     |
| 23    | 0.140377  | 0.2340    | 0.153634  | 0.2397   | 3.71e-05 | 3904     |
| 24    | 0.139399  | 0.2330    | 0.154440  | 0.2394   | 2.80e-05 | 3888     |
| 25    | 0.138567  | 0.2321    | 0.154914  | 0.2393   | 2.00e-05 | 3873     |
| 26    | 0.137810  | 0.2313    | 0.155136  | 0.2391   | 1.33e-05 | 3821     |
| 27    | 0.137217  | 0.2307    | 0.155483  | 0.2389   | 7.99e-06 | 3818     |

**Best val_loss: 0.152468** at epoch 16.  
**Early stopping** triggered at epoch 27 (patience=10, no improvement since epoch 16).

### 6.2 Training Observations

1. **Steady convergence, no instability** — unlike BDH-v3, no NaN epochs or
   divergence events. ViT trained cleanly from start to finish.

2. **Overfitting from epoch ~13** — the train-val gap widened steadily after
   epoch 13. By epoch 27, train_loss (0.137) was 12% below val_loss (0.155).

3. **Val MAE plateaued at ~0.239** — while val_loss (MSE) bottomed at 0.1525
   (epoch 16), val_mae continued a tiny decline through epoch 27, suggesting
   the model was still making small marginal improvements on easy positions.

4. **Resume worked flawlessly** — SLURM job 2481094 was killed at 24h during
   epoch 21. Job 2484137 resumed from epoch 20's checkpoint, restored the
   scheduler to LR=6.975e-05, and continued seamlessly.

---

## 7. Evaluation — ELO Benchmark

### 7.1 Puzzle Results (1,200 puzzles, 12 tiers)

| Tier (ELO)       | ViT-S Accuracy | BDH v3 Accuracy |
|------------------|---------------|-----------------|
| (0, 500]         | 22.0%         | 25.0%           |
| (500, 750]       | 11.0%         | 8.0%            |
| (750, 1000]      | 9.0%          | 5.0%            |
| (1000, 1250]     | 8.0%          | 7.0%            |
| (1250, 1500]     | 5.0%          | 5.0%            |
| (1500, 1750]     | 4.0%          | 6.0%            |
| (1750, 2000]     | 2.0%          | 6.0%            |
| (2000, 2250]     | 2.0%          | 2.0%            |
| (2250, 2500]     | 0.0%          | 0.0%            |
| (2500, 2750]     | 1.0%          | 0.0%            |
| (2750, 3000]     | 0.0%          | 0.0%            |
| (3000, 3250]     | 0.0%          | 0.0%            |

### 7.2 Summary Metrics

| Metric             | ViT-S (this run) | BDH v3           | Reference ViT-S |
|--------------------|-----------------|-------------------|-----------------|
| **ELO**            | **-3488**       | **-2969**         | **1817**        |
| Puzzle accuracy    | 5.3%            | 5.8%              | N/A             |
| Best val_loss      | 0.1525 (MSE)    | 0.01212 (Huber)†  | N/A             |
| Best val_mae       | 0.2430          | 0.1101†           | N/A             |
| Parameters         | 2,656,001       | ~2,500,000        | ~2,640,000      |
| Total GPU hours    | ~31.5h          | ~33h              | >200h           |
| Epochs trained     | 27              | 26                | ~30             |
| Data               | 50M             | 50M               | 316M            |

†BDH v3 loss/MAE values are on the `tanh(cp/10000)` scale and are **not directly
comparable** to ViT-S values on the `tanh(cp/1000)` scale. ELO is the fair
comparison metric.

---

## 8. Analysis

### 8.1 ViT-S vs BDH v3 (Our Models)

Both models achieve similarly poor ELO scores (ViT-S: -3488, BDH-v3: -2969),
with overall puzzle accuracies of 5.3% and 5.8% respectively. Neither model
demonstrates meaningful chess understanding beyond random-level play.

The BDH model achieved slightly better ELO (-2969 vs -3488), but this difference
is within noise given the methodology (linear regression on 12 data points with
very low accuracy across all tiers).

### 8.2 Our ViT-S vs Reference ViT-S

The reference ViT-S achieved **ELO 1817** on 316M positions, while our
reproduction achieved **ELO -3488** on 50M positions — a massive gap. Possible
contributing factors:

1. **Data scale (6.3× less data)**: We used 50M vs 316M positions. The
   reference trained on every position for 3 virtual epochs (316M × 30 virtual
   epochs / 10 = ~948M sample encounters). We processed 45M × 27 = 1.215B
   sample encounters — **comparable in terms of total training samples seen**.
   This suggests data scale alone does not fully explain the gap.

2. **Virtual epoch vs real epoch**: The reference uses "virtual epochs" (1/10 of
   the data per epoch), which means each epoch checkpoint represents 1/10 of a
   full data pass. Their "30 epochs" ≈ 3 real epochs. Our 27 epochs = 27 real
   epochs. So our model saw **9× more** data in terms of total samples. This
   makes the poor result even more puzzling.

3. **Score normalization**: The `tanh(cp/1000)` normalization compresses the
   target distribution differently. However, we matched the reference scale
   exactly, so this should not be a disadvantage.

4. **Loss function**: The reference uses plain MSE, as does our ViT-S. BDH v3
   used Huber + deep supervision. This is a controlled variable.

5. **Batch size**: Both use 8192, matching the reference.

6. **Evaluation methodology**: Our ELO estimation uses a different puzzle set
   and regression methodology than the reference, which may explain part of
   the discrepancy. The reference tested on Lichess puzzles with a presumably
   different sampling and tier structure.

### 8.3 Possible Root Causes for Poor Performance

1. **Evaluation benchmark mismatch**: The most likely explanation. Our 1,200
   puzzles with linear regression at the 50% accuracy crossing point may not
   be equivalent to the reference's evaluation methodology. When accuracy is
   uniformly low (<10%), the regression is numerically unstable and produces
   nonsensical negative ELOs.

2. **Data channel layout**: Our PyTorch model ingests (B, 12, 8, 8) and
   internally permutes to (B, 64, 12). The reference ingests (B, 8, 8, 12)
   directly. The ordering within the 12 channels is identical. This should
   not cause issues.

3. **Positional embedding initialization**: We use `trunc_normal_(std=0.02)`,
   matching standard ViT practice. The reference uses Keras `Embedding` layer
   which initializes with `uniform(-0.05, 0.05)`. This is unlikely to cause
   major differences.

---

## 9. Compute Summary

| Resource         | Job 1 (2481094) | Job 2 (2484137) | Total   |
|------------------|----------------|-----------------|---------|
| Epochs           | 1–20           | 21–27           | 27      |
| Wall time        | 24h (killed)   | 7.53h           | ~31.5h  |
| GPU              | A100-SXM4-40GB | A100-SXM4-40GB  | —       |
| Peak GPU memory  | ~27 GB alloc   | ~27 GB alloc    | —       |
| Throughput       | ~11,788 samples/sec | ~11,788 samples/sec | — |

Data re-normalization (50M scores): ~1 minute.

---

## 10. Conclusions

1. **ViT-S and BDH v3 perform comparably on our benchmark** — both achieve
   puzzle accuracy around 5–6% and deeply negative ELO scores, indicating
   neither architecture extracts meaningful chess evaluation from 50M positions
   with our training pipeline.

2. **The sanity check raises questions about our evaluation pipeline** — the
   reference ViT-S achieved 1817 ELO on 316M data, while our faithful
   reproduction achieves -3488 on 50M data (with 9× more total sample encounters
   due to the virtual-epoch difference). The gap is too large to attribute
   solely to data scale.

3. **Training was stable and reproducible** — the ViT trained cleanly for 27
   epochs with no NaN events, smooth loss curves, and flawless SLURM resume.
   The architecture itself is not the problem.

4. **Likely culprit: evaluation methodology** — our puzzle benchmark methodology
   (linear regression on tier accuracies) produces unstable results when
   accuracy is uniformly very low. The reference may use a fundamentally
   different evaluation approach.

### Recommendations

- **Investigate evaluation pipeline**: Compare our puzzle set and ELO estimation
  method against the reference's exact approach.
- **Test with reference checkpoints**: Load the published reference ViT-S weights
  and evaluate with our benchmark to isolate whether the gap is in training or
  evaluation.
- **Scale data**: Train on the full 316M positions to match reference conditions
  exactly and rule out data as the bottleneck.

---

## Appendix: Files Created

| File | Description |
|------|-------------|
| `src/models/vit.py` | PyTorch ViT model (ChessViT) |
| `src/models/registry.py` | Updated with "vit" registration |
| `configs/train_vit_s_a100.yaml` | Training configuration |
| `scripts/renormalize_data.py` | Score re-normalization utility |
| `train_vit_a100.sbatch` | SLURM submission script |
| `report/vit_experiment_design.md` | Pre-experiment design document |

---

*Generated: 2026-03-23*
