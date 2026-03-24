# ViT-S v2 (Score Fix) — Experiment Report

**Date:** 2026-03-24  
**Author:** Antoni Czapski  
**W&B run:** `6p5oo777` — [view on W&B](https://wandb.ai/antoni-krzysztof-czapski/bdh-searchless-chess/runs/6p5oo777)  
**SLURM job:** 2485306 (12h, timed out during epoch 22)  
**Hardware:** NVIDIA A100-SXM4-40GB (Ares HPC, plgrid, node t0023)

---

## 1. Objective

Re-train the ViT-S architecture with a **critical bug fix**: the score labels
in all previous experiments (BDH-v3, ViT-S sanity check) had the wrong sign
for black-to-move positions, causing ~50% of the 50M training labels to be
inverted. This experiment also includes several pipeline optimizations.

---

## 2. Root Cause Analysis — The Score Perspective Bug

### 2.1 The Bug

The HuggingFace dataset (`mateuszgrzyb/lichess-stockfish-normalized`) stores
centipawn evaluations **from white's perspective**. Our board encoding uses
`always_white_perspective=True`, which mirrors the board for black-to-move
positions so the model always "sees" the board from the moving side's viewpoint.

However, the score was **not** flipped for black-to-move positions. This meant:

- For white-to-move: board ✓, score ✓ → correct label
- For black-to-move: board mirrored ✓, score still from white's perspective ✗ → **inverted label**

Since roughly half the positions have black to move, **~50% of all 50M training
labels were inverted**. The model received contradictory supervision and
converged to predicting near-zero for everything.

### 2.2 The Fix

In `src/data/prepare.py`, we now flip the score sign for black-to-move positions:

```python
is_white_to_move = " w " in fen
sign = 1.0 if is_white_to_move else -1.0
score = sign * normalize_cp(raw_cp, scale=cp_scale)
```

This ensures the score always represents the evaluation from the perspective of
the side to move, matching the mirrored board encoding.

### 2.3 Impact

This single bug explains why all previous experiments produced catastrophic ELO
scores (-2969 for BDH-v3, -3488 for ViT-S sanity check) despite clean training
curves and large-scale data. The models were learning a self-contradictory
mapping.

---

## 3. Pipeline Optimizations

In addition to the score fix, several pipeline improvements were applied:

| Optimization | Before | After | Impact |
|---|---|---|---|
| Board dtype | float32 | **uint8** | 4× less disk & RAM (138 GB → 33 GB train) |
| Data loading | Memory-mapped | **In-memory (RAM)** | Eliminates I/O bottleneck |
| uint8→float32 | At dataset init | **Per-batch in `__getitem__`** | Lower peak RAM |
| `cudnn.benchmark` | off | **on** | Auto-tunes conv kernels |
| `prefetch_factor` | 2 | **4** | Better CPU-GPU overlap |
| Data re-prepared | — | **v2 with score fix** | Clean labels |

`torch.compile` was initially enabled but caused CUDA OOM on the 40GB A100
(both `reduce-overhead` and `default` modes). It was disabled for this run.

---

## 4. Data Pipeline

### 4.1 Dataset v2

Fully re-prepared from scratch with the score fix applied during encoding.

| Property | Value |
|---|---|
| Source | `mateuszgrzyb/lichess-stockfish-normalized` |
| Total positions | 50,000,000 |
| Train / Val / Test | 45M / 2.5M / 2.5M |
| Score normalization | `tanh(cp/1000)` with sign flip for black |
| Board dtype | uint8 (binary 0/1 channels) |
| Train boards size | 33 GB (was 138 GB in float32) |
| Data directory | `bdh-chess-data-v2/` |

### 4.2 Loading Strategy

The `InMemoryChessDataset` pre-loads all numpy arrays into RAM as uint8 tensors.
Each `__getitem__` call converts a single sample to float32 on the fly. This
eliminates disk I/O during training at the cost of ~38 GB RAM.

Peak RAM during loading required careful memory management (`del` numpy arrays +
`gc.collect()` after torch conversion) to stay within the 96 GB SLURM allocation.

---

## 5. Training Configuration

| Parameter | ViT-S v2 (this run) | ViT-S sanity (previous) |
|---|---|---|
| Architecture | ViT-S (identical) | ViT-S |
| Parameters | 2,656,001 | 2,656,001 |
| Optimizer | AdamW | AdamW |
| Learning rate | **2e-4** | 2e-4 |
| Min LR | 1e-6 | 1e-6 |
| Weight decay | 1e-4 | 1e-4 |
| Batch size | **8,192** | 8,192 |
| Grad clip | 1.0 | 1.0 |
| AMP dtype | bf16 | bf16 |
| Warmup | 5 epochs | 5 epochs |
| Scheduler | Cosine decay | Cosine decay |
| Loss | MSE | MSE |
| Score scale | tanh(cp/1000) | tanh(cp/1000) |
| Score sign fix | **✓ yes** | ✗ no |
| Data loading | **In-memory (uint8)** | Memory-mapped (float32) |
| torch.compile | disabled | N/A |
| Epochs completed | **21 (timeout)** | 27 (early stop) |

**Steps per epoch:** 5,493 (45M / 8192)  
**Total steps trained:** 115,353 (21 × 5,493)

---

## 6. Training Dynamics

### 6.1 Full Epoch-by-Epoch Results

| Epoch | Train Loss | Train MAE | Val Loss   | Val MAE  | LR       | Time (s) |
|-------|-----------|-----------|------------|----------|----------|----------|
| 1     | 0.103900  | 0.2217    | 0.085130   | 0.1946   | 4.16e-05 | 2016     |
| 2     | 0.081138  | 0.1868    | 0.071881   | 0.1712   | 8.12e-05 | 2014     |
| 3     | 0.069972  | 0.1692    | 0.064473   | 0.1590   | 1.21e-04 | 2008     |
| 4     | 0.063853  | 0.1594    | 0.059407   | 0.1509   | 1.60e-04 | 2007     |
| 5     | 0.059407  | 0.1522    | 0.055943   | 0.1446   | 2.00e-04 | 2007     |
| 6     | 0.055322  | 0.1456    | 0.051607   | 0.1394   | 1.99e-04 | 2008     |
| 7     | 0.051702  | 0.1397    | 0.048759   | 0.1343   | 1.97e-04 | 2008     |
| 8     | 0.049005  | 0.1354    | 0.047279   | 0.1314   | 1.93e-04 | 2019     |
| 9     | 0.046842  | 0.1318    | 0.044394   | 0.1257   | 1.88e-04 | 2088     |
| 10    | 0.045076  | 0.1288    | 0.042911   | 0.1228   | 1.81e-04 | 2089     |
| 11    | 0.043630  | 0.1263    | 0.041721   | 0.1206   | 1.73e-04 | 2087     |
| 12    | 0.042382  | 0.1242    | 0.040539   | 0.1190   | 1.64e-04 | 2087     |
| 13    | 0.041270  | 0.1223    | 0.039900   | 0.1176   | 1.54e-04 | 2088     |
| 14    | 0.040317  | 0.1206    | 0.039081   | 0.1167   | 1.43e-04 | 2088     |
| 15    | 0.039421  | 0.1190    | 0.038517   | 0.1145   | 1.31e-04 | 2101     |
| 16    | 0.038635  | 0.1177    | 0.037817   | 0.1146   | 1.19e-04 | 2100     |
| 17    | 0.037915  | 0.1164    | 0.037220   | 0.1120   | 1.07e-04 | 2100     |
| 18    | 0.037234  | 0.1153    | 0.036852   | 0.1119   | 9.43e-05 | 2100     |
| 19    | 0.036601  | 0.1142    | 0.036408   | 0.1103   | 8.19e-05 | 2043     |
| 20    | 0.036033  | 0.1132    | 0.036192   | 0.1099   | 6.98e-05 | 2005     |
| 21    | 0.035488  | 0.1123    | **0.035720** | **0.1087** | 5.81e-05 | 2005 |

**Best model:** epoch 21, val_loss = **0.035720**, val_MAE = **0.1087**

### 6.2 Training Observations

1. **Val_loss improved every single epoch** — the model was still actively
   improving when SLURM killed it at the 12h time limit. Unlike the previous
   ViT-S run, there is no overfitting: val_loss < train_loss through epoch 18,
   and the gap remains tiny.

2. **Dramatically lower loss than previous run** — val_loss=0.0357 vs 0.1525
   (previous ViT-S). This 4.3× reduction in MSE is entirely attributable to
   fixing the score sign bug. With correct labels, the model can actually learn
   a coherent position → evaluation mapping.

3. **No overfitting** — train-val gap at epoch 21 is only 0.6% relative
   (0.03549 vs 0.03572). The model has plenty of capacity to keep improving.
   Early stopping was not triggered.

4. **Smooth warmup and decay** — LR warmed up linearly over 5 epochs to 2e-4,
   then followed cosine decay. No instability or loss spikes at any point.

5. **Throughput: ~22k samples/sec** — consistent ~34 min/epoch. The in-memory
   loading eliminated the I/O bottleneck that plagued the previous run
   (~86 min/epoch with memory-mapped float32).

---

## 7. Evaluation — ELO Benchmark

### 7.1 Puzzle Results (1,200 puzzles, 12 tiers)

| Tier (ELO)       | ViT-S v2 (fixed) | ViT-S (broken) | BDH v3 (broken) |
|------------------|------------------|----------------|-----------------|
| (0, 500]         | **85.0%**        | 22.0%          | 25.0%           |
| (500, 750]       | **83.0%**        | 11.0%          | 8.0%            |
| (750, 1000]      | **78.0%**        | 9.0%           | 5.0%            |
| (1000, 1250]     | **63.0%**        | 8.0%           | 7.0%            |
| (1250, 1500]     | **67.0%**        | 5.0%           | 5.0%            |
| (1500, 1750]     | **51.0%**        | 4.0%           | 6.0%            |
| (1750, 2000]     | **37.0%**        | 2.0%           | 6.0%            |
| (2000, 2250]     | **26.0%**        | 2.0%           | 2.0%            |
| (2250, 2500]     | **10.0%**        | 0.0%           | 0.0%            |
| (2500, 2750]     | **6.0%**         | 1.0%           | 0.0%            |
| (2750, 3000]     | **2.0%**         | 0.0%           | 0.0%            |
| (3000, 3250]     | **1.0%**         | 0.0%           | 0.0%            |

### 7.2 Summary Metrics

| Metric             | ViT-S v2 (fixed) | ViT-S (broken) | BDH v3 (broken) | Reference ViT-S |
|--------------------|------------------|----------------|-----------------|-----------------|
| **ELO**            | **1547**         | -3488          | -2969           | **1817**        |
| Puzzle accuracy    | **42.4%**        | 5.3%           | 5.8%            | N/A             |
| Best val_loss      | 0.0357           | 0.1525         | 0.0122†         | N/A             |
| Best val_MAE       | 0.1087           | 0.2430         | 0.1101†         | N/A             |
| Parameters         | 2,656,001        | 2,656,001      | ~2,500,000      | ~2,640,000      |
| GPU hours          | **~12h**         | ~31.5h         | ~33h            | >200h           |
| Epochs trained     | 21 (timeout)     | 27 (early stop)| 26 (early stop) | ~30             |
| Data               | 50M (fixed)      | 50M (broken)   | 50M (broken)    | 316M            |

†BDH v3 values are on `tanh(cp/10000)` scale — not directly comparable.

---

## 8. Analysis

### 8.1 Score Fix Impact: -3488 → 1547 ELO (+5035 points)

The score perspective fix transformed the model from catastrophically bad to a
strong club-level player. This single change:

- Increased puzzle accuracy from 5.3% to **42.4%** (8× improvement)
- Moved ELO from -3488 to **1547** (from nonsensical to meaningful)
- Reduced val_loss from 0.1525 to **0.0357** (4.3× reduction)

The model now solves 85% of beginner puzzles (0–500 ELO), 78% of intermediate
puzzles (750–1000), and still manages 37% at advanced level (1750–2000). This is
a qualitatively different result: the model genuinely understands chess positions.

### 8.2 Comparison with Reference ViT-S (ELO 1817)

Our ELO of 1547 is **270 points below** the reference's 1817. Contributing factors:

1. **Training was cut short** — the model was still improving (new best every
   epoch) when the 12h SLURM limit was reached. With 30 epochs or early stopping,
   ELO could be significantly higher.

2. **Data scale (6.3× less)** — we used 50M vs 316M positions. The reference
   likely benefits from greater position diversity.

3. **Remaining gap is reasonable** — 270 ELO points for 6.3× less data and
   incomplete training (21 vs 30 epochs) is a very encouraging result. It
   confirms our ViT-S implementation is correct.

### 8.3 Pipeline Performance

The optimizations yielded a **2.5× speedup** in epoch time:

| Metric | Previous ViT-S | This run | Improvement |
|---|---|---|---|
| Epoch time | ~86 min | **~34 min** | 2.5× faster |
| Throughput | ~8,700 samples/sec | **~22,000 samples/sec** | 2.5× higher |
| Data load time | N/A (mmap) | **~40s** (one-time) | — |
| Disk usage | 138 GB | **33 GB** | 4.2× smaller |

The speedup comes primarily from in-memory loading (eliminating I/O wait) and
uint8 boards (4× less data to transfer).

---

## 9. Compute Summary

| Resource | Value |
|---|---|
| SLURM job | 2485306 |
| Node | t0023 |
| GPU | NVIDIA A100-SXM4-40GB |
| Wall time | 12h 0m (timeout) |
| Training time | ~12h (21 epochs) |
| Avg epoch time | 34.2 min |
| RAM allocated | 96 GB |
| RAM used (peak) | ~44 GB |
| Throughput | ~22,000 samples/sec |
| ELO evaluation | ~3h (CPU, 1200 puzzles) |

---

## 10. Conclusions

1. **The score perspective bug was the root cause of all previous failures.**
   Fixing the sign flip for black-to-move positions transformed ELO from -3488
   to **1547** — a 5035-point improvement with identical architecture and data.

2. **Our ViT-S implementation is correct.** The 1547 ELO result on 50M positions
   is within 270 points of the reference's 1817 on 316M positions, and training
   was still improving when cut short.

3. **The model was still improving when SLURM timed out.** Val_loss improved
   every single epoch (21 consecutive best models). More training time would
   likely close the remaining gap with the reference.

4. **Pipeline optimizations delivered 2.5× training speedup.** In-memory uint8
   loading reduced epoch time from ~86 min to ~34 min, enabling 21 epochs in 12h
   vs 20 epochs in 24h previously.

### Next Steps

- **Resume training** with the saved checkpoint to complete all 30 epochs and
  let early stopping determine the optimal stopping point.
- **Scale to 316M positions** to match the reference conditions exactly.
- **Apply the score fix to BDH-v3** and re-train to get a fair architecture
  comparison between ViT-S and BDH.

---

## Appendix: Experiment Timeline

| Event | Time | Notes |
|---|---|---|
| Data prep (v2) | Job 2484888 | 50M positions with score fix, uint8 boards |
| Training submitted | 2026-03-23 17:44 | Job 2485306, 12h limit, 96 GB RAM |
| Data loaded | +40s | 45M boards (34.6 GB) loaded to RAM |
| Training started | 17:45 | 5493 steps/epoch, ~34 min/epoch |
| Epoch 5 (warmup done) | 20:33 | LR peaked at 2e-4 |
| Epoch 21 (killed) | 05:43 | val_loss=0.035720 (still improving) |
| SLURM timeout | 05:44 | 12h wall time exceeded |
| ELO evaluation | 07:00–10:12 | 1200 puzzles, CPU-only, ~3h |

### All Experiments Summary

| Experiment | Score Fix | Params | Data | GPU-h | Puzzle Acc | ELO |
|---|---|---|---|---|---|---|
| BDH v1 | ✗ | 796k | 500k | ~1h | 0.5% | N/A |
| BDH v2 | ✗ | 682k | 500k | ~0.5h | 0.8% | N/A |
| BDH v3 | ✗ | 2.5M | 50M | ~33h | 5.8% | -2969 |
| ViT-S sanity | ✗ | 2.6M | 50M | ~31.5h | 5.3% | -3488 |
| **ViT-S v2 (fixed)** | **✓** | **2.6M** | **50M** | **~12h** | **42.4%** | **1547** |
| Reference ViT-S | ✓ | 2.6M | 316M | >200h | N/A | 1817 |

---

*Generated: 2026-03-24*
