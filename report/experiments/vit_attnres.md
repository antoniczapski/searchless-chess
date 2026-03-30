# ViT-S + Attention Residuals — Experiment Report

**Date:** 2026-03-28  
**Model:** `vit_attnres` (ViT-S with Windowed Attention Residuals)  
**Best val_loss:** 0.033787 (epoch 14)  
**Estimated ELO:** **1729** | Puzzle accuracy: **49.3%**  
**W&B run:** `ez54qrk9` (session 1), resumed in session 2  
**SLURM jobs:** 2491255 (epochs 1–7, 12h timeout), 2491801 (epochs 8–14, 12h timeout)  

---

## 1. Motivation

ViT-S v2 achieved ELO 1621 with a standard Vision Transformer (residual connections
via running sum). The "Attention Residuals" paper (Kimi Team, 2025) proposes replacing
fixed residual accumulation with a **learned, depth-wise softmax retrieval**: each
sublayer receives a weighted mixture of previous sublayer outputs, allowing the model
to dynamically select which layer's representation to build upon.

We hypothesized that this mechanism, combined with SwiGLU FFN and RMSNorm, could
improve evaluation quality without increasing model size.

---

## 2. Architectural Changes (vs Base ViT-S v2)

| Component          | ViT-S v2 (base)                 | ViT-S + AttnRes               |
|--------------------|---------------------------------|-------------------------------|
| Residual connection | `x = x + sublayer(norm(x))`   | Windowed AttnRes (W=3)        |
| FFN activation     | GELU (2 projections)            | **SwiGLU** (3 projections)    |
| Normalization      | LayerNorm                       | **RMSNorm**                   |
| Depth mixing       | None (fixed running sum)        | **Learned softmax** per sublayer |
| Parameters         | 2,656,001                       | **2,664,705** (+0.3%)         |

### 2.1 Windowed Attention Residuals

Each sublayer (attention and MLP) receives input from a softmax-weighted mixture
of the **last W=3 depth sources** rather than a fixed running sum. Each mixing point
has a learned pseudo-query (zero-initialized for uniform start) and RMSNorm on
depth keys before scoring.

With 5 transformer blocks × 2 sublayers = 10 AttnRes mixing points, the model
has 10 learned queries that control information routing across layers.

**Memory optimization:** Sources outside the window are discarded (not just detached)
to bound GPU memory at O(W) rather than O(depth). This was necessary because
Full AttnRes (all 11 sources) caused CUDA OOM on A100-40GB even at batch 4096.

### 2.2 SwiGLU FFN

Replaces GELU FFN with a gated linear unit using SiLU activation. Uses 3 projection
matrices (gate, up, down) with hidden dim adjusted to `2/3 × dim × ffn_ratio`
(rounded to multiple of 8) for parameter parity with the 2-projection GELU FFN.

### 2.3 RMSNorm

Replaces LayerNorm everywhere — lighter computation (no mean subtraction, no bias),
consistent with modern transformer practice.

---

## 3. Configuration

| Parameter          | ViT-S v2         | ViT-S + AttnRes   |
|--------------------|------------------|--------------------|
| Architecture       | `vit`            | `vit_attnres`      |
| projection_dim     | 256              | 256                |
| num_heads          | 4                | 4                  |
| transformer_layers | 5                | 5                  |
| ffn_ratio          | 2                | 2                  |
| dropout_rate       | 0.1              | 0.1                |
| Parameters         | 2,656,001        | 2,664,705          |
| batch_size         | 8192             | 4096               |
| learning_rate      | 2e-4             | 2e-4               |
| warmup             | 5 epochs         | 5 epochs           |
| Epochs completed   | **30 (full)**    | **14 (2× timeout)**|
| Data               | 50M positions    | 50M positions      |
| cp_scale           | 1000             | 1000               |
| AMP                | bf16             | bf16               |
| Loss               | MSE              | MSE                |

---

## 4. Training History

### 4.1 Session 1: Job 2491255 (t0011, epochs 1–7)

Started: 2026-03-26 21:42 | Killed: 2026-03-27 09:42 (12h timeout)

### 4.2 Session 2: Job 2491801 (t0037, epochs 8–14)

Started: 2026-03-27 12:41 | Killed: 2026-03-28 00:41 (12h timeout)

### 4.3 Full Epoch-by-Epoch Results

| Epoch | Train Loss | Train MAE | Val Loss   | Val MAE  | LR       | Time (s) |
|-------|-----------|-----------|------------|----------|----------|----------|
| 1     | 0.092753  | 0.2052    | 0.077071   | 0.1805   | 4.16e-05 | 6056     |
| 2     | 0.068443  | 0.1661    | 0.062417   | 0.1546   | 8.12e-05 | 6052     |
| 3     | 0.058181  | 0.1496    | 0.054016   | 0.1427   | 1.21e-04 | 6052     |
| 4     | 0.052451  | 0.1404    | 0.049818   | 0.1341   | 1.60e-04 | 6052     |
| 5     | 0.048717  | 0.1344    | 0.046083   | 0.1305   | 2.00e-04 | 6052     |
| 6     | 0.045448  | 0.1289    | 0.042604   | 0.1236   | 2.00e-04 | 6052     |
| 7     | 0.042665  | 0.1241    | 0.040491   | 0.1193   | 1.98e-04 | 6051     |
| 8     | 0.040557  | 0.1205    | 0.038315   | 0.1151   | 1.96e-04 | 6092     |
| 9     | 0.038924  | 0.1175    | 0.037764   | 0.1143   | 1.94e-04 | 6081     |
| 10    | 0.037619  | 0.1152    | 0.036024   | 0.1104   | 1.90e-04 | 6088     |
| 11    | 0.036512  | 0.1132    | 0.035472   | 0.1093   | 1.86e-04 | 6094     |
| 12    | 0.035577  | 0.1116    | 0.034856   | 0.1099   | 1.81e-04 | 6091     |
| 13    | 0.034729  | 0.1101    | 0.034304   | 0.1073   | 1.75e-04 | 6092     |
| 14    | 0.033970  | 0.1088    | **0.033787**| 0.1062   | 1.69e-04 | 6096     |

**Best model:** epoch 14, val_loss = **0.033787**, val_MAE = **0.1062**

**Key observations:**
1. **val_loss improved every single epoch** — no plateaus. The model was still
   improving when training was terminated by the wall-clock limit.
2. **~101 min/epoch** at batch_size=4096 (vs ~34 min/epoch for base ViT-S at 8192).
   The 3× slowdown comes from half batch size (2×) plus AttnRes/SwiGLU overhead.
3. **Total GPU time:** ~23.6 hours across 2 sessions (14 epochs × ~101 min).
4. **Minimal overfitting** — train-val gap at epoch 14 is only 0.5% relative
   (0.03397 vs 0.03379). Val_loss < train_loss still possible with more epochs.
5. **val_loss=0.033787 already below ViT-S v2's best of 0.034701** at only 14 epochs
   (vs ViT-S v2's 27 epochs to reach that level).

---

## 5. Training Challenges

### 5.1 CUDA OOM with Full Attention Residuals

The original Full AttnRes implementation kept all 11 depth source tensors
(embedding + 2 per block × 5 blocks) alive in the computation graph. At batch
sizes of 4096–8192, this consumed 30–40 GB of VRAM, causing CUDA OOM on the
A100-40GB.

**Solution:** Switched to Windowed AttnRes (W=3) — each sublayer only mixes the
last 3 depth sources. Older sources are discarded entirely from the sources list
(not just detached). This bounds memory at O(W) regardless of depth. Batch 4096
with windowed AttnRes fits comfortably in ~25 GB.

### 5.2 Batch Size Reduction

Due to the memory overhead of AttnRes mixing (stacking W=3 tensors, running
key_norm + softmax) and SwiGLU's 3-projection FFN, batch_size was halved from
8192 to 4096. This doubled steps/epoch from 5,493 to 10,986 and roughly tripled
epoch time (from ~34 min to ~101 min). With the 12h SLURM limit, only 7 epochs
could be completed per session.

---

## 6. Evaluation — ELO Benchmark

### 6.1 Puzzle Results (1,200 puzzles, 12 tiers)

| Tier (ELO)        | ViT-S + AttnRes | ViT-S v2 (base) | BDH v4       | Reference |
|--------------------|-----------------|------------------|--------------|-----------|
| (0, 500]           | **87.0%**       | 87.0%            | 82.0%        | —         |
| (500, 750]         | **97.0%**       | 93.0%            | 66.0%        | —         |
| (750, 1000]        | **88.0%**       | 77.0%            | 60.0%        | —         |
| (1000, 1250]       | **73.0%**       | 64.0%            | 42.0%        | —         |
| (1250, 1500]       | **79.0%**       | 68.0%            | 43.0%        | —         |
| (1500, 1750]       | **64.0%**       | 49.0%            | 29.0%        | —         |
| (1750, 2000]       | **46.0%**       | 37.0%            | 17.0%        | —         |
| (2000, 2250]       | **31.0%**       | 25.0%            | 12.0%        | —         |
| (2250, 2500]       | 13.0%           | **17.0%**        | 10.0%        | —         |
| (2500, 2750]       | **9.0%**        | 6.0%             | 4.0%         | —         |
| (2750, 3000]       | 2.0%            | **4.0%**         | 1.0%         | —         |
| (3000, 3250]       | **3.0%**        | 2.0%             | 6.0%         | —         |
| **ELO**            | **1729**        | 1621             | 1177         | **1817**  |
| Puzzle accuracy    | **49.3%**       | 45.1%            | 31.0%        | N/A       |

### 6.2 Comparison

| Metric              | ViT-S + AttnRes | ViT-S v2 | Δ (AttnRes − v2) |
|----------------------|-----------------|----------|-------------------|
| ELO                  | **1729**        | 1621     | **+108**          |
| Puzzle accuracy      | **49.3%**       | 45.1%    | **+4.2 pp**       |
| Best val_loss        | **0.033787**    | 0.034701 | **−0.000914**     |
| Best val_MAE         | **0.1062**      | 0.1064   | **−0.0002**       |
| Epochs trained       | 14              | 30       | −16               |
| GPU hours            | ~23.6h          | ~17h     | +6.6h             |
| Parameters           | 2,664,705       | 2,656,001| +8,704 (+0.3%)    |

---

## 7. Analysis

### 7.1 AttnRes Improvements: +108 ELO with 0.3% More Parameters

The Windowed Attention Residuals architecture achieves **1729 ELO** compared to
base ViT-S v2's **1621 ELO** — a gain of **108 points** with virtually the same
parameter count (+0.3%). The improvement is broad-based, with higher accuracy
across most puzzle tiers from beginner through advanced.

The gains are strongest in the **mid-range tiers** (750–2000 ELO), where accuracy
improvements are 9–15 percentage points. This suggests AttnRes helps the model
learn more nuanced positional evaluation — exactly the kind of intermediate
complexity where flexible information routing across layers should help most.

### 7.2 Fewer Epochs, Better Loss

At only 14 epochs, AttnRes already achieved val_loss=0.033787, **lower than ViT-S v2's
best of 0.034701 after 30 epochs**. The loss curve was still clearly decreasing with
no signs of plateauing, suggesting the model would improve further with more training.

This faster convergence (better loss in fewer epochs despite same learning rate and
schedule) indicates that the learned depth mixing provides a better optimization
landscape compared to fixed residual connections.

### 7.3 Gap to Reference: 88 ELO

Our model is now **88 ELO below** the reference ViT-S's 1817 ELO (vs 196 ELO gap
for base ViT-S v2). The remaining gap is likely explained by:

1. **Data volume:** We train on 50M positions vs reference's 316M (6.3× less)
2. **Training duration:** Only 14/40 planned epochs completed — val_loss was still
   improving and more epochs would likely close the gap further
3. **Windowed vs Full AttnRes:** The W=3 window limits information routing compared
   to the full depth retrieval described in the paper. A more memory-efficient
   implementation could enable the full mechanism.

### 7.4 AttnRes Overhead: Batch Size Trade-off

The main practical cost of AttnRes is **higher memory usage**, forcing batch_size
from 8192 → 4096. This tripled epoch time (34 → 101 min), meaning the same GPU
budget covers fewer epochs. However, the per-epoch improvement is larger, so the
**information per GPU-hour is competitive**.

---

## 8. Experiment Summary Table

| Experiment       | Score Fix | Params  | Epochs | GPU-h  | Puzzle Acc | ELO    |
|------------------|-----------|---------|--------|--------|------------|--------|
| BDH v3           | ✗         | 2.5M    | 50     | ~33h   | 5.8%       | -2969  |
| ViT-S (broken)   | ✗         | 2.6M    | 30     | ~31.5h | 5.3%       | -3488  |
| BDH v4 (fixed)   | ✓         | 2.5M    | 43     | ~48h   | 31.0%      | 1177   |
| ViT-S v2 (fixed) | ✓         | 2.6M    | 30     | ~17h   | 45.1%      | 1621   |
| **ViT-S+AttnRes**| **✓**     |**2.66M**| **14** |**~24h**| **49.3%**  |**1729**|
| Reference ViT-S  | ✓         | 2.6M    | —      | >200h  | N/A        | 1817   |

---

## 9. Conclusion

The Windowed Attention Residuals architecture demonstrates that **replacing fixed
residual connections with learned depth mixing significantly improves chess
evaluation quality**. With virtually identical parameter count (+0.3%) and only 14
training epochs, we achieved:

- **+108 ELO** over base ViT-S v2 (1729 vs 1621)
- **+4.2 pp** puzzle accuracy (49.3% vs 45.1%)
- **Lower val_loss in fewer epochs** (0.03379 at epoch 14 vs 0.03470 at epoch 27)
- **Only 88 ELO below reference** (vs 196 ELO gap previously)

The model was still improving when terminated — additional training would likely
yield further gains. The key trade-off is memory: windowed AttnRes requires halving
the batch size, tripling epoch time. Future work could explore gradient checkpointing
or a more memory-efficient AttnRes implementation to enable larger batches or the
full (non-windowed) mechanism.
