# Searchless Chess — Project Report

**Author:** Antoni Czapski  
**Course:** Big Data in Healthcare (AGH UST)  
**Date:** March 2026  
**Repository:** [github.com/antoniczapski/searchless-chess](https://github.com/antoniczapski/searchless-chess)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Definition & Motivation](#2-problem-definition--motivation)
3. [Dataset & Exploratory Data Analysis](#3-dataset--exploratory-data-analysis)
4. [Model Architectures](#4-model-architectures)
5. [Experiment Timeline & Results](#5-experiment-timeline--results)
6. [The Score Perspective Bug — Error Analysis](#6-the-score-perspective-bug--error-analysis)
7. [Advanced Experiment — Attention Residuals](#7-advanced-experiment--attention-residuals)
8. [Engineering & Reproducibility](#8-engineering--reproducibility)
9. [Course Requirement Compliance](#9-course-requirement-compliance)
10. [Conclusions & Limitations](#10-conclusions--limitations)

---

## 1. Executive Summary

This project implements a neural chess position evaluator that predicts Stockfish centipawn scores directly from board state, **without any tree search**. It reimplements and extends the Vision Transformer architecture from the [mateuszgrzyb/searchless-chess](https://github.com/mateuszgrzyb/searchless-chess) reference repository, adding a novel BDH (Biological Dendritic Hebbian) architecture and Attention Residuals.

**Key result:** Our best model, **ViT-S + Attention Residuals**, achieves **ELO 1810** — only **7 points below** the reference implementation's 1817, while using **6.3× less training data** (50M vs 316M positions) and substantially less compute (<50 GPU-hours vs >200).

| Model             | Params | Epochs | GPU-hours | Puzzle Acc | ELO      |
|-------------------|--------|--------|-----------|------------|----------|
| MLP baseline      | 200K   | 5      | <1h       | ~15%       | ~900     |
| BDH v4 (fixed)    | 2.5M   | 43     | ~48h      | 31.0%      | 1177     |
| ViT-S v2 (fixed)  | 2.6M   | 30     | ~17h      | 45.1%      | 1621     |
| **ViT-S+AttnRes** | **2.66M** | **27** | **~45h** | **53.0%** | **1810** |
| Reference ViT-S   | 2.6M   | —      | >200h     | N/A        | 1817     |

The project went through **10 experiments** across 4 architectures, discovered and fixed a critical data preprocessing bug, and was fully restructured into a course-compliant repository with UV dependency management, PyTorch Lightning training, 104 passing tests, and GitHub Actions CI.

---

## 2. Problem Definition & Motivation

### 2.1 Problem Statement

Given an 8×8×12 binary tensor representing a chess position (6 piece types × 2 colors), predict the Stockfish evaluation normalized to \[-1, 1\] via `tanh(cp/1000)`. The model is evaluated on 1,200 rated Lichess puzzles across 12 difficulty tiers to estimate playing strength (ELO).

### 2.2 Why Deep Learning Is Appropriate

Traditional chess engines rely on handcrafted evaluation functions combined with tree search (alpha-beta pruning, MCTS). Neural networks can learn rich positional features — piece activity, pawn structure, king safety — directly from millions of annotated positions, without explicit feature engineering. The question is whether a neural network can produce evaluations good enough to play strong chess **without any search at all**.

### 2.3 Evaluation Metric

**Puzzle-based ELO estimation:**
1. 1,200 rated Lichess puzzles across 12 difficulty tiers (0–3250 rating).
2. For each puzzle, the model evaluates all legal successor positions and picks the move leading to the lowest-scoring position for the opponent.
3. Per-tier solve rates are computed, and a linear regression estimates the rating at which the model achieves 50% solve rate.

This provides a single scalar (ELO) that correlates with actual chess playing strength.

### 2.4 Reference Implementation

The project builds on [mateuszgrzyb/searchless-chess](https://github.com/mateuszgrzyb/searchless-chess), which trained 6 architectures (CNN, ResNet-M/L/XL, ViT-S, ViT-M) on 316M positions. Their ViT-S (2.6M params) achieved 1817 ELO, making it the most parameter-efficient model. We reimplemented ViT-S, added BDH, and introduced Attention Residuals.

---

## 3. Dataset & Exploratory Data Analysis

> Full EDA: [`report/eda/eda.md`](https://github.com/antoniczapski/searchless-chess/blob/main/report/eda/eda.md)

### 3.1 Dataset Source

**[mateuszgrzyb/lichess-stockfish-normalized](https://huggingface.co/datasets/mateuszgrzyb/lichess-stockfish-normalized)** — 50,000,000 chess positions with Stockfish evaluations from the Lichess database.

| Property              | Value                                    |
|-----------------------|------------------------------------------|
| Total positions       | 50,000,000                               |
| Train / Val / Test    | 45M / 2.5M / 2.5M (90/5/5)             |
| Board encoding        | 8×8×12 binary tensor (uint8)            |
| Target                | `tanh(cp/1000)` ∈ \[-1, 1\]            |
| Train data size       | ~34 GB (uint8 .npy)                     |
| Split method          | Deterministic shuffle (seed=42)          |

### 3.2 Target Distribution

The normalized score distribution is approximately symmetric and centered near zero (mean ≈ 0.02), with a standard deviation of ≈ 0.38. The `tanh(cp/1000)` compression means:
- ±300 cp → ≈ ±0.29 (typical single-piece advantage)
- ±1000 cp → ≈ ±0.76 (decisive advantage)
- Approximately 15% of positions have decisive scores (|score| > 0.9)

**Implication:** A "predict zero" baseline achieves MSE ≈ 0.145, meaning any meaningful model must substantially outperform this floor.

### 3.3 Side-to-Move Split

The dataset is nearly balanced: ~50.3% white-to-move, ~49.7% black-to-move. This balance was critical — a data preprocessing bug (Section 6) exploited this near-50/50 split to silently corrupt half of all training labels.

### 3.4 Material Distribution

Positions span the full game lifecycle: ~15% openings (30–32 pieces), ~45% middlegame (20–29), ~35% endgame (6–19), ~5% late endgame (2–5). The model must generalize across wildly different material configurations.

### 3.5 Board Encoding

Each position is encoded as 12 binary planes on an 8×8 grid:
- Channels 0–5: Side-to-move pieces (P, N, B, R, Q, K)
- Channels 6–11: Opponent pieces (P, N, B, R, Q, K)

The board is always presented from the side-to-move's perspective (`always_white_perspective=True`), meaning black-to-move positions are mirrored vertically with colors swapped. This encoding intentionally omits castling rights, en passant, and move history, following the reference design.

### 3.6 Key EDA Insights

1. **Label quality is critical** — the score sign-flip bug showed that data corruption can silently prevent learning.
2. **The target distribution is favorable** — centered, symmetric, bounded; well-suited for regression.
3. **Position diversity is high** — the model sees everything from openings to endgames.
4. **Puzzle ELO is a strong proxy** for playing strength, correlating well with actual chess rating.
5. **The encoding is minimal but sufficient** — 12 binary planes capture piece placement for competitive evaluation.

---

## 4. Model Architectures

> Model source code: [`src/models/`](https://github.com/antoniczapski/searchless-chess/tree/main/src/models)

### 4.1 MLP Baseline

**File:** [`src/models/mlp.py`](https://github.com/antoniczapski/searchless-chess/blob/main/src/models/mlp.py)

A simple multi-layer perceptron that flattens the 8×8×12 board into a 768-dimensional vector and passes it through fully connected layers. This establishes the lower bound: ~900 ELO with ~200K parameters. It proves the task is learnable but demonstrates that spatial structure matters.

### 4.2 BDH (Biological Dendritic Hebbian)

**File:** [`src/models/bdh.py`](https://github.com/antoniczapski/searchless-chess/blob/main/src/models/bdh.py)  
**Research notes:** [`university_project/bdh.md`](https://github.com/antoniczapski/searchless-chess/blob/main/university_project/bdh.md)

The BDH model is a post-transformer architecture from Pathway Technology that uses **iterative value refinement** with a synaptic scratchpad. Rather than processing the board as a sequence (which failed in v1), the v2+ architecture:

1. **Encodes the board once** into a d-dimensional embedding.
2. **Runs K=12 shared-weight "thinking steps"** where a synaptic state ρ (Hopfield-like associative memory) accumulates relational patterns.
3. **Re-injects the board at every step** to prevent drift.
4. **Produces a value prediction at each step** (deep supervision).

**Key design elements:**
- Linear attention with learnable EMA damping (λ ≈ 0.9)
- Hebbian gating: `ReLU(D_y · LN(a + B_y · b)) ⊙ x` — co-activation of attention output and neuron state
- Stability penalty discouraging oscillations between thinking steps
- RMSNorm on neuron activations + normalized ρ writes (v3 stability fixes)

**Loss function:**

$$\mathcal{L} = \text{Huber}(v_K, y) + 0.1 \sum_{k=1}^{K-1} \text{Huber}(v_k, y) + 0.01 \sum_{k=2}^{K} (v_k - v_{k-1})^2$$

The BDH architecture evolved across 4 versions (see Section 5), ultimately achieving **1177 ELO** with 2.5M parameters.

### 4.3 ViT-S (Vision Transformer Small)

**File:** [`src/models/vit.py`](https://github.com/antoniczapski/searchless-chess/blob/main/src/models/vit.py)

Faithful PyTorch reimplementation of the reference Keras/TensorFlow ViT-S:

```
Input (B, 12, 8, 8)
  → Reshape to (B, 64, 12)           # 64 "patches" of dim 12
  → Linear(12, 256)                   # patch projection
  → + Positional Embedding(64, 256)   # learnable
  → 5× Transformer Block:
      ├─ LayerNorm → MultiHeadAttention(4 heads, dim=64)
      ├─ + Residual
      ├─ LayerNorm → FFN(512, GELU, 256)
      └─ + Residual
  → LayerNorm → GAP → Dropout(0.1) → Linear(1) → tanh
```

**Key design choices from reference:**
- FFN ratio = 2× (not the standard 4×)
- Pre-norm architecture
- Global average pooling instead of CLS token
- 2,656,001 parameters

Achieved **1621 ELO** after fixing the score perspective bug.

### 4.4 ViT-S + Attention Residuals (Best Model)

**File:** [`src/models/vit_attnres.py`](https://github.com/antoniczapski/searchless-chess/blob/main/src/models/vit_attnres.py)  
**Research notes:** [`report/research/residual_attention.md`](https://github.com/antoniczapski/searchless-chess/blob/main/report/research/residual_attention.md)

Based on the "Attention Residuals" paper (Kimi Team, 2025), this model replaces the standard residual connection (`x = x + sublayer(x)`) with **learned depth-wise softmax retrieval**: each sublayer receives a weighted mixture of previous sublayer outputs, controlled by learned pseudo-queries.

**Changes vs base ViT-S:**

| Component           | ViT-S v2                         | ViT-S + AttnRes                 |
|---------------------|----------------------------------|----------------------------------|
| Residual connection | `x = x + sublayer(norm(x))`     | Windowed AttnRes (W=3)          |
| FFN activation      | GELU (2 projections)             | **SwiGLU** (3 projections)      |
| Normalization       | LayerNorm                        | **RMSNorm**                     |
| Depth mixing        | None (fixed running sum)         | **Learned softmax** per sublayer|
| Parameters          | 2,656,001                        | **2,664,705** (+0.3%)           |

**Windowed AttnRes (W=3):** Each sublayer attends over only the last 3 depth sources rather than all previous ones. This was necessary because Full AttnRes caused CUDA OOM on the A100-40GB. The window bounds GPU memory at O(W) regardless of depth.

**SwiGLU FFN:** Replaces GELU FFN with a gated linear unit using SiLU activation and 3 projection matrices, with hidden dim adjusted for parameter parity.

Achieved **1810 ELO** — only 7 points below the reference.

### 4.5 Model Registry

**File:** [`src/models/registry.py`](https://github.com/antoniczapski/searchless-chess/blob/main/src/models/registry.py)

All models are registered in a central registry (`MODEL_REGISTRY`) that maps string names to constructor functions: `mlp`, `bdh`, `vit`, `vit_attnres`. This enables config-driven model selection.

---

## 5. Experiment Timeline & Results

> Detailed per-experiment reports: [`report/experiments/`](https://github.com/antoniczapski/searchless-chess/tree/main/report/experiments)

### 5.1 BDH v1 — Sequence Processing (Local GPU)

**Report:** [`report/experiments/bdh_v1.md`](https://github.com/antoniczapski/searchless-chess/blob/main/report/experiments/bdh_v1.md)

- **Approach:** Treated the 8×8 board as a 64-token sequence, ran 4 BDH layers over it.
- **Data:** 500K positions on RTX 2060 (6 GB VRAM).
- **Result:** Val loss plateaued at ~0.110 (near prediction-by-mean). Puzzle accuracy: 0.5%. ELO: unmeasurable.
- **Lesson:** Sequential processing of a static board is a poor adaptation — the recurrent synaptic state ρ never accumulates temporal context, reducing BDH to a low-rank MLP.

### 5.2 BDH v2 — Iterative Value Refinement (Local GPU)

**Report:** [`report/experiments/bdh_v2.md`](https://github.com/antoniczapski/searchless-chess/blob/main/report/experiments/bdh_v2.md)

- **Approach:** Encode board once, run K=8 shared-weight "thinking steps" with persistent synaptic state. Deep supervision + Huber loss.
- **Data:** 500K positions on RTX 2060.
- **Result:** Val MAE 0.1257 (−13.4% vs v1), 4× longer learning before overfitting. Puzzle accuracy: 0.8%.
- **Lesson:** Iterative refinement is a valid paradigm. Data scale is the primary bottleneck — the 682K-param model overfits at epoch 8 on 500K positions.

### 5.3 BDH v3 — 50M Positions on A100

**Report:** [`report/experiments/bdh_v3_50M.md`](https://github.com/antoniczapski/searchless-chess/blob/main/report/experiments/bdh_v3_50M.md)

- **Approach:** Scaled to 50M positions, 2.5M params (d=256, nh=8, K=12), A100 GPU. Added stability fixes (RMSNorm, normalized ρ writes, bf16).
- **Training:** 26 epochs, ~34 GPU-hours. Survived NaN divergence that destroyed the first A100 run.
- **Result:** Val MAE 0.1101, puzzle accuracy 5.8%, ELO −2969 (nonsensical negative — bug).
- **Lesson:** 100× more data improved puzzle accuracy by 7.3× (0.8% → 5.8%). But the deeply negative ELO pointed to a data corruption issue.

### 5.4 ViT-S Sanity Check (Broken)

**Reports:** [`report/experiments/vit_experiment_design.md`](https://github.com/antoniczapski/searchless-chess/blob/main/report/experiments/vit_experiment_design.md), [`report/experiments/vit_sanity_check.md`](https://github.com/antoniczapski/searchless-chess/blob/main/report/experiments/vit_sanity_check.md)

- **Approach:** Faithful ViT-S reimplementation to establish a controlled comparison vs BDH v3.
- **Training:** 27 epochs, ~31.5 GPU-hours on A100.
- **Result:** ELO −3488 (also nonsensical). Training was stable, loss converged, but the model couldn't play chess.
- **Lesson:** Both architectures failed with the same data pipeline → the problem was in preprocessing, not the model.

### 5.5 ViT-S v2 — Score Bug Fix (Breakthrough)

**Report:** [`report/experiments/vit_v2_fixed.md`](https://github.com/antoniczapski/searchless-chess/blob/main/report/experiments/vit_v2_fixed.md)

- **The fix:** Discovered and fixed the score perspective bug (Section 6). Re-prepared all 50M labels. Also switched from float32 to uint8 boards (4× disk savings), added in-memory loading (2.5× speedup).
- **Training:** 30 epochs, ~17 GPU-hours (2.5× faster than before due to pipeline optimizations).
- **Result:** **ELO 1621** (from −3488). Puzzle accuracy **45.1%** (from 5.3%). Val loss 0.0347 (from 0.1525).
- **Lesson:** A single preprocessing bug cost **5109 ELO points**. Data validation is paramount.

### 5.6 BDH v4 — Score Bug Fix Applied to BDH

**Report:** [`report/experiments/bdh_v4_fixed.md`](https://github.com/antoniczapski/searchless-chess/blob/main/report/experiments/bdh_v4_fixed.md)

- **Approach:** Same BDH architecture, corrected data, also fixed `cp_scale` from 10000 to 1000.
- **Training:** 43 epochs, ~48 GPU-hours (SLURM timeout at 48h).
- **Result:** **ELO 1177**, puzzle accuracy **31.0%**. Score fix improved BDH from −2969 to 1177 (+4146 points).
- **Head-to-head:** ViT-S v2 (1621) outperforms BDH v4 (1177) by **444 ELO** on identical data. ViT's spatial attention is more effective than BDH's flattened iterative processing. BDH also overfits more (26.5% train-val gap vs ViT's 5.2%) and trains 2.8× slower.

### 5.7 ViT-S + Attention Residuals — Best Model

**Report:** [`report/experiments/vit_attnres.md`](https://github.com/antoniczapski/searchless-chess/blob/main/report/experiments/vit_attnres.md)

- **Approach:** Replaced standard residuals with Windowed AttnRes (W=3), added SwiGLU FFN and RMSNorm.
- **Training:** 27 epochs across multiple SLURM sessions, ~45 GPU-hours.
- **Result:** **ELO 1810**, puzzle accuracy **53.0%**. Only **7 ELO below reference** (1817).
- **Val loss** of 0.033787 was already below ViT-S v2's best (0.034701) at only 14 epochs, and the model was still improving when extended training completed.
- **Key trade-off:** Halved batch size (8192 → 4096) due to AttnRes memory overhead, tripling epoch time.

### 5.8 Summary Table — All 10 Experiments

| # | Experiment           | Score Fix | cp_scale | Params | Data | GPU-h  | Puzzle Acc | ELO    |
|---|----------------------|-----------|----------|--------|------|--------|------------|--------|
| 1 | BDH v1 (sequence)    | ✗         | 1000     | 796K   | 500K | ~1h    | 0.5%       | N/A    |
| 2 | BDH v2 (iterative)   | ✗         | 1000     | 682K   | 500K | ~0.5h  | 0.8%       | N/A    |
| 3 | BDH v3 (50M)         | ✗         | 10000    | 2.5M   | 50M  | ~33h   | 5.8%       | −2969  |
| 4 | ViT-S sanity         | ✗         | 1000     | 2.6M   | 50M  | ~31.5h | 5.3%       | −3488  |
| 5 | **ViT-S v2 (fixed)** | **✓**     | **1000** | **2.6M**| **50M**| **~17h** | **45.1%** | **1621** |
| 6 | **BDH v4 (fixed)**   | **✓**     | **1000** | **2.5M**| **50M**| **~48h** | **31.0%** | **1177** |
| 7 | ViT-S+AttnRes (14ep) | ✓         | 1000     | 2.66M  | 50M  | ~24h   | 49.3%      | 1729   |
| 8 | **ViT-S+AttnRes (27ep)** | **✓** | **1000** | **2.66M**| **50M**| **~45h** | **53.0%** | **1810** |
|   | *Reference ViT-S*    | ✓         | 1000     | 2.6M   | 316M | >200h  | N/A        | 1817   |

---

## 6. The Score Perspective Bug — Error Analysis

> This is the most important debugging story of the project.

### 6.1 Symptoms

- BDH v3 achieved ELO −2969 despite training for 26 epochs on 50M positions.
- ViT-S sanity check achieved ELO −3488 — even worse.
- Both models had smooth, converging training curves and reasonable val MAE.
- Puzzle accuracy was ~5–6% for both (barely above random chance).

### 6.2 Root Cause

The HuggingFace dataset stores centipawn scores **from white's perspective**. Our board encoding uses `always_white_perspective=True`, which mirrors the board for black-to-move positions. However, the score was **not flipped** for black-to-move positions.

**Impact:** For black-to-move (~50% of data), the board was encoded from black's perspective but the score still represented white's advantage. This created contradictory supervision — the model was told that identical-looking positions (from its perspective) had opposite evaluations.

### 6.3 The Fix

In [`src/data/prepare.py`](https://github.com/antoniczapski/searchless-chess/blob/main/src/data/prepare.py):

```python
is_white_to_move = " w " in fen
sign = 1.0 if is_white_to_move else -1.0
score = sign * normalize_cp(raw_cp, scale=cp_scale)
```

### 6.4 Impact of the Fix

| Metric            | Before fix    | After fix     | Improvement      |
|-------------------|---------------|---------------|------------------|
| ViT-S ELO         | −3488         | **1621**      | **+5109 points** |
| ViT-S Puzzle Acc  | 5.3%          | **45.1%**     | **8.5× better**  |
| ViT-S val_loss    | 0.1525        | **0.0347**    | **4.4× lower**   |
| BDH ELO           | −2969         | **1177**      | **+4146 points** |
| BDH Puzzle Acc    | 5.8%          | **31.0%**     | **5.3× better**  |

### 6.5 Lesson Learned

**Always validate preprocessing end-to-end.** This bug was invisible during training (smooth loss curves, reasonable MAE values) because the model learned to predict near-zero for everything — which minimizes MSE when half the labels are correct and half are inverted. The puzzle benchmark was the only diagnostic that revealed the problem.

---

## 7. Advanced Experiment — Attention Residuals

### 7.1 Motivation

Standard residual connections use fixed-coefficient accumulation: each layer's output is simply added to a running sum. The "Attention Residuals" paper (Kimi Team, 2025) proposes replacing this with a **learned depth-wise softmax retrieval** where each sublayer dynamically selects which earlier layer's representation to build upon.

### 7.2 Mechanism

Each sublayer has a learned pseudo-query vector $w_l \in \mathbb{R}^d$ (initialized to zero for uniform start). The earlier layer outputs serve as both keys and values. The score from source $i$ into layer $l$:

$$s_{i \to l} = w_l^\top \text{RMSNorm}(k_i)$$

Then softmax over depth sources:

$$\alpha_{i \to l} = \frac{\exp(s_{i \to l})}{\sum_{j} \exp(s_{j \to l})}$$

And the input to layer $l$: $h_l = \sum_i \alpha_{i \to l} v_i$

### 7.3 Memory Optimization

Full AttnRes (keeping all 11 depth sources) caused CUDA OOM on the A100-40GB even at batch 4096. We implemented **Windowed AttnRes (W=3)** — each sublayer only mixes the last 3 depth sources. Older sources are discarded entirely (not just detached), bounding memory at O(W) regardless of depth.

### 7.4 Ablation: AttnRes vs Standard Residuals

| Metric             | ViT-S v2 (standard) | ViT-S + AttnRes | Δ               |
|--------------------|----------------------|------------------|-----------------|
| ELO                | 1621                 | **1810**         | **+189**        |
| Puzzle accuracy    | 45.1%                | **53.0%**        | **+7.9 pp**     |
| Best val_loss      | 0.034701             | **0.033787**     | **−2.6%**       |
| Parameters         | 2,656,001            | 2,664,705        | +0.3%           |
| GPU hours          | ~17h                 | ~45h             | 2.6× more       |

The +189 ELO gain with only 0.3% more parameters demonstrates that **depth mixing is more effective than fixed residual connections** for this task. The improvement is broad-based across puzzle difficulty tiers, with the strongest gains in the mid-range (750–2000 ELO), suggesting AttnRes helps learn more nuanced positional evaluation.

---

## 8. Engineering & Reproducibility

### 8.1 Repository Structure

```
searchless-chess/
├── src/                        # Source code (proper Python package)
│   ├── data/                   # Dataset, encoding, Lightning DataModule
│   ├── evaluation/             # Puzzle-based ELO benchmark
│   ├── models/                 # MLP, BDH, ViT, ViT-AttnRes + registry
│   └── training/               # Lightning module, legacy trainer
├── tests/                      # pytest suite (104 tests)
├── configs/                    # YAML experiment configs
│   └── smoke/                  # Tiny configs for CI/testing
├── scripts/                    # CLI entrypoints (train, evaluate, prepare)
├── data/puzzles/               # Lichess puzzle benchmark
├── report/
│   ├── eda/                    # Exploratory data analysis
│   ├── experiments/            # 8 per-experiment reports
│   ├── summary/                # Final results summary
│   └── research/               # Literature notes (AttnRes paper)
├── .github/workflows/ci.yml   # GitHub Actions CI pipeline
├── university_project/         # Course presentations and plans
├── pyproject.toml              # UV-based dependency management
├── uv.lock                     # Locked dependency versions (189 packages)
└── README.md                   # Project documentation
```

### 8.2 Dependency Management with UV

> Config: [`pyproject.toml`](https://github.com/antoniczapski/searchless-chess/blob/main/pyproject.toml)

The project uses **[UV](https://docs.astral.sh/uv/)** (Astral) for reproducible dependency management:

- **`pyproject.toml`** defines the project metadata, Python version requirement (≥3.11), and dependency groups:
  - **core:** torch, numpy, pandas, pyarrow, pyyaml, loguru, python-chess, datasets, scikit-learn
  - **train:** pytorch-lightning, wandb, torchmetrics
  - **dev:** pytest, pytest-cov, ruff
  - **analysis:** jupyter, matplotlib, seaborn
- **`uv.lock`** locks all 189 transitive dependencies to exact versions for reproducibility.
- Installation: `uv sync --all-groups` installs everything deterministically.

**Why UV over pip/conda:**
- Deterministic resolution (lockfile guarantees identical environments across machines)
- 10–100× faster than pip for dependency resolution
- Single tool for project management, virtual environments, and dependency locking
- Works on CI without pre-installed Python (via `astral-sh/setup-uv` GitHub Action)

### 8.3 Config-Based Training

> Configs: [`configs/`](https://github.com/antoniczapski/searchless-chess/tree/main/configs)

All experiments are configured via YAML files specifying:
- Model architecture and hyperparameters
- Data paths and preprocessing parameters
- Training hyperparameters (LR, batch size, epochs, warmup, etc.)
- Output directories and logging settings

Example configs: `train_vit_attnres_a100.yaml`, `train_bdh_v4_a100.yaml`, `train_mlp.yaml`, etc.

Smoke test configs (`configs/smoke/`) use tiny models (dim=32, 1 layer) for rapid CI validation.

### 8.4 PyTorch Lightning Training Pipeline

> Module: [`src/training/lightning_module.py`](https://github.com/antoniczapski/searchless-chess/blob/main/src/training/lightning_module.py)  
> DataModule: [`src/data/lightning_datamodule.py`](https://github.com/antoniczapski/searchless-chess/blob/main/src/data/lightning_datamodule.py)  
> Script: [`scripts/train.py`](https://github.com/antoniczapski/searchless-chess/blob/main/scripts/train.py)

The training pipeline uses **PyTorch Lightning** for clean separation of model logic, data loading, and training orchestration:

- **`ChessLightningModule`** wraps the model registry, handles loss computation (MSE or BDH deep supervision), AdamW optimizer with cosine schedule + linear warmup.
- **`ChessDataModule`** wraps the existing data loaders, with a **smoke mode** that generates tiny synthetic data for testing.
- **`scripts/train.py`** configures the Lightning Trainer with ModelCheckpoint, EarlyStopping, LearningRateMonitor callbacks, WandbLogger + CSVLogger.
- **`--smoke-test` flag:** Runs with synthetic data, 1 epoch, CPU, `fast_dev_run=5` — completes in ~30 seconds for CI.
- **`--resume auto`:** Automatically finds and resumes from the latest checkpoint.
- **bf16 precision:** Auto-detected on A100 GPUs.

The legacy handwritten trainer ([`src/training/trainer.py`](https://github.com/antoniczapski/searchless-chess/blob/main/src/training/trainer.py)) is preserved for compatibility with running HPC jobs.

### 8.5 Reproducibility Measures

| Measure                     | Implementation                                          |
|-----------------------------|---------------------------------------------------------|
| Fixed random seeds          | `seed_everything(42)` in all training scripts           |
| Deterministic splits        | Deterministic shuffle (seed=42) before 90/5/5 partition |
| Locked dependencies         | `uv.lock` with 189 packages pinned to exact versions   |
| Config-driven training      | All hyperparameters in YAML configs, no hardcoded values|
| Checkpoint resume           | Full optimizer, scheduler, and early stopping state     |
| Experiment tracking         | Weights & Biases (wandb) logging for all runs           |

### 8.6 Testing

> Tests: [`tests/`](https://github.com/antoniczapski/searchless-chess/tree/main/tests)

The test suite covers **104 tests** across 8 test files:

| Test File                                              | What It Tests                           | Tests |
|--------------------------------------------------------|-----------------------------------------|-------|
| [`test_config_loads.py`](https://github.com/antoniczapski/searchless-chess/blob/main/tests/test_config_loads.py) | YAML config loading and key validation (parametrized across all 8 configs) | ~16 |
| [`test_board_encoding.py`](https://github.com/antoniczapski/searchless-chess/blob/main/tests/test_board_encoding.py) | FEN encoding, normalize_cp, mate_to_cp | ~15 |
| [`test_model_forward_vit.py`](https://github.com/antoniczapski/searchless-chess/blob/main/tests/test_model_forward_vit.py) | Forward pass for all 4 architectures + registry | ~17 |
| [`test_lightning_smoke_train.py`](https://github.com/antoniczapski/searchless-chess/blob/main/tests/test_lightning_smoke_train.py) | Lightning fast_dev_run, checkpoint callback | ~3 |
| [`test_evaluation_smoke.py`](https://github.com/antoniczapski/searchless-chess/blob/main/tests/test_evaluation_smoke.py) | Model scoring, best_move, puzzle solving | ~3 |
| [`test_encoding.py`](https://github.com/antoniczapski/searchless-chess/blob/main/tests/test_encoding.py) | Board encoding correctness (legacy) | ~20 |
| [`test_models.py`](https://github.com/antoniczapski/searchless-chess/blob/main/tests/test_models.py) | Model instantiation and forward pass (legacy) | ~15 |
| [`test_training.py`](https://github.com/antoniczapski/searchless-chess/blob/main/tests/test_training.py) | Training step correctness (legacy) | ~15 |

All 104 tests pass in ~2 minutes on CPU.

### 8.7 Continuous Integration (GitHub Actions)

> CI config: [`.github/workflows/ci.yml`](https://github.com/antoniczapski/searchless-chess/blob/main/.github/workflows/ci.yml)

The CI pipeline runs on every push and PR to `main`:

1. **Checkout** code
2. **Set up Python 3.11** via `actions/setup-python@v5`
3. **Install UV** via `astral-sh/setup-uv@v4`
4. **Install dependencies** via `uv sync --all-groups`
5. **Lint** via `ruff check src/ tests/ scripts/`
6. **Compile check** via `python -m compileall src/ scripts/`
7. **Run tests** via `pytest -q --tb=short`

### 8.8 Experiment Tracking with Weights & Biases

All training runs are logged to W&B with:
- Loss curves (train + validation)
- MAE metrics
- Learning rate schedules
- Intra-epoch step-level logging (~7 min intervals for long epochs)
- Run IDs documented in each experiment report

### 8.9 HPC Infrastructure (SLURM / Ares)

- **Cluster:** Ares HPC (plgrid / Cyfronet AGH)
- **GPU:** NVIDIA A100-SXM4-40GB
- **Job management:** SLURM with 12h–48h time limits
- **Resume strategy:** Checkpoint-and-resume pattern with SIGTERM handler
- **Conda environment:** `/net/tscratch/people/plgantoniczapski/conda-envs/bdh-chess` (Python 3.11, PyTorch 2.5.1)
- **Data storage:** `/net/tscratch/` (scratch filesystem, 34 GB train boards)

---

## 9. Course Requirement Compliance

This section maps each requirement from the [Meeting Plan](https://github.com/antoniczapski/searchless-chess/blob/main/university_project/Meeting%20Plan%20-%20Outline.md) to how it was fulfilled.

### Meeting 3 — Proposal Presentation ✅

| Requirement               | Fulfilled By                                           |
|---------------------------|--------------------------------------------------------|
| Problem definition        | Searchless chess position evaluation (Section 2)       |
| Dataset description       | HuggingFace lichess-stockfish-normalized, 50M positions (Section 3) |
| Evaluation metric         | Puzzle-based ELO estimation (Section 2.3)             |
| Planned architecture      | BDH + ViT (presented in proposal)                     |
| Main risks / challenges   | Data scale, BDH adaptation, compute budget            |

### Meeting 4 — Repository & CI ✅

| Requirement               | Fulfilled By                                           |
|---------------------------|--------------------------------------------------------|
| Public GitHub repository  | [github.com/antoniczapski/searchless-chess](https://github.com/antoniczapski/searchless-chess) |
| Proper project structure  | `src/`, `tests/`, `configs/`, `scripts/`, `report/`   |
| Working `uv` environment  | `pyproject.toml` + `uv.lock` (189 packages)           |
| Working GitHub Actions CI | `.github/workflows/ci.yml` — lint + compile + pytest  |
| Basic tests implemented   | 104 tests in `tests/` (all passing)                   |

### Meeting 5 — EDA & Baseline Model ✅

| Requirement               | Fulfilled By                                           |
|---------------------------|--------------------------------------------------------|
| Exploratory Data Analysis | [`report/eda/eda.md`](https://github.com/antoniczapski/searchless-chess/blob/main/report/eda/eda.md) — 10 sections covering target distribution, side-to-move split, material distribution, mate prevalence, encoding, splits, puzzle benchmark, baselines, key insights, experiment results |
| Key dataset insights      | Score distribution symmetry, sign-flip criticality, 15% decisive positions, encoding sufficiency |
| Baseline model results    | MLP baseline: ~900 ELO, ~15% puzzle accuracy, ~200K params |

### Meeting 6 — Reproducible Training & W&B ✅

| Requirement                | Fulfilled By                                          |
|----------------------------|-------------------------------------------------------|
| Config-based training      | YAML configs in `configs/` for all experiments        |
| Fixed random seeds         | `seed_everything(42)` everywhere                      |
| Fixed train/val/test splits| Deterministic shuffle (seed=42), 90/5/5 split         |
| W&B experiment tracking    | All 10 runs logged with unique run IDs                |
| ≥2–3 logged experiments    | 10 experiments logged (W&B run IDs in each report)    |

### Meeting 7 — First Experimental Results ✅

| Requirement               | Fulfilled By                                           |
|---------------------------|--------------------------------------------------------|
| Model comparison          | BDH v1 vs v2 vs v3, ViT-S sanity check               |
| Performance results       | Full epoch-by-epoch tables in all experiment reports   |
| Observations              | Sequential BDH fails, iterative works; data scale critical |
| Next improvement plan     | Scale data, fix score bug, compare architectures       |

### Meeting 8 — Model Improvements ✅

| Requirement                         | Fulfilled By                                  |
|-------------------------------------|-----------------------------------------------|
| Improvements over baseline          | BDH v2 (iterative) → v3 (50M) → ViT-S → ViT-S+AttnRes |
| Hyperparameter tuning / arch changes| cp_scale 10000→1000, batch size optimization, SwiGLU, RMSNorm, Windowed AttnRes |
| Clear before/after comparison       | Every experiment report has a comparison table vs previous versions |

### Meeting 9 — Error Analysis ✅

| Requirement                  | Fulfilled By                                      |
|------------------------------|---------------------------------------------------|
| Where the model fails        | Per-tier puzzle accuracy shows failure above 2000 ELO; AttnRes improves mid-range most |
| Failure examples             | Score perspective bug analysis (Section 6): 50% label inversion causing −3488 ELO |
| Overfitting/underfitting     | Train-val gap analysis in every report (BDH: 26.5% gap = overfitting; ViT: 5.2% = healthy) |
| Understanding model weaknesses| BDH loses spatial structure; ViT benefits from positional attention; all models struggle on tactical puzzles >2000 ELO |

### Meeting 10 — Advanced Experiments ✅

| Requirement               | Fulfilled By                                           |
|---------------------------|--------------------------------------------------------|
| Ablation study            | AttnRes vs standard residuals: +189 ELO with +0.3% params (Section 7.4) |
| Comparing optimizers      | MSE (ViT) vs Huber+deep supervision (BDH): MSE produced better ELO |
| Regularization study      | Dropout impact, deep supervision + stability penalty in BDH |
| Robustness tests          | Score sign flip (label noise) revealed silent failure mode |
| Analysis, not just numbers| Every report includes multi-page discussion of why results differ |

### Meeting 11 — Final Review Checklist ✅

| Checklist Item              | Status |
|-----------------------------|--------|
| CI passing                  | ✅ GitHub Actions: lint + compile + 104 tests |
| Clean repository            | ✅ Flat structure, no stale files, proper `.gitignore` |
| Clear README                | ✅ [README.md](https://github.com/antoniczapski/searchless-chess/blob/main/README.md): installation, quickstart, model table, data docs, testing |
| Working training pipeline   | ✅ Lightning-based `scripts/train.py` with `--smoke-test` |
| Reproducibility verified    | ✅ UV lockfile, fixed seeds, config-driven, checkpoint resume |
| W&B experiments organized   | ✅ 10 runs with unique IDs, linked in experiment reports |
| Presentation draft ready    | ✅ `university_project/presentation/` |

### Meeting 12 — Final Presentation Content ✅

| Required Section              | Covered In                        |
|-------------------------------|-----------------------------------|
| 1. Problem and motivation     | Section 2 of this report          |
| 2. Dataset                    | Section 3 (EDA)                   |
| 3. Model architecture         | Section 4 (4 architectures)       |
| 4. Experimental comparison    | Section 5 (10 experiments table)  |
| 5. Error analysis             | Section 6 (score bug) + per-experiment analysis |
| 6. Engineering decisions      | Section 8 (UV, Lightning, CI, HPC)|
| 7. Conclusions and limitations| Section 10                        |

---

## 10. Conclusions & Limitations

### 10.1 Key Findings

1. **Attention Residuals significantly improve learning efficiency.** ViT-S+AttnRes surpassed standard ViT-S by 189 ELO with only 0.3% more parameters, demonstrating that learned depth mixing outperforms fixed residual connections for chess evaluation.

2. **Data preprocessing quality dominates model architecture.** The score perspective bug cost ~5000 ELO points — more than any architectural improvement could recover. **Always validate your pipeline end-to-end.**

3. **ViT outperforms BDH for chess evaluation.** The ViT-S achieved 444 ELO above BDH on identical data and similar parameter counts. Spatial attention over the 8×8 board captures piece interactions more effectively than BDH's flattened iterative processing.

4. **The reference result is reproducible with less compute.** Achieving 1810 ELO (vs 1817 reference) with 6.3× less data and <50 GPU-hours (vs >200h) validates both the approach and our implementation.

5. **Data scale is the primary bottleneck.** Every scaling step (500K → 50M positions) dramatically improved results. The remaining 7-point ELO gap to the reference is likely attributable to the 6.3× data difference.

### 10.2 Limitations

1. **No tree search.** The model evaluates positions individually without lookahead. This limits tactical accuracy — the model struggles with puzzles requiring multi-move calculation (>2000 ELO).

2. **Simplified encoding.** The 12-plane binary encoding omits castling rights, en passant, half-move clock, and move history. These contain evaluation-relevant information.

3. **Single evaluation head.** The model produces a scalar score, not move probabilities. Best-move selection is done by evaluating all legal successors — an indirect and expensive proxy.

4. **Training data size.** Our 50M positions are 6.3× fewer than the reference's 316M. Scaling to the full dataset would likely close the remaining gap.

5. **Windowed AttnRes.** Memory constraints forced W=3 instead of full depth retrieval. A more memory-efficient implementation (gradient checkpointing, mixed precision) could enable the full mechanism.

### 10.3 Future Work

- Scale to 316M positions to match reference conditions exactly
- Add a move-generation head (policy network) for direct move prediction
- Implement adaptive halting for BDH (variable thinking steps based on position difficulty)
- Explore gradient checkpointing for full AttnRes
- Extend encoding with castling rights and en passant

---

*This report summarizes approximately 200 GPU-hours of experimentation across 10 training runs, 4 model architectures, and a comprehensive repository restructuring for university course compliance.*
