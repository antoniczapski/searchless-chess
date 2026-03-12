# 🧠 Project Plan: BDH Architecture for Searchless Chess

> **Course:** Project: Deep Learning  
> **Date:** March 9, 2026  
> **Next milestone:** Meeting 3 (March 10) — Proposal Presentation (Graded)  
> **Compute budget:** ≤100 hours on NVIDIA A100

---

## Table of Contents

1. [Problem Definition & Motivation](#1-problem-definition--motivation)
2. [Dataset](#2-dataset)
3. [Target Variable & Evaluation Metric](#3-target-variable--evaluation-metric)
4. [Planned Architecture: BDH-GPU](#4-planned-architecture-bdh-gpu)
5. [Baseline Models](#5-baseline-models)
6. [Engineering & MLOps Plan](#6-engineering--mlops-plan)
7. [Experiment Plan](#7-experiment-plan)
8. [Compute Budget Breakdown](#8-compute-budget-breakdown)
9. [Meeting-by-Meeting Roadmap](#9-meeting-by-meeting-roadmap)
10. [Risks & Mitigations](#10-risks--mitigations)
11. [Proposal Presentation Outline (Meeting 3)](#11-proposal-presentation-outline-meeting-3)

---

## 1. Problem Definition & Motivation

### Problem
Train a neural network to **evaluate chess positions** (predict Stockfish centipawn scores) without any tree search — relying purely on learned "intuition" from static board representations.

### Why Deep Learning is Appropriate
- Chess position evaluation from a static board is a **high-dimensional pattern recognition** task — exactly where deep learning excels.
- Prior work (Google DeepMind, 2024; Grzyb, 2025) has proven that neural networks can achieve master-level play (1960+ ELO) without search.
- The problem has a **clean supervised formulation**: FEN → normalized centipawn score (regression).

### Why BDH (Novel Contribution)
- BDH (Brain-Derived Hebbian) is a **post-transformer architecture** that replaces softmax attention with linear attention + Hebbian fast-state updates.
- Its **weight-tied recurrent layers** naturally support variable inference depth — more "thinking" for harder positions.
- Chess is a **relation-heavy** domain: piece interactions, pins, forks, batteries. BDH's neuron-pair synaptic state is architecturally aligned with this.
- **No prior work** has applied BDH to chess or any combinatorial game domain — this is genuinely novel.
- Compared to ViT (the best architecture in Grzyb's work), BDH offers a different inductive bias: iterative relational reasoning vs. one-shot global attention.

### Research Questions
1. Can BDH-GPU match or exceed ViT performance on searchless chess evaluation?
2. How does BDH scale with parameter count and recurrent depth?
3. Does BDH's recurrent structure yield better position evaluation than single-pass architectures of similar size?

---

## 2. Dataset

### Source
**Lichess-Stockfish-Normalized** — published on HuggingFace by Mateusz Grzyb:
- **URL:** https://huggingface.co/datasets/mateuszgrzyb/lichess-stockfish-normalized
- **License:** CC BY 4.0
- **Original source:** [Lichess/chess-position-evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations) (~784M positions with deep Stockfish analysis)

### Statistics
| Property | Value |
|----------|-------|
| Total positions | 316,072,343 |
| Format | Parquet (10 shards, ~32M each) |
| Deduplication | FEN-based, keeping max depth |
| Key columns | `fen`, `depth`, `knodes`, `cp`, `mate` |

### Why This Dataset
- **Publicly available**, well-documented, already deduplicated — saves weeks of data engineering.
- **Large enough** for serious deep learning — 316M unique positions.
- **Proven** to produce 1960 ELO models (Grzyb's ViT-Medium), so we have direct comparison baselines.
- The existing repo provides a complete **5-stage data pipeline** that we can reference (and partially reuse) for reproducibility.

### Data Usage Plan
- We will **not** train on all 316M positions (compute-constrained). Plan: ~50–100M positions sampled uniformly.
- **Train/Val/Test split:** 90% / 5% / 5%, stratified by cp range to ensure balanced evaluation.
- **Encoding:** FEN → 8×8×12 tensor (binary piece-channel encoding, always-white perspective), matching Grzyb's pipeline for fair comparison.
- Data stored in **TFRecord** format for efficient streaming to GPU.

### Download Script
We will provide a `scripts/data_preparation/download_hf_data.py` script using `datasets` library:
```python
from datasets import load_dataset
dataset = load_dataset("mateuszgrzyb/lichess-stockfish-normalized", split="train")
```

---

## 3. Target Variable & Evaluation Metric

### Target Variable
**Normalized centipawn score** — a continuous value in `[-1, 1]` representing position quality from the side-to-move's perspective.

Conversion pipeline (from Grzyb's repo):
1. Mate-in-N → centipawn equivalent (using `mate_to_cp()`)
2. Centipawn → normalized `[-1, 1]` range (tanh-like scaling)

### Primary Metric: MSE (Mean Squared Error)
- Standard regression loss for position evaluation.
- Directly comparable to Grzyb's models (they use MSE + MAE).

### Secondary Metrics
| Metric | Purpose |
|--------|---------|
| **MAE** | Interpretable average error |
| **ELO estimation** | Via Lichess puzzle benchmark (1200 puzzles, 12 tiers) — the gold standard from Grzyb's methodology |
| **Puzzle accuracy per tier** | Fine-grained strength analysis across difficulty levels |
| **Parameter efficiency** | ELO per million parameters |

### ELO Estimation Methodology
Following Grzyb's proven approach:
1. Model solves 1,200 Lichess puzzles (100 per tier, ratings 399–3213)
2. Compute success rate per tier
3. Linear regression: tier rating vs. accuracy
4. ELO = predicted rating at 50% accuracy threshold

The puzzle dataset is already available in `data/puzzles/test_puzzles.feather`.

---

## 4. Planned Architecture: BDH-GPU

### Architecture Overview

BDH-GPU is a **recurrent linear-attention model with Hebbian state updates**. Core components:

```
Input: 8×8×12 board tensor
  ↓
Patch Embedding (flatten 64 squares → sequence of 64 tokens, project to d dimensions)
  ↓
┌─────────────────────────────────────────────┐
│  BDH-GPU Layer (shared weights, applied K×) │
│                                             │
│  1. Residual ReLU low-rank update of x      │
│  2. Linear attention read from fast state ρ │
│  3. Sparse gated output y                   │
│  4. Write back to fast state ρ (Hebbian)    │
└─────────────────────────────────────────────┘
  ↓  (repeat K times with same weights)
Global Average Pooling
  ↓
MLP Head → tanh → scalar ∈ (-1, 1)
```

### Key Trainable Parameters
| Matrix | Shape | Role |
|--------|-------|------|
| E | (n, d) | Encoding / retrieval from synaptic state |
| D_x | (n, d) | Input-side transformation |
| D_y | (n, d) | Output-side transformation |

Total scalable params ≈ `(3 + o(1)) × n × d`

### Planned Model Sizes
| Variant | n (neurons) | d (rank) | K (depth) | ~Params | Training budget |
|---------|-------------|----------|-----------|---------|-----------------|
| BDH-Tiny | 4096 | 128 | 4 | ~1.6M | 5h |
| BDH-Small | 8192 | 256 | 8 | ~6.3M | 20h |
| BDH-Medium | 16384 | 256 | 12 | ~12.6M | 40h |

### Implementation Strategy
- Implement BDH-GPU from scratch in **Keras 3 / TensorFlow 2** (matching the existing repo's framework).
- Custom `keras.Layer` subclass: `BDHGPUBlock` — one reusable recurrent step.
- Custom `keras.Model` subclass: `BDHChessModel` — wraps embedding, K-step unroll, pooling, head.
- All hyperparameters externalized to YAML config.

### Input Representation
Reuse Grzyb's encoding: `fen_to_tensor()` → 8×8×12 uint8 binary tensor (always-white perspective). This encoding is well-tested and lets us directly compare to baseline ViT/ResNet results.

---

## 5. Baseline Models

For meaningful comparison, we will train (or use pretrained weights from) the following baselines:

| Model | Source | Params | Expected ELO | Purpose |
|-------|--------|--------|-------------|---------|
| **CNN-S** | Grzyb's repo | 1.18M | ~1112 | Weak baseline |
| **ViT-Small** | Grzyb's repo / retrain | 2.64M | ~1817 | Strong baseline (similar param count to BDH-Small) |
| **MLP (flat)** | Our implementation | ~2M | ~800–1000 | Sanity-check "no-structure" baseline |

Baselines serve two purposes:
1. Validate our training pipeline produces known-good results.
2. Provide fair parameter-matched comparisons for BDH.

---

## 6. Engineering & MLOps Plan

> **Teacher emphasis:** "This project is more about engineering than research."

This section is the backbone of the project. Every grading criterion from the course outline is addressed.

### 6.1 Repository Structure

```
searchless-chess/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Lint + test on push/PR
│       └── train-smoke.yml     # Optional: smoke-test training on CPU
├── configs/
│   ├── config.yaml             # Data pipeline config (existing)
│   ├── train_bdh_tiny.yaml     # BDH-Tiny training config
│   ├── train_bdh_small.yaml    # BDH-Small training config
│   ├── train_bdh_medium.yaml   # BDH-Medium training config
│   └── train_vit_baseline.yaml # ViT baseline config
├── data/
│   ├── puzzles/                # Evaluation puzzles (existing)
│   └── samples/                # Debug samples (existing)
├── docs/
│   └── ...
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_results.ipynb
│   ├── 03_bdh_results.ipynb
│   ├── 04_error_analysis.ipynb
│   └── 05_final_comparison.ipynb
├── scripts/
│   ├── data_preparation/       # Existing pipeline
│   └── training/
│       ├── train.py            # Unified, config-driven entry point
│       └── evaluate.py         # ELO estimation script
├── src/
│   ├── models/
│   │   ├── bdh.py              # ★ BDH-GPU implementation
│   │   ├── resnet.py           # Existing
│   │   ├── shallow_cnn.py      # Existing
│   │   └── vision_transformer.py  # Existing
│   ├── data_preparation/       # Existing
│   ├── training/
│   │   ├── callbacks.py        # Existing + W&B callback
│   │   └── trainer.py          # Unified training loop
│   ├── evaluation/
│   │   └── puzzle_evaluator.py # ELO estimation module
│   └── utils/
│       └── tools.py            # Existing
├── tests/
│   ├── test_bdh_model.py       # ★ BDH forward pass, shape checks
│   ├── test_data_downloading.py
│   ├── test_data_processing.py
│   ├── test_training_smoke.py  # ★ 1-batch training smoke test
│   └── test_loss.py            # ★ Loss computation sanity check
├── pyproject.toml
├── requirements.txt
└── README.md
```

### 6.2 Environment Management

**Required by course: `uv`**

The existing repo uses Poetry. We must **migrate to `uv`** to satisfy the course requirement:

```bash
# Target workflow
uv sync
uv run python scripts/training/train.py --config configs/train_bdh_small.yaml
```

Action items:
- [ ] Add `uv.lock` and ensure `uv sync` works
- [ ] Keep `pyproject.toml` as the single source of truth for dependencies
- [ ] Add `wandb` to dependencies
- [ ] Document installation in README

### 6.3 CI/CD (GitHub Actions)

**Current state:** CI exists but only runs `pytest` via Poetry.

**Target state:**
```yaml
# .github/workflows/ci.yml
jobs:
  lint:
    - ruff check src/ tests/
    - ruff format --check src/ tests/
  test:
    - uv sync
    - uv run pytest tests/ -v
  type-check:   # bonus
    - uv run mypy src/ --ignore-missing-imports
```

Action items:
- [ ] Migrate CI from Poetry to `uv`
- [ ] Add `ruff` linter step (course requires ruff or flake8)
- [ ] Ensure CI passes before every milestone meeting
- [ ] Add badge to README

### 6.4 Testing (pytest)

**Course minimum required tests:**
| Test | File | Status |
|------|------|--------|
| Data loading | `test_data_downloading.py` | ✅ Exists |
| Data processing | `test_data_processing.py` | ✅ Exists |
| BDH forward pass | `test_bdh_model.py` | 🔲 To implement |
| BDH output shape | `test_bdh_model.py` | 🔲 To implement |
| Loss computation | `test_loss.py` | 🔲 To implement |
| Training smoke test | `test_training_smoke.py` | 🔲 To implement |
| Evaluation pipeline | `test_evaluation.py` | 🔲 To implement |

Each test must be fast (< 5s) and runnable without GPU.

### 6.5 Experiment Tracking (Weights & Biases)

**Course requirement: ≥5 logged experiments, W&B project link in README.**

W&B integration plan:

```python
# In training loop
import wandb
wandb.init(
    project="bdh-searchless-chess",
    config={...from YAML...},
    tags=["bdh-small", "epoch-100", "lr-1e-4"],
)
wandb.log({"train/loss": loss, "val/loss": val_loss, "val/mae": val_mae})
wandb.log({"elo_estimate": elo, "puzzle_accuracy_tier_6": acc})
```

Planned experiments to log (minimum):
| # | Experiment | Model | Purpose |
|---|-----------|-------|---------|
| 1 | MLP baseline | MLP-2M | Sanity check |
| 2 | ViT baseline | ViT-Small (2.64M) | Reproduce known result |
| 3 | BDH-Tiny K=4 | BDH-Tiny | First BDH run |
| 4 | BDH-Small K=8 | BDH-Small | Main model |
| 5 | BDH-Small K=4 vs K=8 vs K=12 | BDH-Small | Depth ablation |
| 6 | BDH-Medium K=12 | BDH-Medium | Scale-up |
| 7 | BDH-Small + adaptive depth | BDH-Small | Advanced experiment |

### 6.6 Reproducibility

| Requirement | Implementation |
|-------------|---------------|
| Random seeds | `random.seed(S)`, `tf.random.set_seed(S)`, `np.random.seed(S)` — all from YAML config |
| Fixed data splits | Deterministic shuffle with seed, or explicit file-list split saved to JSON |
| Config-driven training | All hyperparams in YAML, logged to W&B |
| Best model saving | `ModelCheckpoint(monitor='val_loss', save_best_only=True)` |
| Reproducible environment | `uv.lock` pinning exact versions |
| Evaluation reproducibility | Fixed puzzle set (`test_puzzles.feather`) |

### 6.7 Configuration System

One YAML config per experiment:

```yaml
# configs/train_bdh_small.yaml
experiment:
  name: "bdh-small-k8"
  seed: 2001
  wandb_project: "bdh-searchless-chess"
  wandb_tags: ["bdh-small", "k8"]

data:
  source: "mateuszgrzyb/lichess-stockfish-normalized"
  num_samples: 50_000_000  # 50M subset
  train_ratio: 0.90
  val_ratio: 0.05
  test_ratio: 0.05
  batch_size: 512
  encoding:
    always_white_perspective: true
    output_shape: [8, 8, 12]

model:
  architecture: "bdh"
  n_neurons: 8192
  d_rank: 256
  k_depth: 8
  dropout: 0.1
  activation: "relu"

training:
  epochs: 100
  learning_rate: 1e-4
  min_learning_rate: 1e-6
  weight_decay: 1e-4
  clipnorm: 1.0
  warmup_epochs: 5
  scheduler: "cosine"
  mixed_precision: true  # fp16 on A100

callbacks:
  early_stopping:
    patience: 5
    monitor: "val_loss"
  checkpoint:
    save_every: 5
    monitor: "val_loss"
  wandb:
    enabled: true
```

### 6.8 Unified Training Entry Point

Replace per-model scripts with a single config-driven entry:

```bash
uv run python scripts/training/train.py --config configs/train_bdh_small.yaml
```

The script:
1. Loads config
2. Sets seeds
3. Initializes W&B
4. Builds model (factory pattern: config → architecture)
5. Loads data
6. Trains with callbacks
7. Evaluates on puzzle set
8. Logs final ELO to W&B

### 6.9 Bonus Points Opportunities

| Bonus | Feasibility | Plan |
|-------|-------------|------|
| Docker | ✅ Easy | Dockerfile with CUDA + uv |
| Deployment (Streamlit) | ✅ Medium | Interactive board + model inference demo |
| ONNX export | ⚠️ Medium | Export best model, benchmark inference |
| Inference benchmarking | ✅ Easy | Time per position, batch throughput |
| Hyperparameter search | ✅ Medium | W&B Sweeps for BDH hyperparams |

---

## 7. Experiment Plan

### Phase 1: Infrastructure & Baselines (Meetings 4–5)
- Set up repo structure, CI, uv, W&B
- Implement data pipeline (reuse/adapt from existing repo)
- Run EDA notebook on the dataset
- Train MLP baseline (sanity check)
- Retrain or load ViT-Small baseline → verify we get ~1817 ELO

### Phase 2: BDH Implementation & First Results (Meetings 6–7)
- Implement `BDHGPUBlock` layer in Keras
- Implement `BDHChessModel` with configurable K (depth)
- Write unit tests (forward pass, shapes, gradient flow)
- Train BDH-Tiny (5h) — quick iteration
- Train BDH-Small (20h) — main model
- Log to W&B, compare with ViT baseline

### Phase 3: Scaling & Ablations (Meetings 8–10)
- **Depth ablation:** BDH-Small with K=4, 8, 12 — does more recurrence help?
- **Scale ablation:** BDH-Tiny vs Small vs Medium
- Train BDH-Medium (40h) — push for best ELO
- **(Advanced)** Adaptive depth: add halting criterion, train with variable K
- Error analysis: which puzzle tiers fail? What positions does BDH get wrong vs. ViT?

### Phase 4: Analysis & Presentation (Meetings 11–12)
- Compile all results
- Error analysis notebook
- Overfitting/underfitting analysis
- Write conclusions: does BDH's inductive bias help for chess?
- Final presentation

---

## 8. Compute Budget Breakdown

| Task | GPU Hours | Cumulative |
|------|-----------|-----------|
| Data pipeline + encoding | 2h | 2h |
| MLP baseline | 2h | 4h |
| ViT-Small baseline (or load pretrained) | 5h | 9h |
| BDH-Tiny (quick iteration) | 5h | 14h |
| BDH-Small (main model) | 20h | 34h |
| BDH depth ablation (K=4,8,12 × short runs) | 15h | 49h |
| BDH-Medium | 40h | 89h |
| Evaluation runs + misc | 5h | 94h |
| **Buffer** | **6h** | **100h** |

This leaves a 6h safety buffer. If BDH-Medium is cut, we save 40h for more ablations.

---

## 9. Meeting-by-Meeting Roadmap

| Meeting | Date | Deliverables | Engineering Focus |
|---------|------|-------------|-------------------|
| **3** | Mar 10 | Proposal presentation | Problem + dataset + metric + architecture + risks |
| **4** | Mar 17 | Repo & CI review | Public GitHub, uv, CI passing, project structure, basic tests |
| **5** | ~Mar 24 | EDA + baseline | EDA notebook, MLP baseline, ViT baseline results |
| **6** | ~Mar 31 | Reproducible training + W&B | Config-driven training, seeds, W&B logging, ≥2 experiments |
| **7** | ~Apr 7 | First BDH results | BDH-Tiny/Small trained, compared to baselines |
| **8** | ~Apr 14 | Model improvements | BDH-Small tuned, depth ablation, before/after comparison |
| **9** | ~Apr 21 | Error analysis | Confusion by tier, failure examples, overfit/underfit analysis |
| **10** | ~Apr 28 | Advanced experiments | Depth ablation study, BDH-Medium, adaptive depth (optional) |
| **11** | ~May 5 | Final review | CI green, clean repo, README polished, W&B organized |
| **12** | ~May 12 | Final presentation | 15–20 min presentation covering all aspects |

---

## 10. Risks & Mitigations

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| **BDH underperforms significantly** | High | Medium | We still have ViT baseline + comprehensive comparison = valid research contribution ("negative results are results"). Pivot messaging to "architecture comparison study." |
| **BDH implementation bugs** | Medium | High | Start with BDH-Tiny, extensive unit tests, verify gradient flow, compare against paper's toy examples if available. |
| **Compute budget overrun** | Medium | Low | Strict hour tracking. BDH-Medium is expendable — cut if behind schedule. |
| **Data pipeline issues** | Low | Low | Reuse proven pipeline from Grzyb's repo. HuggingFace `datasets` library handles downloading. |
| **W&B integration breaks training** | Low | Low | W&B callback is fire-and-forget. Offline mode fallback. |
| **CI flakiness** | Low | Medium | Pin all versions in uv.lock. Tests must be deterministic and fast. |
| **Course requires `uv`, repo uses Poetry** | Medium | Certain | Migrate early (Meeting 4). This is a known, straightforward task. |
| **No BDH reference implementation** | High | Certain | We implement from scratch — this is actually a plus for "engineering quality" grading. Document the implementation thoroughly. |

### Fallback Strategy
If BDH completely fails to learn (no convergence):
1. Debug with tiny dataset + BDH-Tiny (fast iteration cycles)
2. Try simpler variants: remove Hebbian updates, reduce to standard linear attention
3. Worst case: project becomes "Attempting BDH for Chess: Lessons Learned" — still valid with strong engineering

---

## 11. Proposal Presentation Outline (Meeting 3)

> **Required: 5–7 minutes.** Must include: problem, dataset, metric, architecture, risks.

### Slide 1: Title
**"BDH-GPU for Searchless Chess: Training a Post-Transformer Architecture on Position Evaluation"**

### Slide 2: Problem Definition
- Searchless chess: evaluate positions without tree search
- Input: FEN (board state) → Output: position score ∈ [-1, 1]
- Prior art: DeepMind (2895 ELO, 270M params), Grzyb (1960 ELO, 9.5M params, ViT)
- **Our angle:** Can a novel post-transformer architecture (BDH) do it better/differently?

### Slide 3: Dataset
- Lichess-Stockfish-Normalized: 316M positions, HuggingFace
- Encoding: FEN → 8×8×12 tensor
- We use ~50–100M subset (compute-constrained)
- Proven dataset — known to produce 1960 ELO models

### Slide 4: Architecture — BDH-GPU
- Shared-weight recurrent layers with Hebbian state updates
- Linear attention + sparse activations
- Variable depth K = {4, 8, 12}
- Key insight: "iterative reasoning" — more passes for harder positions
- Diagram of BDH block

### Slide 5: Evaluation Metric
- Primary: MSE on validation set
- Gold standard: ELO estimation via 1200 Lichess puzzles (12 tiers, linear regression at 50% accuracy)
- Comparison: BDH vs. ViT-Small (matched params) vs. MLP baseline

### Slide 6: Engineering Plan
- `uv` + CI/CD + ruff + pytest + W&B
- Config-driven, reproducible training pipeline
- ≥7 logged experiments
- Unified training entry point

### Slide 7: Risks & Timeline
- Main risk: BDH is untested for chess → mitigated by strong baselines + engineering focus
- 100h A100 compute budget, carefully allocated
- 10-meeting roadmap from repo setup to final presentation

---

## Appendix A: One-Page Written Proposal (Post-Meeting 3 Submission)

> **Required after Meeting 3:** max 1 page.

**Problem:** Train a neural network to evaluate chess positions (predict Stockfish scores) without search, using BDH-GPU — a novel post-transformer architecture with Hebbian state dynamics.

**Dataset:** Lichess-Stockfish-Normalized (HuggingFace), 316M unique positions with deep Stockfish evaluations. We use a 50-100M subset encoded as 8×8×12 tensors.

**Target variable:** Normalized centipawn score ∈ [-1, 1] (side-to-move perspective).

**Metric:** MSE (training), ELO estimation via Lichess puzzle benchmark (evaluation).

**Planned architecture:** BDH-GPU — a recurrent linear-attention model with shared weights across layers and Hebbian fast-state updates. Three scales: Tiny (1.6M), Small (6.3M), Medium (12.6M). Baselines: MLP, ViT-Small.

---

## Appendix B: Grading Criteria Alignment

| Criterion | Points | Our Coverage |
|-----------|--------|-------------|
| Problem formulation & methodology | 8 | Novel architecture + clear research questions + proven evaluation methodology |
| Data analysis & preprocessing | 8 | EDA notebook + reuse of battle-tested 5-stage pipeline + 316M position dataset |
| Modeling quality | 10 | BDH implementation from scratch + 3 model sizes + baselines + depth ablation |
| Experimental rigor & analysis | 8 | ≥7 W&B experiments, ablation studies, error analysis by puzzle tier |
| Engineering & DevOps quality | 10 | uv, CI/CD, ruff, pytest (≥7 tests), config-driven, reproducible seeds, W&B |
| Presentation | 6 | Well-structured 15-20 min final presentation with visualizations |
| **Total** | **50** | **All criteria explicitly addressed** |

**Bonus opportunities:** Docker (+), Streamlit demo (+), inference benchmarking (+), W&B Sweeps (+).
