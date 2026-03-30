# Searchless Chess — Neural Position Evaluation Without Tree Search

A deep learning approach to chess position evaluation that predicts Stockfish centipawn scores directly from board state, without any tree search. Reimplements and extends the ViT architecture from [mateuszgrzyb/searchless-chess](https://github.com/mateuszgrzyb/searchless-chess).

**Best result: ViT-S + Attention Residuals — ELO 1810** (reference: 1817)

---

## Problem Statement

Given an 8×8×12 binary tensor representing a chess position (6 piece types × 2 colors), predict the Stockfish evaluation normalized to [-1, 1] via `tanh(cp/1000)`. The model is evaluated on 1,200 rated Lichess puzzles to estimate playing strength (ELO).

---

## Repository Structure

```
searchless-chess/
├── src/                        # Source code
│   ├── data/                   # Dataset, encoding, Lightning DataModule
│   ├── evaluation/             # Puzzle-based ELO benchmark
│   ├── models/                 # MLP, BDH, ViT, ViT-AttnRes
│   └── training/               # Lightning module, legacy trainer
├── tests/                      # pytest suite
├── configs/                    # YAML experiment configs
│   └── smoke/                  # Tiny configs for CI/testing
├── scripts/                    # CLI entrypoints (train, evaluate, prepare)
├── data/
│   └── puzzles/                # Lichess puzzle benchmark (87 KB)
├── report/
│   ├── eda/                    # Exploratory data analysis
│   ├── experiments/            # Per-experiment reports
│   ├── summary/                # Final results summary
│   └── research/               # Literature notes
├── .github/workflows/          # CI pipeline
├── university_project/         # Course presentations and plans
├── pyproject.toml              # Project metadata and dependencies
├── uv.lock                     # Locked dependency versions
└── README.md
```

---

## Installation

### Using UV (recommended)

```bash
git clone <repo-url> && cd searchless-chess

# Install all dependencies
uv sync --all-groups

# Or install specific groups
uv sync                     # core only
uv sync --group train       # + pytorch-lightning, wandb
uv sync --group dev         # + pytest, ruff
```

### HPC Notes (Ares / plgrid)

On the HPC cluster, use the existing conda environment:
```bash
source activate /net/tscratch/people/plgantoniczapski/conda-envs/bdh-chess
```

---

## Quickstart

### Run tests
```bash
uv run pytest
```

### Smoke training (CPU, synthetic data, ~30 seconds)
```bash
uv run python scripts/train.py --config configs/smoke/vit_smoke.yaml --smoke-test
```

### Full training (GPU required)
```bash
uv run python scripts/train.py --config configs/train_vit_attnres_a100.yaml
```

### Resume from checkpoint
```bash
uv run python scripts/train.py --config configs/train_vit_attnres_a100.yaml --resume auto
```

### Evaluate a trained model
```bash
uv run python scripts/evaluate.py \
    --config configs/train_vit_attnres_a100.yaml \
    --checkpoint path/to/best_model.pt
```

---

## Models

| Model            | Params | ELO  | Description |
|------------------|--------|------|-------------|
| MLP baseline     | 200K   | ~900 | Flat MLP, no spatial structure |
| BDH v4           | 2.5M   | 1177 | Iterative value refinement with synaptic scratchpad |
| ViT-S v2         | 2.6M   | 1621 | Standard Vision Transformer (5 layers, dim=256) |
| **ViT-S+AttnRes**| **2.66M** | **1810** | **Windowed Attention Residuals (W=3), SwiGLU, RMSNorm** |
| Reference ViT-S  | 2.6M   | 1817 | mateuszgrzyb reference implementation |

---

## Data

- **Source:** [mateuszgrzyb/lichess-stockfish-normalized](https://huggingface.co/datasets/mateuszgrzyb/lichess-stockfish-normalized) (50M positions)
- **Encoding:** 8×8×12 binary tensor (piece placement, side-to-move perspective)
- **Target:** `tanh(centipawn / 1000)` ∈ [-1, 1]
- **Splits:** 90/5/5 train/val/test

### Prepare data
```bash
uv run python scripts/prepare_data.py --config configs/train_mlp.yaml
```

### EDA
See [report/eda/eda.md](report/eda/eda.md) for detailed dataset analysis.

---

## Reports

- **[EDA](report/eda/eda.md)** — Dataset exploration and target distribution analysis
- **[Final Summary](report/summary/final_experiment_summary.md)** — Key findings and model comparison
- **[Experiment Reports](report/experiments/)** — Detailed per-experiment logs:
  - [ViT-AttnRes](report/experiments/vit_attnres.md) — Best model analysis
  - [ViT v2 (fixed)](report/experiments/vit_v2_fixed.md) — Score fix impact
  - [BDH v4 (fixed)](report/experiments/bdh_v4_fixed.md) — BDH final results

---

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run specific test file
uv run pytest tests/test_model_forward_vit.py -v
```

Tests cover:
- Config loading and validation
- Board encoding correctness
- Model forward pass (all architectures)
- Lightning training smoke test
- Evaluation pipeline smoke test

---

## CI

GitHub Actions runs on every push/PR:
- Dependency installation via UV
- Ruff linting
- Full pytest suite

See [.github/workflows/ci.yml](.github/workflows/ci.yml).

---

## Reproducibility

- All experiments are config-driven (YAML files in `configs/`)
- Seeds are fixed (`seed: 42`) for deterministic splits and initialization
- `uv.lock` pins exact dependency versions
- Training supports resume from checkpoint (`--resume auto`)

---

## License

MIT
