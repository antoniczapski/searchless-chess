# ML Pipeline — Searchless Chess

> Reusable training pipeline for position-evaluation models.
> Designed to be model-agnostic — swap architectures via config YAML.

---

## Quick Start

```bash
# 1. Prepare data (downloads from HuggingFace, encodes, splits)
python scripts/prepare_data.py --config configs/train_mlp.yaml

# 2. Train
python scripts/train.py --config configs/train_mlp.yaml

# 3. Evaluate ELO on puzzle benchmark
python scripts/evaluate.py --config configs/train_mlp.yaml \
    --checkpoint outputs/mlp-baseline/checkpoints/best_model.pt

# Run tests
python -m pytest tests/ -v
```

---

## Project Structure

```
project/
├── configs/
│   └── train_mlp.yaml        # experiment config (model, data, training)
├── data/
│   ├── processed/             # generated .npz splits (train/val/test)
│   └── puzzles/
│       └── test_puzzles.feather  # 1200 rated Lichess puzzles
├── scripts/
│   ├── prepare_data.py        # data download & preparation entry point
│   ├── train.py               # training entry point
│   └── evaluate.py            # ELO evaluation entry point
├── src/
│   ├── data/
│   │   ├── encoding.py        # FEN -> 8x8x12 tensor, cp normalization
│   │   ├── prepare.py         # HF dataset download, encode, split
│   │   └── dataset.py         # PyTorch Dataset & DataLoader factory
│   ├── models/
│   │   ├── registry.py        # model factory & registration system
│   │   └── mlp.py             # MLP baseline architecture
│   ├── training/
│   │   └── trainer.py         # config-driven training loop
│   └── evaluation/
│       └── elo_benchmark.py   # puzzle benchmark & ELO estimation
├── tests/
│   ├── test_encoding.py       # 15 tests for FEN encoding & normalization
│   ├── test_models.py         # 8 tests for model registry & forward pass
│   └── test_training.py       # 3 smoke tests for training loop
└── ml_pipeline.md             # this file
```

---

## Architecture

### Data Flow

```
HuggingFace Parquet  →  prepare.py  →  .npz files  →  DataLoader  →  Trainer
       (FEN + cp)        (encode)     (8x8x12 + score)   (batches)    (train loop)
```

1. **Download** — Streams `mateuszgrzyb/lichess-stockfish-normalized` (316M positions)
2. **Encode** — Each FEN becomes an 8×8×12 binary tensor (6 piece types × 2 colors), always from white's perspective (board is rotated for black-to-move)
3. **Normalize** — Centipawn score → `tanh(cp / 10000)` → `[-1, 1]`
4. **Split** — Deterministic shuffle + configurable train/val/test ratios → `.npz`
5. **Load** — `ChessDataset` memory-maps `.npz`, permutes to `(B, 12, 8, 8)` for conv

### Training Pipeline

The `Trainer` class provides:

| Feature              | Details                                         |
|----------------------|-------------------------------------------------|
| Optimizer            | AdamW (configurable lr, weight decay)           |
| Loss                 | MSE                                             |
| LR Schedule          | Linear warmup → cosine decay                    |
| Gradient clipping    | Max norm (default 1.0)                          |
| Mixed precision      | AMP (auto-disabled on CPU)                      |
| Checkpointing        | Best model + periodic + final                   |
| Early stopping       | Patience + min delta                            |
| Experiment tracking  | W&B (optional, toggle via config)               |
| Metrics              | Train/val loss, MAE, learning rate, epoch time  |

### Evaluation (ELO Estimation)

Uses 1200 rated Lichess puzzles (12 tiers × 100 puzzles):

1. For each puzzle: play opponent's moves, ask model for its moves
2. Model picks the move leading to the **lowest** opponent evaluation
3. Compute accuracy per rating tier
4. Fit `LinearRegression(accuracy → mean_tier_rating)`
5. **ELO** = predicted rating at 50% accuracy

---

## Configuration Format

```yaml
experiment:
  name: "my-experiment"
  seed: 42

wandb:
  enabled: false             # toggle W&B tracking
  project: "project-name"
  tags: ["tag1"]

data:
  hf_dataset: "mateuszgrzyb/lichess-stockfish-normalized"
  output_dir: "data/processed"
  num_samples: 50000         # subset size (max 316M)
  train_ratio: 0.80
  val_ratio: 0.10
  test_ratio: 0.10
  cp_scale: 10000.0          # tanh normalization scale
  batch_size: 256
  num_workers: 0

model:
  architecture: "mlp"        # key in MODEL_REGISTRY
  hidden_dim: 256            # passed as **kwargs to model class
  num_layers: 2
  dropout: 0.1

training:
  epochs: 50
  learning_rate: 0.001
  min_learning_rate: 0.000001
  weight_decay: 0.0001
  clipnorm: 1.0
  mixed_precision: false
  warmup_epochs: 3
  save_every_epochs: 10
  early_stopping:
    patience: 5
    min_delta: 0.00001

evaluation:
  puzzle_path: "data/puzzles/test_puzzles.feather"

output:
  dir: "outputs/my-experiment"
```

---

## Adding a New Model

1. Create `src/models/my_model.py`:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, my_param=64, **kwargs):
        super().__init__()
        # Input: (B, 12, 8, 8)  Output: (B, 1)
        ...

    def forward(self, x):
        ...
        return x  # shape (B, 1), range (-1, 1)
```

2. Register it in `src/models/registry.py`:

```python
from src.models.my_model import MyModel
MODEL_REGISTRY["my_model"] = MyModel
```

3. Create a config:

```yaml
model:
  architecture: "my_model"
  my_param: 128
```

4. Train: `python scripts/train.py --config configs/train_my_model.yaml`

---

## Dependencies

| Package     | Purpose                        |
|-------------|--------------------------------|
| torch       | Models, training, tensors      |
| datasets    | HuggingFace data download      |
| chess       | Board/move logic, FEN parsing  |
| pandas      | Puzzle loading, tier grouping  |
| scikit-learn| Linear regression for ELO      |
| pyyaml      | Config parsing                 |
| loguru      | Structured logging             |
| wandb       | Experiment tracking (optional) |
| pytest      | Testing                        |

---

## Baseline Results (MLP, 50k samples, 5 epochs)

| Metric            | Value          |
|-------------------|----------------|
| Parameters        | 262,913        |
| Best val loss     | 0.1095 (MSE)   |
| Puzzle accuracy   | 0.7%           |
| Estimated ELO     | n/a (random)   |

*Expected: a tiny MLP on 50k samples performs at random level.
Real experiments use 10-100M+ positions with deeper architectures.*
