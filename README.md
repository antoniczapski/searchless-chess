# BDH Architecture for Searchless Chess

> **Course:** Deep Learning Project — University, Semester VI  
> **Author:** Antoni Czapski  
> **Date:** March 2026

Train a neural network to **evaluate chess positions** (predict Stockfish centipawn scores) without any tree search — relying purely on learned pattern recognition from static board representations.

The core contribution is adapting **BDH-GPU** (Brain-Derived Hebbian, [Pathway Technology 2025](https://github.com/Pathway-AI/BDH-GPU)) — a post-transformer architecture with linear attention and Hebbian fast-state updates — to the chess evaluation domain using **iterative value refinement**.

---

## Key Idea

Standard transformers process the board as a flat sequence. BDH-Chess instead:

1. **Encodes the board once** into a fixed embedding.
2. **Runs K "thinking steps"** with a single shared BDH block — reinterpreting BDH's time axis as internal refinement.
3. **Maintains a synaptic state ρ** as a working-memory scratchpad that accumulates relational patterns (pins, forks, batteries, etc.) across steps.
4. **Re-injects the board** at every step to prevent drift.

Each thinking step performs: linear attention read → Hebbian gating → decode → neuron update → synaptic write → value prediction.

This preserves BDH's distinctive properties:
- Shared weights across steps (Universal-Transformer style)
- Persistent neuron-aligned synaptic state
- Sparse positive activations
- Variable compute by choosing K

---

## Results Summary

| Model | Params | Data | Val MAE | Puzzle Acc. | VRAM (AMP) |
|---|---|---|---|---|---|
| MLP baseline | 263K | 50K | ~0.145 | 0.7% | ~200 MB |
| BDH-v1 (sequence) | 796K | 500K | 0.1451 | 0.5% | ~1,500 MB |
| **BDH-v2 (iterative)** | **682K** | **500K** | **0.1257** | **0.8%** | **681 MB** |

BDH-v2 vs. v1: **−13.4% MAE**, **−55% VRAM**, **−57% time/epoch**, **4× longer learning before overfitting**.

Full reports in [`project/report/`](project/report/).

> Puzzle accuracy is low because (a) the model predicts *scalar evaluations*, not *moves*, and (b) 500K training positions is ~600× smaller than what reference ViT models use. The primary bottleneck is data scale, not architecture.

---

## Repository Structure

```
searchless-chess/
├── project/                    # ★ Main project code
│   ├── configs/                # YAML experiment configurations
│   │   ├── train_mlp.yaml      # MLP baseline
│   │   ├── train_bdh.yaml      # BDH v2 (RTX 2060, 500K data)
│   │   └── train_bdh_a100.yaml # BDH v3 (A100, 50M data)
│   ├── data/
│   │   ├── puzzles/            # 1200 Lichess rated puzzles (committed)
│   │   └── processed/          # ★ Generated .npz files (gitignored)
│   ├── scripts/
│   │   ├── setup.py            # One-command data download + preparation
│   │   ├── prepare_data.py     # Config-driven data preparation
│   │   ├── train.py            # Config-driven training entry point
│   │   └── evaluate.py         # Puzzle-based ELO estimation
│   ├── src/
│   │   ├── data/               # Encoding, preparation, dataset loading
│   │   ├── models/             # MLP, BDH (with registry)
│   │   ├── training/           # Trainer with AMP, early stopping, W&B
│   │   └── evaluation/         # Puzzle ELO benchmark
│   ├── tests/                  # pytest (26 tests)
│   ├── report/                 # Experiment reports (v1, v2)
│   └── ml_pipeline.md          # Pipeline documentation
├── university_project/         # Course docs, presentations, project plan
└── .gitignore
```

---

## Quick Start (After Cloning)

### 1. Environment Setup

```bash
# Create and activate environment (conda example)
conda create -n bdh-chess python=3.11 -y
conda activate bdh-chess

# Install PyTorch with CUDA (adjust for your GPU)
# A100 / CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install datasets chess python-chess pandas numpy scikit-learn loguru pyyaml pyarrow
```

### 2. Download & Prepare Data

```bash
cd project/

# Default: 500K positions (quick, ~2 min download + ~5 min encoding)
python scripts/setup.py

# Large-scale: 50M positions (for A100, ~30 min)
python scripts/setup.py --num-samples 50000000

# Or use a specific config
python scripts/prepare_data.py --config configs/train_bdh.yaml
```

The setup script will:
- Verify your Python/CUDA environment
- Ensure the puzzle benchmark file exists
- Download positions from HuggingFace (`mateuszgrzyb/lichess-stockfish-normalized`)
- Encode FEN → 8×8×12 tensors and create train/val/test `.npz` splits

### 3. Train

```bash
cd project/

# BDH v2 — RTX 2060 / small GPU (500K data, ~30 min)
python scripts/train.py --config configs/train_bdh.yaml

# BDH v3 — A100 (50M data, larger model)
python scripts/train.py --config configs/train_bdh_a100.yaml

# MLP baseline (50K data, ~2 min)
python scripts/train.py --config configs/train_mlp.yaml
```

### 4. Evaluate

```bash
cd project/

python scripts/evaluate.py \
    --config configs/train_bdh.yaml \
    --checkpoint outputs/bdh-chess-v2/checkpoints/best_model.pt
```

### 5. Run Tests

```bash
cd project/
pytest tests/ -v
```

---

## A100 VM Workflow

For large-scale experiments on an NVIDIA A100 VM:

```bash
# 1. Clone
git clone https://github.com/mateuszgrzyb-pl/searchless-chess.git
cd searchless-chess

# 2. Environment
conda create -n bdh-chess python=3.11 -y
conda activate bdh-chess
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install datasets chess python-chess pandas numpy scikit-learn loguru pyyaml pyarrow

# 3. Download 50M positions + prepare splits (~30 min)
cd project/
python scripts/setup.py --num-samples 50000000

# 4. Train large BDH model
python scripts/train.py --config configs/train_bdh_a100.yaml

# 5. Evaluate
python scripts/evaluate.py \
    --config configs/train_bdh_a100.yaml \
    --checkpoint outputs/bdh-chess-v3-a100/checkpoints/best_model.pt
```

---

## Architecture Details

### BDH-Chess v2 — Iterative Value Refinement

```
Input (B, 12, 8, 8) → flatten → board_encoder → board embedding b ∈ ℝ^d
                                                        │
    ┌───────────────────────────────────────────────────┘
    │   Initialize: x₀ = ReLU(W_init · b), ρ₀ = 0
    │
    │   For k = 1 … K (shared weights):
    │     1. a_k = ρ_{k-1}^T · x_{k-1}         ← linear attention read
    │     2. y_k = ReLU(D_y · LN(a + B_y·b)) ⊙ x  ← Hebbian gating
    │     3. z_k = LN(E · y_k)                  ← decode to embedding
    │     4. x_k = x + ReLU(D_x(z + B_x·b))    ← neuron update
    │     5. ρ_k = λρ_{k-1} + (1-λ)x̂⊗z         ← EMA state write
    │     6. v_k = tanh(head(z_k))               ← value prediction
    │
    └──→ Output: v_K ∈ (-1, 1)
```

**Loss:** Huber + α·(deep supervision on all K steps) + β·(stability penalty)

### Training Setup

| Parameter | RTX 2060 config | A100 config |
|---|---|---|
| `n_embd` | 128 | 256 |
| `n_head` | 4 | 8 |
| `thinking_steps` (K) | 8 | 12 |
| Sparse neurons (total) | 1,024 | 2,048 |
| Parameters | 682K | ~2.5M |
| Batch size | 256 | 1,024 |
| Data | 500K | 50M |
| AMP fp16 | ✓ | ✓ |

---

## Data

**Source:** [mateuszgrzyb/lichess-stockfish-normalized](https://huggingface.co/datasets/mateuszgrzyb/lichess-stockfish-normalized) (316M positions, CC BY 4.0)

| Property | Value |
|---|---|
| Encoding | FEN → 8×8×12 binary tensor (always-white perspective) |
| Target | `tanh(cp / 10000)` ∈ [-1, 1] |
| Puzzle benchmark | 1,200 Lichess puzzles, 12 tiers (399–3213 rating) |

---

## References

- **BDH-GPU:** Pathway Technology (2025). *Brain-Derived Hebbian Learning.* [GitHub](https://github.com/Pathway-AI/BDH-GPU)
- **Searchless Chess:** Grzyb, M. (2025). *Searchless Chess with Neural Networks.* [GitHub](https://github.com/mateuszgrzyb/searchless-chess)
- **Original dataset:** [Lichess/chess-position-evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations) (784M positions)
- **Universal Transformer:** Dehghani et al. (2019). *Universal Transformers.* ICLR.

---

## License

This project is for educational purposes (university course project).  
The dataset is licensed under CC BY 4.0 by Lichess / Mateusz Grzyb.
