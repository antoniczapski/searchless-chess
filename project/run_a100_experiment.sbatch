#!/bin/bash
#SBATCH --job-name=bdh-chess-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgolimpiada2026-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/net/tscratch/people/plgantoniczapski/bdh-chess-outputs/slurm-%j.out
#SBATCH --error=/net/tscratch/people/plgantoniczapski/bdh-chess-outputs/slurm-%j.err
#SBATCH --signal=B:SIGTERM@120

# ──────────────────────────────────────────────
# BDH-Chess v3 — Full experiment pipeline
# Data prep (from existing chunks) + Training + Evaluation
# ──────────────────────────────────────────────

set -euo pipefail

echo "=============================================="
echo "BDH-Chess A100 Experiment"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "=============================================="

# ---- Environment setup ----
module load CUDA/12.1.1
module load Miniconda3/25.7.0-2
eval "$(conda shell.bash hook)"
conda activate /net/tscratch/people/plgantoniczapski/conda-envs/bdh-chess

# Project directory
PROJECT_DIR="/net/people/plgrid/plgantoniczapski/projects/searchless-chess/project"
cd "$PROJECT_DIR"

# Load W&B credentials
if [ -f "../.env" ]; then
    export $(grep -v '^#' ../.env | xargs)
    echo "Loaded .env credentials"
fi
# Map our key name to what wandb expects
export WANDB_API_KEY="${WEIGHTS_AND_BIASES_KEY:-${WANDB_API_KEY:-}}"

# Ensure output dirs exist
mkdir -p /net/tscratch/people/plgantoniczapski/bdh-chess-outputs

# ---- GPU info ----
echo ""
echo "GPU Information:"
nvidia-smi
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
" || echo "GPU info script failed (non-critical)"

CONFIG="configs/train_bdh_a100.yaml"

# ---- Phase 1: Data preparation ----
echo ""
echo "=============================================="
echo "Phase 1: Data preparation (from existing chunks)"
echo "=============================================="
python scripts/prepare_data.py --config "$CONFIG"

# ---- Phase 2: Training ----
echo ""
echo "=============================================="
echo "Phase 2: Training BDH-Chess v3 on A100"
echo "=============================================="
python scripts/train.py --config "$CONFIG" --resume

# ---- Phase 3: Evaluation ----
echo ""
echo "=============================================="
echo "Phase 3: Evaluation on puzzle benchmark"
echo "=============================================="
BEST_CKPT="/net/tscratch/people/plgantoniczapski/bdh-chess-outputs/bdh-chess-v3-a100-v2/checkpoints/best_model.pt"
if [ -f "$BEST_CKPT" ]; then
    python scripts/evaluate.py --config "$CONFIG" --checkpoint "$BEST_CKPT"
else
    echo "WARNING: No best model checkpoint found at $BEST_CKPT"
    # Try latest
    LATEST_CKPT="/net/tscratch/people/plgantoniczapski/bdh-chess-outputs/bdh-chess-v3-a100-v2/checkpoints/latest_checkpoint.pt"
    if [ -f "$LATEST_CKPT" ]; then
        python scripts/evaluate.py --config "$CONFIG" --checkpoint "$LATEST_CKPT"
    fi
fi

echo ""
echo "=============================================="
echo "Experiment complete!"
echo "Date: $(date)"
echo "=============================================="
