#!/bin/bash
#SBATCH --job-name=bdh-resume
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
# BDH-Chess v3 — Resume training + Evaluation
# Resumes from latest checkpoint (epoch 19)
# Remaining: ~11 epochs × 73min ≈ 13.4h
# ──────────────────────────────────────────────

set -euo pipefail

echo "=============================================="
echo "BDH-Chess A100 — Resume Training"
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
export WANDB_API_KEY="${WEIGHTS_AND_BIASES_KEY:-${WANDB_API_KEY:-}}"

mkdir -p /net/tscratch/people/plgantoniczapski/bdh-chess-outputs

# ---- GPU info ----
echo ""
echo "GPU Information:"
nvidia-smi
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'bf16: {torch.cuda.is_bf16_supported()}')
" || true

CONFIG="configs/train_bdh_a100.yaml"
OUTPUT_DIR="/net/tscratch/people/plgantoniczapski/bdh-chess-outputs/bdh-chess-v3-a100-v2"

# ---- Phase 1: Resume Training ----
echo ""
echo "=============================================="
echo "Phase 1: Resuming training from latest checkpoint"
echo "=============================================="

LATEST_CKPT="${OUTPUT_DIR}/checkpoints/latest_checkpoint.pt"
if [ -f "$LATEST_CKPT" ]; then
    echo "Found checkpoint: $LATEST_CKPT"
    python -c "
import torch
ckpt = torch.load('$LATEST_CKPT', map_location='cpu', weights_only=False)
print(f'  Epoch: {ckpt[\"epoch\"]}')
print(f'  Val loss: {ckpt[\"val_loss\"]:.6f}')
print(f'  Best val loss: {ckpt[\"best_val_loss\"]:.6f}')
" || true
    python scripts/train.py --config "$CONFIG" --resume
else
    echo "ERROR: No checkpoint found at $LATEST_CKPT"
    exit 1
fi

# ---- Phase 2: Evaluation ----
echo ""
echo "=============================================="
echo "Phase 2: ELO evaluation on puzzle benchmark"
echo "=============================================="

BEST_CKPT="${OUTPUT_DIR}/checkpoints/best_model.pt"
if [ -f "$BEST_CKPT" ]; then
    python scripts/evaluate.py --config "$CONFIG" --checkpoint "$BEST_CKPT"
else
    echo "WARNING: No best model checkpoint found"
    if [ -f "$LATEST_CKPT" ]; then
        python scripts/evaluate.py --config "$CONFIG" --checkpoint "$LATEST_CKPT"
    fi
fi

echo ""
echo "=============================================="
echo "Experiment complete!"
echo "Date: $(date)"
echo "=============================================="
