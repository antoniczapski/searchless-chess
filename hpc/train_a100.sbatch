#!/bin/bash -l
# ──────────────────────────────────────────────────────────
# BDH-Chess v3 — Large-scale training on A100 via SLURM
# ──────────────────────────────────────────────────────────
#SBATCH -J bdh-chess-v3
#SBATCH -p plgrid-gpu-a100
#SBATCH -A plgolimpiada2026-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bdh-v3_%j.out
#SBATCH --error=logs/bdh-v3_%j.err
#SBATCH --signal=B:SIGINT@120
# ↑ Send SIGINT 120s before wall-time so Python can save checkpoint gracefully

set -euo pipefail

# ── Setup ────────────────────────────────────────────────
REPO_DIR="$SLURM_SUBMIT_DIR"
CONFIG="configs/train_bdh_a100.yaml"
VENV="$HOME/envs/pcg"
SCRATCH="/net/tscratch/people/plgantoniczapski"

# Use scratch for HuggingFace cache (home quota is only 10GB)
export HF_HOME="$SCRATCH/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH/hf_cache/datasets"

mkdir -p "$REPO_DIR/logs" "$HF_HOME" "$HF_DATASETS_CACHE"

echo "========================================"
echo "BDH-Chess v3 — A100 Training"
echo "========================================"
echo "HOSTNAME        = $(hostname)"
echo "SLURM_JOB_ID    = $SLURM_JOB_ID"
echo "CUDA_VISIBLE    = ${CUDA_VISIBLE_DEVICES:-unset}"
echo "REPO_DIR        = $REPO_DIR"
echo "CONFIG          = $CONFIG"
echo "DATE            = $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# ── Activate environment ─────────────────────────────────
source "$VENV/bin/activate"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# ── GPU info ─────────────────────────────────────────────
nvidia-smi || true
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
else:
    print('WARNING: No CUDA device found!')
"

# ── Train (with auto-resume from latest checkpoint) ─────
cd "$REPO_DIR"
echo ""
echo "Starting training with --resume auto ..."
echo "  (Will resume from latest checkpoint if available)"
echo ""

python -u scripts/train.py \
    --config "$CONFIG" \
    --resume auto

echo ""
echo "========================================"
echo "Training finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
