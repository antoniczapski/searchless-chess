#!/bin/bash
#SBATCH --job-name=bdh-v4
#SBATCH --account=plgolimpiada2026-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/bdh-v4_%j.out
#SBATCH --error=logs/bdh-v4_%j.err

# ──────────────────────────────────────────────
# BDH-Chess v4 — Fixed score perspective + optimized
# Iterative value refinement with synaptic scratchpad
# ──────────────────────────────────────────────
# 48h uninterrupted run — full training budget.
#
# Submit:   sbatch train_bdh_a100.sbatch
# Monitor:  squeue -u $USER
# Cancel:   scancel <job_id>

set -euo pipefail

# ── Configuration ──
CONFIG="configs/train_bdh_v4_a100.yaml"
OUTPUT_DIR="/net/tscratch/people/plgantoniczapski/bdh-chess-outputs/bdh-v4-fixed"
CONDA_ENV="/net/tscratch/people/plgantoniczapski/conda-envs/bdh-chess"
PROJECT_DIR="/net/people/plgrid/plgantoniczapski/projects/searchless-chess/project"

# ── Environment ──
echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "Config: $CONFIG"
echo "Time limit: 48h"

# Activate conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Redirect HuggingFace cache to scratch (home dir has quota limits)
export HF_HOME="/net/tscratch/people/plgantoniczapski/hf-cache"

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

cd "$PROJECT_DIR"

# ── Check for resume ──
RESUME_FLAG=""
CHECKPOINT="${OUTPUT_DIR}/checkpoints/latest_checkpoint.pt"
if [[ -f "$CHECKPOINT" ]]; then
    echo "Found checkpoint: $CHECKPOINT — resuming training"
    RESUME_FLAG="--resume $CHECKPOINT"
fi

# ── Load .env for W&B key ──
if [[ -f "$PROJECT_DIR/.env" ]]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
fi

# ── Train ──
echo ""
echo "=== Starting Training ==="
python scripts/train.py \
    --config "$CONFIG" \
    $RESUME_FLAG

echo ""
echo "=== Training Complete ==="
echo "End: $(date)"

# ── Evaluate ──
BEST_CHECKPOINT="${OUTPUT_DIR}/checkpoints/best_model.pt"
if [[ -f "$BEST_CHECKPOINT" ]]; then
    echo ""
    echo "=== Running Evaluation ==="
    python scripts/evaluate.py \
        --config "$CONFIG" \
        --checkpoint "$BEST_CHECKPOINT"
fi

echo ""
echo "=== All Done ==="
echo "End: $(date)"
