#!/bin/bash
#SBATCH --job-name=attnres-eval
#SBATCH --account=plgolimpiada2026-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/attnres-eval_%j.out
#SBATCH --error=logs/attnres-eval_%j.err

set -euo pipefail

CONFIG="configs/train_vit_attnres_a100.yaml"
CHECKPOINT="/net/tscratch/people/plgantoniczapski/bdh-chess-outputs/vit-s-attnres/checkpoints/best_model.pt"
CONDA_ENV="/net/tscratch/people/plgantoniczapski/conda-envs/bdh-chess"
PROJECT_DIR="/net/people/plgrid/plgantoniczapski/projects/searchless-chess"

echo "=== ViT-S+AttnRes Evaluation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Checkpoint: $CHECKPOINT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export HF_HOME="/net/tscratch/people/plgantoniczapski/hf-cache"

cd "$PROJECT_DIR"

if [[ -f "$PROJECT_DIR/.env" ]]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
fi

python scripts/evaluate.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT"

echo ""
echo "=== Done ==="
echo "End: $(date)"
