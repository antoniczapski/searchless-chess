#!/bin/bash -l
# ──────────────────────────────────────────────────────────
# Data preparation job — download + encode 50M positions
# ──────────────────────────────────────────────────────────
#SBATCH -J bdh-data-prep
#SBATCH -p plgrid-gpu-a100
#SBATCH -A plgolimpiada2026-gpu-a100
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/data-prep_%j.out
#SBATCH --error=logs/data-prep_%j.err

set -euo pipefail

REPO_DIR="$SLURM_SUBMIT_DIR"
VENV="$HOME/envs/pcg"
SCRATCH="/net/tscratch/people/plgantoniczapski"

export HF_HOME="$SCRATCH/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH/hf_cache/datasets"

mkdir -p "$REPO_DIR/logs" "$HF_HOME" "$HF_DATASETS_CACHE"

echo "========================================"
echo "Data Preparation — 50M positions"
echo "========================================"
echo "HOSTNAME     = $(hostname)"
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "MEM          = $SLURM_MEM_PER_NODE MB"
echo "DATE         = $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

source "$VENV/bin/activate"
cd "$REPO_DIR"

python -u scripts/setup.py \
    --num-samples 50000000 \
    --train-ratio 0.90 \
    --val-ratio 0.05 \
    --test-ratio 0.05 \
    --skip-env-check \
    --output-dir "$SCRATCH/bdh-chess-data"

echo ""
echo "Data preparation finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Files at: $SCRATCH/bdh-chess-data/"
ls -lh "$SCRATCH/bdh-chess-data/"
