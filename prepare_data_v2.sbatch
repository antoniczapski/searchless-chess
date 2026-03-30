#!/bin/bash -l
# ──────────────────────────────────────────────────────────
# Data preparation v2 — 50M positions with score sign fix
# Boards stored as uint8 (4× smaller than previous float32)
# Score flipped for black-to-move (critical bug fix)
# ──────────────────────────────────────────────────────────
#SBATCH -J bdh-data-prep-v2
#SBATCH -p plgrid-gpu-a100
#SBATCH -A plgolimpiada2026-gpu-a100
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/data-prep-v2_%j.out
#SBATCH --error=logs/data-prep-v2_%j.err

set -euo pipefail

echo "========================================"
echo "Data Preparation v2 — 50M positions"
echo "  Score sign fix + uint8 boards"
echo "========================================"
echo "HOSTNAME     = $(hostname)"
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "DATE         = $(date)"

module load Miniconda3/25.7.0-2
eval "$(conda shell.bash hook)"
conda activate /net/tscratch/people/plgantoniczapski/conda-envs/bdh-chess

# Redirect HuggingFace cache to scratch (home dir has quota limits)
export HF_HOME="/net/tscratch/people/plgantoniczapski/hf-cache"
export HF_DATASETS_CACHE="/net/tscratch/people/plgantoniczapski/hf-cache/datasets"

cd /net/people/plgrid/plgantoniczapski/projects/searchless-chess/project

# Output directory for v2 data
DATA_DIR="/net/tscratch/people/plgantoniczapski/bdh-chess-data-v2"

# Clean up any partial files from previous attempts
rm -f "${DATA_DIR}/train_boards.npy"
rm -f "${DATA_DIR}/train_scores.npy"
rm -f "${DATA_DIR}/val_boards.npy"
rm -f "${DATA_DIR}/val_scores.npy"
rm -f "${DATA_DIR}/test_boards.npy"
rm -f "${DATA_DIR}/test_scores.npy"
rm -f "${DATA_DIR}/split_info.json"
rm -rf "${DATA_DIR}/_chunks"

echo ""
echo "Running data preparation with score sign fix..."
python scripts/prepare_data.py --config configs/train_vit_s_a100.yaml

echo ""
echo "Data preparation complete: $(date)"
echo "Output files:"
ls -lh "${DATA_DIR}"/*.npy 2>/dev/null || echo "No .npy files found"
echo ""
cat "${DATA_DIR}/split_info.json" 2>/dev/null || echo "No split_info.json"

echo ""
echo "Data preparation finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Files at: ${DATA_DIR}/"
ls -lh "${DATA_DIR}/"
