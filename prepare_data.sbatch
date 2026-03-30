#!/bin/bash -l
# ──────────────────────────────────────────────────────────
# Data preparation job — split 50M encoded positions
# Uses existing chunks, creates memory-mapped .npy splits
# ──────────────────────────────────────────────────────────
#SBATCH -J bdh-data-prep
#SBATCH -p plgrid-gpu-a100
#SBATCH -A plgolimpiada2026-gpu-a100
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/net/tscratch/people/plgantoniczapski/bdh-chess-outputs/data-prep_%j.out
#SBATCH --error=/net/tscratch/people/plgantoniczapski/bdh-chess-outputs/data-prep_%j.err

set -euo pipefail

echo "========================================"
echo "Data Preparation — 50M positions"
echo "========================================"
echo "HOSTNAME     = $(hostname)"
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "DATE         = $(date)"

module load Miniconda3/25.7.0-2
eval "$(conda shell.bash hook)"
conda activate /net/tscratch/people/plgantoniczapski/conda-envs/bdh-chess

cd /net/people/plgrid/plgantoniczapski/projects/searchless-chess/project

# Clean up partial files from previous failed attempt
rm -f /net/tscratch/people/plgantoniczapski/bdh-chess-data/train_boards.npy
rm -f /net/tscratch/people/plgantoniczapski/bdh-chess-data/train_scores.npy
rm -f /net/tscratch/people/plgantoniczapski/bdh-chess-data/split_info.json

echo ""
echo "Running data preparation..."
python scripts/prepare_data.py --config configs/train_bdh_a100.yaml

echo ""
echo "Data preparation complete: $(date)"
echo "Output files:"
ls -lh /net/tscratch/people/plgantoniczapski/bdh-chess-data/*.npy
echo ""
cat /net/tscratch/people/plgantoniczapski/bdh-chess-data/split_info.json

echo ""
echo "Data preparation finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Files at: $SCRATCH/bdh-chess-data/"
ls -lh "$SCRATCH/bdh-chess-data/"
