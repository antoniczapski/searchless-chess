"""Evaluate a trained model on the Lichess puzzle benchmark.

Usage:
    python scripts/evaluate.py --config configs/train_mlp.yaml --checkpoint outputs/mlp-baseline/checkpoints/best_model.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from loguru import logger

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.registry import create_model
from src.evaluation.elo_benchmark import estimate_elo


def main():
    parser = argparse.ArgumentParser(description="Evaluate model ELO on puzzles")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint (optional)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Build model
    model = create_model(config["model"])

    # Optionally load weights
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    else:
        logger.warning("No checkpoint provided — evaluating with random weights")

    # Evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = estimate_elo(model, config["evaluation"]["puzzle_path"], device)

    if result["elo"] is not None:
        logger.success(f"Estimated ELO: {result['elo']:.0f}")
    logger.info(f"Overall accuracy: {result['total_accuracy']:.1%}")


if __name__ == "__main__":
    main()
