"""Download and prepare training data from HuggingFace.

Usage:
    python scripts/prepare_data.py --config configs/train_mlp.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml
from loguru import logger

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.prepare import download_and_prepare


def main():
    parser = argparse.ArgumentParser(description="Prepare chess training data")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dc = config["data"]
    info = download_and_prepare(
        hf_dataset=dc["hf_dataset"],
        output_dir=dc["output_dir"],
        num_samples=dc["num_samples"],
        train_ratio=dc["train_ratio"],
        val_ratio=dc["val_ratio"],
        test_ratio=dc["test_ratio"],
        seed=config["experiment"]["seed"],
        cp_scale=dc["cp_scale"],
    )

    logger.success(f"Data ready in {dc['output_dir']}/")
    for split, n in info.items():
        logger.info(f"  {split}: {n}")


if __name__ == "__main__":
    main()
