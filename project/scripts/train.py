"""Train a chess evaluation model.

Usage:
    python scripts/train.py --config configs/train_mlp.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from loguru import logger

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import create_dataloaders
from src.models.registry import create_model, count_parameters
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train chess evaluation model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--resume", type=str, nargs="?", const="auto",
        help="Resume from checkpoint. 'auto' (default if flag given) finds latest_checkpoint.pt "
             "in the output dir. Or provide an explicit path to a .pt file.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Reproducibility
    seed = config["experiment"]["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Data
    dc = config["data"]
    loaders = create_dataloaders(
        data_dir=dc["output_dir"],
        batch_size=dc["batch_size"],
        num_workers=dc["num_workers"],
    )

    logger.info(
        f"Data: train={len(loaders['train'].dataset)}, "
        f"val={len(loaders['val'].dataset)}, "
        f"test={len(loaders['test'].dataset)}"
    )

    # Model
    model = create_model(config["model"])
    logger.info(f"Parameters: {count_parameters(model):,}")

    # Train (with optional resume)
    trainer = Trainer(model, config, output_dir=config["output"]["dir"])

    start_epoch = 1
    if args.resume:
        if args.resume == "auto":
            ckpt_path = Path(config["output"]["dir"]) / "checkpoints" / "latest_checkpoint.pt"
        else:
            ckpt_path = Path(args.resume)

        if ckpt_path.exists():
            start_epoch = trainer.resume_from_checkpoint(str(ckpt_path))
        else:
            logger.warning(f"No checkpoint found at {ckpt_path} — starting from scratch")

    result = trainer.fit(loaders["train"], loaders["val"], start_epoch=start_epoch)

    logger.success(
        f"Training complete — best val_loss={result['best_val_loss']:.6f} "
        f"at epoch {result['final_epoch']}"
    )


if __name__ == "__main__":
    main()
