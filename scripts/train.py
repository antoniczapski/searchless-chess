"""Train a chess evaluation model using PyTorch Lightning.

Usage:
    python scripts/train.py --config configs/train_mlp.yaml
    python scripts/train.py --config configs/train_mlp.yaml --smoke-test
    python scripts/train.py --config configs/train_mlp.yaml --resume path/to/checkpoint.ckpt
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from loguru import logger

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env for W&B credentials etc.
try:
    from dotenv import load_dotenv
    for p in [Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent.parent.parent]:
        if (p / ".env").exists():
            load_dotenv(p / ".env")
            break
except ImportError:
    pass

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from src.training.lightning_module import ChessLightningModule
from src.data.lightning_datamodule import ChessDataModule


def main():
    parser = argparse.ArgumentParser(description="Train chess evaluation model (Lightning)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--resume", type=str, nargs="?", const="auto",
        help="Resume from checkpoint. 'auto' finds latest in output dir, or provide a path.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run a tiny smoke test (synthetic data, 1 epoch, CPU).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Reproducibility
    seed = config["experiment"]["seed"]
    pl.seed_everything(seed, workers=True)

    # ─── Data ─────────────────────────────────────────────────
    dc = config["data"]
    smoke = args.smoke_test

    datamodule = ChessDataModule(
        data_dir=dc.get("output_dir", "data/processed"),
        batch_size=dc["batch_size"] if not smoke else 16,
        num_workers=dc.get("num_workers", 0) if not smoke else 0,
        smoke=smoke,
    )

    # ─── Model ────────────────────────────────────────────────
    model = ChessLightningModule(config)
    logger.info(f"Architecture: {config['model']['architecture']}")

    # ─── Callbacks ────────────────────────────────────────────
    tc = config["training"]
    output_dir = config.get("output", {}).get("dir", "outputs/default")
    if smoke:
        output_dir = "outputs/smoke"

    callbacks = [
        ModelCheckpoint(
            dirpath=f"{output_dir}/checkpoints",
            filename="best-{epoch:03d}-{val/loss:.6f}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=tc.get("early_stopping", {}).get("patience", 5),
            min_delta=tc.get("early_stopping", {}).get("min_delta", 1e-5),
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # ─── Logger ───────────────────────────────────────────────
    wandb_cfg = config.get("wandb", {})
    train_loggers = [
        CSVLogger(save_dir=output_dir, name="csv_logs"),
    ]

    if wandb_cfg.get("enabled", False) and not smoke:
        api_key = os.environ.get("WANDB_API_KEY") or os.environ.get("WEIGHTS_AND_BIASES_KEY")
        if api_key:
            import wandb
            wandb.login(key=api_key, relogin=True)
        train_loggers.append(
            WandbLogger(
                project=wandb_cfg.get("project", "bdh-searchless-chess"),
                name=config["experiment"]["name"],
                tags=wandb_cfg.get("tags", []),
                config=config,
                save_dir=output_dir,
            )
        )

    # ─── Trainer ──────────────────────────────────────────────
    use_amp = tc.get("mixed_precision", False)
    precision = "16-mixed" if use_amp else 32
    # Prefer bf16 on A100+
    if use_amp:
        amp_dtype = tc.get("amp_dtype", "auto")
        if amp_dtype == "bf16" or (amp_dtype == "auto" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            precision = "bf16-mixed"

    trainer = pl.Trainer(
        max_epochs=1 if smoke else tc["epochs"],
        accelerator="auto",
        devices=1,
        precision=precision if not smoke else 32,
        callbacks=callbacks,
        logger=train_loggers,
        gradient_clip_val=tc.get("clipnorm", 1.0),
        enable_progress_bar=True,
        log_every_n_steps=max(1, 50 if not smoke else 1),
        default_root_dir=output_dir,
        fast_dev_run=5 if smoke else False,
    )

    # ─── Resume ───────────────────────────────────────────────
    ckpt_path = None
    if args.resume and not smoke:
        if args.resume == "auto":
            candidate = Path(output_dir) / "checkpoints" / "last.ckpt"
            if candidate.exists():
                ckpt_path = str(candidate)
                logger.info(f"Auto-resume from {ckpt_path}")
            else:
                logger.warning(f"No checkpoint at {candidate} — starting fresh")
        else:
            ckpt_path = args.resume

    # ─── Train ────────────────────────────────────────────────
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    if not smoke:
        logger.success(
            f"Training complete — best val/loss from checkpoint callback"
        )


if __name__ == "__main__":
    main()
