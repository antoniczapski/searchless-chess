"""Smoke test: Lightning training loop on synthetic data."""

import tempfile

import numpy as np
import pytest
import torch

# Guard against pytorch-lightning not being installed
pl = pytest.importorskip("pytorch_lightning")

from src.training.lightning_module import ChessLightningModule
from src.data.lightning_datamodule import ChessDataModule


class TestLightningSmokeTraining:
    """Run a minimal Lightning training loop to verify integration."""

    @pytest.fixture
    def config(self):
        return {
            "experiment": {"name": "lightning-smoke", "seed": 42},
            "wandb": {"enabled": False},
            "data": {
                "output_dir": "data/processed",
                "batch_size": 16,
                "num_workers": 0,
                "hf_dataset": "mateuszgrzyb/lichess-stockfish-normalized",
                "num_samples": 1000,
                "train_ratio": 0.80,
                "val_ratio": 0.10,
                "test_ratio": 0.10,
                "cp_scale": 1000.0,
            },
            "model": {
                "architecture": "mlp",
                "hidden_dim": 32,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "training": {
                "epochs": 2,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "clipnorm": 1.0,
                "mixed_precision": False,
                "warmup_epochs": 0,
                "min_learning_rate": 1e-6,
                "save_every_epochs": 10,
                "early_stopping": {"patience": 5, "min_delta": 0.0},
            },
            "output": {"dir": ""},
        }

    def test_fast_dev_run(self, config, tmp_path):
        """fast_dev_run=2 should complete without errors."""
        config["output"]["dir"] = str(tmp_path)

        model = ChessLightningModule(config)
        dm = ChessDataModule(smoke=True, batch_size=16, num_workers=0)

        trainer = pl.Trainer(
            fast_dev_run=2,
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(model, datamodule=dm)

    def test_checkpoint_callback(self, config, tmp_path):
        """ModelCheckpoint should save a best checkpoint."""
        from pytorch_lightning.callbacks import ModelCheckpoint

        config["output"]["dir"] = str(tmp_path)

        model = ChessLightningModule(config)
        dm = ChessDataModule(smoke=True, batch_size=16, num_workers=0)

        ckpt_cb = ModelCheckpoint(
            dirpath=str(tmp_path / "ckpt"),
            monitor="val/loss",
            mode="min",
            save_top_k=1,
        )

        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            devices=1,
            callbacks=[ckpt_cb],
            enable_progress_bar=False,
            logger=False,
            limit_train_batches=2,
            limit_val_batches=2,
        )
        trainer.fit(model, datamodule=dm)

        assert ckpt_cb.best_model_path, "No checkpoint was saved"
        assert (tmp_path / "ckpt").exists()

    def test_vit_smoke(self, tmp_path):
        """ViT architecture also trains under Lightning."""
        config = {
            "experiment": {"name": "vit-smoke", "seed": 42},
            "wandb": {"enabled": False},
            "data": {"batch_size": 16, "num_workers": 0, "output_dir": "data/processed",
                     "hf_dataset": "x", "num_samples": 100, "train_ratio": 0.8,
                     "val_ratio": 0.1, "test_ratio": 0.1, "cp_scale": 1000.0},
            "model": {
                "architecture": "vit",
                "projection_dim": 32,
                "num_heads": 2,
                "transformer_layers": 1,
                "ffn_ratio": 2,
                "dropout_rate": 0.0,
            },
            "training": {
                "epochs": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "clipnorm": 1.0,
                "mixed_precision": False,
                "warmup_epochs": 0,
                "min_learning_rate": 1e-6,
                "early_stopping": {"patience": 5, "min_delta": 0.0},
            },
            "output": {"dir": str(tmp_path)},
        }

        model = ChessLightningModule(config)
        dm = ChessDataModule(smoke=True, batch_size=16, num_workers=0)

        trainer = pl.Trainer(
            fast_dev_run=2,
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(model, datamodule=dm)
