"""Smoke test for the training loop (1-2 epochs on synthetic data)."""

import tempfile

import numpy as np
import torch
import pytest

from src.data.dataset import ChessDataset, create_dataloaders
from src.models.registry import create_model
from src.training.trainer import Trainer


@pytest.fixture
def synthetic_data_dir():
    """Create a temp directory with tiny .npz files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rng = np.random.default_rng(42)
        for split in ("train", "val", "test"):
            n = 64 if split == "train" else 16
            boards = rng.random((n, 8, 8, 12)).astype(np.float32)
            scores = rng.uniform(-1, 1, (n,)).astype(np.float32)
            np.savez(f"{tmpdir}/{split}.npz", boards=boards, scores=scores)
        yield tmpdir


@pytest.fixture
def minimal_config():
    return {
        "experiment": {"name": "test-smoke", "seed": 42},
        "wandb": {"enabled": False},
        "data": {
            "batch_size": 16,
            "num_workers": 0,
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


class TestTrainerSmokeTest:
    """Ensure training runs without errors on tiny data."""

    def test_fit_runs(self, synthetic_data_dir, minimal_config):
        minimal_config["output"]["dir"] = synthetic_data_dir + "/output"
        loaders = create_dataloaders(synthetic_data_dir, batch_size=16, num_workers=0)
        model = create_model(minimal_config["model"])
        trainer = Trainer(model, minimal_config, output_dir=minimal_config["output"]["dir"])
        result = trainer.fit(loaders["train"], loaders["val"])

        assert "best_val_loss" in result
        assert result["final_epoch"] == 2
        assert len(result["history"]) == 2

    def test_checkpoints_saved(self, synthetic_data_dir, minimal_config):
        minimal_config["output"]["dir"] = synthetic_data_dir + "/output"
        loaders = create_dataloaders(synthetic_data_dir, batch_size=16, num_workers=0)
        model = create_model(minimal_config["model"])
        trainer = Trainer(model, minimal_config, output_dir=minimal_config["output"]["dir"])
        trainer.fit(loaders["train"], loaders["val"])

        ckpt_dir = trainer.checkpoint_dir
        assert (ckpt_dir / "best_model.pt").exists()
        assert (ckpt_dir / "final_model.pt").exists()

    def test_loss_decreases_or_stable(self, synthetic_data_dir, minimal_config):
        """Loss should not explode."""
        minimal_config["output"]["dir"] = synthetic_data_dir + "/output"
        loaders = create_dataloaders(synthetic_data_dir, batch_size=16, num_workers=0)
        model = create_model(minimal_config["model"])
        trainer = Trainer(model, minimal_config, output_dir=minimal_config["output"]["dir"])
        result = trainer.fit(loaders["train"], loaders["val"])

        # Loss should stay reasonable (not NaN/Inf)
        for rec in result["history"]:
            assert np.isfinite(rec["train_loss"])
            assert np.isfinite(rec["val_loss"])
