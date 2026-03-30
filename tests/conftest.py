"""Shared test fixtures for the searchless-chess test suite."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path so `from src.xxx` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def synthetic_data_dir():
    """Create a temp directory with tiny .npz files (train/val/test)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rng = np.random.default_rng(42)
        for split in ("train", "val", "test"):
            n = 64 if split == "train" else 16
            boards = rng.integers(0, 2, size=(n, 8, 8, 12), dtype=np.uint8).astype(np.float32)
            scores = rng.uniform(-1, 1, (n,)).astype(np.float32)
            np.savez(f"{tmpdir}/{split}.npz", boards=boards, scores=scores)
        yield tmpdir


@pytest.fixture
def minimal_config():
    """Minimal experiment config for smoke tests."""
    return {
        "experiment": {"name": "test-smoke", "seed": 42},
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
        "evaluation": {"puzzle_path": "data/puzzles/test_puzzles.feather"},
    }
