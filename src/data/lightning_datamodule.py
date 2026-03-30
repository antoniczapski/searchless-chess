"""PyTorch Lightning DataModule for chess position data.

Wraps the existing dataset/dataloader factory with a standard
LightningDataModule interface. Supports both real data and
synthetic smoke-test data for CI.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.dataset import create_dataloaders, ChessDataset


class ChessDataModule(pl.LightningDataModule):
    """Lightning DataModule for chess position evaluation.

    In normal mode, wraps the existing ``create_dataloaders`` factory.
    In smoke mode, generates tiny synthetic data for testing.

    Args:
        data_dir: Path to the prepared data directory.
        batch_size: Batch size for all splits.
        num_workers: DataLoader workers.
        smoke: If True, generate tiny synthetic data (ignores data_dir).
        smoke_n_train: Number of synthetic training samples.
        smoke_n_val: Number of synthetic validation samples.
    """

    def __init__(
        self,
        data_dir: str = "data/processed",
        batch_size: int = 256,
        num_workers: int = 0,
        smoke: bool = False,
        smoke_n_train: int = 128,
        smoke_n_val: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.smoke = smoke
        self.smoke_n_train = smoke_n_train
        self.smoke_n_val = smoke_n_val

        self._loaders: dict[str, DataLoader] | None = None
        self._tmpdir: tempfile.TemporaryDirectory | None = None

    def setup(self, stage: str | None = None) -> None:
        if self._loaders is not None:
            return  # already set up

        if self.smoke:
            self._setup_smoke()
        else:
            self._loaders = create_dataloaders(
                data_dir=self.data_dir,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )

    def _setup_smoke(self) -> None:
        """Create tiny synthetic .npz data in a temp directory."""
        self._tmpdir = tempfile.TemporaryDirectory()
        tmpdir = self._tmpdir.name
        rng = np.random.default_rng(42)

        for split, n in [
            ("train", self.smoke_n_train),
            ("val", self.smoke_n_val),
            ("test", self.smoke_n_val),
        ]:
            boards = rng.integers(0, 2, size=(n, 8, 8, 12), dtype=np.uint8).astype(np.float32)
            scores = rng.uniform(-1, 1, size=(n,)).astype(np.float32)
            np.savez(f"{tmpdir}/{split}.npz", boards=boards, scores=scores)

        self._loaders = create_dataloaders(
            data_dir=tmpdir,
            batch_size=min(self.batch_size, self.smoke_n_train),
            num_workers=0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loaders["train"]

    def val_dataloader(self) -> DataLoader:
        return self._loaders["val"]

    def test_dataloader(self) -> DataLoader:
        return self._loaders["test"]

    def teardown(self, stage: str | None = None) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None
