from __future__ import annotations
"""PyTorch Dataset and DataLoader factory for chess position data."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ChessDataset(Dataset):
    """PyTorch dataset loading from .npz files (boards + scores).

    Args:
        npz_path: Path to a .npz file with 'boards' and 'scores' arrays.
    """

    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path)
        # boards: (N, 8, 8, 12) -> (N, 12, 8, 8) for PyTorch conv convention
        self.boards = torch.from_numpy(data["boards"]).permute(0, 3, 1, 2).contiguous()
        self.scores = torch.from_numpy(data["scores"]).unsqueeze(1)  # (N, 1)

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.boards[idx], self.scores[idx]


def create_dataloaders(
    data_dir: str,
    batch_size: int = 512,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    """Create train/val/test DataLoaders from prepared .npz files.

    Args:
        data_dir: Directory containing train.npz, val.npz, test.npz.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        pin_memory: Pin memory for GPU transfer.

    Returns:
        Dict with 'train', 'val', 'test' DataLoaders.
    """
    data_path = Path(data_dir)
    loaders = {}

    for split in ["train", "val", "test"]:
        npz_file = data_path / f"{split}.npz"
        if not npz_file.exists():
            raise FileNotFoundError(f"Missing {npz_file}. Run data preparation first.")

        ds = ChessDataset(npz_file)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == "train"),
        )

    return loaders
