from __future__ import annotations
"""PyTorch Dataset and DataLoader factory for chess position data.

Supports two on-disk formats:
  - .npz (small data, loaded fully into RAM)
  - .npy memory-mapped (large data, OS pages on demand)
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger


# ---------------------------------------------------------------------------
# Small-data dataset (.npz, fully in RAM)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Large-data dataset (.npy, memory-mapped)
# ---------------------------------------------------------------------------

class MemoryMappedChessDataset(Dataset):
    """PyTorch dataset backed by memory-mapped .npy files.

    Only the accessed pages are loaded into RAM, enabling datasets
    far larger than available memory.

    Args:
        boards_path: Path to {split}_boards.npy (shape: N, 8, 8, 12).
        scores_path: Path to {split}_scores.npy (shape: N,).
    """

    def __init__(self, boards_path: str | Path, scores_path: str | Path):
        self.boards = np.load(str(boards_path), mmap_mode="r")
        self.scores = np.load(str(scores_path), mmap_mode="r")
        logger.info(
            f"MemoryMappedChessDataset: {len(self.scores):,} samples "
            f"from {Path(boards_path).name}"
        )

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # .copy() needed because mmap arrays are read-only
        board = torch.from_numpy(self.boards[idx].copy())  # (8, 8, 12)
        board = board.permute(2, 0, 1)                      # (12, 8, 8)
        score = torch.tensor([self.scores[idx]], dtype=torch.float32)
        return board, score


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    data_dir: str,
    batch_size: int = 512,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    """Create train/val/test DataLoaders, auto-detecting format.

    Checks for .npy files first (large-scale mmap), falls back to .npz.

    Args:
        data_dir: Directory containing prepared data files.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        pin_memory: Pin memory for GPU transfer.

    Returns:
        Dict with 'train', 'val', 'test' DataLoaders.
    """
    data_path = Path(data_dir)
    loaders = {}

    # Auto-detect format from split_info.json or file existence
    split_info_path = data_path / "split_info.json"
    fmt = "npz"
    if split_info_path.exists():
        info = json.loads(split_info_path.read_text())
        fmt = info.get("format", "npz")
    elif (data_path / "train_boards.npy").exists():
        fmt = "npy"

    logger.info(f"Loading data from {data_path} (format={fmt})")

    for split in ["train", "val", "test"]:
        if fmt == "npy":
            boards_file = data_path / f"{split}_boards.npy"
            scores_file = data_path / f"{split}_scores.npy"
            if not boards_file.exists() or not scores_file.exists():
                raise FileNotFoundError(
                    f"Missing {boards_file} or {scores_file}. Run data preparation first."
                )
            ds = MemoryMappedChessDataset(boards_file, scores_file)
        else:
            npz_file = data_path / f"{split}.npz"
            if not npz_file.exists():
                raise FileNotFoundError(
                    f"Missing {npz_file}. Run data preparation first."
                )
            ds = ChessDataset(npz_file)

        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == "train"),
            persistent_workers=(num_workers > 0),
        )

    return loaders
