from __future__ import annotations
"""PyTorch Dataset and DataLoader factory for chess position data.

Supports three on-disk formats / loading strategies:
  - .npz (small data, loaded fully into RAM)
  - .npy memory-mapped (large data, OS pages on demand)
  - .npy in-memory (large data, pre-loaded into RAM as contiguous tensors)
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
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
        boards = data["boards"]
        if boards.dtype == np.uint8:
            boards = boards.astype(np.float32)
        self.boards = torch.from_numpy(boards).permute(0, 3, 1, 2).contiguous()
        self.scores = torch.from_numpy(data["scores"]).unsqueeze(1)  # (N, 1)

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.boards[idx], self.scores[idx]


# ---------------------------------------------------------------------------
# Large-data dataset (.npy, memory-mapped — legacy fallback)
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
        board = torch.from_numpy(self.boards[idx].copy()).float()  # (8, 8, 12)
        board = board.permute(2, 0, 1)                              # (12, 8, 8)
        score = torch.tensor([self.scores[idx]], dtype=torch.float32)
        return board, score


# ---------------------------------------------------------------------------
# Large-data dataset (.npy, pre-loaded into RAM — fast)
# ---------------------------------------------------------------------------

class InMemoryChessDataset(Dataset):
    """Pre-loads entire .npy dataset into RAM as contiguous torch tensors.

    Much faster than memory-mapped access:
    - No per-item page faults or .copy() overhead
    - Batch indexing is vectorized (boards[indices])
    - Pre-permuted to (N, 12, 8, 8) at init time
    - Supports uint8 boards (converted to float32 on GPU)

    Memory: 45M × 8 × 8 × 12 × 1 byte (uint8) ≈ 34.6 GB.

    Args:
        boards_path: Path to {split}_boards.npy (shape: N, 8, 8, 12).
        scores_path: Path to {split}_scores.npy (shape: N,).
    """

    def __init__(self, boards_path: str | Path, scores_path: str | Path):
        logger.info(f"Loading {Path(boards_path).name} into RAM...")
        import time
        t0 = time.time()

        # Load numpy arrays
        boards_np = np.load(str(boards_path))    # (N, 8, 8, 12), uint8 or float32
        scores_np = np.load(str(scores_path))    # (N,), float32

        # Convert to torch tensors
        # Keep boards as uint8 in RAM (4× smaller), cast to float in __getitem__
        self.boards = torch.from_numpy(boards_np).permute(0, 3, 1, 2).contiguous()
        self.scores = torch.from_numpy(scores_np).unsqueeze(1)  # (N, 1)

        elapsed = time.time() - t0
        mem_gb = (self.boards.nbytes + self.scores.nbytes) / 1e9
        logger.info(
            f"InMemoryChessDataset: {len(self.scores):,} samples loaded "
            f"in {elapsed:.1f}s ({mem_gb:.1f} GB RAM, dtype={self.boards.dtype})"
        )

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        board = self.boards[idx].float()  # uint8 → float32 (or no-op if already float)
        return board, self.scores[idx]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    data_dir: str,
    batch_size: int = 512,
    num_workers: int = 0,
    pin_memory: bool = True,
    in_memory: bool = True,
) -> dict[str, DataLoader]:
    """Create train/val/test DataLoaders, auto-detecting format.

    Checks for .npy files first (large-scale), falls back to .npz.

    Args:
        data_dir: Directory containing prepared data files.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        pin_memory: Pin memory for GPU transfer.
        in_memory: If True and format is npy, pre-load entire dataset
            into RAM for faster training (requires sufficient RAM).

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

    logger.info(f"Loading data from {data_path} (format={fmt}, in_memory={in_memory})")

    for split in ["train", "val", "test"]:
        if fmt == "npy":
            boards_file = data_path / f"{split}_boards.npy"
            scores_file = data_path / f"{split}_scores.npy"
            if not boards_file.exists() or not scores_file.exists():
                raise FileNotFoundError(
                    f"Missing {boards_file} or {scores_file}. Run data preparation first."
                )
            if in_memory:
                ds = InMemoryChessDataset(boards_file, scores_file)
            else:
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
            prefetch_factor=4 if num_workers > 0 else None,
        )

    return loaders
