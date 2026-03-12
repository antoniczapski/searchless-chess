from __future__ import annotations
"""Data download and preparation pipeline.

Downloads chess position data from HuggingFace, encodes it, and saves
train/val/test splits as .npz files for fast loading.
"""

import json
import hashlib
from pathlib import Path

import numpy as np
from loguru import logger

from src.data.encoding import fen_to_tensor, mate_to_cp, normalize_cp


def download_and_prepare(
    hf_dataset: str,
    output_dir: str,
    num_samples: int | None = None,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 2001,
    cp_scale: float = 10_000.0,
) -> dict:
    """Download dataset from HuggingFace and prepare train/val/test splits.

    Args:
        hf_dataset: HuggingFace dataset identifier.
        output_dir: Directory to save prepared .npz files.
        num_samples: Max samples to use (None = all).
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for test.
        seed: Random seed for reproducible splitting.
        cp_scale: Scale factor for cp normalization.

    Returns:
        Dict with split sizes and output paths.
    """
    from datasets import load_dataset

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if already prepared
    split_info_path = output_path / "split_info.json"
    if split_info_path.exists():
        info = json.loads(split_info_path.read_text())
        # Verify files exist
        if all((output_path / f"{s}.npz").exists() for s in ["train", "val", "test"]):
            logger.info(f"Data already prepared at {output_path}. Skipping download.")
            return info

    logger.info(f"Downloading dataset: {hf_dataset}")
    ds = load_dataset(hf_dataset, split="train")

    if num_samples is not None and num_samples < len(ds):
        logger.info(f"Sampling {num_samples} from {len(ds)} total positions.")
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(ds), size=num_samples, replace=False)
        indices.sort()
        ds = ds.select(indices.tolist())

    total = len(ds)
    logger.info(f"Processing {total} positions...")

    # Memory-efficient: encode in chunks and write intermediate files to disk,
    # then concatenate. This avoids holding all 50M encoded positions in RAM
    # alongside the HuggingFace dataset simultaneously.
    CHUNK_SIZE = 5_000_000  # 5M positions per chunk (~9.2 GB per chunk)
    chunk_dir = output_path / "_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunk_boards = np.zeros((CHUNK_SIZE, 8, 8, 12), dtype=np.float32)
    chunk_scores = np.zeros(CHUNK_SIZE, dtype=np.float32)
    chunk_idx = 0  # position within current chunk
    chunk_num = 0  # chunk file counter
    total_valid = 0
    skipped = 0
    chunk_files = []

    def _flush_chunk():
        nonlocal chunk_idx, chunk_num
        if chunk_idx == 0:
            return
        path = chunk_dir / f"chunk_{chunk_num:03d}.npz"
        np.savez(path, boards=chunk_boards[:chunk_idx], scores=chunk_scores[:chunk_idx])
        chunk_files.append((str(path), chunk_idx))
        logger.info(f"  Flushed chunk {chunk_num}: {chunk_idx} samples -> {path}")
        chunk_num += 1
        chunk_idx = 0

    for i, row in enumerate(ds):
        if i % 500_000 == 0 and i > 0:
            logger.info(f"  Encoded {i}/{total} positions... ({total_valid} valid, {skipped} skipped)")

        fen = row["fen"]
        cp = row.get("cp")
        mate = row.get("mate")

        # Resolve score
        if cp is not None and not _is_nan(cp):
            raw_cp = float(cp)
        elif mate is not None and not _is_nan(mate):
            raw_cp = mate_to_cp(mate)
            if raw_cp is None:
                skipped += 1
                continue
            raw_cp = float(raw_cp)
        else:
            skipped += 1
            continue

        chunk_boards[chunk_idx] = fen_to_tensor(fen, always_white_perspective=True)
        chunk_scores[chunk_idx] = normalize_cp(raw_cp, scale=cp_scale)
        chunk_idx += 1
        total_valid += 1

        if chunk_idx >= CHUNK_SIZE:
            _flush_chunk()

    # Flush remaining
    _flush_chunk()

    # Free the HF dataset from memory before loading chunks
    del ds
    import gc
    gc.collect()

    logger.info(f"Encoded {total_valid} positions ({skipped} skipped) in {len(chunk_files)} chunks.")
    logger.info("Loading chunks and concatenating...")

    # Load all chunks and concatenate
    all_boards = []
    all_scores = []
    for path, size in chunk_files:
        data = np.load(path)
        all_boards.append(data["boards"])
        all_scores.append(data["scores"])

    boards_arr = np.concatenate(all_boards, axis=0)
    scores_arr = np.concatenate(all_scores, axis=0)
    del all_boards, all_scores
    gc.collect()

    logger.info(f"Total: {len(boards_arr)} positions loaded.")

    # Deterministic shuffle + split
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(boards_arr))
    boards_arr = boards_arr[perm]
    scores_arr = scores_arr[perm]

    n = len(boards_arr)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": (boards_arr[:n_train], scores_arr[:n_train]),
        "val": (boards_arr[n_train : n_train + n_val], scores_arr[n_train : n_train + n_val]),
        "test": (boards_arr[n_train + n_val :], scores_arr[n_train + n_val :]),
    }

    info = {"seed": seed, "total": n, "splits": {}}
    for name, (b, s) in splits.items():
        path = output_path / f"{name}.npz"
        logger.info(f"  Saving {name}: {len(b)} samples -> {path} ...")
        np.savez(path, boards=b, scores=s)
        info["splits"][name] = {"size": len(b), "path": str(path)}
        logger.info(f"  ✓ Saved {name}")

    split_info_path.write_text(json.dumps(info, indent=2))

    # Clean up chunk files
    import shutil
    shutil.rmtree(chunk_dir, ignore_errors=True)

    logger.success(f"Data preparation complete. Info saved to {split_info_path}")
    return info


def _is_nan(val) -> bool:
    """Check if a value is NaN."""
    try:
        import math
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return False
