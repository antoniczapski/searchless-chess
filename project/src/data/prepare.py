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

    # Pre-allocate arrays to avoid OOM from Python list overhead.
    # We allocate for the full dataset size; actual count may be slightly less
    # due to skipped rows (NaN/missing scores).
    boards_arr = np.zeros((total, 8, 8, 12), dtype=np.float32)
    scores_arr = np.zeros(total, dtype=np.float32)
    valid_count = 0
    skipped = 0

    for i, row in enumerate(ds):
        if i % 500_000 == 0 and i > 0:
            logger.info(f"  Encoded {i}/{total} positions... ({valid_count} valid, {skipped} skipped)")

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

        boards_arr[valid_count] = fen_to_tensor(fen, always_white_perspective=True)
        scores_arr[valid_count] = normalize_cp(raw_cp, scale=cp_scale)
        valid_count += 1

    logger.info(f"Encoded {valid_count} positions ({skipped} skipped).")

    # Trim to actual valid count
    boards_arr = boards_arr[:valid_count]
    scores_arr = scores_arr[:valid_count]

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
    logger.success(f"Data preparation complete. Info saved to {split_info_path}")
    return info


def _is_nan(val) -> bool:
    """Check if a value is NaN."""
    try:
        import math
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return False
