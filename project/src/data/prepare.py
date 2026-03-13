from __future__ import annotations
"""Data download and preparation pipeline.

Downloads chess position data from HuggingFace, encodes it, and saves
train/val/test splits as .npz files for fast loading.
"""

import json
import hashlib
import gc
import shutil
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

    # Check if chunks already exist (resume from OOM during split phase)
    existing_chunks = sorted(chunk_dir.glob("chunk_*.npz"))
    if existing_chunks:
        expected_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
        if len(existing_chunks) >= expected_chunks:
            logger.info(f"Found {len(existing_chunks)} existing chunks — skipping encoding.")
            chunk_files = []
            total_valid = 0
            for ec in existing_chunks:
                data = np.load(ec)
                sz = len(data["scores"])
                chunk_files.append((str(ec), sz))
                total_valid += sz
                del data
            skipped = total - total_valid
            # Skip the encoding loop entirely
            ds_loaded = False
        else:
            logger.info(f"Found {len(existing_chunks)} chunks but expected ~{expected_chunks} — re-encoding.")
            ds_loaded = True
    else:
        ds_loaded = True

    if not ds_loaded:
        # Free the HF dataset — we don't need it, chunks are on disk
        del ds
        gc.collect()

    if ds_loaded:
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
        del chunk_boards, chunk_scores
        gc.collect()

        logger.info(f"Encoded {total_valid} positions ({skipped} skipped) in {len(chunk_files)} chunks.")

    # --- Memory-efficient shuffle + split using index permutation ---
    # Instead of loading all chunks into one giant array, we:
    #  1. Generate a shuffled permutation of indices [0, total_valid)
    #  2. Split indices into train/val/test
    #  3. For each split, load only the needed rows from chunks on demand

    rng = np.random.default_rng(seed)
    perm = rng.permutation(total_valid)

    n_train = int(total_valid * train_ratio)
    n_val = int(total_valid * val_ratio)

    split_indices = {
        "train": np.sort(perm[:n_train]),
        "val": np.sort(perm[n_train : n_train + n_val]),
        "test": np.sort(perm[n_train + n_val :]),
    }

    # Build a map: global_index -> (chunk_file, local_index)
    chunk_boundaries = []  # (start_global_idx, end_global_idx, chunk_path)
    offset = 0
    for path, size in chunk_files:
        chunk_boundaries.append((offset, offset + size, path))
        offset += size

    info = {"seed": seed, "total": total_valid, "splits": {}}

    for split_name, indices in split_indices.items():
        n_split = len(indices)
        boards_out = np.zeros((n_split, 8, 8, 12), dtype=np.float32)
        scores_out = np.zeros(n_split, dtype=np.float32)

        logger.info(f"  Building {split_name}: {n_split} samples ...")

        # Process one chunk at a time to limit memory
        out_pos = 0
        for chunk_start, chunk_end, chunk_path in chunk_boundaries:
            # Find which split indices fall into this chunk
            mask = (indices >= chunk_start) & (indices < chunk_end)
            global_idxs = indices[mask]
            if len(global_idxs) == 0:
                continue

            local_idxs = global_idxs - chunk_start
            data = np.load(chunk_path)
            boards_out[out_pos : out_pos + len(local_idxs)] = data["boards"][local_idxs]
            scores_out[out_pos : out_pos + len(local_idxs)] = data["scores"][local_idxs]
            out_pos += len(local_idxs)
            del data

        # In-place shuffle within the split (the chunk-based extraction
        # produces sorted-by-chunk order, so we shuffle to ensure randomness).
        # Use Fisher-Yates via rng.shuffle on a joint index to avoid copies.
        shuffle_idx = np.arange(n_split)
        rng.shuffle(shuffle_idx)
        boards_out[:] = boards_out[shuffle_idx]
        scores_out[:] = scores_out[shuffle_idx]
        del shuffle_idx

        path = output_path / f"{split_name}.npz"
        logger.info(f"  Saving {split_name}: {n_split} samples -> {path} ...")
        np.savez(path, boards=boards_out, scores=scores_out)
        info["splits"][split_name] = {"size": n_split, "path": str(path)}
        logger.info(f"  ✓ Saved {split_name}")

        # Free memory between splits
        del boards_out, scores_out
        gc.collect()

    split_info_path.write_text(json.dumps(info, indent=2))

    # Clean up chunk files
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
