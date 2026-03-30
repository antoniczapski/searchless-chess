from __future__ import annotations
"""Data download and preparation pipeline.

Downloads chess position data from HuggingFace, encodes it, and saves
train/val/test splits as memory-mapped .npy files (large data) or .npz
files (small data) for fast loading.
"""

import json
import gc
import shutil
import time
from pathlib import Path

import numpy as np
from loguru import logger

from src.data.encoding import fen_to_tensor, mate_to_cp, normalize_cp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_nan(val) -> bool:
    """Check if a value is NaN."""
    try:
        import math
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return False


def _is_data_ready(output_path: Path) -> dict | None:
    """Check if prepared data already exists. Returns split_info or None."""
    split_info_path = output_path / "split_info.json"
    if not split_info_path.exists():
        return None

    info = json.loads(split_info_path.read_text())
    fmt = info.get("format", "npz")

    if fmt == "npy":
        files_ok = all(
            (output_path / f"{s}_boards.npy").exists() and
            (output_path / f"{s}_scores.npy").exists()
            for s in ["train", "val", "test"]
        )
    else:
        files_ok = all(
            (output_path / f"{s}.npz").exists()
            for s in ["train", "val", "test"]
        )

    if files_ok:
        logger.info(f"Data already prepared at {output_path} (format={fmt}). Skipping.")
        return info
    return None


def _fmt_bytes(n: int) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

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

    For large datasets (>2M samples) this uses memory-mapped .npy files
    to avoid OOM during split creation. For small datasets it uses .npz.

    Args:
        hf_dataset: HuggingFace dataset identifier.
        output_dir: Directory to save prepared files.
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
    existing = _is_data_ready(output_path)
    if existing is not None:
        return existing

    # Decide format based on scale
    use_mmap = (num_samples or 0) > 2_000_000
    logger.info(
        f"Format: {'npy/mmap (large-scale)' if use_mmap else 'npz (in-memory)'} "
        f"for {num_samples or 'all'} samples"
    )

    # ------------------------------------------------------------------
    # Phase 1: Encode positions into on-disk chunks
    # ------------------------------------------------------------------
    CHUNK_SIZE = 5_000_000  # 5M positions per chunk
    chunk_dir = output_path / "_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Check if chunks already exist (resume support)
    existing_chunks = sorted(chunk_dir.glob("chunk_*.npz"))
    need_encoding = True
    chunk_files: list[tuple[str, int]] = []
    total_valid = 0

    if existing_chunks:
        # Count total samples in existing chunks
        chunk_files_tmp: list[tuple[str, int]] = []
        total_in_chunks = 0
        for ec in existing_chunks:
            try:
                data = np.load(ec)
                sz = len(data["scores"])
                chunk_files_tmp.append((str(ec), sz))
                total_in_chunks += sz
                del data
            except Exception as e:
                logger.warning(f"Corrupt chunk {ec}: {e}")
                break

        expected_total = num_samples or total_in_chunks
        # Accept chunks if we have approximately the right number of samples
        # (within one chunk size tolerance)
        if total_in_chunks >= expected_total - CHUNK_SIZE:
            logger.info(
                f"Found {len(chunk_files_tmp)} existing chunks with {total_in_chunks:,} "
                f"samples — skipping encoding phase."
            )
            chunk_files = chunk_files_tmp
            total_valid = total_in_chunks
            need_encoding = False
        else:
            logger.info(
                f"Found {len(chunk_files_tmp)} chunks ({total_in_chunks:,} samples) "
                f"but expected ~{expected_total:,} — will re-encode."
            )

    if need_encoding:
        logger.info(f"Downloading dataset: {hf_dataset}")
        ds = load_dataset(hf_dataset, split="train")

        if num_samples is not None and num_samples < len(ds):
            logger.info(f"Sampling {num_samples} from {len(ds):,} total positions.")
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(ds), size=num_samples, replace=False)
            indices.sort()
            ds = ds.select(indices.tolist())

        total = len(ds)
        logger.info(f"Encoding {total:,} positions into chunks of {CHUNK_SIZE:,}...")

        chunk_boards = np.zeros((CHUNK_SIZE, 8, 8, 12), dtype=np.uint8)
        chunk_scores = np.zeros(CHUNK_SIZE, dtype=np.float32)
        chunk_idx = 0
        chunk_num = 0
        total_valid = 0
        skipped = 0
        chunk_files = []
        t_start = time.time()

        def _flush_chunk():
            nonlocal chunk_idx, chunk_num
            if chunk_idx == 0:
                return
            path = chunk_dir / f"chunk_{chunk_num:03d}.npz"
            np.savez(path, boards=chunk_boards[:chunk_idx], scores=chunk_scores[:chunk_idx])
            chunk_files.append((str(path), chunk_idx))
            elapsed = time.time() - t_start
            logger.info(
                f"  Flushed chunk {chunk_num}: {chunk_idx:,} samples -> {path} "
                f"[{elapsed:.0f}s elapsed, {total_valid:,}/{total:,} encoded]"
            )
            chunk_num += 1
            chunk_idx = 0

        for i, row in enumerate(ds):
            if i % 500_000 == 0 and i > 0:
                elapsed = time.time() - t_start
                rate = i / elapsed
                eta = (total - i) / rate if rate > 0 else 0
                logger.info(
                    f"  Encoded {i:,}/{total:,} positions "
                    f"({total_valid:,} valid, {skipped:,} skipped) "
                    f"[{elapsed:.0f}s elapsed, ETA {eta:.0f}s]"
                )

            fen = row["fen"]
            cp = row.get("cp")
            mate = row.get("mate")

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
            # Fix: cp from HuggingFace is from white's perspective.
            # With always_white_perspective=True, the board is mirrored for
            # black so side-to-move is always at bottom. We must also flip
            # the score sign so it means "good for the side-to-move".
            is_white_to_move = " w " in fen
            sign = 1.0 if is_white_to_move else -1.0
            chunk_scores[chunk_idx] = sign * normalize_cp(raw_cp, scale=cp_scale)
            chunk_idx += 1
            total_valid += 1

            if chunk_idx >= CHUNK_SIZE:
                _flush_chunk()

        _flush_chunk()
        del ds, chunk_boards, chunk_scores
        gc.collect()

        elapsed = time.time() - t_start
        logger.info(
            f"Encoding complete: {total_valid:,} positions ({skipped:,} skipped) "
            f"in {len(chunk_files)} chunks [{elapsed:.0f}s]."
        )

    # ------------------------------------------------------------------
    # Phase 2: Shuffle + split into train/val/test
    # ------------------------------------------------------------------
    logger.info(f"Building splits from {total_valid:,} positions...")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(total_valid)

    n_train = int(total_valid * train_ratio)
    n_val = int(total_valid * val_ratio)

    split_indices = {
        "train": np.sort(perm[:n_train]),
        "val": np.sort(perm[n_train : n_train + n_val]),
        "test": np.sort(perm[n_train + n_val :]),
    }

    # Build chunk boundary map
    chunk_boundaries: list[tuple[int, int, str]] = []
    offset = 0
    for path, size in chunk_files:
        chunk_boundaries.append((offset, offset + size, path))
        offset += size

    split_info_path = output_path / "split_info.json"
    info: dict = {
        "seed": seed,
        "total": total_valid,
        "format": "npy" if use_mmap else "npz",
        "splits": {},
    }

    for split_name, indices in split_indices.items():
        n_split = len(indices)
        mem_needed = n_split * 8 * 8 * 12 * 1  # uint8 boards
        logger.info(
            f"  Building {split_name}: {n_split:,} samples "
            f"(~{_fmt_bytes(mem_needed)} for boards)"
        )

        t_split = time.time()

        if use_mmap:
            _write_split_mmap(
                output_path, split_name, indices, n_split,
                chunk_boundaries, rng
            )
            info["splits"][split_name] = {
                "size": n_split,
                "boards_path": str(output_path / f"{split_name}_boards.npy"),
                "scores_path": str(output_path / f"{split_name}_scores.npy"),
            }
        else:
            _write_split_npz(
                output_path, split_name, indices, n_split,
                chunk_boundaries, rng
            )
            info["splits"][split_name] = {
                "size": n_split,
                "path": str(output_path / f"{split_name}.npz"),
            }

        elapsed = time.time() - t_split
        logger.info(f"  ✓ Saved {split_name} ({n_split:,} samples) in {elapsed:.1f}s")

    split_info_path.write_text(json.dumps(info, indent=2))

    # Clean up chunk files
    shutil.rmtree(chunk_dir, ignore_errors=True)
    logger.success(f"Data preparation complete. Info saved to {split_info_path}")
    return info


# ---------------------------------------------------------------------------
# Split writers
# ---------------------------------------------------------------------------

def _write_split_mmap(
    output_path: Path,
    split_name: str,
    indices: np.ndarray,
    n_split: int,
    chunk_boundaries: list[tuple[int, int, str]],
    rng: np.random.Generator,
) -> None:
    """Write a split using memory-mapped .npy files (no OOM for large data).

    Creates {split}_boards.npy and {split}_scores.npy on disk, filling them
    chunk by chunk. Only one chunk is in RAM at a time.
    """
    boards_path = output_path / f"{split_name}_boards.npy"
    scores_path = output_path / f"{split_name}_scores.npy"

    # Create memory-mapped output files (uint8 for boards: 4× smaller)
    boards_out = np.lib.format.open_memmap(
        str(boards_path), mode="w+", dtype=np.uint8,
        shape=(n_split, 8, 8, 12),
    )
    scores_out = np.lib.format.open_memmap(
        str(scores_path), mode="w+", dtype=np.float32,
        shape=(n_split,),
    )

    # Fill from chunks (indices are sorted, so we read chunks sequentially)
    out_pos = 0
    for chunk_start, chunk_end, chunk_path in chunk_boundaries:
        mask = (indices >= chunk_start) & (indices < chunk_end)
        global_idxs = indices[mask]
        if len(global_idxs) == 0:
            continue

        local_idxs = global_idxs - chunk_start
        data = np.load(chunk_path)
        n_here = len(local_idxs)
        boards_out[out_pos : out_pos + n_here] = data["boards"][local_idxs]
        scores_out[out_pos : out_pos + n_here] = data["scores"][local_idxs]
        out_pos += n_here
        del data
        gc.collect()
        logger.debug(
            f"    {split_name}: filled {out_pos:,}/{n_split:,} from {Path(chunk_path).name}"
        )

    # Flush to disk
    boards_out.flush()
    scores_out.flush()
    del boards_out, scores_out
    gc.collect()


def _write_split_npz(
    output_path: Path,
    split_name: str,
    indices: np.ndarray,
    n_split: int,
    chunk_boundaries: list[tuple[int, int, str]],
    rng: np.random.Generator,
) -> None:
    """Write a split as a single .npz file (fine for small data)."""
    boards_out = np.zeros((n_split, 8, 8, 12), dtype=np.uint8)
    scores_out = np.zeros(n_split, dtype=np.float32)

    out_pos = 0
    for chunk_start, chunk_end, chunk_path in chunk_boundaries:
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

    # Shuffle in memory (fine for small data)
    shuffle_idx = np.arange(n_split)
    rng.shuffle(shuffle_idx)
    boards_out[:] = boards_out[shuffle_idx]
    scores_out[:] = scores_out[shuffle_idx]
    del shuffle_idx

    path = output_path / f"{split_name}.npz"
    np.savez(path, boards=boards_out, scores=scores_out)
    del boards_out, scores_out
    gc.collect()
