#!/usr/bin/env python3
"""Re-normalize chess evaluation scores from tanh(cp/10000) to tanh(cp/1000).

The reference repository (mateuszgrzyb-pl/searchless-chess) uses tanh(cp/1000)
for score normalization, while our existing 50M dataset uses tanh(cp/10000).

This script converts scores mathematically without re-downloading:
    new_score = tanh(atanh(old_score) * 10)

Board tensors are symlinked since the encoding is identical.

Usage:
    python scripts/renormalize_data.py \\
        --src /net/tscratch/people/plgantoniczapski/bdh-chess-data \\
        --dst /net/tscratch/people/plgantoniczapski/bdh-chess-data-1k
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from loguru import logger


def renormalize_scores(
    scores: np.ndarray,
    old_scale: float = 10_000.0,
    new_scale: float = 1_000.0,
    clip_eps: float = 1e-7,
) -> np.ndarray:
    """Convert scores from tanh(cp/old_scale) to tanh(cp/new_scale).

    Args:
        scores: Array of normalized scores in (-1, 1).
        old_scale: Original cp normalization scale.
        new_scale: Target cp normalization scale.
        clip_eps: Epsilon for clipping before arctanh (avoids inf).

    Returns:
        Re-normalized scores in (-1, 1).
    """
    # Clip to avoid arctanh(±1) = ±inf
    clipped = np.clip(scores, -1.0 + clip_eps, 1.0 - clip_eps)

    # Recover approximate raw centipawn values
    raw_cp = np.arctanh(clipped) * old_scale

    # Re-normalize with new scale
    new_scores = np.tanh(raw_cp / new_scale).astype(np.float32)

    return new_scores


def main():
    parser = argparse.ArgumentParser(description="Re-normalize chess scores")
    parser.add_argument(
        "--src",
        default="/net/tscratch/people/plgantoniczapski/bdh-chess-data",
        help="Source data directory (tanh/10000)",
    )
    parser.add_argument(
        "--dst",
        default="/net/tscratch/people/plgantoniczapski/bdh-chess-data-1k",
        help="Destination data directory (tanh/1000)",
    )
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]

    for split in splits:
        boards_src = src / f"{split}_boards.npy"
        scores_src = src / f"{split}_scores.npy"

        if not boards_src.exists() or not scores_src.exists():
            logger.warning(f"Skipping {split}: files not found")
            continue

        # Symlink boards (encoding is identical)
        boards_dst = dst / f"{split}_boards.npy"
        if not boards_dst.exists():
            os.symlink(str(boards_src.resolve()), str(boards_dst))
            logger.info(f"Symlinked {boards_dst.name} → {boards_src}")
        else:
            logger.info(f"Boards symlink already exists: {boards_dst}")

        # Re-normalize scores
        scores_dst = dst / f"{split}_scores.npy"
        if scores_dst.exists():
            logger.info(f"Scores already exist: {scores_dst} — skipping")
            continue

        logger.info(f"Loading {scores_src} ...")
        old_scores = np.load(str(scores_src))
        logger.info(f"  Shape: {old_scores.shape}, range: [{old_scores.min():.6f}, {old_scores.max():.6f}]")

        new_scores = renormalize_scores(old_scores)
        logger.info(f"  New range: [{new_scores.min():.6f}, {new_scores.max():.6f}]")

        # Diagnostics
        logger.info(f"  Mean: {old_scores.mean():.6f} → {new_scores.mean():.6f}")
        logger.info(f"  Std:  {old_scores.std():.6f} → {new_scores.std():.6f}")

        np.save(str(scores_dst), new_scores)
        logger.info(f"  Saved {scores_dst}")

    # Write split_info.json
    src_info_path = src / "split_info.json"
    if src_info_path.exists():
        info = json.loads(src_info_path.read_text())
        # Update paths
        for split_name, split_data in info.get("splits", {}).items():
            split_data["boards_path"] = str(dst / f"{split_name}_boards.npy")
            split_data["scores_path"] = str(dst / f"{split_name}_scores.npy")
        info["cp_scale"] = 1000.0
        info["note"] = "Re-normalized from tanh(cp/10000) to tanh(cp/1000)"
        (dst / "split_info.json").write_text(json.dumps(info, indent=2))
        logger.info("Wrote split_info.json")

    logger.success("Done! Re-normalized data is at: " + str(dst))


if __name__ == "__main__":
    main()
