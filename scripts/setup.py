"""Full setup script: install dependencies, download data, prepare splits.

Run after cloning on a fresh machine:

    cd project/
    python scripts/setup.py                       # default: 500k positions
    python scripts/setup.py --num-samples 50000000  # 50M for A100

This script:
  1. Checks Python/CUDA environment.
  2. Downloads the puzzle benchmark (if missing).
  3. Downloads chess positions from HuggingFace.
  4. Encodes positions and creates train/val/test splits.
"""

import argparse
import shutil
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def check_environment():
    """Print environment info and verify critical packages."""
    import platform

    print("=" * 60)
    print("Environment check")
    print("=" * 60)
    print(f"  Python:   {sys.version}")
    print(f"  Platform: {platform.system()} {platform.machine()}")

    try:
        import torch

        print(f"  PyTorch:  {torch.__version__}")
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
            print(f"  GPU:      {gpu} ({mem:.1f} GB)")
        else:
            print("  GPU:      Not available (CPU-only)")
    except ImportError:
        print("  PyTorch:  NOT INSTALLED")
        print("  → Install: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    for pkg in ["datasets", "chess", "pandas", "numpy", "sklearn", "loguru"]:
        try:
            __import__(pkg)
        except ImportError:
            print(f"  ✗ Missing package: {pkg}")
            print(f"  → Install: pip install {pkg}")
            sys.exit(1)

    print("  All dependencies OK ✓")
    print()


def ensure_puzzles(puzzle_path: Path):
    """Ensure the puzzle benchmark file exists.

    The file is small (87 KB) and should be committed to the repo.
    If it's somehow missing, download from the reference repo.
    """
    if puzzle_path.exists():
        print(f"  Puzzles already present: {puzzle_path}")
        return

    print("  Puzzle file missing — downloading from reference repo...")
    puzzle_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import urllib.request

        url = (
            "https://huggingface.co/datasets/mateuszgrzyb/lichess-stockfish-normalized/"
            "resolve/main/test_puzzles.feather"
        )
        urllib.request.urlretrieve(url, str(puzzle_path))
        print(f"  Downloaded puzzles to {puzzle_path}")
    except Exception as e:
        print(f"  ✗ Failed to download puzzles: {e}")
        print("  → You can copy test_puzzles.feather manually from the reference repo.")
        print("    See: https://github.com/mateuszgrzyb/searchless-chess")


def prepare_data(
    num_samples: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    cp_scale: float,
    output_dir: str,
):
    """Download from HuggingFace and create train/val/test .npz splits."""
    from src.data.prepare import download_and_prepare

    info = download_and_prepare(
        hf_dataset="mateuszgrzyb/lichess-stockfish-normalized",
        output_dir=output_dir,
        num_samples=num_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        cp_scale=cp_scale,
    )
    return info


def main():
    parser = argparse.ArgumentParser(
        description="Setup: download data and prepare training splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500_000,
        help="Number of positions to sample from HuggingFace dataset",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.80, help="Training split ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.10, help="Validation split ratio"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.10, help="Test split ratio"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--cp-scale", type=float, default=10_000.0, help="Centipawn normalization scale"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory for .npz output files",
    )
    parser.add_argument(
        "--puzzle-path",
        type=str,
        default="data/puzzles/test_puzzles.feather",
        help="Path to puzzle benchmark file",
    )
    parser.add_argument(
        "--skip-env-check",
        action="store_true",
        help="Skip environment verification",
    )
    args = parser.parse_args()

    # 1. Environment
    if not args.skip_env_check:
        check_environment()

    # 2. Puzzles
    print("Checking puzzle benchmark...")
    ensure_puzzles(Path(args.puzzle_path))
    print()

    # 3. Training data
    print(f"Preparing training data ({args.num_samples:,} positions)...")
    print(f"  Split: {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")
    print(f"  Output: {args.output_dir}/")
    print()

    info = prepare_data(
        num_samples=args.num_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        cp_scale=args.cp_scale,
        output_dir=args.output_dir,
    )

    print()
    print("=" * 60)
    print("Setup complete!")
    print("=" * 60)
    total = info.get("total", sum(s["size"] for s in info.get("splits", {}).values()))
    print(f"  Total positions: {total:,}")
    for name, split in info.get("splits", {}).items():
        print(f"  {name:>5}: {split['size']:,} samples")
    print()
    print("Next steps:")
    print("  python scripts/train.py --config configs/train_bdh.yaml")
    print("  python scripts/evaluate.py --config configs/train_bdh.yaml \\")
    print("      --checkpoint outputs/bdh-chess-v2/checkpoints/best_model.pt")


if __name__ == "__main__":
    main()
