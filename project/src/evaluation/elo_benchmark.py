from __future__ import annotations
"""ELO evaluation via Lichess puzzle benchmark.

Estimates model playing strength by testing on 1200 rated puzzles
and fitting a linear regression at the 50% accuracy threshold.
"""

from pathlib import Path

import chess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.linear_model import LinearRegression

from src.data.encoding import fen_to_tensor


# Tier bin edges for puzzle rating grouping
TIER_BINS = [0, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250]


def load_puzzles(puzzle_path: str) -> pd.DataFrame:
    """Load and prepare the puzzle benchmark dataset.

    Args:
        puzzle_path: Path to test_puzzles.feather.

    Returns:
        DataFrame with columns: fen, moves (list), rating, num_of_moves.
    """
    df = pd.read_feather(puzzle_path)
    df["moves"] = df["moves"].str.split(" ")
    return df


@torch.no_grad()
def model_best_move(
    model: nn.Module,
    board: chess.Board,
    device: torch.device,
) -> chess.Move | None:
    """Ask the model to pick the best legal move.

    For each legal move, push it on the board, encode the resulting position,
    and evaluate. Pick the move that gives the *lowest* score for the opponent
    (= best for us, since scores are from side-to-move perspective).

    Args:
        model: Evaluation model outputting scores in (-1, 1).
        board: Current board state.
        device: Torch device.

    Returns:
        Best move, or None if no legal moves.
    """
    model.eval()
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    tensors = []
    for move in legal_moves:
        board.push(move)
        t = fen_to_tensor(board.fen(), always_white_perspective=True)
        tensors.append(t)
        board.pop()

    batch = np.array(tensors, dtype=np.float32)
    batch_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)

    scores = model(batch_tensor).cpu().numpy().flatten()

    # We want the position *worst* for the opponent (lowest eval after our move)
    best_idx = int(np.argmin(scores))
    return legal_moves[best_idx]


def solve_puzzle(
    model: nn.Module,
    puzzle: pd.Series,
    device: torch.device,
) -> dict:
    """Attempt to solve a single puzzle with the model.

    Puzzle convention:
      - Even-indexed moves (0, 2, 4…) are the opponent's moves (given).
      - Odd-indexed moves (1, 3, 5…) are the engine's expected moves.

    Args:
        model: The evaluation model.
        puzzle: Row from the puzzle DataFrame.
        device: Torch device.

    Returns:
        Dict with 'success' (bool) and 'correct_moves' (int).
    """
    board = chess.Board(puzzle["fen"])
    moves = puzzle["moves"]

    our_total = len([m for i, m in enumerate(moves) if i % 2 == 1])
    correct = 0

    for i, expected_uci in enumerate(moves):
        if i % 2 == 0:
            # Opponent's move — just play it
            board.push(chess.Move.from_uci(expected_uci))
        else:
            # Our move — ask the model
            predicted = model_best_move(model, board, device)
            if predicted and predicted.uci() == expected_uci:
                correct += 1
                board.push(predicted)
            else:
                break  # stop on first mistake

    return {"success": correct == our_total, "correct_moves": correct}


def estimate_elo(
    model: nn.Module,
    puzzle_path: str,
    device: torch.device | None = None,
) -> dict:
    """Run full ELO estimation benchmark.

    1. Solve all 1200 puzzles.
    2. Group accuracy by rating tier.
    3. Fit linear regression: mean_tier_rating = f(accuracy).
    4. ELO = predicted rating at 50% accuracy.

    Args:
        model: The evaluation model.
        puzzle_path: Path to test_puzzles.feather.
        device: Torch device (auto-detected if None).

    Returns:
        Dict with 'elo', 'tier_results' DataFrame, 'total_accuracy'.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    puzzles = load_puzzles(puzzle_path)
    logger.info(f"Evaluating on {len(puzzles)} puzzles...")

    results = []
    for idx, row in puzzles.iterrows():
        r = solve_puzzle(model, row, device)
        results.append(r["success"])

    puzzles["success"] = results
    total_acc = sum(results) / len(results)
    logger.info(f"Overall puzzle accuracy: {total_acc:.1%}")

    # Group by tier
    puzzles["tier"] = pd.cut(puzzles["rating"], bins=TIER_BINS)
    tier_stats = puzzles.groupby("tier", observed=True).agg(
        accuracy=("success", "mean"),
        count=("success", "count"),
        mean_rating=("rating", "mean"),
    )

    logger.info("Tier results:")
    for tier, row in tier_stats.iterrows():
        logger.info(f"  {tier}: acc={row['accuracy']:.1%} (n={int(row['count'])})")

    # Linear regression for ELO estimation
    X = tier_stats["accuracy"].values.reshape(-1, 1)
    y = tier_stats["mean_rating"].values

    if len(X) < 2:
        logger.warning("Not enough tiers to fit regression. Returning accuracy only.")
        return {"elo": None, "tier_results": tier_stats, "total_accuracy": total_acc}

    reg = LinearRegression().fit(X, y)
    elo = float(reg.predict([[0.5]])[0])

    logger.success(f"Estimated ELO: {elo:.0f}")

    return {
        "elo": elo,
        "tier_results": tier_stats,
        "total_accuracy": total_acc,
        "regression_coef": float(reg.coef_[0]),
        "regression_intercept": float(reg.intercept_),
    }
