from __future__ import annotations
"""Chess board encoding utilities.

Converts FEN strings to 8x8x12 tensor representation for neural network input.
"""

import chess
import numpy as np


PIECE_TO_CHANNEL = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}


def fen_to_tensor(fen: str, always_white_perspective: bool = True) -> np.ndarray:
    """Convert FEN to 8x8x12 binary tensor.

    Args:
        fen: FEN notation string.
        always_white_perspective: If True, board is mirrored for black's turn
            so the model always sees the position from the side-to-move.

    Returns:
        Tensor of shape (8, 8, 12) with dtype float32.
        Channels 0-5: side-to-move pieces (P, N, B, R, Q, K).
        Channels 6-11: opponent pieces (P, N, B, R, Q, K).
    """
    board = chess.Board(fen)

    if always_white_perspective and not board.turn:
        board = board.mirror()

    tensor = np.zeros((8, 8, 12), dtype=np.float32)

    for square, piece in board.piece_map().items():
        channel = PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]
        row, col = divmod(square, 8)
        tensor[7 - row, col, channel] = 1.0

    return tensor


def mate_to_cp(
    mate: int | float | None,
    max_cp_no_mate: int = 20_000,
    mate_margin: int = 7_000,
    step: int = 100,
) -> int | None:
    """Convert mate-in-N to centipawn equivalent.

    Args:
        mate: Mate-in-N value. Positive = winning, negative = losing.
        max_cp_no_mate: Base cp for non-mate positions.
        mate_margin: Additional margin for mate conversions.
        step: Centipawn decrement per move to mate.

    Returns:
        Centipawn integer, or None if mate is None/NaN/0.
    """
    if mate is None or mate == 0:
        return None
    try:
        import math
        if math.isnan(mate):
            return None
    except (TypeError, ValueError):
        pass

    sign = 1 if mate > 0 else -1
    cp_value = max_cp_no_mate + mate_margin - step * (abs(int(mate)) - 1)
    return sign * max(abs(cp_value), max_cp_no_mate + step)


def normalize_cp(cp: float, scale: float = 10_000.0) -> float:
    """Normalize centipawn score to [-1, 1] using tanh-like scaling.

    Args:
        cp: Raw centipawn value.
        scale: Scaling factor. cp/scale is passed through tanh.

    Returns:
        Normalized score in [-1, 1].
    """
    return float(np.tanh(cp / scale))
