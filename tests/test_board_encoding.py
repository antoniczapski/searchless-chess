"""Tests for FEN-to-tensor board encoding and score normalization."""

import numpy as np
import pytest

from src.data.encoding import fen_to_tensor, mate_to_cp, normalize_cp


# ─── FEN encoding ──────────────────────────────────────────────

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class TestFenToTensor:
    """Verify board tensor shape, dtype, and piece placement."""

    def test_shape(self):
        t = fen_to_tensor(STARTING_FEN)
        assert t.shape == (8, 8, 12)

    def test_dtype(self):
        t = fen_to_tensor(STARTING_FEN)
        assert t.dtype == np.uint8

    def test_starting_white_pawns(self):
        t = fen_to_tensor(STARTING_FEN)
        # Channel 0 = white pawns; rank 2 (row 6 in array) has 8 pawns
        assert t[6, :, 0].sum() == 8

    def test_empty_board(self):
        t = fen_to_tensor("8/8/8/8/8/8/8/8 w - - 0 1")
        assert t.sum() == 0

    def test_single_king(self):
        t = fen_to_tensor("8/8/8/8/8/8/8/4K3 w - - 0 1")
        assert t[7, 4, 5] == 1  # e1 = king channel
        assert t.sum() == 1

    def test_total_pieces_starting(self):
        t = fen_to_tensor(STARTING_FEN)
        assert t.sum() == 32  # 16 white + 16 black pieces

    def test_black_perspective_flips(self):
        fen_w = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        fen_b = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        tw = fen_to_tensor(fen_w, always_white_perspective=True)
        tb = fen_to_tensor(fen_b, always_white_perspective=True)
        assert not np.array_equal(tw, tb)


# ─── Score normalization ───────────────────────────────────────

class TestNormalizeCp:
    """Verify normalize_cp maps to [-1, 1] correctly."""

    def test_zero(self):
        assert normalize_cp(0) == pytest.approx(0.0)

    def test_range(self):
        for cp in [-50000, -1000, 0, 1000, 50000]:
            assert -1.0 <= normalize_cp(cp) <= 1.0

    def test_symmetry(self):
        assert normalize_cp(500) == pytest.approx(-normalize_cp(-500))

    def test_large_positive_near_one(self):
        assert normalize_cp(30000) > 0.9


class TestMateToCp:
    """Verify mate-to-centipawn conversion."""

    def test_positive_mate(self):
        cp = mate_to_cp(1)
        assert cp is not None and cp > 0

    def test_negative_mate(self):
        cp = mate_to_cp(-1)
        assert cp is not None and cp < 0

    def test_zero_returns_none(self):
        assert mate_to_cp(0) is None

    def test_deeper_mate_lower(self):
        assert mate_to_cp(1) > mate_to_cp(5)
