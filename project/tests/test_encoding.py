"""Unit tests for FEN encoding and score normalization."""

import numpy as np
import pytest

from src.data.encoding import fen_to_tensor, mate_to_cp, normalize_cp


class TestFenToTensor:
    """Tests for fen_to_tensor()."""

    def test_starting_position_shape(self):
        tensor = fen_to_tensor("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        assert tensor.shape == (8, 8, 12)
        assert tensor.dtype == np.float32

    def test_starting_position_white_pieces(self):
        tensor = fen_to_tensor("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        # Row 6 should be all white pawns (channel 0)
        assert tensor[6, :, 0].sum() == 8  # 8 white pawns on rank 2

    def test_empty_board(self):
        tensor = fen_to_tensor("8/8/8/8/8/8/8/8 w - - 0 1")
        assert tensor.sum() == 0.0

    def test_single_piece(self):
        # White king on e1
        tensor = fen_to_tensor("8/8/8/8/8/8/8/4K3 w - - 0 1")
        # e1 = row 7, col 4; King = channel 5
        assert tensor[7, 4, 5] == 1.0
        assert tensor.sum() == 1.0

    def test_always_white_perspective_flips_for_black(self):
        # Same position, black to move — board should be flipped
        fen_w = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        fen_b = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        tw = fen_to_tensor(fen_w, always_white_perspective=True)
        tb = fen_to_tensor(fen_b, always_white_perspective=True)
        # Black perspective should be rotated 180° and colors swapped
        assert not np.array_equal(tw, tb)

    def test_no_perspective_transform(self):
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        t1 = fen_to_tensor(fen, always_white_perspective=False)
        t2 = fen_to_tensor(fen, always_white_perspective=True)
        assert not np.array_equal(t1, t2)


class TestMateToCp:
    """Tests for mate_to_cp()."""

    def test_mate_in_1_positive(self):
        cp = mate_to_cp(1)
        assert cp is not None
        assert cp > 0

    def test_mate_in_1_negative(self):
        cp = mate_to_cp(-1)
        assert cp is not None
        assert cp < 0

    def test_zero_mate_returns_none(self):
        assert mate_to_cp(0) is None

    def test_deeper_mate_lower_value(self):
        cp1 = mate_to_cp(1)
        cp5 = mate_to_cp(5)
        assert cp1 > cp5  # mate in 1 is better (higher cp)


class TestNormalizeCp:
    """Tests for normalize_cp()."""

    def test_zero_cp(self):
        assert normalize_cp(0) == pytest.approx(0.0)

    def test_large_positive(self):
        val = normalize_cp(30000)
        assert 0.9 < val <= 1.0

    def test_large_negative(self):
        val = normalize_cp(-30000)
        assert -1.0 <= val < -0.9

    def test_symmetry(self):
        assert normalize_cp(100) == pytest.approx(-normalize_cp(-100))

    def test_output_range(self):
        for cp in [-50000, -1000, -100, 0, 100, 1000, 50000]:
            val = normalize_cp(cp)
            assert -1.0 <= val <= 1.0
