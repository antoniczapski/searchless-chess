"""Smoke test for the evaluation pipeline (no GPU, no real puzzles needed)."""

import numpy as np
import torch
import pytest

from src.models.registry import create_model


class TestEvaluationSmoke:
    """Verify the model-based move selection pipeline runs without crash."""

    @pytest.fixture
    def model(self):
        cfg = {
            "architecture": "mlp",
            "hidden_dim": 32,
            "num_layers": 1,
            "dropout": 0.0,
        }
        m = create_model(cfg)
        m.eval()
        return m

    def test_model_scores_batch(self, model):
        """Model should score a batch of positions without error."""
        batch = torch.randn(10, 12, 8, 8)
        scores = model(batch)
        assert scores.shape == (10, 1)
        assert torch.isfinite(scores).all()

    def test_model_best_move_returns_move(self, model):
        """model_best_move should return a valid chess.Move on the starting position."""
        import chess
        from src.evaluation.elo_benchmark import model_best_move

        board = chess.Board()
        device = torch.device("cpu")
        move = model_best_move(model, board, device)
        assert move is not None
        assert isinstance(move, chess.Move)
        assert move in board.legal_moves

    def test_solve_puzzle_returns_dict(self, model):
        """solve_puzzle should return a dict with 'success' and 'correct_moves'."""
        import pandas as pd
        from src.evaluation.elo_benchmark import solve_puzzle

        # Create a minimal fake puzzle: starting position, move e2e4
        puzzle = pd.Series({
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "moves": ["e2e4", "e7e5"],  # opponent plays e4, we should play e5
            "rating": 500,
        })

        result = solve_puzzle(model, puzzle, torch.device("cpu"))
        assert "success" in result
        assert "correct_moves" in result
        assert isinstance(result["success"], bool)
