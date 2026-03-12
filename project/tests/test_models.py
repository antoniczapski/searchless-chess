"""Tests for model creation, forward pass, and registry."""

import torch
import pytest

from src.models.registry import create_model, count_parameters, MODEL_REGISTRY


class TestModelRegistry:
    """Tests for the model registry."""

    def test_mlp_in_registry(self):
        assert "mlp" in MODEL_REGISTRY

    def test_create_mlp(self):
        cfg = {"architecture": "mlp", "hidden_dim": 64, "num_layers": 2, "dropout": 0.0}
        model = create_model(cfg)
        assert model is not None

    def test_unknown_architecture_raises(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            create_model({"architecture": "unknown_model_xyz"})


class TestMLPForwardPass:
    """Tests for MLP forward pass."""

    @pytest.fixture
    def model(self):
        cfg = {"architecture": "mlp", "hidden_dim": 64, "num_layers": 2, "dropout": 0.0}
        return create_model(cfg)

    def test_output_shape(self, model):
        x = torch.randn(4, 12, 8, 8)
        out = model(x)
        assert out.shape == (4, 1)

    def test_output_range(self, model):
        """Output should be in (-1, 1) due to Tanh activation."""
        x = torch.randn(16, 12, 8, 8)
        out = model(x)
        assert (out > -1.0).all() and (out < 1.0).all()

    def test_batch_size_one(self, model):
        x = torch.randn(1, 12, 8, 8)
        out = model(x)
        assert out.shape == (1, 1)

    def test_count_parameters(self, model):
        n = count_parameters(model)
        assert n > 0
        assert isinstance(n, int)

    def test_deterministic(self, model):
        model.eval()
        x = torch.randn(2, 12, 8, 8)
        y1 = model(x)
        y2 = model(x)
        torch.testing.assert_close(y1, y2)
