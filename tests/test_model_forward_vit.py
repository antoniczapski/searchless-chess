"""Tests for ViT and ViT-AttnRes model forward pass and properties."""

import torch
import pytest

from src.models.registry import create_model, count_parameters


# ─── ViT ───────────────────────────────────────────────────────

class TestViTForward:
    """Verify ViT instantiation, forward pass, and output properties."""

    @pytest.fixture
    def model(self):
        cfg = {
            "architecture": "vit",
            "projection_dim": 32,
            "num_heads": 2,
            "transformer_layers": 1,
            "ffn_ratio": 2,
            "dropout_rate": 0.0,
        }
        return create_model(cfg)

    def test_output_shape(self, model):
        x = torch.randn(4, 12, 8, 8)
        out = model(x)
        assert out.shape == (4, 1)

    def test_output_bounded(self, model):
        """Output uses tanh, should be in (-1, 1)."""
        model.eval()
        x = torch.randn(8, 12, 8, 8)
        out = model(x)
        assert (out > -1.0).all() and (out < 1.0).all()

    def test_no_nans(self, model):
        model.eval()
        x = torch.randn(4, 12, 8, 8)
        out = model(x)
        assert torch.isfinite(out).all()

    def test_batch_size_one(self, model):
        out = model(torch.randn(1, 12, 8, 8))
        assert out.shape == (1, 1)

    def test_deterministic(self, model):
        model.eval()
        x = torch.randn(2, 12, 8, 8)
        torch.testing.assert_close(model(x), model(x))

    def test_param_count(self, model):
        assert count_parameters(model) > 0


# ─── ViT + AttnRes ────────────────────────────────────────────

class TestViTAttnResForward:
    """Verify ViT-AttnRes instantiation and forward pass."""

    @pytest.fixture
    def model(self):
        cfg = {
            "architecture": "vit_attnres",
            "projection_dim": 32,
            "num_heads": 2,
            "transformer_layers": 2,
            "ffn_ratio": 2,
            "dropout_rate": 0.0,
        }
        return create_model(cfg)

    def test_output_shape(self, model):
        x = torch.randn(4, 12, 8, 8)
        out = model(x)
        assert out.shape == (4, 1)

    def test_output_bounded(self, model):
        model.eval()
        x = torch.randn(8, 12, 8, 8)
        out = model(x)
        assert (out > -1.0).all() and (out < 1.0).all()

    def test_no_nans(self, model):
        model.eval()
        x = torch.randn(4, 12, 8, 8)
        out = model(x)
        assert torch.isfinite(out).all()


# ─── BDH ──────────────────────────────────────────────────────

class TestBDHForward:
    """Verify BDH model instantiation and forward pass."""

    @pytest.fixture
    def model(self):
        cfg = {
            "architecture": "bdh",
            "n_embd": 32,
            "n_head": 2,
            "n_neurons_mult": 4,
            "thinking_steps": 2,
            "dropout": 0.0,
            "init_damping": 0.9,
            "deep_supervision": False,
        }
        return create_model(cfg)

    def test_output_shape(self, model):
        x = torch.randn(4, 12, 8, 8)
        out = model(x)
        assert out.shape == (4, 1)

    def test_output_bounded(self, model):
        model.eval()
        x = torch.randn(8, 12, 8, 8)
        out = model(x)
        assert (out > -1.0).all() and (out < 1.0).all()

    def test_no_nans(self, model):
        model.eval()
        x = torch.randn(4, 12, 8, 8)
        out = model(x)
        assert torch.isfinite(out).all()


# ─── Registry ─────────────────────────────────────────────────

class TestModelRegistry:
    """Verify registry contains all expected architectures."""

    @pytest.mark.parametrize("arch", ["mlp", "bdh", "vit", "vit_attnres"])
    def test_architecture_in_registry(self, arch):
        from src.models.registry import MODEL_REGISTRY
        assert arch in MODEL_REGISTRY

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            create_model({"architecture": "nonexistent_xyz"})
