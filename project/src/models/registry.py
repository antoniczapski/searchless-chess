"""Model registry — factory for creating models by name from config."""

import torch.nn as nn

from src.models.mlp import ChessMLP
from src.models.bdh import BDHChess


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "mlp": ChessMLP,
    "bdh": BDHChess,
}


def create_model(config: dict) -> nn.Module:
    """Instantiate a model from a config dict.

    Args:
        config: Must contain 'architecture' key and architecture-specific params.

    Returns:
        Instantiated nn.Module.

    Raises:
        ValueError: If architecture is not in registry.
    """
    arch = config["architecture"]
    if arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    cls = MODEL_REGISTRY[arch]
    # Pass all model config except 'architecture' as kwargs
    model_kwargs = {k: v for k, v in config.items() if k != "architecture"}
    return cls(**model_kwargs)


def register_model(name: str, cls: type[nn.Module]) -> None:
    """Register a new model class in the registry."""
    MODEL_REGISTRY[name] = cls


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
