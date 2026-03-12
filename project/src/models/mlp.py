from __future__ import annotations
"""Simple MLP baseline for chess position evaluation.

Used as a sanity-check baseline to validate the full pipeline.
"""

import torch
import torch.nn as nn


class ChessMLP(nn.Module):
    """Flat MLP that takes 8x8x12 board tensor and outputs a scalar score.

    Args:
        hidden_dim: Hidden layer dimension.
        num_layers: Number of hidden layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        **kwargs,  # absorb extra config keys gracefully
    ):
        super().__init__()
        input_dim = 8 * 8 * 12  # 768

        layers: list[nn.Module] = []
        in_features = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_features = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Tanh())  # output in (-1, 1)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Board tensor of shape (B, 12, 8, 8).

        Returns:
            Score tensor of shape (B, 1) in (-1, 1).
        """
        b = x.size(0)
        x = x.reshape(b, -1)  # (B, 768)
        return self.net(x)
