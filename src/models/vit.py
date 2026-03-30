from __future__ import annotations
"""Vision Transformer (ViT) for chess position evaluation.

Faithful PyTorch reimplementation of the ViT architecture from
mateuszgrzyb-pl/searchless-chess, adapted for our training infrastructure.

Reference architecture:
  Input(8,8,12) → Reshape(64,12) → LinearProjection(dim)
  → +PosEmbed → N× TransformerBlock → LayerNorm → GlobalAvgPool
  → Dropout → Dense(1) → tanh
"""

import math

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block.

    Architecture (matching reference exactly):
        x → LayerNorm → MultiHeadAttention → + residual
          → LayerNorm → FFN(dim→ffn_dim, GELU) → Dropout → Dense(dim) → + residual

    Args:
        dim: Model dimension (projection_dim).
        num_heads: Number of attention heads.
        ffn_ratio: FFN hidden dim = dim * ffn_ratio. Reference uses 2.
        dropout: Dropout rate for attention and FFN.
    """

    def __init__(self, dim: int, num_heads: int, ffn_ratio: int = 2, dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        ffn_dim = dim * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class ChessViT(nn.Module):
    """Vision Transformer for chess position evaluation.

    Converts an 8×8×12 board representation into a scalar evaluation
    in [-1, 1] using a sequence of transformer encoder blocks.

    The board is treated as 64 "patches" (one per square) with 12 channels
    each (6 piece types × 2 colors). This is equivalent to ViT with 1×1
    patches on an 8×8 image.

    Architecture matches the reference ViT from mateuszgrzyb-pl/searchless-chess.

    Args:
        projection_dim: Transformer model dimension. Default 256 (ViT-S).
        num_heads: Number of attention heads. Default 4 (ViT-S).
        transformer_layers: Number of transformer blocks. Default 5 (ViT-S).
        ffn_ratio: FFN expansion ratio. Default 2 (matching reference).
        dropout_rate: Dropout rate. Default 0.1.
    """

    def __init__(
        self,
        projection_dim: int = 256,
        num_heads: int = 4,
        transformer_layers: int = 5,
        ffn_ratio: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.projection_dim = projection_dim
        num_patches = 64  # 8 × 8 squares
        patch_dim = 12    # 6 piece types × 2 colors

        # Patch projection: linear embedding of each square's 12-channel vector
        self.patch_proj = nn.Linear(patch_dim, projection_dim)

        # Learnable positional embedding (one per square)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, projection_dim))

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(projection_dim, num_heads, ffn_ratio, dropout_rate)
            for _ in range(transformer_layers)
        ])

        # Output head
        self.norm = nn.LayerNorm(projection_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(projection_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following ViT conventions."""
        # Positional embedding: truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Patch projection
        nn.init.trunc_normal_(self.patch_proj.weight, std=0.02)
        nn.init.zeros_(self.patch_proj.bias)

        # Output head: small init for stable training start
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        # Transformer blocks
        for block in self.blocks:
            # FFN layers
            for m in block.ffn:
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Board tensor of shape (B, 12, 8, 8) — PyTorch CHW format.

        Returns:
            Position evaluation of shape (B, 1), range [-1, 1].
        """
        B = x.shape[0]

        # Convert from PyTorch CHW to patch sequence
        # (B, 12, 8, 8) → (B, 8, 8, 12) → (B, 64, 12)
        x = x.permute(0, 2, 3, 1).reshape(B, 64, 12)

        # Linear projection + positional embedding
        x = self.patch_proj(x) + self.pos_embed

        # Transformer encoder
        for block in self.blocks:
            x = block(x)

        # Output: LayerNorm → GlobalAvgPool → Dropout → Head → tanh
        x = self.norm(x)
        x = x.mean(dim=1)       # Global average pooling: (B, dim)
        x = self.dropout(x)
        x = self.head(x)        # (B, 1)
        x = torch.tanh(x)       # Bounded to [-1, 1]

        return x
