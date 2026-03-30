from __future__ import annotations
"""ViT with Attention Residuals for chess position evaluation.

Applies insights from "Attention Residuals" (Kimi Team, 2025):
replace fixed residual accumulation with depth-wise softmax selection.
Each sublayer (attention and MLP) receives a learned weighted mixture
of recent sublayer outputs, rather than the running sum.

Uses a *windowed* variant (window_size=3) instead of full AttnRes to
bound GPU memory: each sublayer only looks at the last W depth sources.
Sources outside the window are detached from the computation graph so
autograd does not keep the entire chain alive.

Key changes from base ChessViT:
  1. Windowed AttnRes: depth-wise softmax retrieval over W recent outputs
  2. Per-sublayer pseudo-query (zero-init) + RMSNorm on depth keys
  3. SwiGLU FFN (replaces GELU FFN for better parameter efficiency)
  4. RMSNorm instead of LayerNorm (matches paper's normalization)

Architecture:
  Input(8,8,12) → Reshape(64,12) → LinearProjection(dim)
  → +PosEmbed → N× AttnResTransformerBlock → RMSNorm → GlobalAvgPool
  → Dropout → Dense(1) → tanh

Each block has 2 sublayers = 2 AttnRes mixing points.
With L blocks and window W=3, memory scales with W, not with L.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More lightweight than LayerNorm — no learnable bias, no mean subtraction.
    Used in the paper for both depth-key normalization and pre-sublayer norms.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class DepthAttnRes(nn.Module):
    """Windowed Attention Residual mixer over depth sources.

    Inspired by "Attention Residuals" (Kimi Team, 2025), but uses a sliding
    window of the W most recent depth sources instead of all previous ones.
    This bounds the memory footprint regardless of model depth while still
    providing the key benefit: learned, weighted mixing of recent sublayer
    outputs rather than a fixed running sum.

    With window_size=3, each sublayer mixes its immediate predecessor plus
    2 earlier sources — enough to skip layers and route information.

    The pseudo-query is zero-initialized so training starts with uniform
    mixing (equivalent to averaged residual), avoiding instability.
    """

    def __init__(self, dim: int, window_size: int = 3):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        # Learned pseudo-query: one vector per sublayer instance
        # Zero-init → uniform attention at start (paper Section 3)
        self.query = nn.Parameter(torch.zeros(dim))
        # RMSNorm for keys before scoring (paper says this matters)
        self.key_norm = RMSNorm(dim)

    def forward(self, sources: list[torch.Tensor]) -> torch.Tensor:
        """Mix depth sources via softmax attention (windowed).

        Args:
            sources: List of S tensors, each (B, T, D), representing
                     previous sublayer outputs (and the initial embedding).
                     Only the last ``window_size`` sources are used.

        Returns:
            Weighted mixture (B, T, D).
        """
        # Take only the last W sources
        window = sources[-self.window_size:]

        if len(window) == 1:
            return window[0]

        # Stack the small window: (W, B, T, D) — at most 3 tensors
        stacked = torch.stack(window, dim=0)

        # Normalize keys for scoring: (W, B, T, D)
        keys_normed = self.key_norm(stacked)

        # Score: dot product of pseudo-query with each normalized source
        # query: (D,) → broadcasts to (W, B, T, D) → sum → (W, B, T)
        scores = (keys_normed * self.query).sum(dim=-1)

        # Softmax over depth axis, per token position
        weights = F.softmax(scores, dim=0).unsqueeze(-1)   # (W, B, T, 1)

        # Weighted sum: (W, B, T, D) * (W, B, T, 1) → (B, T, D)
        return (stacked * weights).sum(dim=0)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network.

    SwiGLU uses a gated linear unit with SiLU activation, which has been
    shown to outperform GELU FFN at equivalent parameter count.

    For comparable params to GELU FFN with expansion ratio R:
      GELU FFN: 2 * dim * (dim * R) = 2 * dim² * R params
      SwiGLU:   3 * dim * hidden = 3 * dim * hidden params
    So hidden = 2/3 * dim * R for param parity.
    """

    def __init__(self, dim: int, ffn_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        # Adjusted hidden dim for param parity with GELU FFN
        hidden = int(2 * dim * ffn_ratio / 3)
        # Round to multiple of 8 for hardware efficiency
        hidden = ((hidden + 7) // 8) * 8

        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class AttnResTransformerBlock(nn.Module):
    """Transformer block with Attention Residuals.

    Instead of `x = x + sublayer(norm(x))`, each sublayer receives its
    input from a depth-wise softmax mixture of all previous sublayer outputs.

    Each block contains 2 sublayers (attention + MLP), each with its own
    AttnRes mixer and pseudo-query.
    """

    def __init__(self, dim: int, num_heads: int, ffn_ratio: int = 2, dropout: float = 0.1):
        super().__init__()

        # AttnRes mixers — one per sublayer
        self.attn_res_attn = DepthAttnRes(dim)
        self.attn_res_mlp = DepthAttnRes(dim)

        # Pre-norm for attention sublayer
        self.norm_attn = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Pre-norm for MLP sublayer
        self.norm_mlp = RMSNorm(dim)
        self.mlp = SwiGLUFFN(dim, ffn_ratio, dropout)

    def forward(
        self, sources: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Forward pass with AttnRes depth mixing.

        Args:
            sources: List of previous depth source tensors, each (B, T, D).

        Returns:
            Tuple of (updated_sources, output_tensor).
            updated_sources is truncated to the last ``window_size`` entries
            to bound GPU memory — older sources are discarded entirely.
        """
        # --- Attention sublayer ---
        h_attn = self.attn_res_attn(sources)
        h_normed = self.norm_attn(h_attn)
        attn_out, _ = self.attn(h_normed, h_normed, h_normed)
        v_attn = h_attn + attn_out
        sources = sources + [v_attn]

        # --- MLP sublayer ---
        h_mlp = self.attn_res_mlp(sources)
        mlp_out = self.mlp(self.norm_mlp(h_mlp))
        v_mlp = h_mlp + mlp_out
        sources = sources + [v_mlp]

        # Truncate sources to the window size — discard older tensors
        # so they can be freed from GPU memory. Each subsequent AttnRes
        # mixer only looks at the last W sources anyway.
        w = max(self.attn_res_attn.window_size, self.attn_res_mlp.window_size)
        if len(sources) > w:
            sources = list(sources[-w:])

        return sources, v_mlp


class ChessViTAttnRes(nn.Module):
    """Vision Transformer with Attention Residuals for chess evaluation.

    Replaces standard residual connections with Full AttnRes — each sublayer
    retrieves a softmax-weighted mixture of all previous sublayer outputs.

    With L transformer blocks (each having 2 sublayers), there are 2L+1
    depth sources: the initial embedding plus 2 per block.

    Args:
        projection_dim: Transformer model dimension.
        num_heads: Number of attention heads.
        transformer_layers: Number of transformer blocks.
        ffn_ratio: FFN expansion ratio.
        dropout_rate: Dropout rate.
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

        # Transformer blocks with AttnRes
        self.blocks = nn.ModuleList([
            AttnResTransformerBlock(projection_dim, num_heads, ffn_ratio, dropout_rate)
            for _ in range(transformer_layers)
        ])

        # Output head
        self.norm = RMSNorm(projection_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(projection_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following ViT + AttnRes conventions."""
        # Positional embedding: truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Patch projection
        nn.init.trunc_normal_(self.patch_proj.weight, std=0.02)
        nn.init.zeros_(self.patch_proj.bias)

        # Output head: small init
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        # Transformer blocks
        for block in self.blocks:
            # MHA weights — default init is usually fine
            # SwiGLU: small init for stability
            for m in [block.mlp.w_gate, block.mlp.w_up, block.mlp.w_down]:
                nn.init.trunc_normal_(m.weight, std=0.02)

            # AttnRes pseudo-queries are already zero-init (paper requirement)
            # RMSNorm weights are already ones-init

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

        # This is depth source 0 (the "embedding" in paper terms)
        sources = [x]

        # Transformer encoder with AttnRes
        for block in self.blocks:
            sources, x = block(sources)

        # Output: RMSNorm → GlobalAvgPool → Dropout → Head → tanh
        x = self.norm(x)
        x = x.mean(dim=1)       # Global average pooling: (B, dim)
        x = self.dropout(x)
        x = self.head(x)        # (B, 1)
        x = torch.tanh(x)       # Bounded to [-1, 1]

        return x
