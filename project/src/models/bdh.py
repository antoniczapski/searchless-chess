"""BDH chess evaluator — iterative value refinement over a fixed board.

Adapts BDH-GPU (Pathway Technology, 2025) for chess position evaluation.

Key insight: instead of treating the board as a token sequence (which reduces
BDH to a peculiar low-rank MLP), we re-interpret BDH's time axis as K internal
"thinking steps" over the **same fixed board embedding**. A synaptic state rho
acts as a working-memory scratchpad that accumulates relational patterns across
thinking steps.

This preserves BDH's distinctive properties:
  - Shared weights across steps (Universal-Transformer style)
  - Persistent neuron-aligned synaptic state rho
  - Sparse positive activations (ReLU gating)
  - Variable compute by choosing K

Architecture per thinking step k:
  1.  a_k = LinearAttentionRead(rho_{k-1}, x_{k-1})     -- read scratchpad
  2.  y_k = ReLU(D_y . LN(a_k + B_y . b)) * x_{k-1}    -- Hebbian gating
  3.  z_k = LN(E . y_k)                                  -- decode to embedding
  4.  x_k = x_{k-1} + ReLU(D_x(z_k + B_x . b))          -- update neurons
  5.  rho_k = lam . rho_{k-1} + (1-lam) . Write(x_k, z_k) -- update scratchpad
  6.  v_k = tanh(head(z_k))                               -- predict value

Board embedding b is injected at every step (not just once).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Deep-supervision loss wrapper
# ---------------------------------------------------------------------------

class DeepSupervisionLoss(nn.Module):
    """Huber loss with deep supervision and stability penalty.

    Used during training to ensure all thinking steps produce useful
    predictions, not just the final one.

    Args:
        model: BDHChess instance (reads model._step_values).
        deep_sup_weight: Weight for intermediate-step losses.
        stability_weight: Weight for consecutive-step oscillation penalty.
        huber_delta: Delta for Huber loss.
    """

    def __init__(
        self,
        model: nn.Module,
        deep_sup_weight: float = 0.1,
        stability_weight: float = 0.01,
        huber_delta: float = 0.5,
    ):
        super().__init__()
        self.model = model
        self.base_loss = nn.HuberLoss(delta=huber_delta)
        self.deep_sup_weight = deep_sup_weight
        self.stability_weight = stability_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Primary loss on final prediction
        loss = self.base_loss(preds, targets)

        step_values = getattr(self.model, "_step_values", [])
        if len(step_values) > 1:
            # Deep supervision: loss on intermediate steps (not final)
            for v_k in step_values[:-1]:
                loss = loss + self.deep_sup_weight * self.base_loss(v_k, targets)

            # Stability: penalize oscillation between consecutive steps
            for i in range(1, len(step_values)):
                diff = (step_values[i] - step_values[i - 1]).pow(2).mean()
                loss = loss + self.stability_weight * diff

        return loss


# ---------------------------------------------------------------------------
# BDH Chess Model
# ---------------------------------------------------------------------------

class BDHChess(nn.Module):
    """BDH chess evaluator with iterative value refinement.

    Input:  (B, 12, 8, 8) -- board tensor
    Output: (B, 1)         -- evaluation in (-1, 1)

    The board is encoded once. A single shared BDH block runs K times,
    maintaining synaptic state rho as a working-memory scratchpad. The board
    embedding is re-injected at every step.

    Args:
        n_embd: Embedding dimension d.
        n_head: Number of attention heads.
        n_neurons_mult: Multiplier for sparse neuron count (N = mult * d / nh).
        thinking_steps: K -- number of iterative refinement steps.
        dropout: Dropout probability.
        init_damping: Initial value for the EMA damping factor lambda.
        deep_supervision: Whether to enable deep supervision loss.
        deep_sup_weight: Weight for intermediate-step losses.
        stability_weight: Weight for stability penalty.
    """

    def __init__(
        self,
        n_embd: int = 128,
        n_head: int = 4,
        n_neurons_mult: int = 8,
        thinking_steps: int = 8,
        dropout: float = 0.1,
        init_damping: float = 0.9,
        deep_supervision: bool = True,
        deep_sup_weight: float = 0.1,
        stability_weight: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        d = n_embd
        nh = n_head
        N = n_neurons_mult * d // nh   # per-head neuron count
        n = nh * N                      # total sparse neurons
        d_head = d // nh

        self.d = d
        self.nh = nh
        self.N = N
        self.n = n
        self.d_head = d_head
        self.K = thinking_steps
        self._deep_supervision = deep_supervision
        self._deep_sup_weight = deep_sup_weight
        self._stability_weight = stability_weight

        # --- Board encoder: (B, 12, 8, 8) -> (B, d) ---
        self.board_encoder = nn.Sequential(
            nn.Linear(768, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Linear(d, d),
        )

        # --- Neuron initialization from board ---
        self.x_init = nn.Linear(d, n)

        # === Shared BDH block parameters (used K times) ===

        # D_y: project from d -> N per head (gating the read result)
        self.D_y = nn.Parameter(torch.zeros(nh, d, N).normal_(std=0.02))

        # E: decode from sparse n -> embedding d
        self.E = nn.Parameter(torch.zeros(n, d).normal_(std=0.02))

        # D_x: project from d -> N per head (neuron update)
        self.D_x = nn.Parameter(torch.zeros(nh, d, N).normal_(std=0.02))

        # Board injection at every step (two separate projections)
        self.B_y = nn.Linear(d, d, bias=False)   # for attention-read path
        self.B_x = nn.Linear(d, d, bias=False)   # for neuron-update path

        # Shared layer norms
        self.ln_a = nn.LayerNorm(d)   # after attention read
        self.ln_z = nn.LayerNorm(d)   # after decode

        # Learnable damping factor lambda in (0, 1)
        self.log_damping = nn.Parameter(
            torch.tensor(math.log(init_damping / (1.0 - init_damping)))
        )

        # Dropout
        self.drop = nn.Dropout(dropout)

        # --- Scalar value head ---
        self.value_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d // 2, 1),
        )

        # For deep supervision: stores per-step predictions during forward
        self._step_values: list[torch.Tensor] = []

        self.apply(self._init_weights)

    # ---- Initialization ----

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ---- Properties ----

    @property
    def damping(self) -> torch.Tensor:
        """EMA damping factor lambda in (0, 1)."""
        return torch.sigmoid(self.log_damping)

    # ---- Loss factory ----

    def create_criterion(self) -> nn.Module:
        """Return the appropriate loss function for this model."""
        if self._deep_supervision:
            return DeepSupervisionLoss(
                self,
                deep_sup_weight=self._deep_sup_weight,
                stability_weight=self._stability_weight,
            )
        return nn.HuberLoss(delta=0.5)

    # ---- Forward ----

    def forward(self, boards: torch.Tensor) -> torch.Tensor:
        """Run iterative value refinement on a batch of board positions.

        Args:
            boards: (B, 12, 8, 8) board tensor.

        Returns:
            (B, 1) evaluation score in (-1, 1).
        """
        B = boards.size(0)

        # ---- Encode board once ----
        flat = boards.reshape(B, -1)                 # (B, 768)
        b = self.board_encoder(flat)                  # (B, d)

        # ---- Initialize sparse neurons ----
        x = F.relu(self.x_init(b))                    # (B, n), sparse positive

        # ---- Initialize synaptic state rho ----
        # Per head: rho_h in R^{N x d_head} -- neuron-to-embedding correlations
        rho = boards.new_zeros(B, self.nh, self.N, self.d_head)

        lam = self.damping                             # scalar in (0, 1)

        # ---- Pre-compute board injections (constant across steps) ----
        b_y = self.B_y(b)                              # (B, d)
        b_x = self.B_x(b)                              # (B, d)

        step_values: list[torch.Tensor] = []

        # ---- K thinking steps with shared weights ----
        for _ in range(self.K):
            x_h = x.view(B, self.nh, self.N)           # (B, nh, N)

            # 1. Linear attention read from rho
            #    a_h = rho_h^T @ x_h -> (B, nh, d_head)
            a_h = torch.einsum("bhnd,bhn->bhd", rho, x_h)
            a = a_h.reshape(B, self.d)                 # (B, d)

            # 2. Board-conditioned Hebbian gating
            a_cond = self.ln_a(a + b_y)                # (B, d)
            g = torch.einsum("bd,hdn->bhn", a_cond, self.D_y)
            g = F.relu(g)                              # sparse positive gate
            y_h = g * x_h                              # Hebbian: gate * activation

            # 3. Decode back to embedding space
            y_flat = y_h.reshape(B, self.n)            # (B, n)
            z = self.ln_z(y_flat @ self.E)             # (B, d)
            z = self.drop(z)

            # 4. Update neurons (residual, with board injection)
            z_cond = z + b_x                           # (B, d)
            dx = torch.einsum("bd,hdn->bhn", z_cond, self.D_x)
            dx = F.relu(dx)                            # sparse positive
            x = x + dx.reshape(B, self.n)              # residual update

            # 5. Write to synaptic state (EMA update)
            x_h_new = x.view(B, self.nh, self.N)
            # Normalize keys before write to prevent blowup
            x_write = x_h_new / (x_h_new.norm(dim=-1, keepdim=True) + 1e-8)
            z_h = z.view(B, self.nh, self.d_head)
            rho = lam * rho + (1.0 - lam) * torch.einsum(
                "bhn,bhd->bhnd", x_write, z_h
            )

            # 6. Predict value at this step
            v_k = torch.tanh(self.value_head(z))       # (B, 1)
            step_values.append(v_k)

        # Store for deep supervision (criterion reads these)
        self._step_values = step_values

        return step_values[-1]                         # (B, 1)
