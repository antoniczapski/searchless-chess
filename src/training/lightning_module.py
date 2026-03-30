"""PyTorch Lightning module wrapping chess evaluation models.

Provides a standard LightningModule interface over the model registry,
handling training/validation/test steps, loss computation, metrics,
and optimizer/scheduler configuration — all driven by the YAML config.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanAbsoluteError, MeanSquaredError

from src.models.registry import create_model, count_parameters


class ChessLightningModule(pl.LightningModule):
    """Lightning wrapper for chess evaluation models.

    Reads model config to instantiate the appropriate architecture and
    training config for optimizer/scheduler setup.

    Args:
        config: Full experiment config dict (same YAML format as before).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # Build model from registry
        self.model = create_model(config["model"])

        # Loss — let model provide custom criterion if available (e.g. BDH deep supervision)
        if hasattr(self.model, "create_criterion"):
            self.criterion = self.model.create_criterion()
        else:
            self.criterion = nn.MSELoss()

        # Metrics
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.val_mse = MeanSquaredError()
        self.test_mae = MeanAbsoluteError()

        n_params = count_parameters(self.model)
        self.log_dict_on_init = {"n_params": float(n_params)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ─── Training ───────────────────────────────────────────────

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        boards, scores = batch
        preds = self.model(boards)
        loss = self.criterion(preds, scores)

        self.train_mae(preds, scores)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=False)

        return loss

    # ─── Validation ─────────────────────────────────────────────

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        boards, scores = batch
        preds = self.model(boards)
        loss = self.criterion(preds, scores)

        self.val_mae(preds, scores)
        self.val_mse(preds, scores)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True)
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True)

    # ─── Test ───────────────────────────────────────────────────

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        boards, scores = batch
        preds = self.model(boards)
        loss = self.criterion(preds, scores)

        self.test_mae(preds, scores)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True)

    # ─── Optimizer / Scheduler ──────────────────────────────────

    def configure_optimizers(self) -> dict[str, Any]:
        tc = self.config["training"]

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=tc["learning_rate"],
            weight_decay=tc.get("weight_decay", 1e-4),
        )

        # Cosine schedule with linear warmup (matches original Trainer)
        warmup_epochs = tc.get("warmup_epochs", 3)
        total_epochs = tc["epochs"]
        min_lr = tc.get("min_learning_rate", 1e-6)

        # Estimate steps per epoch from trainer
        if self.trainer and self.trainer.estimated_stepping_batches:
            total_steps = self.trainer.estimated_stepping_batches
            steps_per_epoch = total_steps // total_epochs
        else:
            steps_per_epoch = 1000  # fallback
            total_steps = steps_per_epoch * total_epochs

        warmup_steps = warmup_epochs * steps_per_epoch

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_steps
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
