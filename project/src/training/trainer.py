from __future__ import annotations
"""Reusable, config-driven training loop with W&B logging and checkpointing."""

import math
import os
import signal
import time
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger

from src.models.registry import count_parameters


def _load_env():
    """Load .env file if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
        # Search upward from cwd for .env
        for p in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
            env_file = p / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                logger.info(f"Loaded environment from {env_file}")
                return
    except ImportError:
        pass


class Trainer:
    """Generic trainer for chess position evaluation models.

    Supports:
        - W&B experiment tracking
        - Periodic & best-model checkpointing
        - Early stopping
        - Learning rate scheduling (cosine with warmup)
        - Mixed precision (AMP)
        - Gradient clipping

    Args:
        model: PyTorch model to train.
        config: Full experiment config dict.
        output_dir: Directory for checkpoints and logs.
    """

    def __init__(self, model: nn.Module, config: dict, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Training config
        tc = config["training"]
        self.epochs = tc["epochs"]
        self.lr = tc["learning_rate"]
        self.weight_decay = tc.get("weight_decay", 1e-4)
        self.clipnorm = tc.get("clipnorm", 1.0)
        self.use_amp = tc.get("mixed_precision", False) and self.device.type == "cuda"

        # Determine AMP dtype: prefer bf16 on A100+ (same range as fp32, no overflow)
        amp_dtype_str = tc.get("amp_dtype", "auto")
        if amp_dtype_str == "bf16":
            self.amp_dtype = torch.bfloat16
        elif amp_dtype_str == "fp16":
            self.amp_dtype = torch.float16
        else:  # auto
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
                logger.info("AMP: using bf16 (auto-detected GPU support)")
            else:
                self.amp_dtype = torch.float16
        # bf16 does not need GradScaler
        self._use_grad_scaler = self.use_amp and (self.amp_dtype == torch.float16)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Loss — let model provide custom criterion if available
        if hasattr(model, "create_criterion"):
            self.criterion = model.create_criterion()
            logger.info(f"Using model-provided criterion: {type(self.criterion).__name__}")
        else:
            self.criterion = nn.MSELoss()

        # AMP scaler — only needed for fp16, not bf16
        try:
            self.scaler = torch.amp.GradScaler(enabled=self._use_grad_scaler)
        except AttributeError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._use_grad_scaler)

        # Early stopping
        es_cfg = tc.get("early_stopping", {})
        self.es_patience = es_cfg.get("patience", 5)
        self.es_min_delta = es_cfg.get("min_delta", 1e-5)
        self._es_counter = 0
        self._es_best = float("inf")

        # Scheduler (cosine with warmup)
        self.warmup_epochs = tc.get("warmup_epochs", 3)
        self.min_lr = tc.get("min_learning_rate", 1e-6)
        self.scheduler = None  # built after we know steps

        # W&B
        _load_env()  # Load .env for WANDB_API_KEY etc.
        self.wandb_run = None
        wandb_cfg = config.get("wandb", {})
        if wandb_cfg.get("enabled", False):
            try:
                import wandb
                # Support WEIGHTS_AND_BIASES_KEY as well as WANDB_API_KEY
                api_key = os.environ.get("WANDB_API_KEY") or os.environ.get("WEIGHTS_AND_BIASES_KEY")
                if api_key:
                    wandb.login(key=api_key, relogin=True)
                run_id = wandb_cfg.get("run_id", None)
                self.wandb_run = wandb.init(
                    project=wandb_cfg.get("project", "bdh-searchless-chess"),
                    name=config["experiment"]["name"],
                    config=config,
                    tags=wandb_cfg.get("tags", []),
                    id=run_id,
                    resume="must" if run_id else None,
                    reinit=True,
                )
                logger.info(f"W&B initialized: {self.wandb_run.url}")
            except Exception as e:
                logger.warning(f"W&B init failed: {e}. Continuing without tracking.")

        # Tracking
        self.best_val_loss = float("inf")
        self.history: list[dict] = []
        self._train_start_time: float | None = None   # wall-clock start
        self._total_train_samples = 0                  # throughput tracking

        n_params = count_parameters(self.model)
        logger.info(f"Model: {config['model']['architecture']} | {n_params:,} params | Device: {self.device}")
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            if self.wandb_run:
                import wandb
                wandb.config.update({"gpu": gpu_name, "gpu_mem_gb": gpu_mem})

    def resume_from_checkpoint(self, checkpoint_path: str) -> int:
        """Load training state from a checkpoint for resumption.

        Call this BEFORE fit(). It restores model, optimizer, early-stopping
        state, and history. The scheduler and scaler states are restored
        inside fit() after the scheduler is built.

        Args:
            checkpoint_path: Path to the checkpoint .pt file.

        Returns:
            The epoch number to resume from (next epoch to train).
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
        self._es_counter = ckpt.get("es_counter", 0)
        self._es_best = ckpt.get("es_best", self.best_val_loss)
        self.history = ckpt.get("history", [])
        self._resume_scheduler_state = ckpt.get("scheduler_state_dict", None)
        self._resume_scaler_state = ckpt.get("scaler_state_dict", None)
        resume_epoch = ckpt["epoch"] + 1
        logger.info(
            f"Resumed from checkpoint: {checkpoint_path} "
            f"(epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}, "
            f"best_val_loss={self.best_val_loss:.6f})"
        )
        return resume_epoch

    def fit(self, train_loader, val_loader, start_epoch: int = 1) -> dict:
        """Run the full training loop.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            start_epoch: Epoch to start from (>1 when resuming).

        Returns:
            Dict with training history and best metrics.
        """
        # Build cosine scheduler
        steps_per_epoch = len(train_loader)
        total_steps = self.epochs * steps_per_epoch
        warmup_steps = self.warmup_epochs * steps_per_epoch

        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.01, total_iters=warmup_steps
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=total_steps - warmup_steps, eta_min=self.min_lr
                ),
            ],
            milestones=[warmup_steps],
        )

        # Restore scheduler / scaler state if resuming
        #
        # NOTE: SequentialLR.load_state_dict() is broken in PyTorch <2.6 —
        # sub-scheduler step counts conflict with optimizer state. Instead,
        # we fast-forward the scheduler to the correct step position.
        if hasattr(self, "_resume_scheduler_state") and self._resume_scheduler_state is not None:
            target_step = (start_epoch - 1) * steps_per_epoch
            logger.info(f"Fast-forwarding scheduler to step {target_step:,}...")
            for _ in range(target_step):
                self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(f"Scheduler restored — LR={current_lr:.6e}")
            self._resume_scheduler_state = None
        if hasattr(self, "_resume_scaler_state") and self._resume_scaler_state is not None:
            self.scaler.load_state_dict(self._resume_scaler_state)
            logger.info("Restored scaler state from checkpoint")
            self._resume_scaler_state = None

        logger.info(
            f"Training for {self.epochs} epochs ({steps_per_epoch} steps/epoch)"
            + (f" — resuming from epoch {start_epoch}" if start_epoch > 1 else "")
        )

        # Initialize global step counter for W&B intra-epoch logging
        self._global_step = (start_epoch - 1) * steps_per_epoch

        # Graceful interrupt handling (Ctrl+C, SLURM SIGINT/SIGTERM)
        self._interrupted = False
        self._train_start_time = time.time()
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def _handle_interrupt(signum, frame):
            sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
            if self._interrupted:
                logger.error(f"Second {sig_name} received — aborting without saving!")
                raise KeyboardInterrupt
            self._interrupted = True
            logger.warning(
                f"{sig_name} received — will save checkpoint after current epoch finishes. "
                "Send again to abort immediately."
            )

        signal.signal(signal.SIGINT, _handle_interrupt)
        signal.signal(signal.SIGTERM, _handle_interrupt)

        try:
            for epoch in range(start_epoch, self.epochs + 1):
                t0 = time.time()

                train_loss, train_mae = self._train_one_epoch(train_loader)
                val_loss, val_mae = self._validate(val_loader)

                elapsed = time.time() - t0
                lr_now = self.optimizer.param_groups[0]["lr"]

                record = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_mae": train_mae,
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "lr": lr_now,
                    "time_s": elapsed,
                }
                self.history.append(record)

                logger.info(
                    f"Epoch {epoch:3d}/{self.epochs} | "
                    f"train_loss={train_loss:.6f} train_mae={train_mae:.4f} | "
                    f"val_loss={val_loss:.6f} val_mae={val_mae:.4f} | "
                    f"lr={lr_now:.2e} | {elapsed:.1f}s"
                )

                # W&B log (enhanced with compute metrics)
                if self.wandb_run:
                    import wandb
                    wb_record = dict(record)
                    # GPU memory
                    if self.device.type == "cuda":
                        wb_record["gpu_mem_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
                        wb_record["gpu_mem_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024**2
                        torch.cuda.reset_peak_memory_stats()
                    # Cumulative compute
                    wall_elapsed = time.time() - self._train_start_time
                    wb_record["wall_time_h"] = wall_elapsed / 3600
                    wb_record["gpu_hours"] = wall_elapsed / 3600  # ~1 GPU
                    # Throughput
                    if hasattr(train_loader, "dataset"):
                        samples_this_epoch = len(train_loader.dataset)
                        wb_record["throughput_samples_per_sec"] = samples_this_epoch / elapsed if elapsed > 0 else 0
                    wandb.log(wb_record, step=self._global_step)

                # NaN epoch detection: abort if loss went NaN
                if not math.isfinite(val_loss):
                    self._nan_epoch_count = getattr(self, "_nan_epoch_count", 0) + 1
                    logger.error(
                        f"Epoch {epoch}: val_loss is NaN/Inf "
                        f"({self._nan_epoch_count} consecutive NaN epochs)"
                    )
                    if self._nan_epoch_count >= 2:
                        logger.error(
                            "Two consecutive NaN epochs — aborting training. "
                            "Best checkpoint preserved."
                        )
                        break
                    continue  # skip checkpointing / early-stop for NaN epoch
                else:
                    self._nan_epoch_count = 0

                # Checkpoint: best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best_model.pt", epoch, val_loss)
                    logger.info(f"  ✓ New best model (val_loss={val_loss:.6f})")

                # Checkpoint: periodic
                save_every = self.config["training"].get("save_every_epochs", 10)
                if epoch % save_every == 0:
                    self._save_checkpoint(f"epoch_{epoch:03d}.pt", epoch, val_loss)

                # Checkpoint: latest (always, for resume support)
                self._save_checkpoint("latest_checkpoint.pt", epoch, val_loss)

                # Early stopping
                if self._check_early_stopping(val_loss):
                    logger.warning(f"Early stopping at epoch {epoch} (patience={self.es_patience})")
                    break

                # Check if interrupted (save was already done above)
                if self._interrupted:
                    logger.warning(f"Graceful shutdown after epoch {epoch} — checkpoint saved.")
                    break

        finally:
            # Restore original signal handlers
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

        # Log final compute summary
        total_wall = time.time() - self._train_start_time
        logger.info(
            f"Total wall time: {total_wall/3600:.2f}h | "
            f"Final epoch: {epoch}"
        )
        if self.device.type == "cuda":
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            logger.info(f"Peak GPU memory allocated: {peak_mb:.0f} MB")

        # Save final
        self._save_checkpoint("final_model.pt", epoch, val_loss)

        if self.wandb_run:
            import wandb
            total_wall = time.time() - self._train_start_time
            wandb.summary["total_gpu_hours"] = total_wall / 3600
            wandb.summary["best_val_loss"] = self.best_val_loss
            if self.device.type == "cuda":
                wandb.summary["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 1024**2
            wandb.finish()

        return {
            "best_val_loss": self.best_val_loss,
            "final_epoch": epoch,
            "history": self.history,
        }

    def _train_one_epoch(self, loader) -> tuple[float, float]:
        """Train for one epoch. Returns (loss, mae)."""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        n_batches = 0
        nan_count = 0
        n_total = len(loader)
        log_every = max(1, n_total // 10)  # log ~10 times per epoch

        # Track the global step for intra-epoch W&B logging
        if not hasattr(self, "_global_step"):
            self._global_step = 0

        for boards, scores in loader:
            boards = boards.to(self.device, non_blocking=True)
            scores = scores.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                preds = self.model(boards)
                loss = self.criterion(preds, scores)

            # NaN detection: skip batch if loss is NaN/Inf
            if not torch.isfinite(loss):
                nan_count += 1
                if nan_count <= 5:
                    logger.warning(
                        f"  [train] NaN/Inf loss at step {n_batches + 1}/{n_total} "
                        f"(nan_count={nan_count}). Skipping batch."
                    )
                if nan_count >= 50:
                    logger.error(
                        f"  [train] Too many NaN batches ({nan_count}). "
                        "Aborting epoch — model likely diverged."
                    )
                    break
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()
                continue

            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipnorm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            total_mae += (preds - scores).abs().mean().item()
            n_batches += 1
            self._global_step += 1

            if n_batches % log_every == 0:
                avg_loss = total_loss / n_batches
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"  [train] step {n_batches:>5d}/{n_total} "
                    f"loss={avg_loss:.6f} lr={current_lr:.2e}"
                    + (f" nan_skipped={nan_count}" if nan_count > 0 else "")
                )
                # Intra-epoch W&B logging for real-time visibility
                if self.wandb_run:
                    import wandb
                    wandb.log({
                        "train/loss_step": avg_loss,
                        "train/lr": current_lr,
                        "train/step": self._global_step,
                    }, step=self._global_step)

        if nan_count > 0:
            logger.warning(f"  [train] Epoch had {nan_count} NaN batches out of {n_total}")

        if n_batches == 0:
            return float("nan"), float("nan")
        return total_loss / n_batches, total_mae / n_batches

    @torch.no_grad()
    def _validate(self, loader) -> tuple[float, float]:
        """Validate. Returns (loss, mae)."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        n_batches = 0
        n_total = len(loader)

        for boards, scores in loader:
            boards = boards.to(self.device, non_blocking=True)
            scores = scores.to(self.device, non_blocking=True)

            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                preds = self.model(boards)
                loss = self.criterion(preds, scores)

            if torch.isfinite(loss):
                total_loss += loss.item()
                total_mae += (preds - scores).abs().mean().item()
                n_batches += 1

        logger.info(f"  [val] {n_batches} batches done")
        if n_batches == 0:
            return float("nan"), float("nan")
        return total_loss / n_batches, total_mae / n_batches

    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        """Save model checkpoint with full training state for resumption."""
        path = self.checkpoint_dir / filename
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "es_counter": self._es_counter,
            "es_best": self._es_best,
            "history": self.history,
            "config": self.config,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler is not None:
            state["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(state, path)
        logger.info(f"Checkpoint saved: {path}")

        # Upload best checkpoint to W&B as artifact for resilience
        if filename == "best_model.pt" and self.wandb_run:
            try:
                import wandb
                art = wandb.Artifact(
                    f"model-{self.wandb_run.id}",
                    type="model",
                    metadata={"epoch": epoch, "val_loss": val_loss},
                )
                art.add_file(str(path))
                self.wandb_run.log_artifact(art)
                logger.info(f"Uploaded best checkpoint to W&B artifact (epoch {epoch})")
            except Exception as e:
                logger.warning(f"Failed to upload checkpoint to W&B: {e}")

    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if training should stop."""
        if val_loss < self._es_best - self.es_min_delta:
            self._es_best = val_loss
            self._es_counter = 0
            return False
        else:
            self._es_counter += 1
            return self._es_counter >= self.es_patience
