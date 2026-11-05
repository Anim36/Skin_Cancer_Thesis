"""The training engine.

:class:`Trainer` implements everything the five training scripts share:

* CUDA auto-detection with a CPU fallback,
* automatic mixed precision (AMP) with gradient scaling,
* gradient clipping,
* AdamW with **separate learning rates** for the backbone and the head,
* cosine / step / plateau schedulers,
* early stopping,
* best and last checkpointing,
* resume-from-checkpoint,
* TensorBoard and CSV history logging,
* reproducible seeding.

Checkpoint contents
===================
A checkpoint stores everything needed to resume or to reproduce an experiment:
the model and optimiser state dicts, the scheduler state, the epoch, the best
metric, the RNG states, the class list and the full configuration.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.classification_metrics import compute_classification_metrics
from utils.hardware import GradScaler, autocast_context
from utils.logging_utils import get_logger

__all__ = ["Trainer", "TrainerConfig", "HistoryTracker"]


@dataclass
class TrainerConfig:
    """Runtime configuration of the trainer."""

    epochs: int = 50
    learning_rate: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    scheduler_config: dict[str, Any] = field(default_factory=dict)
    warmup_epochs: int = 0
    warmup_start_lr: float = 1e-7
    grad_clip_norm: float = 5.0
    amp: bool = True
    channels_last: bool = False
    early_stopping: dict[str, Any] = field(default_factory=dict)
    monitor: str = "val_macro_f1"
    monitor_mode: str = "max"
    output_dir: str = "runs/default"
    checkpoint_dir: str | None = None
    save_best: bool = True
    save_last: bool = True
    log_every: int = 20
    tensorboard: bool = True
    csv_history: bool = True
    device: str = "auto"
    num_classes: int = 7
    class_names: list[str] = field(default_factory=list)
    seed: int = 42
    freeze_backbone_epochs: int = 0
    unfreeze_at_epoch: int | None = None


class HistoryTracker:
    """Accumulates per-epoch metrics and writes them to a CSV."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.rows: list[dict[str, Any]] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: Mapping[str, Any]) -> None:
        self.rows.append(dict(row))

    def flush(self) -> None:
        """Write the accumulated rows to the CSV file."""
        if self.path is None or not self.rows:
            return
        import pandas as pd

        pd.DataFrame(self.rows).to_csv(self.path, index=False)

    def as_frame(self):
        import pandas as pd

        return pd.DataFrame(self.rows)


class Trainer:
    """Generic classification trainer.

    Parameters
    ----------
    model:
        The model to train.  It must accept ``(images, metadata, has_metadata)``
        and return class logits.
    train_loader / val_loader:
        The data loaders.
    criterion:
        The loss function.
    config:
        The trainer configuration.
    metadata_dim:
        Length of the metadata vector.  ``0`` disables the metadata branch.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        config: TrainerConfig,
        metadata_dim: int = 0,
        class_names: Sequence[str] | None = None,
        project_config: Any | None = None,
        model_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.metadata_dim = metadata_dim
        self.class_names = list(class_names or [])
        # The full project configuration is stored in the checkpoint so that
        # inference can rebuild the identical model and preprocessing.
        self.project_config = (
            project_config.to_dict() if hasattr(project_config, "to_dict") else project_config
        )
        # The model configuration is stored so that inference can rebuild the
        # exact architecture the checkpoint was trained with.
        self.model_config = dict(model_config) if model_config else None

        self.logger = get_logger()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(
            config.checkpoint_dir or (self.output_dir / "checkpoints")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # -- device ---------------------------------------------------------
        from utils.hardware import resolve_device

        self.device, self.use_amp, self.amp_dtype = resolve_device(
            preference=config.device,
            allow_amp=config.amp,
        )
        self.model.to(self.device)
        if config.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        self.scaler = GradScaler(self.use_amp)

        # -- optimiser ------------------------------------------------------
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # -- bookkeeping ----------------------------------------------------
        self.start_epoch = 0
        self.best_metric = -float("inf") if config.monitor_mode == "max" else float("inf")
        self.epochs_without_improvement = 0
        self.history = HistoryTracker(
            self.output_dir / "training_history.csv" if config.csv_history else None
        )
        self.writer = self._build_writer()
        self.global_step = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimiser with separate backbone / head learning rates."""
        groups = self._parameter_groups()
        name = self.config.optimizer.lower()
        if name == "adamw":
            return torch.optim.AdamW(
                groups,
                lr=self.config.learning_rate,
                betas=tuple(self.config.betas),
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
            )
        if name == "adam":
            return torch.optim.Adam(
                groups,
                lr=self.config.learning_rate,
                betas=tuple(self.config.betas),
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
            )
        if name == "sgd":
            return torch.optim.SGD(
                groups,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
                nesterov=True,
            )
        raise ValueError(f"Unknown optimizer {self.config.optimizer!r}")

    def _parameter_groups(self) -> list[dict[str, Any]]:
        """Split the parameters into groups with their own learning rates."""
        if hasattr(self.model, "parameter_groups"):
            groups = self.model.parameter_groups(
                self.config.backbone_lr, self.config.learning_rate
            )
            if groups:
                return groups

        # Fallback: everything is one group.
        return [
            {
                "params": [p for p in self.model.parameters() if p.requires_grad],
                "lr": self.config.learning_rate,
            }
        ]

    def _build_scheduler(self):
        """Create the learning-rate scheduler."""
        name = (self.config.scheduler or "none").lower()
        settings = dict(self.config.scheduler_config or {})

        if name in {"none", ""}:
            return None
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(settings.get("t_max", self.config.epochs)),
                eta_min=float(settings.get("t_min", 1e-6)),
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(settings.get("step_size", 20)),
                gamma=float(settings.get("gamma", 0.1)),
            )
        if name == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=str(settings.get("mode", "max")),
                factor=float(settings.get("factor", 0.5)),
                patience=int(settings.get("patience", 5)),
            )
        raise ValueError(f"Unknown scheduler {self.config.scheduler!r}")

    def _build_writer(self):
        if not self.config.tensorboard:
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter

            return SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        except Exception as exc:  # pragma: no cover
            self.logger.warning("TensorBoard unavailable (%s); continuing without it.", exc)
            return None

    # ------------------------------------------------------------------
    # Epochs
    # ------------------------------------------------------------------
    def _prepare_batch(self, batch: Mapping[str, Any]) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None
    ]:
        """Move a batch to the device and extract the model inputs."""
        images = batch["image"].to(self.device, non_blocking=True)
        labels = batch["label"].to(self.device, non_blocking=True)

        metadata = batch.get("metadata")
        has_metadata = None
        if metadata is not None and self.metadata_dim > 0:
            metadata = metadata.to(self.device, non_blocking=True)
            has_metadata = metadata.abs().sum(dim=1) > 0
            metadata = metadata.float()
        else:
            metadata = None

        if self.config.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        return images, labels, metadata, has_metadata

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch and return the summary metrics."""
        self.model.train()
        self._apply_freeze_schedule(epoch)

        running_loss = 0.0
        correct = 0
        total = 0
        all_labels: list[int] = []
        all_preds: list[int] = []
        start = time.time()

        for step, batch in enumerate(self.train_loader):
            images, labels, metadata, has_metadata = self._prepare_batch(batch)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast_context(self.device, self.use_amp, self.amp_dtype):
                logits = self.model(images, metadata, has_metadata)
                loss = self.criterion(logits, labels)

            if self.scaler is not None and self.use_amp:
                self.scaler.scale(loss).backward()
                if self.config.grad_clip_norm and self.config.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.grad_clip_norm and self.config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                self.optimizer.step()

            batch_size = labels.size(0)
            running_loss += float(loss.detach()) * batch_size
            preds = logits.detach().argmax(dim=1)
            correct += int((preds == labels).sum())
            total += batch_size
            all_labels.extend(labels.detach().cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

            self.global_step += 1
            if self.config.log_every and step % self.config.log_every == 0:
                self.logger.info(
                    "epoch %d | step %d/%d | loss %.4f | acc %.4f",
                    epoch,
                    step,
                    len(self.train_loader),
                    float(loss.detach()),
                    correct / max(1, total),
                )

        metrics = {
            "train_loss": running_loss / max(1, total),
            "train_accuracy": correct / max(1, total),
            "train_macro_f1": _macro_f1(all_labels, all_preds),
            "epoch_time": time.time() - start,
        }
        return metrics

    @torch.no_grad()
    def validate(self, epoch: int = 0) -> dict[str, float]:
        """Run the validation pass and return the summary metrics."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels: list[int] = []
        all_preds: list[int] = []

        for batch in self.val_loader:
            images, labels, metadata, has_metadata = self._prepare_batch(batch)

            with autocast_context(self.device, self.use_amp, self.amp_dtype):
                logits = self.model(images, metadata, has_metadata)
                loss = self.criterion(logits, labels)

            batch_size = labels.size(0)
            running_loss += float(loss.detach()) * batch_size
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum())
            total += batch_size
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

        metrics = {
            "val_loss": running_loss / max(1, total),
            "val_accuracy": correct / max(1, total),
            "val_macro_f1": _macro_f1(all_labels, all_preds),
        }
        return metrics

    def _apply_freeze_schedule(self, epoch: int) -> None:
        """Implement the freeze / unfreeze schedule."""
        backbone = getattr(self.model, "backbone", None)
        if backbone is None:
            return

        # The freeze schedule only applies to the project's EfficientNetV2
        # backbone, which exposes a ``config`` attribute and the
        # ``set_backbone_mode`` method.  A plain test model without them is
        # left untouched.
        backbone_config = getattr(backbone, "config", None)
        if backbone_config is None or not hasattr(backbone, "set_backbone_mode"):
            return

        freeze_epochs = self.config.freeze_backbone_epochs
        unfreeze_at = self.config.unfreeze_at_epoch

        should_freeze = False
        if freeze_epochs and epoch < freeze_epochs:
            should_freeze = True
        if unfreeze_at is not None and epoch >= unfreeze_at:
            should_freeze = False

        if should_freeze and not backbone_config.freeze_backbone:
            backbone_config.freeze_backbone = True
            backbone.set_backbone_mode("frozen")
            self.logger.info("epoch %d: backbone frozen", epoch)
            self._rebuild_optimizer()
        elif (
            not should_freeze
            and backbone_config.freeze_backbone
            and freeze_epochs
            and epoch >= freeze_epochs
        ):
            backbone_config.freeze_backbone = False
            backbone.set_backbone_mode("partial")
            self.logger.info("epoch %d: backbone unfrozen", epoch)
            self._rebuild_optimizer()

    def _rebuild_optimizer(self) -> None:
        """Rebuild the optimiser so the parameter groups match the new state."""
        self.optimizer = self._build_optimizer()
        if self.scheduler is not None:
            self.scheduler = self._build_scheduler()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(
        self,
        name: str,
        epoch: int,
        metrics: Mapping[str, Any],
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        """Write a checkpoint and return its path.

        The checkpoint stores the *project* configuration (not only the trainer
        settings) so that inference reproduces the exact architecture and the
        exact preprocessing that were used during training.
        """
        path = self.checkpoint_dir / f"{name}.pth"
        payload: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "best_metric": self.best_metric,
            "metrics": dict(metrics),
            "trainer_config": self.config.__dict__.copy(),
            "config": self.project_config,
            "model_config": self.model_config,
            "class_names": list(self.class_names),
            "num_classes": self.config.num_classes,
            "metadata_dim": self.metadata_dim,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "global_step": self.global_step,
        }
        if extra:
            payload.update(dict(extra))

        torch.save(payload, path)
        self.logger.info("saved checkpoint -> %s", path)
        return path

    def load_checkpoint(self, path: str | Path, resume: bool = True) -> dict[str, Any]:
        """Load a checkpoint.

        When ``resume`` is true the optimiser, scheduler and RNG states are
        restored as well, so training continues exactly where it stopped.
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)

        self.model.load_state_dict(payload["model_state_dict"], strict=True)

        if resume:
            if "optimizer_state_dict" in payload and payload["optimizer_state_dict"]:
                self.optimizer.load_state_dict(payload["optimizer_state_dict"])
            if (
                self.scheduler is not None
                and payload.get("scheduler_state_dict") is not None
            ):
                self.scheduler.load_state_dict(payload["scheduler_state_dict"])
            if self.scaler is not None and payload.get("scaler_state_dict") is not None:
                self.scaler.load_state_dict(payload["scaler_state_dict"])
            if payload.get("torch_rng_state") is not None:
                torch.set_rng_state(payload["torch_rng_state"])
            if payload.get("numpy_rng_state") is not None:
                try:
                    np.random.set_state(payload["numpy_rng_state"])
                except (ValueError, TypeError):
                    pass
            self.start_epoch = int(payload.get("epoch", -1)) + 1
            self.best_metric = float(payload.get("best_metric", self.best_metric))
            self.global_step = int(payload.get("global_step", 0))
            self.logger.info(
                "resumed from %s at epoch %d (best metric %.4f)",
                path,
                self.start_epoch,
                self.best_metric,
            )

        return payload

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def fit(self, resume_from: str | Path | None = None) -> dict[str, Any]:
        """Run the full training loop.

        Returns
        -------
        dict
            A summary containing the best epoch, the best metric and the path
            of the best checkpoint.
        """
        if resume_from:
            self.load_checkpoint(resume_from, resume=True)

        self.logger.info(
            "training on %s | device=%s | amp=%s | epochs=%d",
            self.device,
            self.device,
            self.use_amp,
            self.config.epochs,
        )

        best_epoch = self.start_epoch
        early = self.config.early_stopping or {}
        patience = int(early.get("patience", 10)) if early.get("enabled", True) else None
        min_delta = float(early.get("min_delta", 0.0))
        monitor = str(early.get("monitor", self.config.monitor))
        mode = str(early.get("mode", self.config.monitor_mode))

        for epoch in range(self.start_epoch, self.config.epochs):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate(epoch)

            row = {"epoch": epoch, **train_metrics, **val_metrics}
            self.history.append(row)
            self.history.flush()

            self._log_epoch(epoch, row)

            metric = row.get(monitor)
            if metric is None:
                metric = row.get(self.config.monitor, 0.0)

            improved = (
                metric > self.best_metric + min_delta
                if mode == "max"
                else metric < self.best_metric - min_delta
            )
            if improved:
                self.best_metric = float(metric)
                best_epoch = epoch
                self.epochs_without_improvement = 0
                if self.config.save_best:
                    self.save_checkpoint("best", epoch, row)
            else:
                self.epochs_without_improvement += 1

            if self.config.save_last:
                self.save_checkpoint("last", epoch, row)

            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()

            if (
                patience is not None
                and self.epochs_without_improvement >= patience
            ):
                self.logger.info(
                    "early stopping at epoch %d (no improvement for %d epochs)",
                    epoch,
                    self.epochs_without_improvement,
                )
                break

        summary = {
            "best_epoch": best_epoch,
            "best_metric": self.best_metric,
            "monitor": monitor,
            "epochs_run": len(self.history.rows),
            "best_checkpoint": str(self.checkpoint_dir / "best.pth"),
            "last_checkpoint": str(self.checkpoint_dir / "last.pth"),
        }
        self._write_summary(summary)
        return summary

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log_epoch(self, epoch: int, row: Mapping[str, Any]) -> None:
        self.logger.info(
            "epoch %3d/%d | train_loss %.4f | train_acc %.4f | train_f1 %.4f | "
            "val_loss %.4f | val_acc %.4f | val_f1 %.4f",
            epoch,
            self.config.epochs,
            row.get("train_loss", 0.0),
            row.get("train_accuracy", 0.0),
            row.get("train_macro_f1", 0.0),
            row.get("val_loss", 0.0),
            row.get("val_accuracy", 0.0),
            row.get("val_macro_f1", 0.0),
        )

        if self.writer is not None:
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"metrics/{key}", value, epoch)

    def _write_summary(self, summary: Mapping[str, Any]) -> None:
        path = self.output_dir / "training_summary.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(dict(summary), handle, indent=2, default=str)
        self.logger.info("training summary -> %s", path)


def _macro_f1(labels: Sequence[int], predictions: Sequence[int]) -> float:
    """Macro F1 that tolerates classes missing from the batch."""
    from sklearn.metrics import f1_score

    if not labels:
        return 0.0
    return float(
        f1_score(labels, predictions, average="macro", zero_division=0)
    )
