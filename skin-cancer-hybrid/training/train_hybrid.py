"""Train the complete hybrid classifier.

Usage
-----
::

    python training/train_hybrid.py --data-dir data/raw
    python training/train_hybrid.py --data-dir data/raw --include-scc
    python training/train_hybrid.py --set training.epochs=50 training.batch_size=16
    python training/train_hybrid.py --resume runs/hybrid/checkpoints/last.pth

What this script does
---------------------
1. Builds the leakage-safe data pipeline (see :mod:`data.pipeline`).
2. Fits the metadata encoder **and** the class weights on the training split
   only.
3. Instantiates the hybrid model - optionally with U-Net segmentation and
   attention, depending on the configuration.
4. Trains with AdamW, separate backbone/head learning rates, AMP, gradient
   clipping, early stopping and checkpointing.
5. Writes TensorBoard logs, a CSV history and a JSON summary.

The test split is **not** touched here.  It is evaluated once, by
``evaluation/evaluate.py``, after the model configuration is final.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.pipeline import build_dataloaders, prepare_data
from models.hybrid_model import (
    HybridModelConfig,
    HybridSkinCancerModel,
    model_config_to_dict,
)
from models.losses import build_loss
from training.cli_utils import build_config_from_args, make_run_directory, parse_args, print_banner
from training.trainer import Trainer, TrainerConfig
from utils.logging_utils import setup_logging
from utils.seeding import seed_everything


def build_trainer_config(config, run_dir: Path, args) -> TrainerConfig:
    """Assemble the trainer configuration from the project configuration."""
    training = config.get("training", {}) or {}
    early = training.get("early_stopping", {}) or {}
    scheduler_config = training.get("scheduler_config", {}) or {}

    return TrainerConfig(
        epochs=int(args.epochs or training.get("epochs", 50)),
        learning_rate=float(training.get("lr_head", 1e-4)),
        backbone_lr=float(training.get("lr_backbone", 1e-5)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        betas=tuple(training.get("betas", (0.9, 0.999))),
        eps=float(training.get("eps", 1e-8)),
        optimizer=str(training.get("optimizer", "adamw")),
        scheduler=str(training.get("scheduler", "cosine")),
        scheduler_config=dict(scheduler_config),
        warmup_epochs=int(training.get("warmup_epochs", 0)),
        warmup_start_lr=float(training.get("warmup_start_lr", 1e-7)),
        grad_clip_norm=float(training.get("grad_clip_norm", 5.0)),
        amp=bool(training.get("amp", True)),
        channels_last=bool(training.get("channels_last", False)),
        early_stopping=dict(early),
        monitor=str(early.get("monitor", "val_macro_f1")),
        monitor_mode=str(early.get("mode", "max")),
        output_dir=str(run_dir),
        checkpoint_dir=str(run_dir / "checkpoints"),
        save_best=bool(training.get("save_best", True)),
        save_last=bool(training.get("save_last", True)),
        log_every=int(config.get("logging.log_every", 20)),
        tensorboard=bool(config.get("logging.tensorboard", True)),
        csv_history=bool(config.get("logging.csv_history", True)),
        device=str(config.get("hardware.device", "auto")),
        seed=int(config.get("project.seed", 42)),
        freeze_backbone_epochs=int(training.get("freeze_backbone_epochs", 0)),
        unfreeze_at_epoch=training.get("unfreeze_at_epoch", None),
    )


def main() -> int:
    print_banner("Hybrid skin-lesion classification training")
    args = parse_args("Train the complete hybrid model.")
    config = build_config_from_args(args)

    run_dir = make_run_directory(config, "hybrid")
    logger = setup_logging("skin-cancer-hybrid", run_dir, filename="train_hybrid.log")

    seed = int(config.get("project.seed", 42))
    seed_everything(seed, deterministic=bool(config.get("project.deterministic", True)))
    logger.info("seed=%d output=%s", seed, run_dir)

    # -- data ---------------------------------------------------------------
    prepared = prepare_data(
        config,
        datasets=args.datasets,
        raw_dir=args.data_dir,
        include_scc=args.include_scc,
    )
    for message in prepared.warnings:
        logger.warning("%s", message)
    logger.info(
        "data ready | train=%d val=%d test=%d | classes=%s",
        len(prepared.train_frame),
        len(prepared.val_frame),
        len(prepared.test_frame),
        prepared.class_names,
    )

    if len(prepared.train_frame) == 0:
        logger.error("The training split is empty - check the dataset directory.")
        return 1

    train_loader, val_loader, _ = build_dataloaders(prepared, config)

    # -- model --------------------------------------------------------------
    metadata_dim = prepared.metadata_encoder.output_dim if prepared.metadata_encoder else 0
    model_config = HybridModelConfig.from_config(config, num_classes=prepared.num_classes)
    model_config.metadata_dim = metadata_dim
    model_config.use_metadata = bool(metadata_dim > 0)

    if model_config.use_segmentation:
        logger.info("segmentation stage ENABLED (checkpoint: %s)", model_config.unet_checkpoint)

    model = HybridSkinCancerModel(model_config)
    logger.info(
        "model ready | params=%s | classes=%d | metadata_dim=%d",
        f"{sum(p.numel() for p in model.parameters()):,}",
        prepared.num_classes,
        metadata_dim,
    )

    # -- loss ---------------------------------------------------------------
    training = config.get("training", {}) or {}
    focal = training.get("focal", {}) or {}
    criterion = build_loss(
        name=str(training.get("loss", "weighted_focal")),
        num_classes=prepared.num_classes,
        class_weights=prepared.class_weights,
        gamma=float(focal.get("gamma", 2.0)),
        alpha=focal.get("alpha", None),
        label_smoothing=float(focal.get("label_smoothing", 0.0)),
    )
    logger.info(
        "loss=%s gamma=%s | class weights (from train split): %s",
        training.get("loss", "weighted_focal"),
        focal.get("gamma", 2.0),
        [round(float(w), 3) for w in prepared.class_weights]
        if prepared.class_weights is not None
        else "none",
    )

    # -- train --------------------------------------------------------------
    trainer_config = build_trainer_config(config, run_dir, args)
    trainer_config.num_classes = prepared.num_classes
    trainer_config.class_names = prepared.class_names

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=trainer_config,
        metadata_dim=metadata_dim,
        class_names=prepared.class_names,
        project_config=config,
        model_config=model_config_to_dict(model_config),
    )

    summary = trainer.fit(resume_from=args.resume)
    logger.info("training finished | best metric %.4f", summary["best_metric"])

    # Persist the metadata encoder next to the checkpoint so that inference
    # applies exactly the same transformation.
    if prepared.metadata_encoder is not None:
        encoder_path = run_dir / "checkpoints" / "metadata_encoder.pkl"
        prepared.metadata_encoder.save(encoder_path)
        logger.info("metadata encoder saved -> %s", encoder_path)

    with (run_dir / "data_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(prepared.summary(), handle, indent=2, default=str)

    logger.info("now evaluate the test split with: python evaluation/evaluate.py --checkpoint %s", summary["best_checkpoint"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
