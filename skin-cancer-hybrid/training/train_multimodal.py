"""Train the multimodal (image + clinical metadata) classifier.

Usage
-----
::

    python training/train_multimodal.py --data-dir data/raw
    python training/train_multimodal.py --datasets pad_ufes20 --data-dir data/raw

This script is the dedicated entry point for the image + metadata fusion
experiments.  It behaves like ``train_hybrid.py`` but forces the metadata
branch on and the segmentation branch off, which isolates the contribution of
the clinical metadata.

Important behaviour
===================
* Only PAD-UFES-20 ships clinical metadata.  When it is not part of the run,
  the script warns clearly and falls back to image-only training rather than
  fabricating metadata.
* The metadata encoder is fitted on the training split only and is saved next
  to the checkpoint so inference uses the identical transformation.
* Samples without metadata (all datasets except PAD-UFES-20) receive a
  learned "no metadata" embedding instead of an imputed clinical record.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
from training.train_hybrid import build_trainer_config
from utils.logging_utils import setup_logging
from utils.seeding import seed_everything


def main() -> int:
    print_banner("Multimodal (image + clinical metadata) training")
    args = parse_args("Train the image + metadata fusion model.")
    config = build_config_from_args(args)

    run_dir = make_run_directory(config, "multimodal")
    logger = setup_logging("skin-cancer-hybrid", run_dir, filename="train_multimodal.log")

    seed = int(config.get("project.seed", 42))
    seed_everything(seed, deterministic=bool(config.get("project.deterministic", True)))

    prepared = prepare_data(
        config,
        datasets=args.datasets,
        raw_dir=args.data_dir,
        include_scc=args.include_scc,
    )
    for message in prepared.warnings:
        logger.warning("%s", message)

    metadata_dim = prepared.metadata_encoder.output_dim if prepared.metadata_encoder else 0
    if metadata_dim == 0:
        logger.warning(
            "No clinical metadata is available in this run. The metadata branch "
            "is disabled and the model trains on images only. Include "
            "PAD-UFES-20 to exercise the fusion path."
        )
    else:
        logger.info(
            "metadata encoder ready | output_dim=%d | fitted on the training split only",
            metadata_dim,
        )

    logger.info(
        "data ready | train=%d val=%d | classes=%s",
        len(prepared.train_frame),
        len(prepared.val_frame),
        prepared.class_names,
    )

    if len(prepared.train_frame) == 0:
        logger.error("The training split is empty.")
        return 1

    train_loader, val_loader, _ = build_dataloaders(prepared, config)

    # -- model --------------------------------------------------------------
    model_config = HybridModelConfig.from_config(config, num_classes=prepared.num_classes)
    model_config.use_metadata = bool(metadata_dim > 0)
    model_config.metadata_dim = metadata_dim
    model_config.use_segmentation = False   # isolated multimodal experiment
    model_config.use_attention = bool(config.get("attention.enabled", True))

    model = HybridSkinCancerModel(model_config)
    logger.info(
        "multimodal model ready | params=%s | metadata_dim=%d",
        f"{sum(p.numel() for p in model.parameters()):,}",
        metadata_dim,
    )

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
    logger.info("multimodal training finished | best metric %.4f", summary["best_metric"])

    if prepared.metadata_encoder is not None:
        encoder_path = run_dir / "checkpoints" / "metadata_encoder.pkl"
        prepared.metadata_encoder.save(encoder_path)
        logger.info("metadata encoder saved -> %s", encoder_path)

    with (run_dir / "data_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(prepared.summary(), handle, indent=2, default=str)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
