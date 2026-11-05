"""Run the full ablation study.

Usage
-----
::

    python training/ablation_study.py --data-dir data/raw
    python training/ablation_study.py --only A_baseline B_attention
    python training/ablation_study.py --data-dir data/raw --epochs 10

The six experiments defined in ``configs/evaluation.yaml`` are:

======  ==========================================================
Code    Configuration
======  ==========================================================
A       EfficientNetV2-S baseline (frozen backbone linear probe)
B       + CBAM attention
C       + clinical metadata fusion
D       + U-Net lesion segmentation
E       U-Net + attention
F       Full hybrid: U-Net + attention + metadata + focal loss
======  ==========================================================

Fairness guarantees
===================
* All experiments use the **same split file**, so the comparison is paired.
* All experiments use the same preprocessing, optimizer, schedule and seed.
* The only thing that changes between experiments is the component under test.

Results are written to ``experiments/ablation/`` as JSON and CSV.  No result
is ever fabricated: if an experiment cannot run (for example because no U-Net
checkpoint exists yet), it is recorded as ``skipped`` with the reason.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

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
from training.trainer import Trainer
from training.train_hybrid import build_trainer_config
from utils.logging_utils import setup_logging
from utils.seeding import seed_everything


def _run_experiment(
    name: str,
    spec: dict[str, Any],
    prepared,
    config,
    run_root: Path,
    logger,
    epochs_override: int | None = None,
) -> dict[str, Any]:
    """Run a single ablation experiment and return its summary."""
    run_dir = make_run_directory(config, f"ablation/{name}")
    logger.info("=== experiment %s: %s ===", name, spec.get("description", ""))

    # -- model overrides ---------------------------------------------------
    metadata_dim = prepared.metadata_encoder.output_dim if prepared.metadata_encoder else 0
    model_config = HybridModelConfig.from_config(config, num_classes=prepared.num_classes)
    model_config.use_attention = bool(spec.get("use_attention", False))
    model_config.use_metadata = bool(spec.get("use_metadata", False)) and metadata_dim > 0
    model_config.metadata_dim = metadata_dim if model_config.use_metadata else 0
    model_config.use_segmentation = bool(spec.get("use_segmentation", False))
    model_config.freeze_backbone = bool(spec.get("freeze_backbone", False))

    if model_config.use_segmentation:
        checkpoint = model_config.unet_checkpoint
        if not checkpoint or not Path(checkpoint).exists():
            message = (
                "segmentation is enabled but no trained U-Net checkpoint was "
                "found. Train one first with: python training/train_unet.py"
            )
            logger.warning("skipping %s: %s", name, message)
            return {
                "name": name,
                "status": "skipped",
                "reason": message,
                "description": spec.get("description", ""),
            }

    model = HybridSkinCancerModel(model_config)

    # -- loss --------------------------------------------------------------
    training = config.get("training", {}) or {}
    focal = training.get("focal", {}) or {}
    loss_name = str(spec.get("loss", training.get("loss", "weighted_focal")))
    criterion = build_loss(
        name=loss_name,
        num_classes=prepared.num_classes,
        class_weights=prepared.class_weights,
        gamma=float(focal.get("gamma", 2.0)),
        alpha=focal.get("alpha", None),
        label_smoothing=float(focal.get("label_smoothing", 0.0)),
    )

    # -- data --------------------------------------------------------------
    train_loader, val_loader, _ = build_dataloaders(prepared, config)

    # -- trainer -----------------------------------------------------------
    trainer_config = build_trainer_config(config, run_dir, None)
    trainer_config.epochs = epochs_override or trainer_config.epochs
    trainer_config.num_classes = prepared.num_classes
    trainer_config.class_names = prepared.class_names
    trainer_config.monitor_mode = "max"

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=trainer_config,
        metadata_dim=model_config.metadata_dim,
        class_names=prepared.class_names,
        project_config=config,
        model_config=model_config_to_dict(model_config),
    )

    summary = trainer.fit()

    # -- collect the final validation metrics ------------------------------
    final = trainer.validate()
    result = {
        "name": name,
        "status": "completed",
        "description": spec.get("description", ""),
        "configuration": {
            "use_attention": model_config.use_attention,
            "use_metadata": model_config.use_metadata,
            "use_segmentation": model_config.use_segmentation,
            "freeze_backbone": model_config.freeze_backbone,
            "loss": loss_name,
        },
        "best_epoch": summary["best_epoch"],
        "best_metric": summary["best_metric"],
        "final_val_metrics": final,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "run_dir": str(run_dir),
    }
    logger.info(
        "experiment %s finished | best_macro_f1=%.4f | val_acc=%.4f",
        name,
        result["best_metric"],
        final.get("val_accuracy", 0.0),
    )
    return result


def main() -> int:
    print_banner("Ablation study")
    args = parse_args("Run the component ablation study.")
    config = build_config_from_args(args)

    experiments_dir = Path(config.get("paths.experiments_dir", "experiments")) / "ablation"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(
        "skin-cancer-hybrid", experiments_dir, filename="ablation.log"
    )

    seed = int(config.get("project.seed", 42))
    seed_everything(seed, deterministic=bool(config.get("project.deterministic", True)))

    # -- data (prepared ONCE so every experiment uses the same split) ------
    prepared = prepare_data(
        config,
        datasets=args.datasets,
        raw_dir=args.data_dir,
        include_scc=args.include_scc,
    )
    for message in prepared.warnings:
        logger.warning("%s", message)
    logger.info(
        "shared split | train=%d val=%d test=%d | classes=%s",
        len(prepared.train_frame),
        len(prepared.val_frame),
        len(prepared.test_frame),
        prepared.class_names,
    )

    if len(prepared.train_frame) == 0:
        logger.error("The training split is empty - nothing to ablate.")
        return 1

    # -- run every experiment ----------------------------------------------
    specs = config.get("ablation.experiments", []) or []
    if args.only:
        wanted = set(args.only)
        specs = [s for s in specs if str(s.get("name")) in wanted]
        if not specs:
            logger.error(
                "No experiment matched --only %s. Available: %s",
                ", ".join(args.only),
                ", ".join(
                    str(s.get("name")) for s in config.get("ablation.experiments", []) or []
                ),
            )
            return 1
    results: list[dict[str, Any]] = []

    for spec in specs:
        name = str(spec.get("name", "experiment"))
        try:
            result = _run_experiment(
                name, spec, prepared, config, experiments_dir, logger, args.epochs
            )
        except Exception as exc:  # pragma: no cover - depends on the data
            logger.error("experiment %s failed: %s", name, exc)
            logger.debug(traceback.format_exc())
            result = {
                "name": name,
                "status": "failed",
                "reason": str(exc),
                "description": spec.get("description", ""),
            }
        results.append(result)

    # -- write the comparison table ----------------------------------------
    import pandas as pd

    rows = []
    for result in results:
        metrics = result.get("final_val_metrics", {}) or {}
        rows.append(
            {
                "experiment": result["name"],
                "status": result["status"],
                "description": result.get("description", ""),
                "attention": result.get("configuration", {}).get("use_attention"),
                "metadata": result.get("configuration", {}).get("use_metadata"),
                "segmentation": result.get("configuration", {}).get("use_segmentation"),
                "loss": result.get("configuration", {}).get("loss"),
                "best_macro_f1": result.get("best_metric"),
                "val_macro_f1": metrics.get("val_macro_f1"),
                "val_accuracy": metrics.get("val_accuracy"),
                "val_loss": metrics.get("val_loss"),
                "parameters": result.get("num_parameters"),
                "reason": result.get("reason", ""),
            }
        )

    table = pd.DataFrame(rows)
    csv_path = experiments_dir / "ablation_results.csv"
    table.to_csv(csv_path, index=False)

    json_path = experiments_dir / "ablation_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "split": {
                    "train": len(prepared.train_frame),
                    "val": len(prepared.val_frame),
                    "test": len(prepared.test_frame),
                    "classes": prepared.class_names,
                },
                "results": results,
            },
            handle,
            indent=2,
            default=str,
        )

    logger.info("ablation results -> %s", csv_path)
    if not table.empty:
        print("\n" + table.to_string(index=False))
    logger.info(
        "NOTE: these numbers come from the validation split only. The test "
        "split is evaluated once with evaluation/evaluate.py."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
