"""Train the EfficientNetV2 baseline (ablation experiment A).

Usage
-----
::

    python training/train_baseline.py --data-dir data/raw
    python training/train_baseline.py --set training.loss=weighted_ce
    python training/train_baseline.py --set training.loss=ce

This script trains **only the backbone + classification head**:
no attention, no metadata fusion, no segmentation.  It is the reference point
against which every other component is measured in the ablation study.

Because the baseline is the same architecture without the extra blocks, the
comparison is controlled: the same split, the same preprocessing, the same
optimizer and the same schedule are used.

Loss comparison
---------------
``--set training.loss=ce`` / ``weighted_ce`` / ``weighted_focal`` allows the
three loss functions to be compared on an identical split.
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
from models.classifier import build_classifier_head
from models.efficientnetv2 import BackboneConfig, EfficientNetV2Backbone
from models.losses import build_loss
from training.cli_utils import build_config_from_args, make_run_directory, parse_args, print_banner
from training.trainer import Trainer, TrainerConfig
from training.train_hybrid import build_trainer_config
from utils.logging_utils import setup_logging
from utils.seeding import seed_everything


class BaselineModel(torch.nn.Module):
    """EfficientNetV2 feature extractor + linear/MLP classification head."""

    def __init__(self, config, num_classes: int) -> None:
        super().__init__()
        backbone_config = BackboneConfig.from_config(config)
        self.backbone = EfficientNetV2Backbone(backbone_config)
        self.head = build_classifier_head(
            in_dim=self.backbone.feature_dim,
            num_classes=num_classes,
            config=config,
        )
        self.metadata_dim = 0

    def forward(self, images, metadata=None, has_metadata=None):
        features = self.backbone.forward_features(images)
        return self.head(features)

    def parameter_groups(self, lr_backbone: float, lr_head: float):
        groups = self.backbone.parameter_groups(lr_backbone, lr_head)
        head_parameters = [p for p in self.head.parameters() if p.requires_grad]
        if head_parameters:
            groups.append({"params": head_parameters, "lr": lr_head, "name": "head"})
        return groups

    def to_config_dict(self) -> dict:
        """Serialise the architecture so a checkpoint can rebuild it exactly.

        The baseline model is not a :class:`HybridSkinCancerModel`, so its
        architecture is described explicitly.
        """
        return {
            "kind": "baseline",
            "backbone_name": self.backbone.config.name,
            "feature_dim": self.backbone.feature_dim,
            "num_classes": self.head.num_classes,
            "head_dims": [layer.out_features for layer in self.head.hidden
                          if isinstance(layer, torch.nn.Linear)],
            "head_dropout": self.head.hidden[-1].p
            if isinstance(self.head.hidden[-1], torch.nn.Dropout) else 0.0,
            "metadata_dim": 0,
        }


def main() -> int:
    print_banner("EfficientNetV2 baseline training (ablation A)")
    args = parse_args("Train the EfficientNetV2 baseline classifier.")
    config = build_config_from_args(args)

    run_dir = make_run_directory(config, "baseline")
    logger = setup_logging("skin-cancer-hybrid", run_dir, filename="train_baseline.log")

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

    model = BaselineModel(config, prepared.num_classes)
    logger.info(
        "baseline model ready | params=%s",
        f"{sum(p.numel() for p in model.parameters()):,}",
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
        metadata_dim=0,
        class_names=prepared.class_names,
        project_config=config,
        model_config=model.to_config_dict(),
    )

    summary = trainer.fit(resume_from=args.resume)
    logger.info("baseline training finished | best metric %.4f", summary["best_metric"])

    with (run_dir / "data_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(prepared.summary(), handle, indent=2, default=str)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
