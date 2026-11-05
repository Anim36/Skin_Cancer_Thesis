"""Train the U-Net lesion segmenter on the ISIC 2018 segmentation masks.

Usage
-----
::

    python training/train_unet.py --data-dir data/raw
    python training/train_unet.py --data-dir data/raw --epochs 50 --batch-size 8
    python training/train_unet.py --resume checkpoints/unet/last.pth

The script trains the segmentation model **independently** of the classifier.
The resulting checkpoint is what enables the "U-Net segmentation" stage of the
hybrid pipeline; without it the hybrid model refuses to enable segmentation
rather than using a meaningless randomly initialised segmentor.

Loss
====
``dice_bce`` (the default) combines

* **Dice loss** ``1 - 2|P n G| / (|P| + |G|)``, which directly optimises the
  region overlap and is robust to the extreme foreground/background imbalance
  of lesion masks, and
* **Binary cross entropy** on the logits, which gives a per-pixel gradient
  signal even where the Dice term is nearly flat.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.augmentation import AugmentationConfig
from data.segmentation_dataset import (
    SegmentationDataset,
    split_segmentation_pairs,
    validate_mask_pair,
)
from models.unet import UNet
from training.cli_utils import build_config_from_args, make_run_directory, parse_args, print_banner
from utils.hardware import GradScaler, autocast_context, resolve_device
from utils.logging_utils import setup_logging
from utils.seeding import seed_everything


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
class DiceBCELoss(nn.Module):
    """Combination of the soft Dice loss and binary cross entropy.

    Parameters
    ----------
    dice_weight:
        Weight of the Dice term.
    bce_weight:
        Weight of the BCE term.
    smooth:
        Smoothing constant of the Dice term.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the combined loss."""
        probabilities = torch.sigmoid(logits)
        flat_probs = probabilities.reshape(-1)
        flat_target = target.reshape(-1)
        intersection = (flat_probs * flat_target).sum()
        dice = 1.0 - (
            (2.0 * intersection + self.smooth)
            / (flat_probs.sum() + flat_target.sum() + self.smooth)
        )
        bce = self.bce(logits, target)
        return self.dice_weight * dice + self.bce_weight * bce


class DiceLoss(nn.Module):
    """Pure soft Dice loss."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        flat_probs = probabilities.reshape(-1)
        flat_target = target.reshape(-1)
        intersection = (flat_probs * flat_target).sum()
        return 1.0 - (
            (2.0 * intersection + self.smooth)
            / (flat_probs.sum() + flat_target.sum() + self.smooth)
        )


def build_segmentation_loss(name: str, dice_weight: float, bce_weight: float) -> nn.Module:
    """Factory for the segmentation loss."""
    name = (name or "dice_bce").lower()
    if name == "dice_bce":
        return DiceBCELoss(dice_weight, bce_weight)
    if name == "dice":
        return DiceLoss()
    if name in {"bce", "bce_with_logits"}:
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unknown segmentation loss {name!r}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _load_original_rgb(path: str) -> np.ndarray:
    """Read an image from disk as an RGB uint8 array.

    Used only for the qualitative overlay figure.  The model input is
    normalised, so it cannot be reused for display.
    """
    import cv2

    array = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if array is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)


def main() -> int:
    print_banner("U-Net lesion segmentation training")
    args = parse_args("Train the U-Net segmenter on ISIC 2018 masks.")
    config = build_config_from_args(args)

    run_dir = make_run_directory(config, "unet")
    logger = setup_logging("skin-cancer-hybrid", run_dir, filename="train_unet.log")

    seed = int(config.get("project.seed", 42))
    seed_everything(seed, deterministic=bool(config.get("project.deterministic", True)))

    device, use_amp, amp_dtype = resolve_device(
        preference=str(config.get("hardware.device", "auto")),
        allow_amp=bool(config.get("hardware.amp", True)),
        amp_dtype=str(config.get("hardware.amp_dtype", "float16")),
    )
    logger.info("device=%s amp=%s dtype=%s", device, use_amp, amp_dtype)

    # -- data ---------------------------------------------------------------
    from data.loaders.isic2018 import ISIC2018Loader

    raw_dir = Path(config.get("paths.raw_dir", "data/raw"))
    isic_root = raw_dir / "isic2018"
    if not isic_root.exists():
        logger.error(
            "ISIC 2018 not found at %s. Download it as described in "
            "DATASET_GUIDE.md and re-run.",
            isic_root,
        )
        return 1

    loader = ISIC2018Loader(root=isic_root)
    if not loader.is_available():
        logger.error("ISIC 2018 directory %s is empty.", isic_root)
        return 1

    pairs = loader.build_segmentation_pairs()
    logger.info("found %d image/mask pairs", len(pairs))

    # Validate a sample of the pairs so that a broken pairing is caught early.
    sample = pairs.head(min(50, len(pairs)))
    invalid = [
        report
        for report in (
            validate_mask_pair(row["image_path"], row["mask_path"])
            for _, row in sample.iterrows()
        )
        if not report["valid"]
    ]
    if invalid:
        logger.warning(
            "%d of %d sampled image/mask pairs are invalid (first: %s)",
            len(invalid),
            len(sample),
            invalid[0]["reason"],
        )

    # The segmentation split is derived from the *classification* split so
    # that the segmentor and the classifier see the same training images.
    # Any split manifest is accepted; the newest one is used.
    splits_dir = Path(config.get("paths.splits_dir", "data/splits"))
    split_manifest = None
    candidates = sorted(splits_dir.glob("split_manifest*.csv"))
    if candidates:
        split_file = max(candidates, key=lambda p: p.stat().st_mtime)
        import pandas as pd

        split_manifest = pd.read_csv(split_file, keep_default_na=False, dtype=str)
        logger.info("using the classification split %s", split_file)

    train_pairs, val_pairs = split_segmentation_pairs(pairs, split_manifest)
    logger.info("segmentation split: %d train / %d val", len(train_pairs), len(val_pairs))

    if len(train_pairs) == 0:
        logger.error("No training pairs available for the segmentation stage.")
        return 1

    image_size = int(config.get("unet_training.image_size", 224))
    augmentation_config = AugmentationConfig.from_config(config)

    train_dataset = SegmentationDataset(
        train_pairs, image_size=image_size, train=True, augmentation_config=augmentation_config
    )
    val_dataset = SegmentationDataset(
        val_pairs, image_size=image_size, train=False, augmentation_config=augmentation_config
    )

    batch_size = int(config.get("unet_training.batch_size", 8))
    num_workers = int(config.get("hardware.num_workers", 4))
    pin_memory = bool(config.get("hardware.pin_memory", True))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=len(train_dataset) > batch_size,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # -- model --------------------------------------------------------------
    model = UNet(
        in_channels=int(config.get("segmentation.in_channels", 3)),
        out_channels=int(config.get("segmentation.out_channels", 1)),
        base_channels=int(config.get("segmentation.base_channels", 32)),
        depth=int(config.get("segmentation.depth", 4)),
        bilinear=bool(config.get("segmentation.bilinear", True)),
        attention=True,
    ).to(device)
    logger.info(
        "U-Net parameters: %s",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    criterion = build_segmentation_loss(
        str(config.get("unet_training.loss", "dice_bce")),
        float(config.get("unet_training.dice_weight", 1.0)),
        float(config.get("unet_training.bce_weight", 1.0)),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("unet_training.lr", 1e-4)),
        weight_decay=float(config.get("unet_training.weight_decay", 1e-4)),
    )
    epochs = int(args.epochs or config.get("unet_training.epochs", 30))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(use_amp)

    from evaluation.segmentation_metrics import (
        batch_segmentation_metrics,
    )

    checkpoint_dir = run_dir / "checkpoints"
    best_dice = -1.0
    start_epoch = 0
    history: list[dict[str, float]] = []

    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        if payload.get("optimizer_state_dict"):
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if payload.get("scheduler_state_dict"):
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        best_dice = float(payload.get("best_dice", -1.0))
        logger.info("resumed from %s at epoch %d", args.resume, start_epoch)

    early = config.get("unet_training.early_stopping", {}) or {}
    patience = int(early.get("patience", 8)) if early.get("enabled", True) else None
    epochs_without_improvement = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        total = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, use_amp, amp_dtype):
                logits = model(images)
                if isinstance(logits, list):
                    logits = logits[0]
                loss = criterion(logits, masks)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            running_loss += float(loss.detach()) * images.shape[0]
            total += images.shape[0]

        # -- validation ----------------------------------------------------
        model.eval()
        val_losses: list[float] = []
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                with autocast_context(device, use_amp, amp_dtype):
                    logits = model(images)
                    if isinstance(logits, list):
                        logits = logits[0]
                    val_losses.append(float(criterion(logits, masks)))
                predictions.append(torch.sigmoid(logits).cpu().numpy())
                targets.append(masks.cpu().numpy())

        seg_metrics = batch_segmentation_metrics(
            [p for p in predictions], [t for t in targets]
        )
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, total),
            "val_loss": val_loss,
            "val_dice": seg_metrics["dice"],
            "val_iou": seg_metrics["iou"],
            "val_pixel_accuracy": seg_metrics["pixel_accuracy"],
        }
        history.append(row)
        logger.info(
            "epoch %3d/%d | train_loss %.4f | val_loss %.4f | val_dice %.4f | val_iou %.4f",
            epoch,
            epochs,
            row["train_loss"],
            val_loss,
            row["val_dice"],
            row["val_iou"],
        )

        scheduler.step()

        # -- qualitative check ---------------------------------------------
        # A single overlay figure per epoch makes systematic failures
        # (leaking into the background, missing pale lesions, latching onto
        # the ruler) visible at a glance.  It is cheap and it is the fastest
        # way to tell a broken segmentor from an undertrained one.
        #
        # The *original* images are read from disk rather than reusing the
        # model inputs: the model input is normalised (mean 0, std 1), which
        # would render as an unrecognisable grey mess.
        if len(val_pairs):
            try:
                from explainability.segmentation_viz import save_segmentation_grid

                sample_predictions = np.concatenate(predictions, axis=0)[:6]
                sample_targets = np.concatenate(targets, axis=0)[:6]
                originals = []
                for path in val_pairs["image_path"].tolist()[: len(sample_predictions)]:
                    originals.append(_load_original_rgb(path))
                if originals:
                    save_segmentation_grid(
                        originals,
                        sample_predictions,
                        target_masks=sample_targets,
                        output_path=run_dir / "figures" / f"segmentation_epoch_{epoch:03d}.png",
                    )
            except Exception as exc:  # pragma: no cover - visualisation is optional
                logger.warning("could not write the segmentation figure: %s", exc)

        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_dice": max(best_dice, row["val_dice"]),
            "metrics": row,
            "image_size": image_size,
            "config": config.to_dict(),
        }
        torch.save(payload, checkpoint_dir / "last.pth")
        if row["val_dice"] > best_dice:
            best_dice = row["val_dice"]
            torch.save(payload, checkpoint_dir / "best.pth")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience is not None and epochs_without_improvement >= patience:
            logger.info("early stopping at epoch %d", epoch)
            break

    # -- persist history ----------------------------------------------------
    import pandas as pd

    pd.DataFrame(history).to_csv(run_dir / "unet_training_history.csv", index=False)
    summary = {
        "best_dice": best_dice,
        "epochs_run": len(history),
        "best_checkpoint": str(checkpoint_dir / "best.pth"),
        "last_checkpoint": str(checkpoint_dir / "last.pth"),
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "device": str(device),
    }
    with (run_dir / "unet_training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    logger.info("segmentation training finished | best dice %.4f", best_dice)
    logger.info(
        "Use this checkpoint in the hybrid model: "
        "--set hybrid.unet_checkpoint=%s hybrid.use_segmentation=True",
        checkpoint_dir / "best.pth",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
