"""Segmentation dataset helpers (image/mask pairing and validation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from data.datasets import SegmentationDataset

__all__ = [
    "SegmentationDataset",
    "build_segmentation_pairs",
    "validate_mask_pair",
    "split_segmentation_pairs",
]


def build_segmentation_pairs(
    loader: Any,
    manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return validated image/mask pairs for a segmentation dataset.

    Parameters
    ----------
    loader:
        An :class:`~data.loaders.isic2018.ISIC2018Loader` instance.
    manifest:
        Optional manifest used to restrict the pairs to the samples that
        belong to the requested split.

    Returns
    -------
    pandas.DataFrame
        Columns ``image_path``, ``mask_path``, ``image_id``.
    """
    pairs = loader.build_segmentation_pairs()
    if manifest is not None and "split" in manifest.columns:
        allowed = set(
            manifest.loc[manifest["split"].isin(["train", "val"]), "sample_id"].astype(str)
        )
        # Pair rows carry the image id; match it against the manifest ids.
        pairs = pairs[pairs["image_id"].astype(str).isin(
            {sid.split(":", 1)[-1] for sid in allowed}
        )]
    return pairs.reset_index(drop=True)


def validate_mask_pair(image_path: str, mask_path: str, threshold: float = 0.5) -> dict[str, Any]:
    """Validate a single image/mask pair.

    Returns a report describing whether the mask is usable, its positive
    pixel ratio and whether its shape matches the image.
    """
    import cv2
    import numpy as np

    report: dict[str, Any] = {
        "image_path": image_path,
        "mask_path": mask_path,
        "image_exists": Path(image_path).exists(),
        "mask_exists": Path(mask_path).exists(),
        "shape_match": False,
        "positive_ratio": 0.0,
        "valid": False,
        "reason": "",
    }

    if not report["image_exists"]:
        report["reason"] = "image missing"
        return report
    if not report["mask_exists"]:
        report["reason"] = "mask missing"
        return report

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if image is None or mask is None:
        report["reason"] = "unreadable file"
        return report

    if mask.ndim == 3:
        mask = mask[..., 0]

    report["shape_match"] = image.shape[:2] == mask.shape[:2]
    positive = float(np.mean((mask > 127) if mask.dtype == np.uint8 else (mask > threshold)))
    report["positive_ratio"] = positive

    if not report["shape_match"]:
        report["reason"] = "shape mismatch"
    elif positive <= 0.0:
        report["reason"] = "empty mask"
    elif positive >= 0.98:
        report["reason"] = "mask covers the whole image"
    else:
        report["valid"] = True
        report["reason"] = "ok"

    return report


def split_segmentation_pairs(
    pairs: pd.DataFrame,
    split_manifest: pd.DataFrame,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split image/mask pairs into a training and a validation set.

    The split is derived from the *existing* classification split manifest
    wherever possible, so the segmentation model is trained on exactly the
    same training images as the classification model and the comparison
    between the two stages is fair.
    """
    import numpy as np

    if split_manifest is not None and len(split_manifest):
        id_to_split = {
            str(sid).split(":", 1)[-1]: str(split)
            for sid, split in zip(
                split_manifest["sample_id"], split_manifest["split"]
            )
        }
        pairs = pairs.copy()
        pairs["split"] = pairs["image_id"].astype(str).map(id_to_split).fillna("train")
        train_pairs = pairs[pairs["split"] == "train"].drop(columns=["split"])
        val_pairs = pairs[pairs["split"].isin(["val", "test"])].drop(columns=["split"])
        if len(val_pairs) == 0:
            rng = np.random.default_rng(seed)
            indices = rng.permutation(len(pairs))
            cut = max(1, int(len(pairs) * (1 - val_ratio)))
            train_pairs = pairs.iloc[indices[:cut]].drop(columns=["split"])
            val_pairs = pairs.iloc[indices[cut:]].drop(columns=["split"])
        return train_pairs.reset_index(drop=True), val_pairs.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(pairs))
    cut = max(1, int(len(pairs) * (1 - val_ratio)))
    return (
        pairs.iloc[indices[:cut]].reset_index(drop=True),
        pairs.iloc[indices[cut:]].reset_index(drop=True),
    )
