"""Torch datasets that consume the harmonised manifest.

Two dataset classes are provided:

:class:`SkinCancerDataset`
    The classification dataset.  Reads rows from the harmonised manifest,
    applies the deterministic preprocessing pipeline (plus train-only
    augmentation) and optionally attaches the clinical metadata vector.

:class:`SegmentationDataset`
    The image/mask dataset used to train the U-Net.  Reads the image/mask
    pairs produced by the ISIC 2018 loader.

Both are thin wrappers around the manifest so that the split file - not the
filesystem layout - defines what is training data.  This makes the
train/validation/test partition auditable and reproducible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.augmentation import (
    AugmentationConfig,
    build_train_augmentations,
    build_val_augmentations,
)
from data.metadata import MetadataEncoder
from data.preprocessing import PreprocessConfig, PreprocessingPipeline

__all__ = ["SkinCancerDataset", "SegmentationDataset", "collate_batch"]


def _load_image(path: str) -> np.ndarray:
    """Read an image from disk as an RGB ``uint8`` array."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class SkinCancerDataset(Dataset):
    """Classification dataset over a harmonised manifest.

    Parameters
    ----------
    frame:
        Manifest rows restricted to one split (or to a subset of it).
    preprocess_config:
        Configuration of the deterministic preprocessing stage.
    augmentation_config:
        Configuration of the train-only augmentation stage.
    train:
        When ``True`` the augmentation pipeline is applied; when ``False`` the
        image only receives the deterministic preprocessing.
    metadata_encoder:
        Fitted encoder that converts the clinical metadata into a vector.
        When ``None`` the samples are treated as having no metadata.
    label_column:
        Manifest column holding the integer class index.
    return_metadata:
        Whether the metadata vector should be returned alongside the image.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        preprocess_config: PreprocessConfig | None = None,
        augmentation_config: AugmentationConfig | None = None,
        train: bool = False,
        metadata_encoder: MetadataEncoder | None = None,
        label_column: str = "class_index",
        return_metadata: bool = True,
        image_column: str = "image_path",
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.train = train
        self.metadata_encoder = metadata_encoder
        self.label_column = label_column
        self.return_metadata = return_metadata and metadata_encoder is not None
        self.image_column = image_column

        self.pipeline = PreprocessingPipeline(preprocess_config or PreprocessConfig())
        self.augment_config = augmentation_config or AugmentationConfig()
        size = self.pipeline.config.image_size
        self._augment = (
            build_train_augmentations(self.augment_config, size)
            if train and self.augment_config.enabled
            else build_val_augmentations(size)
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.frame)

    def _metadata_for(self, row: Mapping[str, Any]) -> dict[str, Any]:
        if self.metadata_encoder is None:
            return {}
        raw = row.get("metadata")
        if isinstance(raw, str):
            import json

            try:
                return json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                return {}
        if isinstance(raw, Mapping):
            return dict(raw)
        return {}

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        path = str(row[self.image_column])

        image = _load_image(path)

        if self.train and self.augment_config.enabled:
            # Augmentation operates on the raw uint8 RGB image, then the
            # deterministic pipeline (hair removal, CLAHE, normalisation) is
            # applied afterwards so that val and test see identical treatment.
            augmented = self._augment(image=image)
            image = augmented["image"]

        tensor_image = self.pipeline(image)

        label = row.get(self.label_column, -1)
        try:
            label = int(label)
        except (TypeError, ValueError):
            label = -1

        sample: dict[str, Any] = {
            # PyTorch expects the channel dimension first: (C, H, W).
            "image": torch.from_numpy(
                np.ascontiguousarray(tensor_image.transpose(2, 0, 1))
            ),
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": str(row.get("sample_id", index)),
            "dataset": str(row.get("dataset", "")),
        }

        if self.return_metadata and self.metadata_encoder is not None:
            metadata = self._metadata_for(row)
            vector = self.metadata_encoder.transform(metadata)
            sample["metadata"] = torch.from_numpy(np.ascontiguousarray(vector))
        else:
            sample["metadata"] = torch.zeros(
                self.metadata_encoder.output_dim if self.metadata_encoder else 0,
                dtype=torch.float32,
            )

        return sample

    # ------------------------------------------------------------------
    def class_counts(self) -> dict[int, int]:
        """Number of samples per integer class index (for class weighting)."""
        counts: dict[int, int] = {}
        for value in self.frame[self.label_column]:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if index >= 0:
                counts[index] = counts.get(index, 0) + 1
        return counts


class SegmentationDataset(Dataset):
    """Image / mask pairs for U-Net training.

    Parameters
    ----------
    frame:
        DataFrame with ``image_path`` and ``mask_path`` columns.
    image_size:
        Spatial size both the image and the mask are resized to.
    train:
        Whether to apply the train-only augmentation pipeline.
    augmentation_config:
        Augmentation configuration.
    mask_threshold:
        Binarisation threshold applied after resizing the mask.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        image_size: int = 224,
        train: bool = False,
        augmentation_config: AugmentationConfig | None = None,
        mask_threshold: float = 0.5,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_size = image_size
        self.train = train
        self.mask_threshold = mask_threshold
        self.augment_config = augmentation_config or AugmentationConfig()

        if train and self.augment_config.enabled:
            self._augment = build_train_augmentations(
                self.augment_config, image_size
            )
        else:
            self._augment = build_val_augmentations(image_size)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image = _load_image(str(row["image_path"]))

        mask = cv2.imread(str(row["mask_path"]), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {row['mask_path']}")
        if mask.ndim == 3:
            mask = mask[..., 0]

        if self.train and self.augment_config.enabled:
            # The same geometric transform must be applied to image and mask.
            # Albumentations handles this through a shared transform call.
            result = self._augment(image=image, mask=mask)
            image = result["image"]
            mask = result["mask"]

        # Resize the image (bilinear) and the mask (nearest - never bilinear).
        image = cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )
        mask = cv2.resize(
            mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST
        )

        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        mask = (mask > 127).astype(np.float32) if mask.dtype == np.uint8 else (
            mask > self.mask_threshold
        ).astype(np.float32)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "mask": torch.from_numpy(np.ascontiguousarray(mask))[None, ...],
            "sample_id": str(row.get("image_id", index)),
        }


def collate_batch(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Default collate that keeps the string fields as plain lists."""
    from torch.utils.data import default_collate

    images = default_collate([item["image"] for item in batch])
    labels = default_collate([item["label"] for item in batch])
    metadata = default_collate([item["metadata"] for item in batch])
    sample_ids = [item.get("sample_id", "") for item in batch]
    datasets = [item.get("dataset", "") for item in batch]
    return {
        "image": images,
        "label": labels,
        "metadata": metadata,
        "sample_id": sample_ids,
        "dataset": datasets,
    }