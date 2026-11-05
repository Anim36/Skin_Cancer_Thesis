"""Train-only image augmentation built on Albumentations.

Design constraints for dermoscopy
=================================
* **Colour carries diagnosis.**  Pigment, erythema and vascular patterns are
  diagnostic, so hue/saturation jitter is kept small and no channel shuffle is
  used.
* **Orientation is not diagnostic.**  Dermoscopic images have no canonical
  orientation, so flips and rotations are safe.
* **Texture is diagnostic.**  Elastic warping and heavy blur are disabled by
  default; they destroy the fine structures the model must learn.

Augmentations are only ever applied to the *training* split.  Validation and
test pipelines are built with ``augment=False`` and therefore use the exact
deterministic preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

__all__ = ["AugmentationConfig", "build_train_augmentations", "build_val_augmentations"]


@dataclass
class AugmentationConfig:
    """Configuration for the training-time augmentation pipeline."""

    enabled: bool = True
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.0
    rotation_limit: int = 30
    scale_limit: float = 0.15
    translate_limit: float = 0.05
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    hue_shift: float = 0.02
    saturation_shift: float = 0.15
    random_crop: bool = True
    crop_scale_limit: float = 0.9
    elastic_transform: bool = False
    gaussian_blur_prob: float = 0.0
    gaussian_noise_prob: float = 0.0
    coarse_dropout_prob: float = 0.0

    @classmethod
    def from_config(cls, config: Any) -> "AugmentationConfig":
        def _get(key: str, default: Any) -> Any:
            if hasattr(config, "get"):
                return config.get(f"augmentation.{key}", default)
            if isinstance(config, dict):
                return config.get(key, default)
            return default

        return cls(
            enabled=bool(_get("enabled", True)),
            horizontal_flip=float(_get("horizontal_flip", 0.5)),
            vertical_flip=float(_get("vertical_flip", 0.0)),
            rotation_limit=int(_get("rotation_limit", 30)),
            scale_limit=float(_get("scale_limit", 0.15)),
            translate_limit=float(_get("translate_limit", 0.05)),
            brightness_limit=float(_get("brightness_limit", 0.2)),
            contrast_limit=float(_get("contrast_limit", 0.2)),
            hue_shift=float(_get("hue_shift", 0.02)),
            saturation_shift=float(_get("saturation_shift", 0.15)),
            random_crop=bool(_get("random_crop", True)),
            crop_scale_limit=float(_get("crop_scale_limit", 0.9)),
            elastic_transform=bool(_get("elastic_transform", False)),
            gaussian_blur_prob=float(_get("gaussian_blur_prob", 0.0)),
            gaussian_noise_prob=float(_get("gaussian_noise_prob", 0.0)),
            coarse_dropout_prob=float(_get("coarse_dropout_prob", 0.0)),
        )


def _import_albumentations():
    try:
        # Albumentations performs a network version check on import, which is
        # noisy in an offline or batch environment.  It is harmless, so it is
        # switched off rather than being allowed to slow every import down.
        import os

        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        import albumentations as A
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Albumentations is required for augmentation. "
            "Install it with: pip install albumentations"
        ) from exc
    return A


def build_train_augmentations(
    config: AugmentationConfig | Any,
    image_size: int = 224,
) -> Any:
    """Build the Albumentations pipeline used for the training split.

    Parameters
    ----------
    config:
        An :class:`AugmentationConfig` or the project configuration object.
    image_size:
        Side length of the output image.

    Returns
    -------
    albumentations.Compose
    """
    if not isinstance(config, AugmentationConfig):
        config = AugmentationConfig.from_config(config)

    A = _import_albumentations()

    if not config.enabled:
        return A.Compose([A.Resize(image_size, image_size)])

    transforms: list[Any] = []

    # Geometric transforms first so that photometric jitter is applied to the
    # already-warped image (matching how a real camera produces the data).
    if config.random_crop:
        # Albumentations 2.x renamed the ``height``/``width`` arguments of
        # RandomResizedCrop to a single ``size`` argument.  Both spellings are
        # supported here so the project runs with either major version.
        try:
            transforms.append(
                A.RandomResizedCrop(
                    size=(image_size, image_size),
                    scale=(config.crop_scale_limit, 1.0),
                    ratio=(0.9, 1.1),
                    p=0.8,
                )
            )
        except (TypeError, ValueError):
            transforms.append(
                A.RandomResizedCrop(
                    height=image_size,
                    width=image_size,
                    scale=(config.crop_scale_limit, 1.0),
                    ratio=(0.9, 1.1),
                    p=0.8,
                )
            )
    else:
        transforms.append(A.Resize(image_size, image_size))

    transforms.append(
        A.Affine(
            rotate=(-config.rotation_limit, config.rotation_limit),
            scale=(1.0 - config.scale_limit, 1.0 + config.scale_limit),
            translate_percent=(-config.translate_limit, config.translate_limit),
            shear=(-5, 5),
            # Albumentations 1.4.x renamed ``border_mode`` to ``mode``;
            # passing the old name is silently ignored and warns.
            mode=cv2.BORDER_CONSTANT,
            p=0.7,
        )
    )
    transforms.append(A.HorizontalFlip(p=config.horizontal_flip))
    transforms.append(A.VerticalFlip(p=config.vertical_flip))

    if config.elastic_transform:
        transforms.append(
            A.ElasticTransform(alpha=1, sigma=50, p=0.2)
        )

    # Photometric jitter - deliberately mild to preserve diagnostic colour.
    transforms.append(
        A.RandomBrightnessContrast(
            brightness_limit=config.brightness_limit,
            contrast_limit=config.contrast_limit,
            p=0.5,
        )
    )
    transforms.append(
        A.HueSaturationValue(
            hue_shift_limit=int(config.hue_shift * 180),
            sat_shift_limit=int(config.saturation_shift * 255),
            val_shift_limit=0,
            p=0.4,
        )
    )

    if config.gaussian_blur_prob > 0:
        transforms.append(
            A.GaussianBlur(blur_limit=(3, 5), p=config.gaussian_blur_prob)
        )
    if config.gaussian_noise_prob > 0:
        transforms.append(
            A.GaussNoise(var_limit=(5.0, 25.0), p=config.gaussian_noise_prob)
        )
    if config.coarse_dropout_prob > 0:
        transforms.append(
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.05, 0.15),
                hole_width_range=(0.05, 0.15),
                fill=0,
                p=config.coarse_dropout_prob,
            )
        )

    # Final resize guarantees a constant output shape even when
    # RandomResizedCrop was skipped.
    transforms.append(A.Resize(image_size, image_size))

    return A.Compose(transforms)


def build_val_augmentations(image_size: int = 224) -> Any:
    """Build the validation/test pipeline - deterministic resize only."""
    A = _import_albumentations()
    return A.Compose([A.Resize(image_size, image_size)])
