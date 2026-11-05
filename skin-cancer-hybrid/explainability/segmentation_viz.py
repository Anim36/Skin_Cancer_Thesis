"""Visualisation helpers for lesion segmentation.

The classification pipeline explains *what* the model looked at with Grad-CAM.
The segmentation stage explains *where* the lesion is, which is a different
question and needs different tooling: an overlay of the predicted mask on the
original dermoscopic image, next to the expert annotation, so that the
quality of the U-Net can be judged by eye.

The functions here are deliberately dependency-light (``matplotlib`` +
``numpy`` only) so that they can be used from a training script, from the
evaluation CLI, or from a notebook on a machine without a display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = [
    "overlay_mask",
    "mask_contours",
    "save_segmentation_overlay",
    "save_segmentation_grid",
]


def _to_uint8_rgb(image: Any) -> np.ndarray:
    """Convert an image-like object to an ``(H, W, 3)`` uint8 RGB array."""
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        # CHW -> HWC
        array = np.transpose(array, (1, 2, 0))
    if array.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D image, got shape {array.shape}")

    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]

    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        # Heuristic: values above 1.0 are assumed to be in 0-255 range.
        if array.max() > 1.0:
            array = array / 255.0
        array = np.clip(array, 0.0, 1.0) * 255.0
        array = array.astype(np.uint8)
    return array


def _to_bool_mask(mask: Any, threshold: float = 0.5) -> np.ndarray:
    """Convert a mask-like object to a boolean ``(H, W)`` array."""
    array = np.asarray(mask)
    if array.ndim == 3:
        if array.shape[0] == 1:
            array = array[0]
        elif array.shape[-1] == 1:
            array = array[..., 0]
        else:
            # A multi-channel mask is reduced by taking the maximum.
            array = array.max(axis=-1)
    if array.dtype != bool:
        array = array > threshold
    return array.astype(bool)


def overlay_mask(
    image: Any,
    mask: Any,
    colour: Sequence[float] = (1.0, 0.0, 0.0),
    alpha: float = 0.45,
    threshold: float = 0.5,
) -> np.ndarray:
    """Blend a binary mask over an image.

    Parameters
    ----------
    image:
        RGB image as ``(H, W, 3)`` uint8 (or anything convertible to it).
    mask:
        Binary or probability mask as ``(H, W)``.
    colour:
        RGB colour of the overlay in the 0-1 range.
    alpha:
        Opacity of the overlay; ``0`` leaves the image untouched, ``1``
        replaces the masked pixels completely.
    threshold:
        Binarisation threshold used when ``mask`` holds probabilities.

    Returns
    -------
    np.ndarray
        ``(H, W, 3)`` uint8 image with the mask blended in.
    """
    base = _to_uint8_rgb(image).astype(np.float32) / 255.0
    binary = _to_bool_mask(mask, threshold)

    if binary.shape != base.shape[:2]:
        raise ValueError(
            f"Mask shape {binary.shape} does not match image shape {base.shape[:2]}"
        )

    tint = np.array(colour, dtype=np.float32).reshape(1, 1, 3)
    blended = base.copy()
    selected = binary[..., None]
    blended = blended * (1.0 - alpha * selected) + tint * (alpha * selected)
    return (np.clip(blended, 0.0, 1.0) * 255.0).astype(np.uint8)


def mask_contours(
    mask: Any,
    thickness: int = 2,
    threshold: float = 0.5,
) -> np.ndarray:
    """Return the contour of a mask as a boolean array of the same shape.

    Outlining rather than filling keeps the underlying texture of the lesion
    visible, which matters when judging whether the segmentation follows the
    true lesion border.
    """
    import cv2

    binary = _to_bool_mask(mask, threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = np.zeros_like(binary, dtype=np.uint8)
    if contours:
        cv2.drawContours(outline, contours, -1, 1, thickness=thickness)
    return outline.astype(bool)


def save_segmentation_overlay(
    image: Any,
    predicted_mask: Any,
    target_mask: Any | None = None,
    output_path: str | Path = "segmentation.png",
    alpha: float = 0.45,
    threshold: float = 0.5,
    title: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Save a single image with the predicted (and optional true) mask.

    The ground-truth mask is drawn as a green contour and the prediction as a
    red contour, so the two can be compared directly in one picture.

    Parameters
    ----------
    image:
        The original RGB image.
    predicted_mask:
        The predicted probability or binary mask.
    target_mask:
        Optional ground-truth mask.
    output_path:
        Where to write the PNG.
    alpha:
        Opacity of the filled prediction overlay.
    threshold:
        Binarisation threshold for probability masks.
    title:
        Optional title.  When ``metrics`` is given the Dice/IoU values are
        appended automatically.
    metrics:
        Optional metric dictionary (as returned by
        :func:`evaluation.segmentation_metrics.compute_segmentation_metrics`).

    Returns
    -------
    Path
        The path the image was written to.
    """
    import matplotlib.pyplot as plt

    from explainability.visualization import set_plot_style

    set_plot_style()

    canvas = overlay_mask(image, predicted_mask, alpha=alpha, threshold=threshold)

    if target_mask is not None:
        import cv2

        base = _to_uint8_rgb(image)
        truth = _to_bool_mask(target_mask, threshold)
        if truth.shape == base.shape[:2]:
            contours, _ = cv2.findContours(
                truth.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(base, contours, -1, (0, 255, 0), 2)
        canvas = base

    heading = title or "Lesion segmentation"
    if metrics:
        dice = metrics.get("dice")
        iou = metrics.get("iou")
        parts = []
        if dice is not None:
            parts.append(f"Dice {float(dice):.3f}")
        if iou is not None:
            parts.append(f"IoU {float(iou):.3f}")
        if parts:
            heading = f"{heading} - " + ", ".join(parts)

    figure, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(canvas)
    axis.set_title(heading, fontsize=11)
    axis.axis("off")
    figure.tight_layout()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return destination


def save_segmentation_grid(
    images: Sequence[Any],
    predicted_masks: Sequence[Any],
    target_masks: Sequence[Any] | None = None,
    output_path: str | Path = "segmentation_grid.png",
    metrics: Sequence[dict[str, Any]] | None = None,
    threshold: float = 0.5,
    max_columns: int = 4,
) -> Path:
    """Save a grid of image / prediction / ground-truth triplets.

    This is the "eyeball check" figure for a segmentation model: it shows
    several lesions at once so that systematic failures (segmenting the ruler
    markings, leaking into the background, missing pale lesions) are obvious
    at a glance.
    """
    import matplotlib.pyplot as plt

    from explainability.visualization import set_plot_style

    set_plot_style()

    count = len(images)
    if count == 0:
        raise ValueError("No images supplied")
    if len(predicted_masks) != count:
        raise ValueError("images and predicted_masks must have the same length")
    if target_masks is not None and len(target_masks) != count:
        raise ValueError("images and target_masks must have the same length")

    columns = min(max_columns, count)
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for index in range(columns * rows):
        axis = axes[index]
        axis.axis("off")
        if index >= count:
            continue

        canvas = overlay_mask(
            images[index], predicted_masks[index], alpha=0.45, threshold=threshold
        )
        if target_masks is not None:
            import cv2

            base = _to_uint8_rgb(images[index])
            truth = _to_bool_mask(target_masks[index], threshold)
            if truth.shape == base.shape[:2]:
                contours, _ = cv2.findContours(
                    truth.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(base, contours, -1, (0, 255, 0), 2)
            canvas = base

        axis.imshow(canvas)
        heading = f"#{index + 1}"
        if metrics is not None and index < len(metrics):
            dice = metrics[index].get("dice")
            if dice is not None:
                heading += f"  Dice {float(dice):.3f}"
        axis.set_title(heading, fontsize=9)

    figure.suptitle("Red = predicted lesion mask, green = expert annotation",
                     fontsize=11)
    figure.tight_layout()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return destination
