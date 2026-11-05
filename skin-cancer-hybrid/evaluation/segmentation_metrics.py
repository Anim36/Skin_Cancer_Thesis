"""Segmentation metrics: Dice coefficient, IoU and pixel-level statistics.

The Dice coefficient and the Intersection-over-Union are the two metrics that
the ISIC lesion-segmentation literature reports, so both are implemented
here.  Pixel-level accuracy, sensitivity and specificity are included as
supporting evidence.

All functions accept either probability maps or binary masks and return a
float in ``[0, 1]``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = [
    "dice_coefficient",
    "iou_coefficient",
    "pixel_accuracy",
    "pixel_sensitivity",
    "pixel_specificity",
    "compute_segmentation_metrics",
    "batch_segmentation_metrics",
]


def _prepare(
    prediction: Any, target: Any, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Convert prediction/target to flat boolean numpy arrays."""
    pred = np.asarray(prediction)
    true = np.asarray(target)

    if pred.shape != true.shape:
        raise ValueError(
            f"Prediction shape {pred.shape} does not match target shape {true.shape}"
        )

    if pred.dtype != bool:
        pred = pred > threshold
    if true.dtype != bool:
        true = true > threshold

    return pred.reshape(-1), true.reshape(-1)


def dice_coefficient(
    prediction: Any, target: Any, threshold: float = 0.5, smooth: float = 1e-7
) -> float:
    """Dice coefficient (F1 score on the pixels).

    .. math::
        D = \\frac{2|P \\cap G|}{|P| + |G|}

    Parameters
    ----------
    prediction:
        Predicted probabilities or binary mask.
    target:
        Ground-truth mask.
    threshold:
        Binarisation threshold when probabilities are supplied.
    smooth:
        Added to the denominator so that two empty masks score 1.0 instead of
        dividing by zero.
    """
    pred, true = _prepare(prediction, target, threshold)
    intersection = float(np.sum(pred & true))
    total = float(np.sum(pred) + np.sum(true))
    return float((2.0 * intersection + smooth) / (total + smooth))


def iou_coefficient(
    prediction: Any, target: Any, threshold: float = 0.5, smooth: float = 1e-7
) -> float:
    """Intersection over Union (Jaccard index).

    .. math::
        J = \\frac{|P \\cap G|}{|P \\cup G|}
    """
    pred, true = _prepare(prediction, target, threshold)
    intersection = float(np.sum(pred & true))
    union = float(np.sum(pred | true))
    return float((intersection + smooth) / (union + smooth))


def pixel_accuracy(prediction: Any, target: Any, threshold: float = 0.5) -> float:
    """Fraction of pixels classified correctly."""
    pred, true = _prepare(prediction, target, threshold)
    if pred.size == 0:
        return 0.0
    return float(np.mean(pred == true))


def pixel_sensitivity(prediction: Any, target: Any, threshold: float = 0.5) -> float:
    """Recall over the positive (lesion) pixels."""
    pred, true = _prepare(prediction, target, threshold)
    positives = true.sum()
    if positives == 0:
        return 0.0
    return float(np.sum(pred & true) / positives)


def pixel_specificity(prediction: Any, target: Any, threshold: float = 0.5) -> float:
    """Recall over the negative (background) pixels."""
    pred, true = _prepare(prediction, target, threshold)
    negatives = (~true).sum()
    if negatives == 0:
        return 0.0
    return float(np.sum((~pred) & (~true)) / negatives)


def compute_segmentation_metrics(
    prediction: Any,
    target: Any,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the full segmentation metric suite for one sample or batch."""
    return {
        "dice": dice_coefficient(prediction, target, threshold),
        "iou": iou_coefficient(prediction, target, threshold),
        "pixel_accuracy": pixel_accuracy(prediction, target, threshold),
        "sensitivity": pixel_sensitivity(prediction, target, threshold),
        "specificity": pixel_specificity(prediction, target, threshold),
    }


def batch_segmentation_metrics(
    predictions: Sequence[Any],
    targets: Sequence[Any],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Average the segmentation metrics over a batch of samples."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    if not len(predictions):
        return {
            "dice": 0.0,
            "iou": 0.0,
            "pixel_accuracy": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
        }

    totals: dict[str, list[float]] = {}
    for prediction, target in zip(predictions, targets):
        for key, value in compute_segmentation_metrics(
            prediction, target, threshold
        ).items():
            totals.setdefault(key, []).append(value)

    return {key: float(np.mean(values)) for key, values in totals.items()}
