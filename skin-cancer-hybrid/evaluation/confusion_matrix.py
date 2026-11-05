"""Confusion matrix helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = [
    "build_confusion_matrix",
    "normalise_confusion_matrix",
    "analyse_confusion_matrix",
]


def build_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int | None = None,
) -> np.ndarray:
    """Build the confusion matrix.

    Rows are the true classes and columns are the predicted classes.
    """
    from evaluation.classification_metrics import compute_confusion_matrix

    return compute_confusion_matrix(y_true, y_pred, list(range(num_classes)) if num_classes else None)


def normalise_confusion_matrix(matrix: np.ndarray, axis: int = 1) -> np.ndarray:
    """Normalise a confusion matrix along an axis.

    ``axis=1`` gives the row-normalised matrix (each true class sums to 1),
    which is the form that shows which classes get confused with which.
    """
    matrix = np.asarray(matrix, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalised = matrix / matrix.sum(axis=axis, keepdims=True)
    return np.nan_to_num(normalised)


def analyse_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    top_k: int = 5,
) -> dict[str, Any]:
    """Summarise the most frequent confusions.

    The most clinically relevant confusions are those involving the focus
    class (melanoma), because a melanoma misclassified as a benign nevus is
    the failure mode that matters most in screening.
    """
    matrix = np.asarray(matrix, dtype=float)
    normalised = normalise_confusion_matrix(matrix, axis=1)

    confusions: list[dict[str, Any]] = []
    for true_index in range(matrix.shape[0]):
        for pred_index in range(matrix.shape[1]):
            if true_index == pred_index:
                continue
            if matrix[true_index, pred_index] <= 0:
                continue
            confusions.append(
                {
                    "true_class": class_names[true_index]
                    if true_index < len(class_names)
                    else str(true_index),
                    "predicted_class": class_names[pred_index]
                    if pred_index < len(class_names)
                    else str(pred_index),
                    "count": int(matrix[true_index, pred_index]),
                    "rate_within_true_class": float(normalised[true_index, pred_index]),
                }
            )

    confusions.sort(key=lambda item: item["count"], reverse=True)

    return {
        "top_confusions": confusions[:top_k],
        "matrix": matrix.tolist(),
        "normalised_matrix": normalised.tolist(),
        "class_names": list(class_names),
    }
