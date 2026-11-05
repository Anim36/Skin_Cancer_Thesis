"""Classification metrics for skin-lesion classification.

Every metric that the thesis reports is implemented here, including the
melanoma-specific block that matters most clinically.

Why specificity and sensitivity are reported explicitly
========================================================
A model that never predicts melanoma but predicts "nevus" for everything
achieves ~67% accuracy on HAM10000 while being clinically useless.  The
metrics that matter for screening are:

* **Sensitivity (recall) for melanoma** - how many melanomas are caught.
* **Specificity for melanoma** - how many non-melanomas are correctly
  dismissed.
* **Negative predictive value** - of the patients the model dismisses, how
  many truly do not have melanoma.

These are reported in a dedicated "melanoma focus" section of every
evaluation report.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "compute_classification_metrics",
    "compute_melanoma_metrics",
    "compute_confusion_matrix",
    "safe_auc",
    "classification_report_dict",
]


def _to_numpy(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def compute_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int] | None = None,
) -> np.ndarray:
    """Return the confusion matrix as a numpy array.

    Parameters
    ----------
    y_true:
        Ground-truth class indices.
    y_pred:
        Predicted class indices.
    labels:
        The label space.  When ``None`` it is inferred from the data.
    """
    from sklearn.metrics import confusion_matrix

    true = _to_numpy(y_true).astype(int)
    pred = _to_numpy(y_pred).astype(int)

    if labels is None:
        labels = list(range(int(max(true.max(), pred.max())) + 1)) if len(true) else [0]
    return confusion_matrix(true, pred, labels=list(labels))


def safe_auc(
    y_true: Sequence[int],
    y_score: Sequence[float] | Sequence[Sequence[float]],
    class_index: int,
    multi_class: bool = False,
) -> float | None:
    """Compute the ROC-AUC for one class, returning ``None`` when undefined.

    ROC-AUC is undefined when a class has no positive examples in ``y_true``
    (which happens for the rarer classes in a stratified split).  Returning
    ``None`` instead of a misleading 0.0 or 0.5 keeps the report honest.
    """
    from sklearn.metrics import roc_auc_score

    true = _to_numpy(y_true).astype(int)
    score = _to_numpy(y_score)

    if multi_class:
        if score.ndim != 2 or score.shape[1] <= class_index:
            return None
        try:
            return float(
                roc_auc_score(true, score, multi_class="ovr", average="macro")
            )
        except ValueError:
            return None

    if score.ndim == 2:
        if score.shape[1] <= class_index:
            return None
        positives = score[:, class_index]
    else:
        positives = score

    if len(np.unique(true)) < 2:
        return None
    try:
        return float(roc_auc_score((true == class_index).astype(int), positives))
    except ValueError:
        return None


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[Sequence[float]] | None = None,
    class_names: Sequence[str] | None = None,
    num_classes: int | None = None,
    average: str = "macro",
    focus_class: str | None = "MEL",
) -> dict[str, Any]:
    """Compute the full metric suite.

    Parameters
    ----------
    y_true:
        Ground-truth class indices.
    y_pred:
        Predicted class indices.
    y_score:
        Class probabilities, used for ROC-AUC and the PR curve.  Optional.
    class_names:
        Human-readable class names, ordered by class index.
    num_classes:
        Size of the label space.  Inferred from the data when omitted.
    average:
        Averaging strategy for precision/recall/F1 (``macro`` | ``weighted`` |
        ``micro``).
    focus_class:
        The clinically critical class (melanoma) for the dedicated block.

    Returns
    -------
    dict
        All computed metrics plus the confusion matrix and per-class report.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        f1_score,
        precision_score,
        recall_score,
    )

    true = _to_numpy(y_true).astype(int)
    pred = _to_numpy(y_pred).astype(int)

    if num_classes is None:
        num_classes = int(max(true.max(), pred.max())) + 1 if len(true) else 1
    class_names = list(class_names) if class_names else [
        f"class_{i}" for i in range(num_classes)
    ]
    labels = list(range(num_classes))

    metrics: dict[str, Any] = {}

    # -- global metrics ---------------------------------------------------
    metrics["accuracy"] = float(accuracy_score(true, pred)) if len(true) else 0.0
    metrics["balanced_accuracy"] = (
        float(balanced_accuracy_score(true, pred)) if len(true) else 0.0
    )
    metrics["kappa"] = float(cohen_kappa_score(true, pred)) if len(true) else 0.0
    metrics["num_samples"] = int(len(true))

    # -- per-class averaged metrics ---------------------------------------
    metrics["precision"] = float(
        precision_score(true, pred, labels=labels, average=average, zero_division=0)
    )
    metrics["recall"] = float(
        recall_score(true, pred, labels=labels, average=average, zero_division=0)
    )
    metrics["f1"] = float(
        f1_score(true, pred, labels=labels, average=average, zero_division=0)
    )
    metrics["macro_f1"] = float(
        f1_score(true, pred, labels=labels, average="macro", zero_division=0)
    )
    metrics["weighted_f1"] = float(
        f1_score(true, pred, labels=labels, average="weighted", zero_division=0)
    )
    metrics["micro_f1"] = float(
        f1_score(true, pred, labels=labels, average="micro", zero_division=0)
    )

    # -- per-class detail -------------------------------------------------
    per_class: dict[str, Any] = {}
    for index, label in enumerate(labels):
        mask = true == label
        support = int(mask.sum())
        tp = int(((pred == label) & mask).sum())
        fp = int(((pred == label) & ~mask).sum())
        fn = int(((pred != label) & mask).sum())
        tn = int(((pred != label) & ~mask).sum())

        class_precision = tp / (tp + fp) if (tp + fp) else 0.0
        class_recall = tp / (tp + fn) if (tp + fn) else 0.0
        class_f1 = (
            2 * class_precision * class_recall / (class_precision + class_recall)
            if (class_precision + class_recall)
            else 0.0
        )
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        npv = tn / (tn + fn) if (tn + fn) else 0.0

        entry: dict[str, Any] = {
            "class_index": index,
            "class_name": class_names[index] if index < len(class_names) else f"class_{index}",
            "support": support,
            "precision": float(class_precision),
            "recall": float(class_recall),
            "sensitivity": float(class_recall),
            "specificity": float(specificity),
            "f1": float(class_f1),
            "npv": float(npv),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

        if y_score is not None:
            score = _to_numpy(y_score)
            if score.ndim == 2 and score.shape[1] > index:
                entry["roc_auc"] = safe_auc(true, score, index)
                try:
                    from sklearn.metrics import average_precision_score

                    entry["average_precision"] = float(
                        average_precision_score((true == index).astype(int), score[:, index])
                    )
                except ValueError:
                    entry["average_precision"] = None
            else:
                entry["roc_auc"] = None
                entry["average_precision"] = None

        per_class[entry["class_name"]] = entry

    metrics["per_class"] = per_class

    # -- confusion matrix --------------------------------------------------
    metrics["confusion_matrix"] = compute_confusion_matrix(true, pred, labels).tolist()
    metrics["class_names"] = class_names

    # -- ROC-AUC -----------------------------------------------------------
    if y_score is not None:
        score = _to_numpy(y_score)
        if score.ndim == 2:
            metrics["roc_auc_ovr_macro"] = safe_auc(true, score, 0, multi_class=True)
            metrics["roc_auc_per_class"] = {
                name: entry.get("roc_auc") for name, entry in per_class.items()
            }

    # -- melanoma focus ----------------------------------------------------
    if focus_class and focus_class in per_class:
        metrics["focus_class"] = focus_class
        metrics["focus"] = per_class[focus_class]

    return metrics


def compute_melanoma_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    melanoma_index: int = 0,
) -> dict[str, Any]:
    """Compute the melanoma-specific binary decision metrics.

    The multiclass prediction is collapsed into a binary decision
    ("is this melanoma?") so that the numbers are directly comparable with
    the sensitivity/specificity reported in the literature.
    """
    true = _to_numpy(y_true).astype(int)
    pred = _to_numpy(y_pred).astype(int)

    positive_true = (true == melanoma_index)
    positive_pred = (pred == melanoma_index)

    tp = int((positive_true & positive_pred).sum())
    fp = int((~positive_true & positive_pred).sum())
    fn = int((positive_true & ~positive_pred).sum())
    tn = int((~positive_true & ~positive_pred).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity)
        else 0.0
    )

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "npv": float(npv),
        "f1": float(f1),
        "prevalence": float(positive_true.mean()) if len(true) else 0.0,
        "note": (
            "These are screening-oriented operating characteristics. A model "
            "is a research prototype and NOT a diagnostic device."
        ),
    }


def classification_report_dict(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    num_classes: int | None = None,
) -> dict[str, Any]:
    """Return scikit-learn's text classification report as a dictionary."""
    from sklearn.metrics import classification_report

    true = _to_numpy(y_true).astype(int)
    pred = _to_numpy(y_pred).astype(int)
    labels = list(range(num_classes or (int(max(true.max(), pred.max())) + 1)))

    return classification_report(
        true,
        pred,
        labels=labels,
        target_names=list(class_names)[: len(labels)],
        output_dict=True,
        zero_division=0,
    )
