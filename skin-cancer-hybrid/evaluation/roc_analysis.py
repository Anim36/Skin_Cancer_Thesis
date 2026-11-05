"""ROC and precision-recall analysis, plus probability calibration.

Calibration matters for a screening tool: a model that reports 90% confidence
should be right 90% of the time.  A well-ranked model can still be badly
calibrated, which is why the reliability diagram is produced alongside the
ROC curves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = [
    "save_roc_and_pr_curves",
    "calibration_metrics",
    "save_calibration_plot",
    "expected_calibration_error",
]


def save_roc_and_pr_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Sequence[str],
    output_dir: str | Path,
    stem: str = "curves",
) -> tuple[Path, Path]:
    """Save the ROC and precision-recall plots.

    Returns
    -------
    tuple
        ``(roc_path, pr_path)``.
    """
    from explainability.visualization import save_pr_plot, save_roc_plot

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    roc_path = save_roc_plot(y_true, y_score, class_names, output_dir / f"{stem}_roc.png")
    pr_path = save_pr_plot(y_true, y_score, class_names, output_dir / f"{stem}_pr.png")
    return roc_path, pr_path


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    num_bins: int = 10,
) -> float:
    """Compute the Expected Calibration Error (ECE).

    ECE is the mean absolute difference between the accuracy of the samples in
    each confidence bin and the mean confidence of that bin.  Lower is better;
    0 means the reported confidences are exactly right.
    """
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidence > low) & (confidence <= high)
        if not mask.any():
            continue
        bin_accuracy = correct[mask].mean()
        bin_confidence = confidence[mask].mean()
        ece += mask.sum() / len(confidence) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def calibration_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    num_bins: int = 10,
) -> dict[str, Any]:
    """Compute calibration metrics and the reliability curve data."""
    from sklearn.metrics import brier_score_loss

    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    curve: list[dict[str, float]] = []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidence > low) & (confidence <= high)
        if not mask.any():
            curve.append(
                {
                    "bin_lower": float(low),
                    "bin_upper": float(high),
                    "mean_confidence": float("nan"),
                    "accuracy": float("nan"),
                    "count": 0,
                }
            )
            continue
        curve.append(
            {
                "bin_lower": float(low),
                "bin_upper": float(high),
                "mean_confidence": float(confidence[mask].mean()),
                "accuracy": float(correct[mask].mean()),
                "count": int(mask.sum()),
            }
        )

    return {
        "ece": expected_calibration_error(y_true, probabilities, num_bins),
        "brier_score": float(brier_score_loss(y_true, probabilities.argmax(axis=1))),
        "mean_confidence": float(confidence.mean()),
        "accuracy": float(correct.mean()),
        "reliability_curve": curve,
    }


def save_calibration_plot(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output_dir: str | Path,
    stem: str = "calibration",
    num_bins: int = 10,
) -> tuple[Path, dict[str, Any]]:
    """Save the reliability diagram and return its path plus the metrics."""
    import matplotlib.pyplot as plt

    from explainability.visualization import set_plot_style

    set_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = calibration_metrics(y_true, probabilities, num_bins)
    curve = metrics["reliability_curve"]

    figure, axis = plt.subplots(figsize=(6.5, 6))
    xs = [
        0.5 * (point["bin_lower"] + point["bin_upper"])
        for point in curve
        if point["count"] > 0
    ]
    ys = [point["accuracy"] for point in curve if point["count"] > 0]
    axis.plot(xs, ys, "o-", color="#c0392b", label="model")
    axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    axis.set_xlabel("Mean predicted confidence")
    axis.set_ylabel("Empirical accuracy")
    axis.set_title(
        f"Reliability diagram (ECE={metrics['ece']:.4f}, "
        f"Brier={metrics['brier_score']:.4f})"
    )
    axis.legend()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    path = output_dir / f"{stem}.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path, metrics
