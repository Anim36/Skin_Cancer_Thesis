"""Visualisation helpers for explainability outputs and evaluation plots.

Every figure produced here is saved to disk with a title and, where relevant,
an explicit legend so that it can be included in the thesis without further
editing.

Colour convention
=================
Heatmaps use the ``jet`` colormap, where **red marks the regions the model
relied on most** and blue marks the regions it relied on least.  A colour bar
is always drawn next to the heatmap so the reader is not left guessing the
scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = [
    "save_heatmap",
    "save_overlay",
    "save_gradcam_figure",
    "save_prediction_figure",
    "save_confusion_matrix_plot",
    "save_roc_plot",
    "save_pr_plot",
    "save_training_curves",
    "save_per_class_plot",
    "set_plot_style",
]


def set_plot_style() -> None:
    """Apply a consistent, publication-friendly matplotlib style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.autolayout": True,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


# ---------------------------------------------------------------------------
# Grad-CAM figures
# ---------------------------------------------------------------------------
def save_heatmap(
    heatmap: np.ndarray,
    path: str | Path,
    title: str | None = None,
    cmap: str = "jet",
) -> Path:
    """Save a standalone heatmap with a colour bar."""
    import matplotlib.pyplot as plt

    set_plot_style()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(5, 4.5))
    image = axis.imshow(heatmap, cmap=cmap, vmin=0.0, vmax=1.0)
    axis.set_title(title or "Grad-CAM heatmap")
    axis.set_xlabel("x (pixels)")
    axis.set_ylabel("y (pixels)")
    axis.grid(False)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="relative activation")
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    return destination


def save_overlay(
    overlay: np.ndarray,
    path: str | Path,
    title: str | None = None,
) -> Path:
    """Save the heatmap overlay."""
    import matplotlib.pyplot as plt

    set_plot_style()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(5, 4.5))
    axis.imshow(overlay)
    axis.set_title(title or "Grad-CAM overlay")
    axis.set_xlabel("x (pixels)")
    axis.set_ylabel("y (pixels)")
    axis.grid(False)
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    return destination


def save_gradcam_figure(
    image: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    path: str | Path,
    class_name: str,
    confidence: float,
    extra_text: str = "",
) -> Path:
    """Save a three-panel figure: original, heatmap and overlay.

    The figure includes the predicted class and the confidence so that the
    explanation is self-contained.
    """
    import matplotlib.pyplot as plt

    set_plot_style()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    axes[0].imshow(image)
    axes[0].set_title("Original image")
    axes[1].imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("Grad-CAM heatmap")
    image_artist = axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)

    subtitle = f"predicted: {class_name}  |  confidence: {confidence:.3f}"
    if extra_text:
        subtitle = f"{subtitle}\n{extra_text}"
    figure.suptitle(subtitle, fontsize=12)
    figure.colorbar(image_artist, ax=axes[2], fraction=0.046, pad=0.04,
                    label="relative activation")
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    return destination


def save_prediction_figure(
    image: np.ndarray,
    path: str | Path,
    class_names: Sequence[str],
    probabilities: Sequence[float],
    top_k: int = 3,
) -> Path:
    """Save the original image next to a top-k probability bar chart."""
    import matplotlib.pyplot as plt

    set_plot_style()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    order = np.argsort(probabilities)[::-1][:top_k]
    labels = [class_names[i] for i in order]
    values = [float(probabilities[i]) for i in order]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].imshow(image)
    axes[0].set_title("Input image")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].grid(False)

    axes[1].barh(labels[::-1], values[::-1], color="#c0392b")
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("probability")
    axes[1].set_title(f"Top-{top_k} predictions")
    for index, value in enumerate(values[::-1]):
        axes[1].text(value + 0.01, index, f"{value:.3f}", va="center", fontsize=9)

    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    return destination


# ---------------------------------------------------------------------------
# Evaluation plots
# ---------------------------------------------------------------------------
def save_confusion_matrix_plot(
    matrix: np.ndarray,
    class_names: Sequence[str],
    path: str | Path,
    normalize: bool = True,
    title: str = "Confusion matrix",
) -> Path:
    """Save the confusion matrix as a heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    set_plot_style()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    matrix = np.asarray(matrix, dtype=float)
    if normalize and matrix.sum() > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            plot_matrix = matrix / matrix.sum(axis=1, keepdims=True)
        plot_matrix = np.nan_to_num(plot_matrix)
        fmt = ".2f"
        title = f"{title} (row-normalised)"
    else:
        plot_matrix = matrix
        fmt = "d"

    figure, axis = plt.subplots(figsize=(1.1 * len(class_names) + 3, 1.0 * len(class_names) + 2.5))
    sns.heatmap(
        plot_matrix,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        ax=axis,
        square=False,
    )
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    return destination


def save_roc_plot(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Sequence[str],
    path: str | Path,
    title: str = "ROC curves (one-vs-rest)",
) -> Path:
    """Save the one-vs-rest ROC curves."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    set_plot_style()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 6))
    for index, name in enumerate(class_names):
        if index >= y_score.shape[1]:
            break
        positives = (y_true == index).astype(int)
        if positives.sum() == 0 or positives.sum() == len(positives):
            continue
        fpr, tpr, _ = roc_curve(positives, y_score[:, index])
        auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(
            np.trapz(tpr, fpr)
        )
        axis.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    axis.set_xlabel("False positive rate (1 - specificity)")
    axis.set_ylabel("True positive rate (sensitivity)")
    axis.set_title(title)
    axis.legend(loc="lower right", fontsize=8)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    return destination


def save_pr_plot(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Sequence[str],
    path: str | Path,
    title: str = "Precision-Recall curves (one-vs-rest)",
) -> Path:
    """Save the one-vs-rest precision-recall curves."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve

    set_plot_style()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 6))
    for index, name in enumerate(class_names):
        if index >= y_score.shape[1]:
            break
        positives = (y_true == index).astype(int)
        if positives.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(positives, y_score[:, index])
        ap = float(average_precision_score(positives, y_score[:, index]))
        axis.plot(recall, precision, label=f"{name} (AP={ap:.3f})")

    axis.set_xlabel("Recall (sensitivity)")
    axis.set_ylabel("Precision")
    axis.set_title(title)
    axis.legend(loc="lower left", fontsize=8)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    return destination


def save_training_curves(
    history: Any,
    output_dir: str | Path,
    stem: str = "training_curves",
) -> list[Path]:
    """Save the loss, accuracy and macro-F1 training curves."""
    import matplotlib.pyplot as plt

    set_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(history, "as_frame"):
        frame = history.as_frame()
    else:
        import pandas as pd

        frame = pd.DataFrame(history)

    if "epoch" not in frame.columns:
        frame["epoch"] = range(len(frame))

    paths: list[Path] = []

    # -- loss --------------------------------------------------------------
    figure, axis = plt.subplots(figsize=(7, 5))
    if "train_loss" in frame:
        axis.plot(frame["epoch"], frame["train_loss"], label="train loss")
    if "val_loss" in frame:
        axis.plot(frame["epoch"], frame["val_loss"], label="validation loss")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.set_title("Training and validation loss")
    axis.legend()
    path = output_dir / f"{stem}_loss.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    # -- accuracy ----------------------------------------------------------
    figure, axis = plt.subplots(figsize=(7, 5))
    if "train_accuracy" in frame:
        axis.plot(frame["epoch"], frame["train_accuracy"], label="train accuracy")
    if "val_accuracy" in frame:
        axis.plot(frame["epoch"], frame["val_accuracy"], label="validation accuracy")
    axis.set_xlabel("epoch")
    axis.set_ylabel("accuracy")
    axis.set_ylim(0, 1)
    axis.set_title("Training and validation accuracy")
    axis.legend()
    path = output_dir / f"{stem}_accuracy.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    # -- macro F1 ----------------------------------------------------------
    figure, axis = plt.subplots(figsize=(7, 5))
    if "train_macro_f1" in frame:
        axis.plot(frame["epoch"], frame["train_macro_f1"], label="train macro F1")
    if "val_macro_f1" in frame:
        axis.plot(frame["epoch"], frame["val_macro_f1"], label="validation macro F1")
    axis.set_xlabel("epoch")
    axis.set_ylabel("macro F1")
    axis.set_ylim(0, 1)
    axis.set_title("Training and validation macro F1")
    axis.legend()
    path = output_dir / f"{stem}_macro_f1.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    return paths


def save_per_class_plot(
    per_class: dict[str, dict[str, Any]],
    path: str | Path,
    title: str = "Per-class performance",
) -> Path:
    """Save a grouped bar chart of per-class precision, recall and F1."""
    import matplotlib.pyplot as plt

    set_plot_style()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    names = list(per_class.keys())
    precision = [per_class[n].get("precision", 0.0) for n in names]
    recall = [per_class[n].get("recall", 0.0) for n in names]
    f1 = [per_class[n].get("f1", 0.0) for n in names]

    positions = np.arange(len(names))
    width = 0.27

    figure, axis = plt.subplots(figsize=(max(8, 1.4 * len(names)), 5))
    axis.bar(positions - width, precision, width, label="precision", color="#2980b9")
    axis.bar(positions, recall, width, label="recall (sensitivity)", color="#27ae60")
    axis.bar(positions + width, f1, width, label="F1", color="#c0392b")

    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=20, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("score")
    axis.set_title(title)
    axis.legend()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    return destination
