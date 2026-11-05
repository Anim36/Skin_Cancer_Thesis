"""Explainability: Grad-CAM and the visualisation helpers."""

from explainability.gradcam import GradCAM, GradCAMResult, resolve_target_layer
from explainability.segmentation_viz import (
    mask_contours,
    overlay_mask,
    save_segmentation_grid,
    save_segmentation_overlay,
)
from explainability.visualization import (
    save_heatmap,
    save_overlay,
    save_gradcam_figure,
    save_prediction_figure,
    save_confusion_matrix_plot,
    save_roc_plot,
    save_pr_plot,
    save_training_curves,
    save_per_class_plot,
)

__all__ = [
    "GradCAM",
    "GradCAMResult",
    "resolve_target_layer",
    "overlay_mask",
    "mask_contours",
    "save_segmentation_overlay",
    "save_segmentation_grid",
    "save_heatmap",
    "save_overlay",
    "save_gradcam_figure",
    "save_prediction_figure",
    "save_confusion_matrix_plot",
    "save_roc_plot",
    "save_pr_plot",
    "save_training_curves",
    "save_per_class_plot",
]
