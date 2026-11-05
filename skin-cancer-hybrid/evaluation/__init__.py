"""Evaluation: metrics, plots and per-dataset reporting."""

from evaluation.classification_metrics import (
    compute_classification_metrics,
    compute_melanoma_metrics,
    compute_confusion_matrix,
    classification_report_dict,
    safe_auc,
)
from evaluation.segmentation_metrics import (
    dice_coefficient,
    iou_coefficient,
    pixel_accuracy,
    pixel_sensitivity,
    pixel_specificity,
    compute_segmentation_metrics,
    batch_segmentation_metrics,
)
from evaluation.confusion_matrix import (
    build_confusion_matrix,
    normalise_confusion_matrix,
    analyse_confusion_matrix,
)
from evaluation.roc_analysis import (
    save_roc_and_pr_curves,
    calibration_metrics,
    expected_calibration_error,
    save_calibration_plot,
)
from evaluation.dataset_reports import (
    write_dataset_report,
    build_external_validation_section,
    format_metric_table,
)

__all__ = [
    "compute_classification_metrics",
    "compute_melanoma_metrics",
    "compute_confusion_matrix",
    "classification_report_dict",
    "safe_auc",
    "dice_coefficient",
    "iou_coefficient",
    "pixel_accuracy",
    "pixel_sensitivity",
    "pixel_specificity",
    "compute_segmentation_metrics",
    "batch_segmentation_metrics",
    "build_confusion_matrix",
    "normalise_confusion_matrix",
    "analyse_confusion_matrix",
    "save_roc_and_pr_curves",
    "calibration_metrics",
    "expected_calibration_error",
    "save_calibration_plot",
    "write_dataset_report",
    "build_external_validation_section",
    "format_metric_table",
]
