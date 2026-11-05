"""Import every project module to catch syntax and import errors early."""

import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODULES = [
    "utils.config",
    "utils.seeding",
    "utils.hardware",
    "utils.logging_utils",
    "data.label_mapping",
    "data.preprocessing",
    "data.augmentation",
    "data.metadata",
    "data.loaders.base",
    "data.loaders.ham10000",
    "data.loaders.isic2018",
    "data.loaders.isic2019",
    "data.loaders.pad_ufes20",
    "data.dataset_harmonization",
    "data.deduplication",
    "data.splitting",
    "data.datasets",
    "data.segmentation_dataset",
    "data.pipeline",
    "models.unet",
    "models.efficientnetv2",
    "models.attention",
    "models.metadata_encoder",
    "models.fusion",
    "models.classifier",
    "models.hybrid_model",
    "models.losses",
    "training.trainer",
    "training.cli_utils",
    "training.train_unet",
    "training.train_hybrid",
    "training.train_baseline",
    "training.train_multimodal",
    "training.ablation_study",
    "training.hyperparameter_search",
    "evaluation.classification_metrics",
    "evaluation.segmentation_metrics",
    "evaluation.confusion_matrix",
    "evaluation.roc_analysis",
    "evaluation.dataset_reports",
    "evaluation.evaluate",
    "explainability.gradcam",
    "explainability.visualization",
    "inference.predict",
    "inference.predict_batch",
]

failures = []
for name in MODULES:
    try:
        importlib.import_module(name)
        print(f"OK   {name}")
    except Exception as exc:
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        failures.append(name)

print()
if failures:
    print(f"{len(failures)} MODULE(S) FAILED: {failures}")
    sys.exit(1)
print(f"ALL {len(MODULES)} MODULES IMPORTED SUCCESSFULLY")
