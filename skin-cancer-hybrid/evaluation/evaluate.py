"""Evaluate a trained checkpoint on the test split.

Usage
-----
::

    python evaluation/evaluate.py --checkpoint runs/hybrid/checkpoints/best.pth
    python evaluation/evaluate.py --checkpoint runs/hybrid/checkpoints/best.pth \
        --data-dir data/raw --split test
    python evaluation/evaluate.py --checkpoint ... --split val --per-dataset

Outputs
=======
* A JSON report with every metric (global, per class, per dataset).
* Markdown summary tables.
* Confusion matrix, ROC and precision-recall plots.
* A calibration (reliability) plot.

The report is deliberately explicit about **per-dataset** performance: a
single pooled accuracy that hides the fact that the model generalises poorly
to one of the datasets is not an acceptable result for this thesis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.pipeline import build_dataloaders, prepare_data
from evaluation.classification_metrics import (
    compute_classification_metrics,
    compute_melanoma_metrics,
)
from evaluation.roc_analysis import save_roc_and_pr_curves
from evaluation.dataset_reports import write_dataset_report
from models.hybrid_model import HybridModelConfig, HybridSkinCancerModel
from utils.hardware import resolve_device
from utils.logging_utils import setup_logging


def load_model(checkpoint_path: Path, config: Any, num_classes: int, metadata_dim: int):
    """Rebuild the model from a checkpoint and load its weights.

    The checkpoint stores the model configuration, so the architecture is
    reconstructed exactly as it was during training.
    """
    from models.hybrid_model import build_model_from_checkpoint_payload

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    try:
        model, class_names, saved_num_classes, saved_metadata_dim = (
            build_model_from_checkpoint_payload(payload, config)
        )
    except ValueError:
        # An older checkpoint without a model configuration.
        from models.hybrid_model import HybridModelConfig, HybridSkinCancerModel

        model_config = HybridModelConfig.from_config(config, num_classes=num_classes)
        model_config.pretrained = False
        model_config.metadata_dim = metadata_dim
        model_config.use_metadata = metadata_dim > 0
        model = HybridSkinCancerModel(model_config)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        class_names = payload.get("class_names") or []
        saved_num_classes = int(payload.get("num_classes", num_classes))
        saved_metadata_dim = int(payload.get("metadata_dim", metadata_dim))

    model.eval()
    return model, payload, class_names, saved_num_classes, saved_metadata_dim


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader,
    device,
    use_amp: bool,
) -> dict[str, Any]:
    """Run the model over a loader and collect the predictions."""
    from utils.hardware import autocast_context

    model.eval()
    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[np.ndarray] = []
    all_samples: list[str] = []
    all_datasets: list[str] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        metadata = batch.get("metadata")
        if metadata is not None and metadata.shape[1] > 0:
            metadata = metadata.to(device, non_blocking=True)
        else:
            metadata = None

        with autocast_context(device, use_amp, "float16"):
            logits = model(images, metadata, None)
            probabilities = torch.softmax(logits.float(), dim=1)

        predictions = probabilities.argmax(dim=1)
        all_labels.extend(batch["label"].cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_probabilities.append(probabilities.cpu().numpy())
        all_samples.extend(batch.get("sample_id", []))
        all_datasets.extend(batch.get("dataset", []))

    return {
        "labels": np.array(all_labels),
        "predictions": np.array(all_predictions),
        "probabilities": np.concatenate(all_probabilities, axis=0)
        if all_probabilities
        else np.zeros((0, 0)),
        "sample_ids": all_samples,
        "datasets": all_datasets,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint.")
    parser.add_argument("--config", default=None, help="Configuration file.")
    parser.add_argument("--data-dir", default=None, help="Dataset root directory.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate.")
    parser.add_argument("--output-dir", default=None, help="Where to write the report.")
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--datasets", nargs="*", default=None,
                        choices=["ham10000", "isic2018", "isic2019", "pad_ufes20"])
    parser.add_argument("--include-scc", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip the figure generation.")
    args = parser.parse_args()

    from utils.config import load_config
    from training.cli_utils import build_config_from_args  # noqa: F401  (shared parsing)

    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config) if args.config else project_root / "configs" / "default.yaml"
    overrides = []
    if args.data_dir:
        overrides.append(f"paths.raw_dir={args.data_dir!r}")
    if args.device:
        overrides.append(f"hardware.device={args.device!r}")
    if args.batch_size:
        overrides.append(f"evaluation.batch_size={args.batch_size}")
    config = load_config(config_path, overrides=overrides, root=project_root)

    output_dir = Path(
        args.output_dir or (Path(config.get("paths.reports_dir", "reports")) / "evaluation")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging("skin-cancer-hybrid", output_dir, filename="evaluate.log")

    # -- data ---------------------------------------------------------------
    prepared = prepare_data(
        config,
        datasets=args.datasets,
        raw_dir=args.data_dir,
        include_scc=args.include_scc,
    )
    logger.info(
        "data ready | train=%d val=%d test=%d | classes=%s",
        len(prepared.train_frame),
        len(prepared.val_frame),
        len(prepared.test_frame),
        prepared.class_names,
    )

    frames = {
        "train": prepared.train_frame,
        "val": prepared.val_frame,
        "test": prepared.test_frame,
    }
    frame = frames[args.split]
    if len(frame) == 0:
        logger.error("The %s split is empty.", args.split)
        return 1

    from data.datasets import SkinCancerDataset, collate_batch
    from data.preprocessing import PreprocessConfig

    dataset = SkinCancerDataset(
        frame,
        preprocess_config=PreprocessConfig.from_config(config),
        train=False,
        metadata_encoder=prepared.metadata_encoder,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size or config.get("evaluation.batch_size", 32)),
        shuffle=False,
        num_workers=int(config.get("hardware.num_workers", 4)),
        pin_memory=bool(config.get("hardware.pin_memory", True)),
        collate_fn=collate_batch,
    )

    # -- model --------------------------------------------------------------
    device, use_amp, _ = resolve_device(
        preference=str(args.device or config.get("hardware.device", "auto")),
        allow_amp=bool(config.get("hardware.amp", True)),
    )
    checkpoint_path = Path(args.checkpoint)
    model, payload, class_names, num_classes, metadata_dim = load_model(
        checkpoint_path, config, prepared.num_classes,
        prepared.metadata_encoder.output_dim if prepared.metadata_encoder else 0,
    )
    model.to(device)
    if not class_names:
        class_names = prepared.class_names
    logger.info("loaded checkpoint %s | classes=%s", checkpoint_path, class_names)

    # -- predict ------------------------------------------------------------
    results = predict(model, loader, device, use_amp)
    logger.info("evaluated %d samples", len(results["labels"]))

    # -- metrics ------------------------------------------------------------
    focus_index = class_names.index("MEL") if "MEL" in class_names else 0
    metrics = compute_classification_metrics(
        results["labels"],
        results["predictions"],
        results["probabilities"],
        class_names=class_names,
        num_classes=len(class_names),
        focus_class="MEL" if "MEL" in class_names else None,
    )
    metrics["melanoma_focus"] = compute_melanoma_metrics(
        results["labels"], results["predictions"], melanoma_index=focus_index
    )
    metrics["split"] = args.split
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["device"] = str(device)

    # -- per-dataset breakdown ---------------------------------------------
    per_dataset: dict[str, Any] = {}
    dataset_array = np.array(results["datasets"])
    for dataset_name in sorted(set(results["datasets"])):
        mask = dataset_array == dataset_name
        if mask.sum() == 0:
            continue
        per_dataset[dataset_name] = compute_classification_metrics(
            results["labels"][mask],
            results["predictions"][mask],
            results["probabilities"][mask],
            class_names=class_names,
            num_classes=len(class_names),
            focus_class="MEL" if "MEL" in class_names else None,
        )
        per_dataset[dataset_name]["num_samples"] = int(mask.sum())
    metrics["per_dataset"] = per_dataset

    # -- write the report ---------------------------------------------------
    report = {
        "summary": {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "melanoma_sensitivity": metrics["melanoma_focus"]["sensitivity"],
            "melanoma_specificity": metrics["melanoma_focus"]["specificity"],
            "num_samples": metrics["num_samples"],
        },
        "metrics": metrics,
        "class_names": class_names,
        "per_dataset_summary": {
            name: {
                "accuracy": values.get("accuracy"),
                "macro_f1": values.get("macro_f1"),
                "num_samples": values.get("num_samples"),
                "melanoma_sensitivity": (values.get("focus") or {}).get("sensitivity"),
            }
            for name, values in per_dataset.items()
        },
        "disclaimer": (
            "This is a research prototype. The numbers are experimental "
            "results on the listed dataset(s) and do NOT constitute a medical "
            "diagnosis or a clinically validated device."
        ),
    }

    json_path = output_dir / f"evaluation_{args.split}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    logger.info("metrics written -> %s", json_path)

    write_dataset_report(
        report,
        output_dir,
        stem=f"evaluation_report_{args.split}",
        class_names=class_names,
    )

    # -- plots ---------------------------------------------------------------
    if not args.no_plots:
        try:
            save_roc_and_pr_curves(
                results["labels"],
                results["probabilities"],
                class_names,
                output_dir,
                stem=f"curves_{args.split}",
            )
            from explainability.visualization import (
                save_confusion_matrix_plot,
                save_per_class_plot,
            )

            save_confusion_matrix_plot(
                np.array(metrics["confusion_matrix"]),
                class_names,
                output_dir / f"confusion_matrix_{args.split}.png",
            )
            save_per_class_plot(
                metrics["per_class"],
                output_dir / f"per_class_performance_{args.split}.png",
            )
            logger.info("plots written -> %s", output_dir)
        except Exception as exc:  # pragma: no cover
            logger.warning("Plot generation failed: %s", exc)

    print("\n" + "=" * 70)
    print(f"EVALUATION SUMMARY ({args.split} split)")
    print("=" * 70)
    print(f"  accuracy            : {metrics['accuracy']:.4f}")
    print(f"  balanced accuracy   : {metrics['balanced_accuracy']:.4f}")
    print(f"  macro F1            : {metrics['macro_f1']:.4f}")
    print(f"  weighted F1         : {metrics['weighted_f1']:.4f}")
    print(f"  melanoma sensitivity: {metrics['melanoma_focus']['sensitivity']:.4f}")
    print(f"  melanoma specificity: {metrics['melanoma_focus']['specificity']:.4f}")
    print(f"  samples             : {metrics['num_samples']}")
    if per_dataset:
        print("\n  per-dataset performance:")
        for name, values in per_dataset.items():
            print(
                f"    {name:<14} n={values['num_samples']:<6} "
                f"acc={values['accuracy']:.4f}  macroF1={values['macro_f1']:.4f}"
            )
    print("\n  NOTE: research prototype - not a medical device.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
