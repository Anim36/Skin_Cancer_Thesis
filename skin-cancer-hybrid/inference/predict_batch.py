"""Batch inference over a directory of images.

Usage
-----
::

    python inference/predict_batch.py --checkpoint runs/hybrid/checkpoints/best.pth \
        --image-dir data/raw/ham10000/HAM10000_images --output-dir runs/predictions

    python inference/predict_batch.py --checkpoint ... --image-dir images/ \
        --metadata metadata.csv --output-dir runs/predictions

The optional ``--metadata`` CSV must contain an ``image`` (or ``image_id``)
column that matches the image file name, plus the clinical columns
``age``, ``sex``, ``localization`` and ``skin_cancer_history``.  Rows without
a matching image are reported and skipped.

Outputs
=======
* ``predictions.csv`` - one row per image with the class, the confidence and
  the full probability vector.
* ``predictions.json`` - the same content as JSON.
* ``gradcam/`` - one Grad-CAM figure per image when ``--gradcam`` is used.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.predict import DISCLAIMER, SkinCancerPredictor

__all__ = ["main", "batch_predict"]

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _load_metadata_csv(path: Path) -> dict[str, dict[str, Any]]:
    """Read the optional metadata CSV into a lookup keyed by image name."""
    import pandas as pd

    frame = pd.read_csv(path, keep_default_na=False, dtype=str)
    id_column = None
    for candidate in ("image", "image_id", "image_name", "filename", "file"):
        if candidate in frame.columns:
            id_column = candidate
            break
    if id_column is None:
        raise ValueError(
            f"{path} has no image identifier column (expected one of: image, "
            "image_id, image_name, filename, file)."
        )

    lookup: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        key = str(row[id_column]).strip()
        if not key:
            continue
        lookup[key] = {k: v for k, v in row.items() if k != id_column}
        lookup[Path(key).stem] = lookup[key]
    return lookup


def batch_predict(
    predictor: SkinCancerPredictor,
    image_paths: list[Path],
    metadata_lookup: dict[str, dict[str, Any]] | None = None,
    gradcam: bool = False,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Predict a batch of images and optionally save the explanations."""
    results: list[dict[str, Any]] = []

    for index, image_path in enumerate(image_paths, start=1):
        metadata = None
        if metadata_lookup:
            metadata = metadata_lookup.get(image_path.name) or metadata_lookup.get(
                image_path.stem
            )

        try:
            result = predictor.predict(image_path, metadata=metadata, gradcam=False)
        except Exception as exc:
            print(f"[{index}/{len(image_paths)}] FAILED {image_path}: {exc}", file=sys.stderr)
            continue

        row: dict[str, Any] = {
            "image": str(image_path),
            "image_name": image_path.name,
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "metadata_used": result["metadata_used"],
        }
        for name, probability in result["probabilities"].items():
            row[f"prob_{name}"] = probability

        if gradcam and output_dir is not None:
            try:
                explanation = predictor.explain(image_path, metadata)
                written = predictor.save_prediction(
                    result, output_dir / "gradcam", stem=image_path.stem, gradcam=explanation
                )
                row["gradcam_figure"] = written.get("gradcam_figure", "")
            except Exception as exc:
                row["gradcam_figure"] = ""
                print(f"[{index}] Grad-CAM failed for {image_path}: {exc}", file=sys.stderr)

        results.append(row)
        if index % 25 == 0 or index == len(image_paths):
            print(f"[{index}/{len(image_paths)}] processed", flush=True)

    return results


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Batch prediction over a directory.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-dir", required=True, help="Directory with the images.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--metadata", default=None, help="Optional metadata CSV.")
    parser.add_argument("--output-dir", default="runs/predictions")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM figures.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N images.")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Image directory not found: {image_dir}", file=sys.stderr)
        return 2

    image_paths = sorted(
        p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        print(f"No images found in {image_dir}", file=sys.stderr)
        return 2

    print(f"Found {len(image_paths)} images in {image_dir}")

    metadata_lookup = None
    if args.metadata:
        metadata_lookup = _load_metadata_csv(Path(args.metadata))
        print(f"Loaded metadata for {len(metadata_lookup)} entries")

    predictor = SkinCancerPredictor.from_checkpoint(
        args.checkpoint, config_path=args.config, device=args.device
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = batch_predict(
        predictor,
        image_paths,
        metadata_lookup=metadata_lookup,
        gradcam=args.gradcam,
        output_dir=output_dir,
    )

    if not results:
        print("No predictions were produced.", file=sys.stderr)
        return 1

    csv_path = output_dir / "predictions.csv"
    fieldnames = list(results[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    json_path = output_dir / "predictions.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"predictions": results, "disclaimer": DISCLAIMER},
            handle,
            indent=2,
            default=str,
        )

    # -- summary ------------------------------------------------------------
    counts: dict[str, int] = {}
    for row in results:
        counts[row["predicted_class"]] = counts.get(row["predicted_class"], 0) + 1

    summary = {
        "num_images": len(results),
        "predicted_class_counts": counts,
        "mean_confidence": float(np.mean([r["confidence"] for r in results])),
        "disclaimer": DISCLAIMER,
    }
    summary_path = output_dir / "predictions_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"Processed {len(results)} images")
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<10} {count}")
    print(f"  mean confidence: {summary['mean_confidence']:.4f}")
    print(f"\n{DISCLAIMER}")
    print("=" * 70)
    print(f"results -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
