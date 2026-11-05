"""Validate the datasets and the prepared manifest.

Usage
-----
::

    python scripts/validate_datasets.py --data-dir data/raw
    python scripts/validate_datasets.py --check-images --limit 200

Checks performed
================
* the dataset folders exist and contain images,
* the metadata files can be parsed,
* every metadata row resolves to an existing image file,
* the image files are readable and have a sane size,
* (for ISIC 2018) every segmentation mask exists and matches its image,
* the manifest is internally consistent (unique ids, valid class indices),
* no sample appears in more than one split.

The script exits with a non-zero status when a **blocking** problem is
found, so it can be used as a CI gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loaders import LOADERS


def validate_images(paths: list[str], limit: int | None = None) -> dict[str, Any]:
    """Check that the image files exist, are readable and have a sane size."""
    import cv2

    checked = paths if limit is None else paths[:limit]
    missing: list[str] = []
    unreadable: list[str] = []
    too_small: list[str] = []
    shapes: dict[str, int] = {}

    for path in checked:
        file_path = Path(path)
        if not file_path.exists():
            missing.append(path)
            continue
        try:
            if file_path.stat().st_size < 1024:
                too_small.append(path)
                continue
            image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
            if image is None:
                unreadable.append(path)
                continue
            key = f"{image.shape[0]}x{image.shape[1]}"
            shapes[key] = shapes.get(key, 0) + 1
        except Exception:
            unreadable.append(path)

    return {
        "checked": len(checked),
        "missing": len(missing),
        "unreadable": len(unreadable),
        "too_small": len(too_small),
        "distinct_shapes": len(shapes),
        "examples_missing": missing[:5],
        "examples_unreadable": unreadable[:5],
    }


def validate_manifest(manifest_path: Path) -> dict[str, Any]:
    """Check the internal consistency of the prepared manifest."""
    import pandas as pd

    issues: list[str] = []
    if not manifest_path.exists():
        return {"exists": False, "issues": ["manifest not found"], "ok": False}

    frame = pd.read_csv(manifest_path, keep_default_na=False, dtype=str)

    if frame["sample_id"].duplicated().any():
        duplicates = frame["sample_id"][frame["sample_id"].duplicated()].unique()
        issues.append(f"{len(duplicates)} duplicated sample_id values")

    invalid_class = 0
    for value in frame["class_index"]:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if index < -1:
            invalid_class += 1
    if invalid_class:
        issues.append(f"{invalid_class} rows with an invalid class index")

    if "split" in frame.columns:
        allowed = {"train", "val", "test", ""}
        unexpected = set(frame["split"]) - allowed
        if unexpected:
            issues.append(f"unexpected split values: {unexpected}")
        empty_split = (frame["split"] == "").sum()
        if empty_split:
            issues.append(f"{empty_split} rows without a split assignment")

    # Cross-split group leakage.
    if "group_id" in frame.columns and "split" in frame.columns:
        grouped = frame[frame["group_id"].astype(str).str.len() > 0]
        leaking = 0
        for _, sub in grouped.groupby("group_id"):
            if sub["split"].nunique() > 1:
                leaking += 1
        if leaking:
            issues.append(f"{leaking} patient/lesion groups span multiple splits")

    return {
        "exists": True,
        "num_rows": int(len(frame)),
        "datasets": sorted(set(frame["dataset"])),
        "splits": {
            str(k): int(v) for k, v in frame["split"].value_counts().items()
        } if "split" in frame.columns else {},
        "issues": issues,
        "ok": not issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the datasets.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--datasets", nargs="*", default=None,
                        choices=["ham10000", "isic2018", "isic2019", "pad_ufes20"])
    parser.add_argument("--check-images", action="store_true",
                        help="Read and verify the image files (slow on large datasets).")
    parser.add_argument("--limit", type=int, default=200,
                        help="Maximum number of images to verify per dataset.")
    parser.add_argument("--include-scc", action="store_true")
    args = parser.parse_args()

    from utils.config import load_config

    config_path = Path(args.config) if args.config else ROOT / "configs" / "default.yaml"
    config = load_config(config_path, root=ROOT)
    if args.data_dir:
        config.set("paths.raw_dir", args.data_dir)

    from data.pipeline import dataset_roots_from_config

    roots = dataset_roots_from_config(config, args.data_dir)
    dataset_names = args.datasets or list(LOADERS)

    print("=" * 78)
    print("DATASET VALIDATION")
    print("=" * 78)

    blocking_issues: list[str] = []
    report: dict[str, Any] = {"datasets": {}}

    for name in dataset_names:
        root = roots[name]
        print(f"\n--- {name} ---")
        print(f"  root: {root}")

        if not root.exists():
            print("  status: NOT FOUND (skipped)")
            report["datasets"][name] = {"present": False}
            continue

        loader = LOADERS[name](root=root, include_scc=args.include_scc)
        try:
            manifest = loader.build_manifest()
        except Exception as exc:
            print(f"  status: FAILED - {type(exc).__name__}: {exc}")
            blocking_issues.append(f"{name}: {exc}")
            report["datasets"][name] = {"present": True, "error": str(exc)}
            continue

        print(f"  samples      : {len(manifest)}")
        print(f"  classes      : {manifest.class_list}")

        distribution = manifest.class_distribution()
        if distribution:
            print("  distribution :")
            for code, count in distribution.items():
                print(f"      {code:<8} {count:>7}")

        dataset_report: dict[str, Any] = {
            "present": True,
            "num_samples": len(manifest),
            "class_list": manifest.class_list,
            "class_distribution": distribution,
            "notes": manifest.notes,
        }

        # -- image verification ---------------------------------------------
        if args.check_images:
            paths = [record.image_path for record in manifest.records]
            image_report = validate_images(paths, args.limit)
            dataset_report["images"] = image_report
            print(
                f"  images       : checked {image_report['checked']}, "
                f"missing {image_report['missing']}, "
                f"unreadable {image_report['unreadable']}"
            )
            if image_report["missing"] or image_report["unreadable"]:
                blocking_issues.append(
                    f"{name}: {image_report['missing']} missing and "
                    f"{image_report['unreadable']} unreadable image files"
                )

        # -- segmentation pairs ---------------------------------------------
        if name == "isic2018":
            try:
                pairs = loader.build_segmentation_pairs()
                dataset_report["num_segmentation_pairs"] = len(pairs)
                print(f"  segmentation : {len(pairs)} image/mask pairs")
            except Exception as exc:
                dataset_report["segmentation_error"] = str(exc)
                print(f"  segmentation : unavailable ({exc})")

        report["datasets"][name] = dataset_report

    # -- manifest validation -------------------------------------------------
    splits_dir = Path(config.get("paths.splits_dir", "data/splits"))
    candidates = sorted(splits_dir.glob("split_manifest*.csv"))
    split_path = (
        max(candidates, key=lambda p: p.stat().st_mtime) if candidates
        else splits_dir / "split_manifest.csv"
    )
    print(f"\n--- manifest ---\n  {split_path}")
    manifest_report = validate_manifest(split_path)
    report["manifest"] = manifest_report
    if manifest_report.get("exists"):
        print(f"  rows    : {manifest_report['num_rows']}")
        print(f"  splits  : {manifest_report['splits']}")
        for issue in manifest_report["issues"]:
            print(f"  ISSUE   : {issue}")
        if not manifest_report["ok"]:
            blocking_issues.extend(f"manifest: {i}" for i in manifest_report["issues"])
    else:
        print("  not prepared yet - run scripts/prepare_datasets.py")

    # -- write the report ----------------------------------------------------
    report_dir = Path(config.get("paths.reports_dir", "reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "dataset_validation_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"\nreport -> {report_path}")

    print("\n" + "=" * 78)
    if blocking_issues:
        print(f"VALIDATION FAILED - {len(blocking_issues)} blocking issue(s):")
        for issue in blocking_issues:
            print(f"  - {issue}")
        return 1

    print("VALIDATION PASSED")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
