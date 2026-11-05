"""Inspect the datasets that are present on disk and report their structure.

Usage
-----
::

    python scripts/inspect_datasets.py --data-dir data/raw
    python scripts/inspect_datasets.py --data-dir data/raw --include-scc

This script is the **discovery** step of the project.  It does not assume
anything about the datasets: it opens the real metadata files, reports what
columns and classes it actually found, and writes the dataset compatibility
report.  Everything the harmonisation code later relies on comes from this
inspection.

Outputs (in ``reports/``)
-------------------------
* ``dataset_compatibility_report.md`` / ``.json`` - the label-space analysis.
* ``dataset_inspection.json`` - the raw observed structure of each dataset.

The script works with whatever subset of the four datasets is present.
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

from data.dataset_harmonization import (
    build_compatibility_report,
    harmonise_manifests,
    write_compatibility_report,
)
from data.label_mapping import CANONICAL_CLASSES, CLASS_DESCRIPTIONS
from data.loaders import LOADERS


def inspect_dataset(name: str, root: Path, include_scc: bool) -> dict[str, Any]:
    """Inspect one dataset and return a structured description of it."""
    info: dict[str, Any] = {
        "dataset": name,
        "root": str(root),
        "exists": root.exists(),
        "available": False,
        "error": None,
    }

    if not root.exists():
        return info

    loader = LOADERS[name](root=root, include_scc=include_scc)
    try:
        info["available"] = loader.is_available()
    except Exception as exc:
        info["error"] = str(exc)
        return info

    if not info["available"]:
        return info

    try:
        manifest = loader.build_manifest()
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info

    info["num_samples"] = len(manifest)
    info["notes"] = manifest.notes

    native_counts: dict[str, int] = {}
    metadata_fields: set[str] = set()
    samples_with_metadata = 0
    group_types: dict[str, int] = {}

    for record in manifest.records:
        native_counts[record.native_label] = native_counts.get(record.native_label, 0) + 1
        group_types[record.group_type] = group_types.get(record.group_type, 0) + 1
        if record.has_metadata:
            samples_with_metadata += 1
            metadata_fields.update(record.metadata.keys())

    info["native_label_counts"] = dict(
        sorted(native_counts.items(), key=lambda kv: -kv[1])
    )
    info["num_native_classes"] = len(native_counts)
    info["group_types"] = group_types
    info["num_with_metadata"] = samples_with_metadata
    info["metadata_fields"] = sorted(metadata_fields)
    info["class_list"] = manifest.class_list

    # -- segmentation pairs (ISIC 2018) ------------------------------------
    if name == "isic2018":
        try:
            pairs = loader.build_segmentation_pairs()
            info["num_segmentation_pairs"] = len(pairs)
        except Exception as exc:
            info["num_segmentation_pairs"] = 0
            info["segmentation_error"] = str(exc)

    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the datasets on disk.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-dir", default=None, help="Directory containing the datasets.")
    parser.add_argument("--include-scc", action="store_true",
                        help="Use the 8-class label space.")
    parser.add_argument("--output-dir", default=None, help="Where to write the reports.")
    args = parser.parse_args()

    from utils.config import load_config

    config_path = Path(args.config) if args.config else ROOT / "configs" / "default.yaml"
    config = load_config(config_path, root=ROOT)
    if args.data_dir:
        config.set("paths.raw_dir", args.data_dir)

    output_dir = Path(args.output_dir or config.get("paths.reports_dir", "reports"))
    output_dir.mkdir(parents=True, exist_ok=True)

    from data.pipeline import dataset_roots_from_config

    roots = dataset_roots_from_config(config, args.data_dir)

    print("=" * 78)
    print("DATASET INSPECTION")
    print("=" * 78)

    inspections: list[dict[str, Any]] = []
    manifests = []

    for name in LOADERS:
        root = roots[name]
        info = inspect_dataset(name, root, args.include_scc)
        inspections.append(info)

        status = "OK" if info.get("available") else "MISSING"
        print(f"\n--- {name} ({status}) ---")
        print(f"    root: {root}")

        if info.get("error"):
            print(f"    error: {info['error']}")
            continue
        if not info.get("available"):
            print("    (not present on disk - see DATASET_GUIDE.md)")
            continue

        print(f"    samples: {info.get('num_samples', 0)}")
        print(f"    native classes: {info.get('num_native_classes', 0)}")
        for label, count in (info.get("native_label_counts") or {}).items():
            canonical = _canonical_for(label)
            suffix = f"  -> {canonical}" if canonical else "  -> UNMAPPED"
            print(f"      {label:<28} {count:>7}{suffix}")
        print(f"    grouping: {info.get('group_types', {})}")
        if info.get("metadata_fields"):
            print(f"    metadata fields: {info['metadata_fields']}")
            print(f"    samples with metadata: {info.get('num_with_metadata', 0)}")
        if "num_segmentation_pairs" in info:
            print(f"    segmentation pairs: {info['num_segmentation_pairs']}")
        for note in info.get("notes", []):
            print(f"    note: {note}")

        # Build the manifest for the harmonisation report.
        try:
            loader = LOADERS[name](root=root, include_scc=args.include_scc)
            manifests.append(loader.build_manifest())
        except Exception as exc:
            print(f"    (manifest build failed: {exc})")

    # -- inspection JSON ----------------------------------------------------
    inspection_path = output_dir / "dataset_inspection.json"
    with inspection_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "raw_dir": str(config.get("paths.raw_dir", "data/raw")),
                "include_scc": args.include_scc,
                "datasets": inspections,
            },
            handle,
            indent=2,
            default=str,
        )
    print(f"\ninspection written -> {inspection_path}")

    # -- compatibility report ----------------------------------------------
    if manifests:
        harmonisation = harmonise_manifests(manifests)
        report = build_compatibility_report(harmonisation, include_scc=args.include_scc)
        json_path, md_path = write_compatibility_report(report, output_dir)
        print(f"compatibility report -> {md_path}")
        print(f"compatibility report -> {json_path}")

        print("\n" + "=" * 78)
        print("CLASS COVERAGE")
        print("=" * 78)
        print(f"{'class':<8}{'description':<48}datasets")
        for code, sources in report["class_coverage"].items():
            description = CLASS_DESCRIPTIONS.get(code, code)[:46]
            print(f"{code:<8}{description:<48}{', '.join(sources) or '-'}")
        if report["warnings"]:
            print("\nWARNINGS:")
            for warning in report["warnings"]:
                print(f"  - {warning}")
    else:
        print("\nNo dataset could be loaded, so no compatibility report was produced.")
        print("See DATASET_GUIDE.md for the expected layout of each dataset.")

    return 0


def _canonical_for(label: str) -> str | None:
    """Resolve a native label to its canonical code (for display)."""
    from data.label_mapping import normalise_label, _ALIAS_LOOKUP

    return _ALIAS_LOOKUP.get(normalise_label(label))


if __name__ == "__main__":
    raise SystemExit(main())
