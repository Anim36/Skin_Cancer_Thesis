"""Prepare the datasets: build the manifest, deduplicate and create the split.

Usage
-----
::

    python scripts/prepare_datasets.py --data-dir data/raw
    python scripts/prepare_datasets.py --data-dir data/raw --include-scc
    python scripts/prepare_datasets.py --data-dir data/raw --force

This is the script to run **once** after downloading the datasets and again
whenever the data changes.  It produces:

* ``data/processed/<dataset>/manifest.csv`` - the harmonised manifest per
  dataset.
* ``data/processed/combined_manifest.csv`` - all datasets in one table.
* ``data/splits/split_manifest_<selection>.csv`` - the train/val/test
  assignment.  The file name encodes the dataset selection, so a split made
  for one subset of the datasets can never be reused for another.
* ``reports/leakage_audit_report.md`` / ``.json`` - the leakage audit.

The split file is reused by every later run so that all experiments and all
ablation variants are compared on exactly the same partition.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.pipeline import prepare_data
from data.deduplication import write_leakage_audit
from data.splitting import load_split_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the datasets.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-dir", default=None, help="Directory containing the datasets.")
    parser.add_argument("--include-scc", action="store_true",
                        help="Use the 8-class label space so SCC samples are used.")
    parser.add_argument("--datasets", nargs="*", default=None,
                        choices=["ham10000", "isic2018", "isic2019", "pad_ufes20"])
    parser.add_argument("--force", action="store_true",
                        help="Recreate the split even if one already exists.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    from utils.config import load_config

    config_path = Path(args.config) if args.config else ROOT / "configs" / "default.yaml"
    config = load_config(config_path, root=ROOT)
    if args.data_dir:
        config.set("paths.raw_dir", args.data_dir)

    print("=" * 78)
    print("DATASET PREPARATION")
    print("=" * 78)

    try:
        prepared = prepare_data(
            config,
            datasets=args.datasets,
            raw_dir=args.data_dir,
            include_scc=args.include_scc,
            reuse_split=not args.force,
        )
    except RuntimeError as exc:
        print(f"\nPREPARATION FAILED:\n{exc}")
        return 1

    for message in prepared.warnings:
        print(f"  ! {message}")

    processed_dir = Path(config.get("paths.processed_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)

    # -- per-dataset manifests ----------------------------------------------
    manifest = prepared.manifest
    for dataset_name, sub in manifest.groupby("dataset"):
        target = processed_dir / dataset_name
        target.mkdir(parents=True, exist_ok=True)
        path = target / "manifest.csv"
        sub.to_csv(path, index=False)
        print(f"  wrote {path} ({len(sub)} rows)")

    combined_path = processed_dir / "combined_manifest.csv"
    manifest.to_csv(combined_path, index=False)
    print(f"  wrote {combined_path} ({len(manifest)} rows)")

    # -- split manifest -----------------------------------------------------
    split_path = prepared.split_path
    print(f"\n  split file: {split_path}")
    counts = prepared.split_result.counts()
    for split_name, count in counts.items():
        print(f"    {split_name:<6} {count}")

    distribution = prepared.split_result.class_distribution()
    if distribution:
        print("\n  class distribution per split:")
        classes = sorted({c for counts in distribution.values() for c in counts})
        header = "    " + " " * 8 + "".join(f"{name:>9}" for name in classes)
        print(header)
        for split_name, counts in distribution.items():
            row = "    " + f"{split_name:<8}"
            for class_name in classes:
                row += f"{counts.get(class_name, 0):>9}"
            print(row)

    # -- leakage audit ------------------------------------------------------
    report_dir = Path(config.get("paths.reports_dir", "reports"))
    json_path, md_path = write_leakage_audit(
        manifest,
        dedup_result=prepared.dedup,
        output_dir=report_dir,
    )
    print(f"\n  leakage audit -> {md_path}")
    print(f"  leakage audit -> {json_path}")

    with json_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    status = "PASSED" if audit.get("passed") else "LEAKAGE DETECTED"
    print(f"  leakage audit status: {status}")
    if not audit.get("passed"):
        for key in ("group_leakage", "exact_duplicate_leakage", "near_duplicate_leakage"):
            if audit.get(key):
                print(f"    {key}: {len(audit[key])} issue(s)")

    # -- summary ------------------------------------------------------------
    summary = {
        "prepared": prepared.summary(),
        "leakage_audit": audit,
        "deduplication": prepared.dedup.summary() if prepared.dedup else None,
        "split_path": str(split_path),
    }
    summary_path = processed_dir / "preparation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print("\n" + "=" * 78)
    print("READY")
    print("=" * 78)
    print("Next: python training/train_hybrid.py --data-dir", args.data_dir or "data/raw")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
