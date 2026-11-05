"""Audit the prepared split for data leakage.

Usage
-----
::

    python scripts/audit_leakage.py
    python scripts/audit_leakage.py --data-dir data/raw --rebuild

The audit checks:

1. **Group leakage** - a patient/lesion group appearing in more than one split.
2. **Exact duplicate leakage** - the same file hash in more than one split.
3. **Near-duplicate leakage** - perceptually identical images across splits.
4. **Cross-dataset duplicates** - images shared between two source datasets.
5. **Class coverage** - a class that is missing from one of the splits.

The report is written to ``reports/leakage_audit_report.md`` and ``.json``.
The script exits with a non-zero status when leakage is detected, so it can
gate a training run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.deduplication import DedupConfig, find_duplicates, write_leakage_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the split for data leakage.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split-file", default=None, help="Path to the split manifest.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Recompute the duplicate hashes before auditing.")
    parser.add_argument("--include-scc", action="store_true")
    args = parser.parse_args()

    from utils.config import load_config

    config_path = Path(args.config) if args.config else ROOT / "configs" / "default.yaml"
    config = load_config(config_path, root=ROOT)
    if args.data_dir:
        config.set("paths.raw_dir", args.data_dir)

    splits_dir = Path(config.get("paths.splits_dir", "data/splits"))
    if args.split_file:
        split_path = Path(args.split_file)
    else:
        # Accept any prepared split manifest; the newest one is used.
        candidates = sorted(splits_dir.glob("split_manifest*.csv"))
        split_path = (
            max(candidates, key=lambda p: p.stat().st_mtime) if candidates
            else splits_dir / "split_manifest.csv"
        )

    print("=" * 78)
    print("LEAKAGE AUDIT")
    print("=" * 78)

    if not split_path.exists():
        print(f"\nSplit manifest not found: {split_path}")
        print("Run scripts/prepare_datasets.py first.")
        return 1

    import pandas as pd

    frame = pd.read_csv(split_path, keep_default_na=False, dtype=str)
    print(f"  split file : {split_path}")
    print(f"  rows       : {len(frame)}")

    # -- (re)compute the duplicate columns ---------------------------------
    dedup_result = None
    if args.rebuild or "exact_hash" not in frame.columns:
        print("  computing duplicate hashes (this can take a while)...")
        dedup_config = DedupConfig.from_config(config)
        dedup_result = find_duplicates(frame, dedup_config)
        frame = dedup_result.frame
        frame.to_csv(split_path, index=False)
        print(f"  hashes written back to {split_path}")
        print(
            f"  exact duplicates: {dedup_result.num_exact_duplicates}, "
            f"near duplicates: {dedup_result.num_near_duplicates}, "
            f"cross-dataset: {dedup_result.num_cross_dataset}"
        )
    elif "exact_hash" in frame.columns:
        # Recompute the grouping summary from the stored hashes.
        dedup_result = find_duplicates(
            frame.assign(
                exact_hash=frame["exact_hash"].fillna(""),
                phash=frame["phash"].fillna("") if "phash" in frame.columns else "",
            ),
            DedupConfig.from_config(config),
        )

    # -- run the audit --------------------------------------------------------
    report_dir = Path(config.get("paths.reports_dir", "reports"))
    json_path, md_path = write_leakage_audit(
        frame,
        dedup_result=dedup_result,
        output_dir=report_dir,
    )

    with json_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)

    print("\n" + "-" * 78)
    print("RESULTS")
    print("-" * 78)
    print(f"  samples audited           : {audit['num_samples']}")
    print(f"  split sizes               : {audit['split_counts']}")
    print(f"  group leakage             : {len(audit['group_leakage'])}")
    print(f"  exact duplicate leakage   : {len(audit['exact_duplicate_leakage'])}")
    print(f"  near-duplicate leakage    : {len(audit['near_duplicate_leakage'])}")
    print(f"  cross-dataset duplicates  : {len(audit['cross_dataset_leakage'])}")
    print(f"  class coverage problems   : {len(audit['class_coverage_problems'])}")
    print()

    for split_name, counts in audit["class_distribution_per_split"].items():
        print(f"  {split_name}:")
        for code, count in counts.items():
            print(f"      {code:<8} {count:>7}")

    print(f"\n  report -> {md_path}")
    print(f"  report -> {json_path}")

    if audit["passed"]:
        print("\n" + "=" * 78)
        print("AUDIT PASSED - no leakage detected.")
        print("=" * 78)
        return 0

    print("\n" + "=" * 78)
    print("AUDIT FAILED - leakage detected. See the report for the details.")
    print("=" * 78)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
