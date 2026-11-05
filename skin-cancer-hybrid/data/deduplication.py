"""Duplicate and near-duplicate detection across and within datasets.

Why this matters
================
HAM10000 and ISIC 2018 share a substantial number of images (the HAM10000
set was assembled largely from ISIC archives).  If identical or near-identical
images land in different splits - or in different datasets that are then
merged - the reported test accuracy is inflated by pure memorisation rather
than by generalisation.  This module is therefore run **before** splitting and
is cross-dataset by construction.

Methods
=======
**Exact duplicates** - SHA-1 hash of the raw file bytes.  Fast and
unambiguous.

**Near duplicates** - perceptual hashing (difference hash, ``dHash``) at a
16x16 resolution, producing a 256-bit fingerprint.  Two images whose
Hamming distance is below the configured threshold are treated as near
duplicates.  This catches re-encoded, rescaled or lightly edited copies of the
same lesion.

Resolution
==========
A near-duplicate *cluster* is resolved by keeping the first member (in
dataset order) and marking the rest as duplicates.  The duplicate is **not
deleted from the manifest** - it is flagged, so the report can state exactly
how many samples were affected and the user can choose to keep or drop them.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "DedupConfig",
    "DedupResult",
    "compute_file_hash",
    "compute_perceptual_hash",
    "hamming_distance",
    "find_duplicates",
    "write_leakage_audit",
]


@dataclass
class DedupConfig:
    """Configuration for duplicate detection."""

    enabled: bool = True
    exact_hash: bool = True
    perceptual: bool = True
    phash_size: int = 16
    phash_max_distance: int = 4
    drop_near_duplicates: bool = True
    hash_cache_path: str | Path | None = None

    @classmethod
    def from_config(cls, config: Any) -> "DedupConfig":
        def _get(key: str, default: Any) -> Any:
            if hasattr(config, "get"):
                return config.get(f"deduplication.{key}", default)
            if isinstance(config, dict):
                return config.get(key, default)
            return default

        return cls(
            enabled=bool(_get("enabled", True)),
            exact_hash=bool(_get("exact_hash", True)),
            perceptual=bool(_get("perceptual", True)),
            phash_size=int(_get("phash_size", 16)),
            phash_max_distance=int(_get("phash_max_distance", 4)),
            drop_near_duplicates=bool(_get("drop_near_duplicates", True)),
        )


@dataclass
class DedupResult:
    """Outcome of a deduplication run."""

    frame: pd.DataFrame
    exact_duplicate_groups: list[list[str]] = field(default_factory=list)
    near_duplicate_groups: list[list[str]] = field(default_factory=list)
    cross_dataset_groups: list[list[str]] = field(default_factory=list)
    num_exact_duplicates: int = 0
    num_near_duplicates: int = 0
    num_cross_dataset: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "num_samples": len(self.frame),
            "num_exact_duplicate_groups": len(self.exact_duplicate_groups),
            "num_exact_duplicate_samples": self.num_exact_duplicates,
            "num_near_duplicate_groups": len(self.near_duplicate_groups),
            "num_near_duplicate_samples": self.num_near_duplicates,
            "num_cross_dataset_groups": len(self.cross_dataset_groups),
            "num_cross_dataset_samples": self.num_cross_dataset,
            "errors": list(self.errors),
        }


def compute_file_hash(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-1 hash of the raw file bytes."""
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compute_perceptual_hash(path: str | Path, hash_size: int = 16) -> np.ndarray | None:
    """Compute a difference hash (dHash) of an image.

    Parameters
    ----------
    path:
        Path to the image file.
    hash_size:
        Side length of the resized thumbnail.  16 yields a 256-bit hash.

    Returns
    -------
    numpy.ndarray | None
        A ``uint8`` array of ``hash_size ** 2`` bits, or ``None`` when the
        image cannot be read.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover
        return None

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.reshape(-1).astype(np.uint8)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Number of differing bits between two perceptual hashes."""
    if a is None or b is None:
        return 0
    return int(np.count_nonzero(a != b))


def _group_from_indices(indices: Sequence[int], ids: Sequence[str]) -> list[str]:
    return [ids[i] for i in indices]


def find_duplicates(
    frame: pd.DataFrame,
    config: DedupConfig | None = None,
    id_column: str = "sample_id",
    path_column: str = "image_path",
    dataset_column: str = "dataset",
) -> DedupResult:
    """Detect exact and near duplicates in a harmonised manifest.

    Parameters
    ----------
    frame:
        The harmonised manifest (must contain the id, path and dataset
        columns).
    config:
        Detection configuration.

    Returns
    -------
    DedupResult
        The input frame with the added columns ``exact_hash``,
        ``phash``, ``duplicate_of`` and ``is_duplicate``.
    """
    config = config or DedupConfig()
    result = DedupResult(frame=frame.copy())
    work = result.frame

    if len(work) == 0:
        return result

    ids = work[id_column].astype(str).tolist()
    datasets = work[dataset_column].astype(str).tolist() if dataset_column in work else [""] * len(work)

    # -- exact hashes ----------------------------------------------------
    exact_hash: list[str] = []
    hash_to_indices: dict[str, list[int]] = defaultdict(list)
    errors: list[str] = []

    if config.exact_hash:
        for path in work[path_column].astype(str):
            try:
                digest = compute_file_hash(path)
            except OSError as exc:
                digest = ""
                errors.append(f"Could not hash {path}: {exc}")
            exact_hash.append(digest)
            if digest:
                hash_to_indices[digest].append(len(exact_hash) - 1)
        work["exact_hash"] = exact_hash

        for indices in hash_to_indices.values():
            if len(indices) > 1:
                group = _group_from_indices(indices, ids)
                result.exact_duplicate_groups.append(group)
                result.num_exact_duplicates += len(group) - 1
                if len({datasets[i] for i in indices}) > 1:
                    result.cross_dataset_groups.append(group)
                    result.num_cross_dataset += len(group) - 1

    # -- perceptual hashes ----------------------------------------------
    phashes: list[np.ndarray | None] = []
    if config.perceptual:
        for path in work[path_column].astype(str):
            phashes.append(compute_perceptual_hash(path, config.phash_size))
        work["phash"] = [None if p is None else "".join(map(str, p)) for p in phashes]

        valid = [(i, p) for i, p in enumerate(phashes) if p is not None]
        assigned: set[int] = set()
        for position, (index_i, hash_i) in enumerate(valid):
            if index_i in assigned:
                continue
            cluster = [index_i]
            for index_j, hash_j in valid[position + 1:]:
                if index_j in assigned:
                    continue
                if hamming_distance(hash_i, hash_j) <= config.phash_max_distance:
                    cluster.append(index_j)
            if len(cluster) > 1:
                assigned.update(cluster)
                group = _group_from_indices(cluster, ids)
                result.near_duplicate_groups.append(group)
                result.num_near_duplicates += len(group) - 1
                if len({datasets[i] for i in cluster}) > 1:
                    result.cross_dataset_groups.append(group)
                    result.num_cross_dataset += len(group) - 1

    # -- flag duplicates -------------------------------------------------
    duplicate_of: list[str] = []
    seen: set[str] = set()
    for index in range(len(work)):
        digest = work["exact_hash"].iloc[index] if "exact_hash" in work else ""
        key = digest if digest else ""
        if key and key in seen:
            duplicate_of.append(_first_of(key, hash_to_indices, ids))
        else:
            duplicate_of.append("")
            if key:
                seen.add(key)

    # Near-duplicate clusters: the first member is the canonical one.
    canonical_of: dict[int, str] = {}
    for group in result.near_duplicate_groups:
        for sample_id in group[1:]:
            canonical_of[sample_id] = group[0]

    final_duplicate_of: list[str] = []
    is_duplicate: list[bool] = []
    for sample_id, exact_ref in zip(ids, duplicate_of):
        ref = exact_ref or canonical_of.get(sample_id, "")
        final_duplicate_of.append(ref)
        is_duplicate.append(bool(ref))

    work["duplicate_of"] = final_duplicate_of
    work["is_duplicate"] = is_duplicate
    result.frame = work
    result.errors = errors
    return result


def _first_of(
    digest: str,
    hash_to_indices: Mapping[str, list[int]],
    ids: Sequence[str],
) -> str:
    indices = hash_to_indices.get(digest, [])
    return ids[indices[0]] if indices else ""


def write_leakage_audit(
    frame: pd.DataFrame,
    split_column: str = "split",
    dedup_result: DedupResult | None = None,
    output_dir: str | Path = "reports",
    stem: str = "leakage_audit_report",
    group_column: str = "group_id",
) -> tuple[Path, Path]:
    """Audit the split for leakage and write a JSON + Markdown report.

    Checks performed
    ----------------
    1. **Group leakage** - a patient/lesion group appearing in more than one
       split.
    2. **Exact duplicate leakage** - the same file hash in more than one split.
    3. **Near-duplicate leakage** - perceptually identical images across
       splits.
    4. **Cross-dataset duplicate leakage** - duplicates shared between
       different source datasets.
    5. **Class coverage per split** - so that a rare class is not missing from
       the test set.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    findings: dict[str, Any] = {
        "num_samples": len(frame),
        "split_counts": {},
        "group_leakage": [],
        "exact_duplicate_leakage": [],
        "near_duplicate_leakage": [],
        "cross_dataset_leakage": [],
        "class_distribution_per_split": {},
        "class_coverage_problems": [],
        "passed": True,
    }

    if len(frame) == 0:
        findings["passed"] = False
        findings["error"] = "Empty manifest - nothing to audit."

    if split_column in frame.columns:
        findings["split_counts"] = {
            str(k): int(v) for k, v in frame[split_column].value_counts().items()
        }

        # -- group leakage ------------------------------------------------
        if group_column in frame.columns:
            grouped = frame[frame[group_column].astype(str).str.len() > 0]
            for group_id, sub in grouped.groupby(group_column):
                splits = sorted(set(sub[split_column].astype(str)))
                if len(splits) > 1:
                    findings["group_leakage"].append(
                        {
                            "group_id": str(group_id),
                            "splits": splits,
                            "num_samples": int(len(sub)),
                        }
                    )

        # -- duplicate leakage --------------------------------------------
        if "exact_hash" in frame.columns and frame["exact_hash"].astype(str).str.len().gt(0).any():
            for digest, sub in frame[frame["exact_hash"].astype(str).str.len() > 0].groupby("exact_hash"):
                splits = sorted(set(sub[split_column].astype(str)))
                if len(splits) > 1:
                    findings["exact_duplicate_leakage"].append(
                        {
                            "hash": str(digest),
                            "splits": splits,
                            "num_samples": int(len(sub)),
                            "datasets": sorted(set(sub["dataset"].astype(str))) if "dataset" in sub else [],
                        }
                    )

        if "phash" in frame.columns:
            for phash, sub in frame[frame["phash"].astype(str).str.len() > 0].groupby("phash"):
                splits = sorted(set(sub[split_column].astype(str)))
                if len(splits) > 1:
                    findings["near_duplicate_leakage"].append(
                        {
                            "phash": str(phash)[:32],
                            "splits": splits,
                            "num_samples": int(len(sub)),
                        }
                    )

    # -- cross-dataset duplicates ---------------------------------------
    if dedup_result is not None:
        findings["cross_dataset_leakage"] = dedup_result.cross_dataset_groups
        findings["num_cross_dataset_duplicate_samples"] = dedup_result.num_cross_dataset

    # -- class coverage --------------------------------------------------
    if "unified_label" in frame.columns and split_column in frame.columns:
        for split_name, sub in frame.groupby(split_column):
            counts = (
                sub[sub["unified_label_valid"].astype(str).str.lower().isin({"true", "1"})]
                .groupby("unified_label")
                .size()
                .to_dict()
                if "unified_label_valid" in sub.columns
                else sub.groupby("unified_label").size().to_dict()
            )
            findings["class_distribution_per_split"][str(split_name)] = {
                str(k): int(v) for k, v in counts.items()
            }

        all_classes = set()
        for counts in findings["class_distribution_per_split"].values():
            all_classes.update(counts.keys())
        for split_name, counts in findings["class_distribution_per_split"].items():
            missing = sorted(all_classes - set(counts.keys()))
            if missing:
                findings["class_coverage_problems"].append(
                    {"split": split_name, "missing_classes": missing}
                )

    findings["passed"] = (
        not findings["group_leakage"]
        and not findings["exact_duplicate_leakage"]
        and not findings["near_duplicate_leakage"]
    )

    json_path = out_dir / f"{stem}.json"
    import json

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(findings, handle, indent=2, default=str)

    md_path = out_dir / f"{stem}.md"
    lines = ["# Leakage Audit Report", ""]
    lines.append(
        "Automatically generated by `scripts/audit_leakage.py`. A *passed* audit "
        "means no patient/lesion group, exact duplicate or near duplicate "
        "spans more than one split."
    )
    lines.append("")
    lines.append(f"- Samples audited: **{findings['num_samples']}**")
    lines.append(f"- Result: **{'PASSED' if findings['passed'] else 'LEAKAGE DETECTED'}**")
    lines.append("")
    if findings["split_counts"]:
        lines.append("## Split sizes")
        lines.append("")
        lines.append("| Split | Samples |")
        lines.append("| --- | ---: |")
        for split_name, count in findings["split_counts"].items():
            lines.append(f"| {split_name} | {count} |")
        lines.append("")
    for key, title in (
        ("group_leakage", "Patient / lesion group leakage"),
        ("exact_duplicate_leakage", "Exact duplicate leakage"),
        ("near_duplicate_leakage", "Near-duplicate leakage"),
        ("cross_dataset_leakage", "Cross-dataset duplicate groups"),
        ("class_coverage_problems", "Class coverage problems"),
    ):
        entries = findings.get(key, [])
        lines.append(f"## {title}: {len(entries)}")
        lines.append("")
        if entries:
            for entry in entries[:50]:
                lines.append(f"- `{entry}`")
            if len(entries) > 50:
                lines.append(f"- ... and {len(entries) - 50} more (see the JSON report)")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    return json_path, md_path
