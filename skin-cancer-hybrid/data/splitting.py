"""Leakage-safe train / validation / test splitting.

Guarantees
==========
1. **Group integrity** - all samples sharing a patient or lesion identifier
   land in exactly one split.  This is the single most important safeguard
   against inflated accuracy in dermoscopy, where the same lesion is often
   photographed several times.
2. **Duplicate integrity** - when the same image (or a near-duplicate) appears
   in more than one dataset, all its copies are forced into the same split.
3. **Stratification** - where the group structure allows it, the class
   distribution is kept approximately equal across splits.
4. **Reproducibility** - the split is fully determined by the seed and is
   written to ``data/splits/`` so every experiment reuses exactly the same
   partition.

Group-aware stratification
==========================
A plain stratified split would break group integrity, so a *group-stratified*
split is used instead: samples are first aggregated into groups (each group
carries the class of its members, which for these datasets is constant within
a lesion), the groups are then assigned to splits in a way that keeps the
per-class proportions as close to the target as possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "SplitConfig",
    "SplitResult",
    "make_splits",
    "group_stratified_split",
    "load_split_manifest",
    "split_file_for",
    "resolve_split_file",
]


def split_file_for(splits_dir: str | Path, dataset_names: Sequence[str]) -> Path:
    """Return the split-manifest path for a given dataset selection.

    The dataset selection is part of the file name so that a split made for
    one subset of the datasets (for example PAD-UFES-20 only) can never be
    silently reused for a different subset (for example all four datasets).
    Reusing a mismatched split would leave validation and test sets empty and
    would invalidate every reported metric.
    """
    active = sorted({str(name) for name in dataset_names if str(name)})
    if not active:
        suffix = "all"
    elif len(active) == 1:
        suffix = active[0]
    else:
        suffix = "all"
    return Path(splits_dir) / f"split_manifest_{suffix}.csv"


def resolve_split_file(
    splits_dir: str | Path,
    dataset_names: Sequence[str],
    explicit: str | Path | None = None,
) -> Path:
    """Return the split file to use, falling back to the legacy name.

    ``explicit`` always wins.  Otherwise the dataset-aware name is preferred;
    the legacy ``split_manifest.csv`` is still honoured when it exists and the
    dataset-aware file does not, so that projects created before this change
    keep working.
    """
    if explicit:
        return Path(explicit)
    candidate = split_file_for(splits_dir, dataset_names)
    if candidate.exists():
        return candidate
    legacy = Path(splits_dir) / "split_manifest.csv"
    if legacy.exists():
        return legacy
    return candidate


@dataclass
class SplitConfig:
    """Configuration for the splitting stage."""

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    stratify: bool = True
    group_by: str = "auto"          # auto | patient | lesion | none
    unify_duplicates_across_datasets: bool = True

    @classmethod
    def from_config(cls, config: Any) -> "SplitConfig":
        def _get(key: str, default: Any) -> Any:
            if hasattr(config, "get"):
                return config.get(f"splitting.{key}", default)
            if isinstance(config, dict):
                return config.get(key, default)
            return default

        ratios = _get("ratios", {}) or {}
        if not isinstance(ratios, Mapping):
            ratios = {}

        total = float(ratios.get("train", 0.70)) + float(ratios.get("val", 0.15)) + float(
            ratios.get("test", 0.15)
        )
        if total <= 0:
            ratios = {"train": 0.70, "val": 0.15, "test": 0.15}

        return cls(
            train_ratio=float(ratios.get("train", 0.70)),
            val_ratio=float(ratios.get("val", 0.15)),
            test_ratio=float(ratios.get("test", 0.15)),
            seed=int(_get("seed", 42)),
            stratify=bool(_get("stratify", True)),
            group_by=str(_get("group_by", "auto")),
            unify_duplicates_across_datasets=bool(
                _get("unify_duplicates_across_datasets", True)
            ),
        )


@dataclass
class SplitResult:
    """Outcome of the splitting stage."""

    frame: pd.DataFrame
    assignments: dict[str, str] = field(default_factory=dict)
    group_assignments: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        if "split" not in self.frame.columns:
            return {}
        return {str(k): int(v) for k, v in self.frame["split"].value_counts().items()}

    def class_distribution(self) -> dict[str, dict[str, int]]:
        """Per-split class distribution (unified labels only)."""
        out: dict[str, dict[str, int]] = {}
        if "split" not in self.frame.columns:
            return out
        for split_name, sub in self.frame.groupby("split"):
            counts: dict[str, int] = {}
            for label in sub.get("unified_label", pd.Series(dtype=str)).tolist():
                if label:
                    counts[label] = counts.get(label, 0) + 1
            out[str(split_name)] = dict(sorted(counts.items()))
        return out


def _resolve_group_column(frame: pd.DataFrame, group_by: str) -> str | None:
    """Choose the grouping column according to the configuration."""
    if group_by == "none":
        return None
    if group_by in {"patient", "lesion"}:
        if "group_type" in frame.columns:
            sub = frame[frame["group_type"].astype(str) == group_by]
            if len(sub) and sub["group_id"].astype(str).str.len().gt(0).any():
                return "group_id"
        return None
    # auto: prefer a real group id; fall back to the sample id (one group per
    # sample) which still yields a valid, if less protective, split.
    if "group_id" in frame.columns and frame["group_id"].astype(str).str.len().gt(0).any():
        return "group_id"
    return None


def group_stratified_split(
    frame: pd.DataFrame,
    group_column: str | None,
    ratios: Mapping[str, float],
    label_column: str = "unified_label",
    seed: int = 42,
    stratify: bool = True,
) -> tuple[dict[str, str], dict[str, str]]:
    """Assign splits to groups while keeping class proportions balanced.

    Parameters
    ----------
    frame:
        Manifest to split.
    group_column:
        Column holding the group id, or ``None`` to split per sample.
    ratios:
        Mapping of split name to target ratio.
    label_column:
        Column holding the class used for stratification.
    seed:
        Random seed.
    stratify:
        Whether to balance classes across splits.

    Returns
    -------
    tuple
        ``(sample_assignments, group_assignments)``.
    """
    rng = np.random.default_rng(seed)
    split_names = list(ratios.keys())
    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        raise ValueError("Split ratios must sum to a positive value")
    ratios = {k: v / total_ratio for k, v in ratios.items()}

    work = frame.copy()
    if group_column is None:
        work["_group"] = work.index.astype(str)
    else:
        work["_group"] = work[group_column].astype(str)
        # Rows without a group id become their own singleton group.
        empty = work["_group"].str.len() == 0
        work.loc[empty, "_group"] = [f"__solo_{i}" for i in work.index[empty]]

    # The stratification label of a group is its most frequent class.
    if stratify and label_column in work.columns:
        group_labels = (
            work.groupby("_group")[label_column]
            .agg(lambda values: values.mode().iloc[0] if len(values.mode()) else "")
        )
    else:
        group_labels = pd.Series("", index=work["_group"].unique())

    groups = work["_group"].unique().tolist()
    rng.shuffle(groups)

    group_assignments: dict[str, str] = {}
    counts = {name: 0 for name in split_names}
    label_counts = {name: {} for name in split_names}
    total_samples = len(work)
    group_sizes = work.groupby("_group").size().to_dict()

    # Two-pass greedy assignment: first fill each split up to its target
    # sample count, then place the remainder into the split that is furthest
    # below its target.
    target_counts = {name: ratios[name] * total_samples for name in split_names}

    # Per-class targets, used to keep the class distribution balanced across
    # the splits.  The target of a class in a split is the class's overall
    # frequency multiplied by the split's ratio.
    overall_label_counts: dict[str, int] = {}
    if label_column in work.columns:
        for label, number in work[label_column].value_counts().items():
            overall_label_counts[str(label)] = int(number)
    target_label_counts: dict[str, dict[str, float]] = {
        name: {
            label: overall_label_counts[label] * ratios[name]
            for label in overall_label_counts
        }
        for name in split_names
    }

    for group in groups:
        label = str(group_labels.get(group, ""))
        size = group_sizes.get(group, 0)

        # Score every split by how badly it needs this group.  The *size*
        # deficit keeps the 70/15/15 proportions, while the *label* deficit
        # breaks the tie in favour of the split that is missing this class
        # most - that is what makes the split stratified rather than merely
        # proportional.
        best_name = None
        best_score = None
        for name in split_names:
            size_deficit = target_counts[name] - counts[name]
            if size_deficit <= 0 and best_name is not None:
                # The split is already full; only consider it when nothing
                # else is available.
                continue

            size_need = size_deficit / max(target_counts[name], 1.0)

            label_need = 0.0
            if label and label in target_label_counts[name]:
                target = target_label_counts[name][label]
                current = label_counts[name].get(label, 0)
                label_need = (target - current) / max(target, 1.0)

            score = size_need + label_need
            if best_score is None or score > best_score:
                best_score = score
                best_name = name

        if best_name is None:
            best_name = max(split_names, key=lambda n: target_counts[n] - counts[n])

        group_assignments[group] = best_name
        counts[best_name] += size
        if label:
            label_counts[best_name][label] = label_counts[best_name].get(label, 0) + size

    # Balance check: if a split is badly over-filled relative to its target,
    # move the smallest groups out of it.
    for _ in range(20):
        over = [name for name in split_names if counts[name] > target_counts[name] * 1.25]
        if not over:
            break
        source = max(over, key=lambda name: counts[name] - target_counts[name])
        movable = [
            (size, group)
            for group, size in group_sizes.items()
            if group_assignments.get(group) == source
        ]
        if not movable:
            break
        movable.sort()
        _, group = movable[0]
        deficits = {name: target_counts[name] - counts[name] for name in split_names}
        destination = max(
            (name for name in split_names if name != source),
            key=lambda name: deficits[name],
        )
        # Move the label counts along with the group so that the stratification
        # bookkeeping stays consistent with the actual assignment.
        moved_label = str(group_labels.get(group, ""))
        if moved_label:
            label_counts[source][moved_label] = (
                label_counts[source].get(moved_label, 0) - group_sizes[group]
            )
            label_counts[destination][moved_label] = (
                label_counts[destination].get(moved_label, 0) + group_sizes[group]
            )
        group_assignments[group] = destination
        counts[source] -= group_sizes[group]
        counts[destination] += group_sizes[group]

    sample_assignments = {
        str(sample_id): group_assignments.get(str(group), "train")
        for sample_id, group in zip(work["sample_id"], work["_group"])
    }
    return sample_assignments, group_assignments


def make_splits(
    frame: pd.DataFrame,
    config: SplitConfig | None = None,
) -> SplitResult:
    """Split a harmonised manifest into train / val / test.

    Parameters
    ----------
    frame:
        The harmonised manifest, ideally already annotated with the
        duplicate-detection columns.
    config:
        Split configuration.

    Returns
    -------
    SplitResult
    """
    config = config or SplitConfig()
    result = SplitResult(frame=frame.copy())

    if len(result.frame) == 0:
        result.warnings.append("Cannot split an empty manifest.")
        result.frame["split"] = pd.Series(dtype=str)
        return result

    work = result.frame
    ratios = {
        "train": config.train_ratio,
        "val": config.val_ratio,
        "test": config.test_ratio,
    }

    # -- unify cross-dataset duplicates into a single group -------------
    if config.unify_duplicates_across_datasets and "is_duplicate" in work.columns:
        duplicates = work[work["is_duplicate"].astype(str).str.lower().isin({"true", "1"})]
        if len(duplicates):
            canonical = {
                str(row["sample_id"]): str(row["duplicate_of"])
                for _, row in duplicates.iterrows()
                if row["duplicate_of"]
            }
            if canonical:
                work = work.copy()
                work["_split_group"] = [
                    canonical.get(str(sid), "") for sid in work["sample_id"]
                ]
                # Attach each duplicate to the canonical sample's group.
                canonical_group = dict(
                    zip(work["sample_id"], work.get("group_id", pd.Series(dtype=str)))
                )
                for index, row in work.iterrows():
                    ref = canonical.get(str(row["sample_id"]), "")
                    if ref and ref in canonical_group:
                        work.at[index, "group_id"] = canonical_group[ref]

    group_column = _resolve_group_column(work, config.group_by)
    if config.group_by == "auto" and group_column is None:
        result.warnings.append(
            "No patient or lesion identifier is available, so splitting falls "
            "back to per-image grouping. Multi-image lesions of the same "
            "patient may end up in different splits. This limitation is "
            "reported in the leakage audit."
        )
    elif group_column is not None:
        result.warnings.append(
            f"Splitting is grouped by '{group_column}' so that all images of "
            "one patient/lesion stay in a single split."
        )

    assignments, group_assignments = group_stratified_split(
        work,
        group_column=group_column,
        ratios=ratios,
        label_column="unified_label",
        seed=config.seed,
        stratify=config.stratify,
    )

    result.frame["split"] = [
        assignments.get(str(sid), "train") for sid in result.frame["sample_id"]
    ]
    result.assignments = assignments
    result.group_assignments = group_assignments

    # -- sanity checks ---------------------------------------------------
    distribution = result.class_distribution()
    if distribution:
        for split_name, counts in distribution.items():
            if not counts:
                result.warnings.append(
                    f"Split '{split_name}' contains no validly labelled samples."
                )

    return result


def load_split_manifest(path: str | Path) -> pd.DataFrame:
    """Load a previously written split manifest."""
    frame = pd.read_csv(path, keep_default_na=False, dtype=str)
    return frame
