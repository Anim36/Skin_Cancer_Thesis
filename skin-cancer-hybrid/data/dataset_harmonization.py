"""Cross-dataset harmonisation and the dataset compatibility report.

This module is responsible for the scientifically critical question: *can the
four datasets be combined without corrupting the label semantics?*

It produces:

* a combined manifest across all available datasets,
* a per-dataset label compatibility table,
* an explicit list of classes that are excluded from the unified head,
* a report explaining every decision (written to ``reports/``).

The report is generated from the **real** metadata of whatever datasets are
actually present on disk - never from assumptions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from data.label_mapping import (
    CANONICAL_CLASSES,
    CLASS_DESCRIPTIONS,
    UNIFIED_CLASSES,
    UNIFIED_PLUS_SCC_CLASSES,
    LabelMapping,
    build_label_mapping,
)
from data.loaders.base import DatasetManifest

__all__ = [
    "DatasetCompatibility",
    "HarmonisationResult",
    "build_compatibility_report",
    "harmonise_manifests",
    "write_compatibility_report",
]


@dataclass
class DatasetCompatibility:
    """Compatibility of one dataset with the unified label space."""

    dataset: str
    available: bool = False
    num_samples: int = 0
    native_classes: list[str] = field(default_factory=list)
    mapped_classes: list[str] = field(default_factory=list)
    excluded_classes: list[str] = field(default_factory=list)
    unmapped_classes: list[str] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)
    excluded_counts: dict[str, int] = field(default_factory=dict)
    has_metadata: bool = False
    metadata_fields: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    root: str = ""

    @property
    def fully_compatible(self) -> bool:
        """True when every native class maps cleanly into the unified head."""
        return self.available and not self.unmapped_classes

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "available": self.available,
            "num_samples": self.num_samples,
            "native_classes": self.native_classes,
            "mapped_classes": self.mapped_classes,
            "excluded_classes": self.excluded_classes,
            "unmapped_classes": self.unmapped_classes,
            "class_counts": self.class_counts,
            "excluded_counts": self.excluded_counts,
            "has_metadata": self.has_metadata,
            "metadata_fields": self.metadata_fields,
            "fully_compatible": self.fully_compatible,
            "notes": self.notes,
            "root": self.root,
        }


@dataclass
class HarmonisationResult:
    """Outcome of harmonising one or more dataset manifests."""

    frame: pd.DataFrame
    compatibilities: list[DatasetCompatibility] = field(default_factory=list)
    label_mappings: dict[str, LabelMapping] = field(default_factory=dict)
    class_list: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_samples(self) -> int:
        return len(self.frame)

    def usable(self) -> pd.DataFrame:
        """Samples that can enter the unified head."""
        if "unified_label_valid" not in self.frame.columns:
            return self.frame
        mask = self.frame["unified_label_valid"].astype(str).str.lower().isin(
            {"true", "1"}
        )
        return self.frame[mask]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "usable_samples": len(self.usable()),
            "class_list": list(self.class_list),
            "compatibilities": [c.to_dict() for c in self.compatibilities],
            "warnings": list(self.warnings),
        }


def _compatibility_from_manifest(
    manifest: DatasetManifest,
    mapping: LabelMapping,
) -> DatasetCompatibility:
    """Derive the compatibility summary for a single dataset."""
    counts: dict[str, int] = {}
    excluded_counts: dict[str, int] = {}
    metadata_fields: set[str] = set()
    has_metadata = False

    for record in manifest.records:
        if record.unified_label_valid and record.unified_label:
            counts[record.unified_label] = counts.get(record.unified_label, 0) + 1
        else:
            excluded_counts[record.native_label] = (
                excluded_counts.get(record.native_label, 0) + 1
            )
        if record.has_metadata:
            has_metadata = True
            metadata_fields.update(record.metadata.keys())

    return DatasetCompatibility(
        dataset=manifest.dataset,
        available=True,
        num_samples=len(manifest),
        native_classes=list(mapping.native_classes),
        mapped_classes=sorted(set(mapping.native_to_unified.values())),
        excluded_classes=list(mapping.excluded_labels),
        unmapped_classes=list(mapping.unmapped_labels),
        class_counts=dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        excluded_counts=dict(sorted(excluded_counts.items(), key=lambda kv: -kv[1])),
        has_metadata=has_metadata,
        metadata_fields=sorted(metadata_fields),
        notes=list(manifest.notes),
        root=manifest.root,
    )


def harmonise_manifests(
    manifests: Sequence[DatasetManifest],
    target_classes: Sequence[str] | None = None,
) -> HarmonisationResult:
    """Combine per-dataset manifests into one harmonised frame.

    Parameters
    ----------
    manifests:
        The manifests produced by the four loaders (only the datasets that are
        actually present should be passed in).
    target_classes:
        The unified label space.  Defaults to the 7-class taxonomy.

    Returns
    -------
    HarmonisationResult
    """
    target = list(target_classes or UNIFIED_CLASSES)
    warnings: list[str] = []
    compatibilities: list[DatasetCompatibility] = []
    label_mappings: dict[str, LabelMapping] = {}

    frames: list[pd.DataFrame] = []
    for manifest in manifests:
        if len(manifest) == 0:
            warnings.append(f"{manifest.dataset}: manifest is empty, skipped.")
            compatibilities.append(
                DatasetCompatibility(dataset=manifest.dataset, available=False)
            )
            continue

        mapping = build_label_mapping(
            dataset=manifest.dataset,
            native_labels=[r.native_label for r in manifest.records],
            target_classes=target,
            exclude_from_unified={"SCC"},
        )
        label_mappings[manifest.dataset] = mapping

        if mapping.unmapped_labels:
            warnings.append(
                f"{manifest.dataset}: {len(mapping.unmapped_labels)} native class(es) "
                f"could not be mapped to the unified taxonomy: "
                f"{mapping.unmapped_labels}. These samples are retained in the "
                "manifest with unified_label_valid=False and are excluded from "
                "unified-head training."
            )

        # Recompute the unified index using the shared target space so that all
        # datasets share one integer encoding.
        frame = manifest.to_frame()
        frame["unified_label"] = [
            mapping.to_unified(r.native_label) or "" for r in manifest.records
        ]
        frame["unified_label_valid"] = [
            mapping.to_unified(r.native_label) is not None for r in manifest.records
        ]
        frame["class_index"] = [
            mapping.to_unified_index(r.native_label) for r in manifest.records
        ]
        frames.append(frame)

        compatibilities.append(_compatibility_from_manifest(manifest, mapping))

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame()

    # Only classes actually present in the combined data get an index, but the
    # index must stay stable across the whole project, so it follows the
    # canonical order of the target space.
    present = set()
    for frame in frames:
        present.update(
            frame.loc[frame["unified_label_valid"], "unified_label"].tolist()
        )
    class_list = [code for code in target if code in present]

    if not class_list:
        warnings.append(
            "No sample could be mapped into the unified label space. Check the "
            "dataset metadata and the alias table in configs/dataset.yaml."
        )

    # Re-index the samples onto ``class_list``.
    #
    # ``class_index`` is initially the index into the *full* target space, but
    # the classifier head is sized by ``len(class_list)`` - and ``class_list``
    # only contains the classes that are actually present.  When a class in the
    # middle of the taxonomy is absent (for example PAD-UFES-20 has no DF and
    # no VASC, so with --include-scc the class list is
    # [MEL, NV, BCC, AKIEC, BKL, SCC] while the full target space has 8
    # entries) the two numbering schemes diverge and the raw index would point
    # outside the head.  The samples are therefore re-indexed onto
    # ``class_list`` here so that index == position in ``class_list`` always.
    index_of = {code: index for index, code in enumerate(class_list)}
    reindexed: list[int] = []
    remapped = 0
    for frame in frames:
        for label, valid in zip(frame["unified_label"], frame["unified_label_valid"]):
            if valid and label in index_of:
                reindexed.append(index_of[label])
            else:
                reindexed.append(-1)
                remapped += 1
    if frames:
        combined["class_index"] = reindexed

    if remapped:
        warnings.append(
            f"{remapped} sample(s) carry class_index=-1 because their class is "
            "not part of the active label space; they are excluded from "
            "training but retained in the manifest."
        )

    return HarmonisationResult(
        frame=combined,
        compatibilities=compatibilities,
        label_mappings=label_mappings,
        class_list=class_list,
        warnings=warnings,
    )


def build_compatibility_report(
    result: HarmonisationResult,
    include_scc: bool = False,
) -> dict[str, Any]:
    """Build the machine-readable dataset compatibility report.

    The report is the scientific backbone of the "combined training" decision:
    it states exactly which classes each dataset contributes, which are
    excluded, and why.
    """
    target_space = UNIFIED_PLUS_SCC_CLASSES if include_scc else UNIFIED_CLASSES

    coverage: dict[str, list[str]] = {code: [] for code in CANONICAL_CLASSES}
    for compat in result.compatibilities:
        for code in compat.mapped_classes:
            if code in coverage:
                coverage[code].append(compat.dataset)

    missing_from_all = [code for code, sources in coverage.items() if not sources]

    return {
        "unified_label_space": target_space,
        "canonical_vocabulary": CANONICAL_CLASSES,
        "class_descriptions": CLASS_DESCRIPTIONS,
        "total_samples": result.total_samples,
        "usable_samples": len(result.usable()),
        "class_list_actually_used": result.class_list,
        "datasets": [c.to_dict() for c in result.compatibilities],
        "class_coverage": coverage,
        "classes_not_present_in_any_dataset": missing_from_all,
        "warnings": result.warnings,
        "decisions": [
            "SCC (squamous cell carcinoma) is an invasive carcinoma and is "
            "NOT merged with AKIEC (a pre-invasive actinic keratosis). It is "
            "either excluded from the shared 7-class head or trained as its "
            "own 8th class.",
            "Every sample keeps its native label in the manifest; nothing is "
            "silently discarded. Samples outside the unified head carry "
            "unified_label_valid=False.",
            "Metadata is used only where the dataset actually provides it "
            "(PAD-UFES-20). Missing metadata is never invented.",
            "The final test split is never used for tuning, threshold "
            "selection or hyperparameter search.",
        ],
    }


def write_compatibility_report(
    report: Mapping[str, Any],
    output_dir: str | Path,
    stem: str = "dataset_compatibility_report",
) -> tuple[Path, Path]:
    """Write the compatibility report as JSON and Markdown.

    Returns
    -------
    tuple
        ``(json_path, markdown_path)``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{stem}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    md_path = out_dir / f"{stem}.md"
    lines: list[str] = []
    lines.append("# Dataset Compatibility Report")
    lines.append("")
    lines.append(
        "Generated automatically by `scripts/inspect_datasets.py` from the "
        "**actual** dataset metadata found on disk. All numbers below are "
        "measured, not assumed."
    )
    lines.append("")
    lines.append(f"- Unified label space: `{', '.join(report['unified_label_space'])}`")
    lines.append(f"- Total samples discovered: **{report['total_samples']}**")
    lines.append(f"- Samples usable in the unified head: **{report['usable_samples']}**")
    lines.append("")

    lines.append("## Per-dataset compatibility")
    lines.append("")
    for dataset in report["datasets"]:
        lines.append(f"### {dataset['dataset']}")
        lines.append("")
        if not dataset["available"]:
            lines.append("_Dataset not present on disk._")
            lines.append("")
            continue
        lines.append(f"- Samples: **{dataset['num_samples']}**")
        lines.append(f"- Native classes: `{', '.join(dataset['native_classes']) or 'n/a'}`")
        lines.append(f"- Mapped into unified head: "
                     f"`{', '.join(dataset['mapped_classes']) or 'none'}`")
        if dataset["excluded_classes"]:
            lines.append(
                f"- **Excluded from unified head**: "
                f"`{', '.join(dataset['excluded_classes'])}`"
            )
        if dataset["unmapped_classes"]:
            lines.append(
                f"- **Unmapped**: `{', '.join(dataset['unmapped_classes'])}`"
            )
        lines.append(f"- Clinical metadata: "
                     f"{'yes' if dataset['has_metadata'] else 'no'}")
        if dataset["class_counts"]:
            lines.append("")
            lines.append("| Class | Samples |")
            lines.append("| --- | ---: |")
            for code, count in dataset["class_counts"].items():
                lines.append(f"| {code} | {count} |")
        if dataset["excluded_counts"]:
            lines.append("")
            lines.append("| Excluded native label | Samples |")
            lines.append("| --- | ---: |")
            for code, count in dataset["excluded_counts"].items():
                lines.append(f"| {code} | {count} |")
        if dataset["notes"]:
            lines.append("")
            for note in dataset["notes"]:
                lines.append(f"> {note}")
        lines.append("")

    lines.append("## Class coverage across datasets")
    lines.append("")
    lines.append("| Class | Description | Datasets contributing |")
    lines.append("| --- | --- | --- |")
    for code, sources in report["class_coverage"].items():
        description = CLASS_DESCRIPTIONS.get(code, code)
        lines.append(f"| {code} | {description} | {', '.join(sources) or '-'} |")
    lines.append("")

    if report["warnings"]:
        lines.append("## Warnings")
        lines.append("")
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("## Documented decisions")
    lines.append("")
    for decision in report["decisions"]:
        lines.append(f"- {decision}")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    return json_path, md_path
