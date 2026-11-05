"""End-to-end data pipeline shared by every training script.

:func:`prepare_data` performs the whole preparation chain exactly once:

1. discover the datasets that are present on disk,
2. build the harmonised manifest,
3. detect duplicates and near-duplicates,
4. create the leakage-safe train/val/test split,
5. fit the metadata encoder **on the training split only**,
6. compute the class weights **from the training split only**,
7. build the PyTorch datasets and DataLoaders.

Steps 5 and 6 are the two places where leakage would most easily creep in, so
they are implemented here once and used everywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.augmentation import AugmentationConfig
from data.dataset_harmonization import (
    HarmonisationResult,
    build_compatibility_report,
    harmonise_manifests,
    write_compatibility_report,
)
from data.deduplication import DedupConfig, DedupResult, find_duplicates
from data.label_mapping import UNIFIED_CLASSES, UNIFIED_PLUS_SCC_CLASSES
from data.loaders import LOADERS, DatasetManifest
from data.metadata import MetadataConfig, MetadataEncoder
from data.preprocessing import PreprocessConfig
from data.splitting import SplitConfig, SplitResult, make_splits, resolve_split_file
from utils.seeding import get_generator, seed_worker

__all__ = [
    "PreparedData",
    "prepare_data",
    "build_dataloaders",
    "dataset_roots_from_config",
]


@dataclass
class PreparedData:
    """Everything the training scripts need to start an experiment."""

    manifest: pd.DataFrame
    split_result: SplitResult
    harmonisation: HarmonisationResult
    dedup: DedupResult | None
    class_names: list[str]
    metadata_encoder: MetadataEncoder | None
    class_weights: torch.Tensor | None
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    split_path: Path | None = None
    compatibility_report_path: tuple[Path, Path] | None = None

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def summary(self) -> dict[str, Any]:
        return {
            "total_samples": int(len(self.manifest)),
            "train_samples": int(len(self.train_frame)),
            "val_samples": int(len(self.val_frame)),
            "test_samples": int(len(self.test_frame)),
            "class_names": list(self.class_names),
            "num_classes": self.num_classes,
            "metadata_dim": self.metadata_encoder.output_dim if self.metadata_encoder else 0,
            "warnings": list(self.warnings),
        }


def dataset_roots_from_config(
    config: Any,
    raw_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Resolve the expected root directory of each dataset.

    The root of a dataset is ``<raw_dir>/<dataset_name>`` unless the
    configuration overrides it explicitly.  Every path is configurable and no
    absolute path is hardcoded.
    """
    raw_root = Path(raw_dir) if raw_dir else Path(config.get("paths.raw_dir", "data/raw"))
    roots: dict[str, Path] = {}
    for name in LOADERS:
        override = config.get(f"datasets.roots.{name}", None)
        roots[name] = Path(override) if override else raw_root / name
    return roots


def _loader_subdir(name: str) -> str:
    loader_class = LOADERS[name]
    return loader_class.default_subdir


def prepare_data(
    config: Any,
    datasets: Sequence[str] | None = None,
    raw_dir: str | Path | None = None,
    include_scc: bool = False,
    split_path: str | Path | None = None,
    reuse_split: bool = True,
    write_reports: bool = True,
) -> PreparedData:
    """Build the complete data pipeline.

    Parameters
    ----------
    config:
        The project configuration.
    datasets:
        Which datasets to include.  Defaults to the enabled ones.
    raw_dir:
        Override for the raw data directory.
    include_scc:
        Whether to use the 8-class label space (ISIC 2019 and PAD-UFES-20
        contain SCC; HAM10000 and ISIC 2018 do not).
    split_path:
        Where the split manifest is written / read from.
    reuse_split:
        Whether to reuse an existing split file.  Reusing the same split is
        essential for a fair ablation comparison.
    write_reports:
        Whether to write the compatibility report.

    Returns
    -------
    PreparedData
    """
    warnings: list[str] = []
    roots = dataset_roots_from_config(config, raw_dir)

    if datasets is None:
        enabled = config.get("datasets.enabled", {}) or {}
        datasets = [
            name for name in LOADERS if enabled.get(name, True)
        ]

    target_classes = (
        UNIFIED_PLUS_SCC_CLASSES if include_scc else UNIFIED_CLASSES
    )

    # -- 1. discover and load --------------------------------------------
    manifests: list[DatasetManifest] = []
    for name in datasets:
        root = roots.get(name, Path("data/raw") / name)
        loader = LOADERS[name](root=root, include_scc=include_scc)
        if not loader.is_available():
            warnings.append(
                f"{name}: dataset not found at {root} - skipped. "
                "See DATASET_GUIDE.md for the expected layout."
            )
            continue
        try:
            manifest = loader.build_manifest()
            manifests.append(manifest)
            warnings.append(
                f"{name}: {len(manifest)} samples loaded from {root}"
            )
        except Exception as exc:  # pragma: no cover - depends on real data
            warnings.append(f"{name}: failed to load ({exc}) - skipped.")

    if not manifests:
        raise RuntimeError(
            "No dataset could be loaded. Place at least one dataset under "
            f"{config.get('paths.raw_dir', 'data/raw')} and run "
            "scripts/prepare_datasets.py first.\n" + "\n".join(warnings)
        )

    # -- 2. harmonise ------------------------------------------------------
    harmonisation = harmonise_manifests(manifests, target_classes=target_classes)
    manifest_frame = harmonisation.frame
    warnings.extend(harmonisation.warnings)

    # -- 3. deduplicate ----------------------------------------------------
    dedup_result: DedupResult | None = None
    dedup_config = DedupConfig.from_config(config)
    if dedup_config.enabled and len(manifest_frame):
        dedup_result = find_duplicates(manifest_frame, dedup_config)
        manifest_frame = dedup_result.frame
        if dedup_result.num_exact_duplicates or dedup_result.num_near_duplicates:
            warnings.append(
                f"duplicates detected: {dedup_result.num_exact_duplicates} exact, "
                f"{dedup_result.num_near_duplicates} near-duplicate, "
                f"{dedup_result.num_cross_dataset} cross-dataset. "
                "Duplicates are grouped so they stay in the same split."
            )

    # -- 4. split ----------------------------------------------------------
    split_config = SplitConfig.from_config(config)
    active_datasets = sorted(manifest_frame["dataset"].unique())
    split_file = resolve_split_file(
        Path(config.get("paths.splits_dir", "data/splits")),
        active_datasets,
        explicit=split_path,
    )

    if reuse_split and split_file.exists():
        existing = pd.read_csv(split_file, keep_default_na=False, dtype=str)
        if "sample_id" in existing.columns and "split" in existing.columns:
            mapping = dict(zip(existing["sample_id"], existing["split"]))
            manifest_frame["split"] = [
                mapping.get(str(sid), "") for sid in manifest_frame["sample_id"]
            ]
            missing = (manifest_frame["split"] == "").sum()
            if missing:
                # A stale split file must never silently move new samples into
                # the training split: that would leave validation and test
                # empty and make the reported metrics meaningless.  When a
                # significant part of the manifest is unknown to the split
                # file, the split is recomputed instead.
                coverage = len(manifest_frame) - missing
                coverage_ratio = coverage / max(len(manifest_frame), 1)
                if coverage_ratio < 0.5:
                    warnings.append(
                        f"The existing split file {split_file} only covers "
                        f"{coverage}/{len(manifest_frame)} samples "
                        f"({coverage_ratio:.1%}). The split is recomputed so "
                        "that validation and test sets are not left empty."
                    )
                    split_result = make_splits(manifest_frame, split_config)
                    manifest_frame = split_result.frame
                else:
                    warnings.append(
                        f"{missing} sample(s) are not present in the existing "
                        "split file and are assigned to the train split."
                    )
                    manifest_frame.loc[
                        manifest_frame["split"] == "", "split"
                    ] = "train"
                    split_result = SplitResult(frame=manifest_frame)
            else:
                split_result = SplitResult(frame=manifest_frame)
            split_result.warnings.append(
                f"Reused the existing split file {split_file}."
            )
        else:
            split_result = make_splits(manifest_frame, split_config)
    else:
        split_result = make_splits(manifest_frame, split_config)

    manifest_frame = split_result.frame
    warnings.extend(split_result.warnings)

    split_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_frame.to_csv(split_file, index=False)

    # -- 4b. exclude samples outside the active label space ----------------
    # A sample whose native label has no counterpart in the active head (for
    # example SCC when the 7-class space is used) keeps ``class_index = -1``.
    # Such rows must never reach the loss, because a negative target makes
    # CrossEntropy / FocalLoss raise "index -1 is out of bounds".
    #
    # The exclusion happens *after* splitting, so the split covers every
    # sample, and only the training frames are filtered: the manifest keeps
    # every row (nothing is silently discarded) and the excluded samples stay
    # visible in the manifest and in the compatibility report.
    if "unified_label_valid" in manifest_frame.columns:
        valid_mask = (
            manifest_frame["unified_label_valid"]
            .astype(str)
            .str.lower()
            .isin({"true", "1"})
        )
    else:
        valid_mask = pd.Series(True, index=manifest_frame.index)

    if "class_index" in manifest_frame.columns:
        index_values = pd.to_numeric(
            manifest_frame["class_index"], errors="coerce"
        ).fillna(-1).astype(int)
        valid_mask = valid_mask & (index_values >= 0)

    excluded = manifest_frame[~valid_mask]
    if len(excluded):
        by_class = excluded["native_label"].value_counts().to_dict()
        warnings.append(
            f"{len(excluded)} sample(s) excluded from the active label space "
            f"(not remapped, retained in the manifest): {by_class}. "
            "They are reported in the compatibility report and can be trained "
            "by enabling the 8-class space (--include-scc)."
        )

    # -- 5. metadata encoder (train split only) ----------------------------
    metadata_config = MetadataConfig.from_config(config)
    metadata_encoder: MetadataEncoder | None = None

    train_frame = manifest_frame[
        (manifest_frame["split"] == "train") & valid_mask
    ].copy()
    val_frame = manifest_frame[
        (manifest_frame["split"] == "val") & valid_mask
    ].copy()
    test_frame = manifest_frame[
        (manifest_frame["split"] == "test") & valid_mask
    ].copy()

    has_metadata_column = "metadata" in manifest_frame.columns
    any_metadata = False
    if has_metadata_column and len(train_frame):
        for raw in train_frame["metadata"]:
            if raw and str(raw).strip() not in {"", "nan", "{}", "None"}:
                any_metadata = True
                break

    if any_metadata:
        metadata_encoder = MetadataEncoder(metadata_config)
        records = _extract_metadata_records(train_frame)
        metadata_encoder.fit(records)
        warnings.append(
            f"metadata encoder fitted on the training split only "
            f"({len(records)} records, output_dim={metadata_encoder.output_dim})"
        )
    else:
        warnings.append(
            "No clinical metadata found in the training split - the metadata "
            "branch is disabled. Only PAD-UFES-20 provides metadata."
        )

    # -- 6. class weights (train split only) ------------------------------
    weighting = config.get("training.class_weighting", {}) or {}
    strategy = str(weighting.get("strategy", "effective_number"))
    beta = float(weighting.get("beta", 0.999))
    cap = float(weighting.get("cap", 10.0))
    normalize = bool(weighting.get("normalize_weights", True))

    class_names = list(harmonisation.class_list)
    train_labels = [
        int(value)
        for value in train_frame["class_index"].tolist()
        if str(value) not in {"", "nan", "None"} and int(value) >= 0
    ]
    class_weights = None
    if class_names and train_labels:
        from models.losses import compute_class_weights

        class_weights = compute_class_weights(
            train_labels,
            num_classes=len(class_names),
            strategy=strategy,
            beta=beta,
            cap=cap,
            normalize=normalize,
        )

    # -- 7. reports --------------------------------------------------------
    compatibility_report = None
    if write_reports:
        report = build_compatibility_report(harmonisation, include_scc=include_scc)
        report_dir = Path(config.get("paths.reports_dir", "reports"))
        compatibility_report = write_compatibility_report(report, report_dir)

    return PreparedData(
        manifest=manifest_frame,
        split_result=split_result,
        harmonisation=harmonisation,
        dedup=dedup_result,
        class_names=class_names,
        metadata_encoder=metadata_encoder,
        class_weights=class_weights,
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
        warnings=warnings,
        split_path=split_file,
        compatibility_report_path=compatibility_report,
    )


def _extract_metadata_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Parse the JSON metadata column into a list of dictionaries."""
    records: list[dict[str, Any]] = []
    for raw in frame.get("metadata", pd.Series(dtype=str)).tolist():
        record: dict[str, Any] = {}
        if raw and isinstance(raw, str) and raw.strip() not in {"", "nan", "None"}:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    record = parsed
            except json.JSONDecodeError:
                record = {}
        records.append(record)
    return records


def build_dataloaders(
    prepared: PreparedData,
    config: Any,
    batch_size: int | None = None,
    train: bool = True,
    shuffle_train: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build the train, validation and test DataLoaders.

    The metadata encoder is attached **only** to the loaders whose split
    carries clinical metadata; the other datasets keep an all-zero metadata
    vector so the model learns the dedicated "no metadata" embedding.
    """
    from data.datasets import SkinCancerDataset, collate_batch

    preprocess_config = PreprocessConfig.from_config(config)
    augmentation_config = AugmentationConfig.from_config(config)

    batch_size = batch_size or int(config.get("training.batch_size", 16))
    num_workers = int(config.get("hardware.num_workers", 4))
    seed = int(config.get("project.seed", 42))
    pin_memory = bool(config.get("hardware.pin_memory", True))

    def _make(frame: pd.DataFrame, is_train: bool) -> DataLoader:
        dataset = SkinCancerDataset(
            frame,
            preprocess_config=preprocess_config,
            augmentation_config=augmentation_config,
            train=is_train,
            metadata_encoder=prepared.metadata_encoder,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train and shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=collate_batch,
            worker_init_fn=seed_worker,
            generator=get_generator(seed) if (is_train and shuffle_train) else None,
            persistent_workers=bool(
                num_workers > 0 and config.get("hardware.persistent_workers", True)
            ),
        )

    train_loader = _make(prepared.train_frame, True)
    val_loader = _make(prepared.val_frame, False)
    test_loader = _make(prepared.test_frame, False)
    return train_loader, val_loader, test_loader
