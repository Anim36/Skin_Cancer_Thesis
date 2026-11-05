"""Shared dataset abstractions: manifests, records and the base loader.

The four datasets differ in their file layout and metadata schema, but they all
produce the same canonical record:

``SampleRecord``
    One row of the harmonised manifest, carrying the image path, the *native*
    label, the *unified* label, the dataset name, the original identifier, the
    patient/lesion group identifier (when the dataset provides one), and the
    raw clinical metadata (when available).

Every loader stores its manifest as a CSV in ``data/processed/<dataset>/`` so
that a run can be reproduced exactly and so that the splitting stage never has
to re-read the raw metadata.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

__all__ = [
    "SampleRecord",
    "DatasetManifest",
    "BaseDatasetLoader",
    "ManifestSchema",
]


#: Column names of the harmonised manifest.  Every loader emits exactly these.
MANIFEST_COLUMNS: list[str] = [
    "sample_id",           # globally unique id: "<dataset>:<original_id>"
    "dataset",             # source dataset name
    "image_path",          # path to the image file (absolute, reproducible)
    "relative_path",       # path relative to the dataset root
    "original_id",         # identifier as given by the source dataset
    "native_label",        # label exactly as found in the source metadata
    "native_label_norm",   # normalised native label
    "unified_label",       # canonical code or -1 when excluded
    "unified_label_valid", # bool: usable in the unified head
    "class_index",         # integer index in the training label space
    "group_id",            # patient/lesion group for leakage-safe splitting
    "group_type",          # patient | lesion | none
    "has_metadata",        # bool: clinical metadata present
    "metadata",            # JSON string with the clinical fields
    "source",              # provenance string (dataset + file reference)
]


class ManifestSchema:
    """Helper that guarantees a manifest has the expected columns."""

    COLUMNS = MANIFEST_COLUMNS

    @staticmethod
    def normalise(frame: pd.DataFrame) -> pd.DataFrame:
        """Add any missing column (filled with an empty value) and order them."""
        for column in MANIFEST_COLUMNS:
            if column not in frame.columns:
                frame[column] = None
        extra = [c for c in frame.columns if c not in MANIFEST_COLUMNS]
        return frame[MANIFEST_COLUMNS + extra]


@dataclass
class SampleRecord:
    """A single harmonised sample."""

    sample_id: str
    dataset: str
    image_path: str
    relative_path: str
    original_id: str
    native_label: str
    native_label_norm: str
    unified_label: str
    unified_label_valid: bool
    class_index: int
    group_id: str
    group_type: str
    has_metadata: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def to_row(self) -> dict[str, Any]:
        """Serialise to a flat dictionary for the manifest CSV."""
        row = asdict(self)
        row["metadata"] = json.dumps(self.metadata, sort_keys=True)
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "SampleRecord":
        """Rebuild a record from a manifest row."""
        raw_meta = row.get("metadata")
        if isinstance(raw_meta, str):
            try:
                metadata = json.loads(raw_meta) if raw_meta else {}
            except json.JSONDecodeError:
                metadata = {}
        elif isinstance(raw_meta, Mapping):
            metadata = dict(raw_meta)
        else:
            metadata = {}

        def _as_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if value is None:
                return False
            return str(value).strip().lower() in {"true", "1", "yes"}

        def _as_int(value: Any, default: int = -1) -> int:
            try:
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    return default
                return int(value)
            except (TypeError, ValueError):
                return default

        return cls(
            sample_id=str(row.get("sample_id", "")),
            dataset=str(row.get("dataset", "")),
            image_path=str(row.get("image_path", "")),
            relative_path=str(row.get("relative_path", "")),
            original_id=str(row.get("original_id", "")),
            native_label=str(row.get("native_label", "")),
            native_label_norm=str(row.get("native_label_norm", "")),
            unified_label=str(row.get("unified_label", "") or ""),
            unified_label_valid=_as_bool(row.get("unified_label_valid")),
            class_index=_as_int(row.get("class_index")),
            group_id=str(row.get("group_id", "") or ""),
            group_type=str(row.get("group_type", "none") or "none"),
            has_metadata=_as_bool(row.get("has_metadata")),
            metadata=metadata,
            source=str(row.get("source", "") or ""),
        )


@dataclass
class DatasetManifest:
    """Container for a dataset's harmonised records."""

    dataset: str
    records: list[SampleRecord] = field(default_factory=list)
    root: str = ""
    class_list: list[str] = field(default_factory=list)
    metadata_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def to_frame(self) -> pd.DataFrame:
        """Convert to the canonical manifest DataFrame."""
        if not self.records:
            return ManifestSchema.normalise(pd.DataFrame(columns=MANIFEST_COLUMNS))
        frame = pd.DataFrame([r.to_row() for r in self.records])
        return ManifestSchema.normalise(frame)

    def save(self, path: str | Path) -> Path:
        """Write the manifest CSV."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(destination, index=False)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        """Read a manifest CSV back into a :class:`DatasetManifest`."""
        frame = pd.read_csv(path, keep_default_na=False, dtype=str)
        records = [SampleRecord.from_row(row) for _, row in frame.iterrows()]
        return cls(
            dataset=str(frame["dataset"].iloc[0]) if len(frame) else "",
            records=records,
        )

    def class_distribution(self) -> dict[str, int]:
        """Count of samples per unified class (valid samples only)."""
        counts: dict[str, int] = {}
        for record in self.records:
            if not record.unified_label_valid:
                continue
            counts[record.unified_label] = counts.get(record.unified_label, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def summary(self) -> dict[str, Any]:
        """JSON-serialisable dataset summary used by the reports."""
        return {
            "dataset": self.dataset,
            "root": self.root,
            "num_samples": len(self.records),
            "class_list": list(self.class_list),
            "metadata_columns": list(self.metadata_columns),
            "class_distribution": self.class_distribution(),
            "num_with_metadata": sum(1 for r in self.records if r.has_metadata),
            "notes": list(self.notes),
        }


class BaseDatasetLoader:
    """Base class shared by the four concrete dataset loaders.

    Subclasses must implement :meth:`_discover`, which returns the raw
    metadata rows, and may override :meth:`_post_process`.

    The base class handles:
    * discovering the dataset root (configurable, never hardcoded),
    * building a :class:`LabelMapping` from the *observed* labels,
    * assigning unified indices and the ``unified_label_valid`` flag,
    * writing the harmonised manifest.
    """

    #: Canonical dataset name.
    name: str = "base"
    #: Subdirectory under ``data/raw`` where the dataset is expected.
    default_subdir: str = ""
    #: Column names that identify a patient (first match wins).
    patient_id_columns: tuple[str, ...] = ()
    #: Column names that identify a lesion.
    lesion_id_columns: tuple[str, ...] = ()
    #: Column names holding the native label.
    label_columns: tuple[str, ...] = ()
    #: Column names holding clinical metadata.
    metadata_columns: tuple[str, ...] = ()

    def __init__(
        self,
        root: str | Path,
        label_mapping: Any | None = None,
        target_classes: Sequence[str] | None = None,
        include_scc: bool = False,
        use_metadata: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.target_classes = list(target_classes) if target_classes else None
        self.include_scc = include_scc
        self._label_mapping = label_mapping
        self.use_metadata = use_metadata
        self.notes: list[str] = []

    # ------------------------------------------------------------------
    # To be implemented by subclasses
    # ------------------------------------------------------------------
    def _discover(self) -> pd.DataFrame:
        """Return the raw metadata table for this dataset.

        The returned frame must contain at least a path column and a label
        column.  The concrete column names are dataset specific.
        """
        raise NotImplementedError

    def _image_path_column(self) -> str:
        """Name of the column in the discovered frame holding the image path."""
        return "image_path"

    def _label_column(self) -> str:
        """Name of the column in the discovered frame holding the native label."""
        return "label"

    # ------------------------------------------------------------------
    # Shared machinery
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """Whether the dataset root exists and looks populated."""
        return self.root.exists() and any(self.root.iterdir())

    def _resolve_label_mapping(self, labels: Iterable[Any]):
        """Build (or reuse) the label mapping for this dataset."""
        if self._label_mapping is not None:
            return self._label_mapping
        from data.label_mapping import build_label_mapping

        target = self.target_classes
        if target is None:
            from data.label_mapping import resolve_dataset_class_list

            target = resolve_dataset_class_list(
                self.name, labels, include_scc=self.include_scc
            )
        self._label_mapping = build_label_mapping(
            dataset=self.name,
            native_labels=list(labels),
            target_classes=target,
            exclude_from_unified=set() if self.include_scc else {"SCC"},
        )
        return self._label_mapping

    def _group_for(
        self,
        row: Mapping[str, Any],
        patient_columns: Sequence[str],
        lesion_columns: Sequence[str],
    ) -> tuple[str, str]:
        """Return ``(group_id, group_type)`` for a row.

        A patient id is preferred because images of the same patient may show
        different lesions but still share clinical risk factors.  When only a
        lesion id exists, that is used instead.
        """
        for column in patient_columns:
            value = row.get(column)
            if value is not None and str(value).strip() not in {"", "nan", "None"}:
                return f"{self.name}_patient_{value}", "patient"
        for column in lesion_columns:
            value = row.get(column)
            if value is not None and str(value).strip() not in {"", "nan", "None"}:
                return f"{self.name}_lesion_{value}", "lesion"
        return "", "none"

    def build_manifest(self) -> DatasetManifest:
        """Discover the dataset and build the harmonised manifest."""
        if not self.root.exists():
            raise FileNotFoundError(
                f"{self.name}: dataset root does not exist: {self.root}\n"
                f"See DATASET_GUIDE.md for the expected layout."
            )

        frame = self._discover()
        if frame is None or len(frame) == 0:
            raise RuntimeError(
                f"{self.name}: no samples discovered under {self.root}. "
                "Check the dataset layout against DATASET_GUIDE.md."
            )

        path_column = self._image_path_column()
        label_column = self._label_column()

        label_mapping = self._resolve_label_mapping(frame[label_column].tolist())

        records: list[SampleRecord] = []
        for _, row in frame.iterrows():
            raw_path = Path(str(row[path_column]))
            absolute = raw_path if raw_path.is_absolute() else (self.root / raw_path)
            native_label = str(row[label_column])
            unified = label_mapping.to_unified(native_label)
            valid = unified is not None
            class_index = label_mapping.to_unified_index(native_label)

            group_id, group_type = self._group_for(
                row, self.patient_id_columns, self.lesion_id_columns
            )

            metadata = self._extract_metadata(row)
            has_metadata = bool(metadata) and any(
                str(v).strip() not in {"", "nan", "None", "<MISSING>"}
                for v in metadata.values()
            )

            original_id = self._original_id(row, raw_path)
            sample_id = f"{self.name}:{original_id}"

            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    dataset=self.name,
                    image_path=str(absolute.resolve()) if absolute.exists() else str(absolute),
                    relative_path=str(raw_path),
                    original_id=original_id,
                    native_label=native_label,
                    native_label_norm=native_label.strip().lower(),
                    unified_label=unified or "",
                    unified_label_valid=valid,
                    class_index=class_index,
                    group_id=group_id,
                    group_type=group_type,
                    has_metadata=has_metadata,
                    metadata=metadata,
                    source=f"{self.name}:{self.relative_source(raw_path)}",
                )
            )

        manifest = DatasetManifest(
            dataset=self.name,
            records=records,
            root=str(self.root),
            class_list=list(label_mapping.target_classes),
            metadata_columns=list(self.metadata_columns),
            notes=list(self.notes),
        )
        return manifest

    def _original_id(self, row: Mapping[str, Any], path: Path) -> str:
        """Extract the dataset's own identifier for a sample."""
        for column in ("image_id", "id", "sample_id", "original_id"):
            if column in row:
                value = row[column]
                if value is not None and str(value).strip() not in {"", "nan"}:
                    return str(value)
        return path.stem

    def _extract_metadata(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Extract clinical metadata from a row (empty when the dataset has none).

        Metadata is only extracted when the loader is configured to use it.
        HAM10000, ISIC 2018 and ISIC 2019 redistribute clinical fields in some
        mirrors, but they are *not* validated clinical annotations comparable
        to PAD-UFES-20, so they are excluded by default to keep the multimodal
        comparison controlled.
        """
        if not self.metadata_columns or not self.use_metadata:
            return {}
        out: dict[str, Any] = {}
        for column in self.metadata_columns:
            if column in row:
                value = row[column]
                out[column] = None if value is None else value
        return out

    def relative_source(self, path: Path) -> str:
        """Path of ``path`` relative to the dataset root (for provenance)."""
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)
