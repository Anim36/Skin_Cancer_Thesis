"""HAM10000 dataset loader.

Official layout
===============
The HAM10000 archive ships a ``metadata.csv`` (or ``HAM10000_metadata.csv``)
and an image folder.  The metadata contains at least::

    image_id, diagnosis, dx, dx_type, age, sex, localization
    (and in the Kaggle version: lesion_id, ...)

The loader is deliberately tolerant: it accepts both the original HAM10000
release layout and the widely redistributed Kaggle mirror, and it never
*assumes* a column exists - it discovers what is actually present and records
that in the dataset report.

Grouping
========
When a ``lesion_id`` column is present (Kaggle mirror), all images sharing a
lesion id are placed in the **same** split so that multi-image lesions cannot
leak between train and test.  HAM10000 contains a handful of lesions imaged
more than once (this is exactly the kind of leakage that inflates reported
accuracy), so this matters.

When no lesion or patient identifier exists, the loader falls back to the
per-image grouping and the limitation is reported explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from data.loaders.base import BaseDatasetLoader, DatasetManifest

__all__ = ["HAM10000Loader"]


#: File name patterns searched for the HAM10000 metadata table.
_METADATA_PATTERNS = (
    "metadata.csv",
    "HAM10000_metadata.csv",
    "ham10000_metadata.csv",
    "*metadata*.csv",
)

#: Directory patterns searched for the HAM10000 images.
_IMAGE_DIR_PATTERNS = (
    "HAM10000_images",
    "ham10000_images",
    "HAM10000",
    "images",
    "HAM10000_images",
    "all_images",
)


class HAM10000Loader(BaseDatasetLoader):
    """Loader for the HAM10000 dermoscopy dataset."""

    name = "ham10000"
    default_subdir = "ham10000"
    label_columns = ("dx", "diagnosis", "dx_type")
    patient_id_columns = ("lesion_id", "patient_id")
    lesion_id_columns = ("lesion_id",)
    metadata_columns = ("age", "sex", "localization")

    def __init__(
        self,
        root: str | Path,
        label_mapping: Any | None = None,
        target_classes: Sequence[str] | None = None,
        include_scc: bool = False,
        use_metadata: bool = False,
    ) -> None:
        super().__init__(root, label_mapping, target_classes, include_scc, use_metadata)
        self.metadata_path: Path | None = None
        self.image_dir: Path | None = None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def _find_metadata_file(self) -> Path:
        for pattern in _METADATA_PATTERNS:
            matches = sorted(self.root.rglob(pattern))
            if matches:
                # Prefer the shallowest match: the top-level CSV is the real
                # metadata, a nested one is usually a per-split copy.
                matches.sort(key=lambda p: (len(p.parts), str(p)))
                return matches[0]
        raise FileNotFoundError(
            f"{self.name}: no metadata CSV found under {self.root}. Expected one "
            "of: " + ", ".join(_METADATA_PATTERNS)
        )

    def _find_image_dir(self) -> Path:
        candidates: list[Path] = []
        for pattern in _IMAGE_DIR_PATTERNS:
            for path in self.root.rglob(pattern):
                if path.is_dir():
                    candidates.append(path)
        if candidates:
            candidates.sort(key=lambda p: (len(p.parts), str(p)))
            return candidates[0]
        # Fall back to the root itself if it holds image files.
        image_suffixes = {".jpg", ".jpeg", ".png"}
        if any(p.suffix.lower() in image_suffixes for p in self.root.iterdir()):
            return self.root
        raise FileNotFoundError(
            f"{self.name}: no image directory found under {self.root}"
        )

    def _discover(self) -> pd.DataFrame:
        self.metadata_path = self._find_metadata_file()
        self.image_dir = self._find_image_dir()

        frame = pd.read_csv(self.metadata_path)
        frame.columns = [str(c).strip() for c in frame.columns]

        label_column = self._detect_label_column(frame)
        id_column = self._detect_id_column(frame)

        self.notes = self._build_notes(frame)

        rows: list[dict[str, Any]] = []
        image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        available = {
            p.stem.lower(): p
            for p in self.image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in image_suffixes
        }

        for _, row in frame.iterrows():
            image_id = str(row[id_column]).strip()
            path = available.get(image_id.lower())
            if path is None:
                # The metadata may reference an id that is not a valid file
                # name; try a direct hit before giving up on the row.
                direct = self.image_dir / image_id
                if direct.exists():
                    path = direct
            if path is None:
                continue

            record: dict[str, Any] = {
                "image_path": path,
                "image_id": image_id,
                "label": str(row[label_column]).strip(),
            }
            for column in ("lesion_id", "patient_id", "age", "sex", "localization"):
                if column in frame.columns:
                    record[column] = row[column]
            rows.append(record)

        if not rows:
            raise RuntimeError(
                f"{self.name}: metadata contained {len(frame)} rows but none of "
                f"them matched an image file in {self.image_dir}."
            )

        result = pd.DataFrame(rows)
        result.attrs["skipped_rows"] = len(frame) - len(result)
        return result

    # ------------------------------------------------------------------
    # Column detection
    # ------------------------------------------------------------------
    def _detect_label_column(self, frame: pd.DataFrame) -> str:
        """Find the column holding the diagnostic class.

        Priority: ``dx`` (the short code), then ``diagnosis`` (the full
        description).  ``dx_type`` is only used as a last resort because in
        some mirrors it holds slightly different wording than ``dx``.
        """
        for candidate in self.label_columns:
            if candidate in frame.columns:
                if candidate == "dx_type":
                    continue
                return candidate
        raise RuntimeError(
            f"{self.name}: could not find a label column. Available columns: "
            f"{list(frame.columns)}"
        )

    def _detect_id_column(self, frame: pd.DataFrame) -> str:
        for candidate in ("image_id", "id", "image"):
            if candidate in frame.columns:
                return candidate
        raise RuntimeError(
            f"{self.name}: could not find an image id column. Available "
            f"columns: {list(frame.columns)}"
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def _build_notes(self, frame: pd.DataFrame) -> list[str]:
        notes = [
            f"metadata file: {self.metadata_path.name if self.metadata_path else 'n/a'}",
            f"metadata columns: {', '.join(map(str, frame.columns))}",
        ]
        if "lesion_id" in frame.columns:
            counts = frame["lesion_id"].value_counts()
            multi = int((counts > 1).sum())
            notes.append(
                f"lesion-level grouping available: {multi} lesions have more "
                "than one image (these are kept inside a single split)"
            )
        else:
            notes.append(
                "LIMITATION: no lesion_id/patient_id column found; grouping "
                "falls back to per-image, so multi-image lesions cannot be "
                "protected from leakage. Consider using the Kaggle HAM10000 "
                "mirror which ships lesion_id."
            )
        if "age" in frame.columns:
            notes.append(
                "NOTE: HAM10000 age/sex/localization are present in some "
                "redistributions but are NOT used as metadata features by "
                "default - only PAD-UFES-20 metadata is used, to keep the "
                "multimodal comparison controlled."
            )
        return notes
