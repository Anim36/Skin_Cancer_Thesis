"""PAD-UFES-20 dataset loader (images + clinical metadata).

Official layout
===============
PAD-UFES-20 ("A new dataset for skin lesion image classification") ships as a
folder of images plus a metadata CSV with one row per image.  The metadata
contains the image filename and the clinical fields:

* ``image_id`` / ``image``      - image filename
* ``dx``                        - diagnosis (native label)
* ``age``                       - patient age in years
* ``sex``                       - patient sex
* ``localization``              - anatomical lesion site
* ``skin_cancer_history``       - previous skin cancer (yes / no)

Metadata availability
=====================
PAD-UFES-20 is the **only** dataset in this project that ships genuine
clinical metadata.  It is therefore the primary dataset for the multimodal
(image + metadata) experiments.

Label space
===========
PAD-UFES-20 has six diagnostic classes.  The mapping to the seven-class
unified taxonomy is derived at runtime from the actual metadata; classes that
have no counterpart in the unified taxonomy are excluded rather than forced
into an incorrect bucket, and every exclusion is reported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from data.loaders.base import BaseDatasetLoader

__all__ = ["PADUFES20Loader"]


_METADATA_PATTERNS = (
    "metadata_pad_ufes20.csv",
    "metadata.csv",
    "METADATA_PAD_UFES20.csv",
    "*pad*ufes*.csv",
    "*metadata*.csv",
)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

#: Clinical fields extracted from the metadata, in canonical order.
CLINICAL_FIELDS = ("age", "sex", "localization", "skin_cancer_history")


class PADUFES20Loader(BaseDatasetLoader):
    """Loader for the PAD-UFES-20 multimodal dataset."""

    name = "pad_ufes20"
    default_subdir = "pad_ufes20"
    label_columns = ("dx", "diagnosis", "dx_type")
    patient_id_columns = ("patient_id",)
    lesion_id_columns = ("image_id", "image")
    metadata_columns = CLINICAL_FIELDS

    def __init__(
        self,
        root: str | Path,
        label_mapping: Any | None = None,
        target_classes: Sequence[str] | None = None,
        include_scc: bool = False,
        use_metadata: bool = True,
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
                matches.sort(key=lambda p: (len(p.parts), str(p)))
                return matches[0]
        raise FileNotFoundError(
            f"{self.name}: no metadata CSV found under {self.root}. Expected a "
            "PAD-UFES-20 metadata file (see DATASET_GUIDE.md)."
        )

    def _find_image_dir(self) -> Path:
        for path in sorted(self.root.rglob("*")):
            if not path.is_dir():
                continue
            if any(p.suffix.lower() in _IMAGE_SUFFIXES for p in path.iterdir()):
                return path
        raise FileNotFoundError(f"{self.name}: no image directory found under {self.root}")

    def _discover(self) -> pd.DataFrame:
        self.metadata_path = self._find_metadata_file()
        self.image_dir = self._find_image_dir()

        frame = pd.read_csv(self.metadata_path)
        frame.columns = [str(c).strip() for c in frame.columns]

        label_column = self._detect_label_column(frame)
        id_column = self._detect_id_column(frame)

        images = {
            p.stem.lower(): p
            for p in self.image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        }

        rows: list[dict[str, Any]] = []
        missing_images = 0
        for _, row in frame.iterrows():
            raw_id = str(row[id_column]).strip()
            path = images.get(raw_id.lower())
            if path is None:
                # The id may include the file extension.
                path = images.get(Path(raw_id).stem.lower())
            if path is None:
                missing_images += 1
                continue
            record: dict[str, Any] = {
                "image_path": path,
                "image_id": raw_id,
                "label": str(row[label_column]).strip(),
            }
            for field in CLINICAL_FIELDS:
                if field in frame.columns:
                    record[field] = row[field]
            rows.append(record)

        if not rows:
            raise RuntimeError(
                f"{self.name}: metadata rows did not match any image file in "
                f"{self.image_dir}."
            )

        self.notes = [
            f"metadata file: {self.metadata_path.name}",
            f"image dir: {self.image_dir.name}",
            f"clinical fields available: "
            f"{[f for f in CLINICAL_FIELDS if f in frame.columns]}",
            f"metadata rows without a matching image: {missing_images}",
        ]
        if missing_images:
            self.notes.append(
                "NOTE: rows whose image file could not be found are excluded "
                "from the manifest and counted above."
            )
        if "patient_id" not in frame.columns:
            self.notes.append(
                "LIMITATION: no patient_id column found; splitting falls back "
                "to per-image grouping."
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Column detection
    # ------------------------------------------------------------------
    def _detect_label_column(self, frame: pd.DataFrame) -> str:
        for candidate in self.label_columns:
            if candidate in frame.columns:
                return candidate
        raise RuntimeError(
            f"{self.name}: no label column found. Available: {list(frame.columns)}"
        )

    def _detect_id_column(self, frame: pd.DataFrame) -> str:
        for candidate in ("image_id", "image", "id", "image_name"):
            if candidate in frame.columns:
                return candidate
        raise RuntimeError(
            f"{self.name}: no image identifier column found. Available: "
            f"{list(frame.columns)}"
        )
