"""ISIC 2019 dataset loader.

Official layout
===============
The ISIC 2019 challenge archive contains several task folders.  The
classification task used here is Task 1, whose ground truth CSV
(``ISIC_2019_Training_GroundTruth.csv``) has the wide format::

    image, MEL, NV, BCC, AKIEC, BKL, DF, VASC, SCC

i.e. one **one-hot column per class** rather than a single label column.  The
loader therefore supports both the wide one-hot format and the long
``image, diagnosis`` format, detecting which one is present at runtime.

The eighth class
================
ISIC 2019 contains **SCC (squamous cell carcinoma)** in addition to the seven
classic categories.  SCC is a *distinct* disease entity and is never merged
into AKIEC.  Two training options are supported:

``include_scc=False`` (default)
    The shared 7-class head is used and SCC samples are excluded from the
    unified experiments.  They are still written to the manifest with
    ``unified_label_valid = False`` so nothing is silently lost.

``include_scc=True``
    The 8-class head is used and **every** ISIC 2019 sample contributes to
    training.  This is the "maximum data use" configuration.

Both configurations use the same loader code; only the resulting label space
differs.  The choice is recorded in the dataset compatibility report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from data.loaders.base import BaseDatasetLoader

__all__ = ["ISIC2019Loader"]


_METADATA_PATTERNS = (
    "ISIC_2019_Training_GroundTruth.csv",
    "GT_2019.csv",
    "*GroundTruth*.csv",
    "*ground_truth*.csv",
    "*2019*label*.csv",
    "ISIC_2019_Training_Metadata.csv",
)

_IMAGE_DIR_PATTERNS = (
    "ISIC_2019_Training_Input",
    "ISIC_2019_Training_Input/images",
    "ISIC_2019_Training_Input",
)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class ISIC2019Loader(BaseDatasetLoader):
    """Loader for ISIC 2019 (Task 1 classification, 8 classes)."""

    name = "isic2019"
    default_subdir = "isic2019"
    label_columns = ("diagnosis", "dx", "label")
    lesion_id_columns = ("image", "image_id")
    metadata_columns = ()

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
        self._one_hot_columns: list[str] = []

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
            f"{self.name}: no ground-truth CSV found under {self.root}. "
            "Download the ISIC 2019 challenge data (see DATASET_GUIDE.md)."
        )

    def _find_image_dir(self) -> Path:
        for pattern in _IMAGE_DIR_PATTERNS:
            for path in self.root.rglob(pattern):
                if path.is_dir() and any(
                    p.suffix.lower() in _IMAGE_SUFFIXES for p in path.iterdir()
                ):
                    return path
        for path in sorted(self.root.rglob("*")):
            if path.is_dir() and any(
                p.suffix.lower() in _IMAGE_SUFFIXES for p in path.iterdir()
            ):
                return path
        raise FileNotFoundError(f"{self.name}: no image directory found under {self.root}")

    def _discover(self) -> pd.DataFrame:
        self.metadata_path = self._find_metadata_file()
        self.image_dir = self._find_image_dir()

        frame = pd.read_csv(self.metadata_path)
        frame.columns = [str(c).strip() for c in frame.columns]

        images = {
            p.stem.lower(): p
            for p in self.image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        }

        rows = self._rows_from_wide_format(frame, images)
        if rows is None:
            rows = self._rows_from_long_format(frame, images)

        if not rows:
            raise RuntimeError(
                f"{self.name}: metadata rows did not match any image file in "
                f"{self.image_dir}. Expected either a wide one-hot CSV or an "
                "'image,diagnosis' CSV."
            )

        self.notes = [
            f"metadata file: {self.metadata_path.name}",
            f"image dir: {self.image_dir.name}",
            f"one-hot class columns detected: {self._one_hot_columns or 'none'}",
        ]
        if "SCC" in self._one_hot_columns:
            self.notes.append(
                "ISIC 2019 contains the 8th class SCC (squamous cell "
                "carcinoma). SCC is NOT merged into AKIEC: it is either kept as "
                "a separate class (include_scc=True) or excluded from the "
                "shared 7-class head (include_scc=False)."
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Format handling
    # ------------------------------------------------------------------
    def _rows_from_wide_format(
        self, frame: pd.DataFrame, images: dict[str, Path]
    ) -> list[dict[str, Any]] | None:
        """Handle the ``image, MEL, NV, ...`` one-hot format."""
        from data.label_mapping import CANONICAL_CLASSES

        id_column = None
        for candidate in ("image", "image_id", "id"):
            if candidate in frame.columns:
                id_column = candidate
                break
        if id_column is None:
            return None

        class_columns = [c for c in frame.columns if str(c).upper() in CANONICAL_CLASSES]
        if not class_columns:
            return None

        self._one_hot_columns = [str(c) for c in class_columns]
        rows: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            key = str(row[id_column]).strip()
            path = images.get(key.lower())
            if path is None:
                continue
            # A row may carry more than one positive one-hot flag; take the
            # first in canonical order and record the ambiguity.
            positives = [
                str(c).upper()
                for c in class_columns
                if float(row[c]) == 1.0
            ]
            if not positives:
                continue
            label = positives[0]
            if len(positives) > 1:
                self.notes.append(
                    f"WARNING: image {key} carries multiple one-hot labels "
                    f"({positives}); the first in canonical order ({label}) was "
                    "used."
                )
            rows.append(
                {
                    "image_path": path,
                    "image_id": key,
                    "label": label,
                    "ambiguous_labels": ",".join(positives) if len(positives) > 1 else "",
                }
            )
        return rows

    def _rows_from_long_format(
        self, frame: pd.DataFrame, images: dict[str, Path]
    ) -> list[dict[str, Any]]:
        """Handle the ``image, diagnosis`` long format."""
        id_column = None
        for candidate in ("image", "image_id", "id", "image_name"):
            if candidate in frame.columns:
                id_column = candidate
                break
        label_column = None
        for candidate in self.label_columns:
            if candidate in frame.columns:
                label_column = candidate
                break
        if id_column is None or label_column is None:
            raise RuntimeError(
                f"{self.name}: could not interpret the metadata CSV. Columns: "
                f"{list(frame.columns)}"
            )

        rows: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            key = str(row[id_column]).strip()
            path = images.get(key.lower())
            if path is None:
                continue
            rows.append(
                {
                    "image_path": path,
                    "image_id": key,
                    "label": str(row[label_column]).strip(),
                    "ambiguous_labels": "",
                }
            )
        return rows
