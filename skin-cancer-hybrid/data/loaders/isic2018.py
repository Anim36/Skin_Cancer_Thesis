"""ISIC 2018 dataset loader (classification + segmentation masks).

Official layout
===============
The ISIC 2018 challenge archive contains several task folders.  The two this
project uses are:

* Task 1 - **Lesion Boundary Segmentation**
  ``ISIC2018_Training_Input/`` (images) and
  ``ISIC2018_Training_GroundTruth/`` (masks, PNG with a 0/255 mask).

* Task 3 - **Lesion Classification**
  ``ISIC2018_Training_Input/`` plus a ground truth CSV with columns
  ``image, diagnosis`` (and ``diagnosis_confirm_type`` in some releases).

Both tasks share the same image folder, so one loader serves both.  The
loader discovers the layout at runtime rather than assuming it, because the
archive is redistributed in several different folder arrangements.

Masks
=====
The official ISIC 2018 Task-1 masks are named ``ISIC_xxxxxxx_segmentation.png``
while the images are named ``ISIC_xxxxxxx.jpg``.  The two therefore do **not**
share a filename stem, so the lookup normalises both sides: the ``_segmentation``
suffix is stripped before the stems are compared.  A mask that really is named
``ISIC_xxxxxxx.png`` is matched as well.

When a mask is missing for an image the sample is still usable for
classification, but it is excluded from the segmentation split and the
discrepancy is reported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from data.loaders.base import BaseDatasetLoader, DatasetManifest

__all__ = ["ISIC2018Loader"]


_METADATA_PATTERNS = (
    "ISIC2018_Task3_Training_GroundTruth.csv",
    "ISIC2018_Training_GroundTruth.csv",
    "GT_2018.csv",
    "*Task3*GroundTruth*.csv",
    "*Task3*GroundTruth*.csv",
    "*GroundTruth*.csv",
    "*ground_truth*.csv",
    "*2018*label*.csv",
    "*classification*.csv",
)

#: Folder names used by the official ISIC 2018 challenge archive.  Task 1-2
#: holds the *segmentation* images plus their masks, Task 3 holds the
#: *classification* images and ground truth.
_IMAGE_DIR_PATTERNS = (
    "ISIC2018_Task3_Training_Input",
    "ISIC2018_Task1-2_Training_Input",
    "ISIC2018_Task1_Training_Input",
    "ISIC2018_Training_Input",
)

_MASK_DIR_PATTERNS = (
    "ISIC2018_Task1-2_Training_GroundTruth",
    "ISIC2018_Task1_Training_GroundTruth",
    "ISIC2018_Training_GroundTruth",
)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_MASK_SUFFIXES = {".png", ".tif", ".tiff", ".bmp"}

#: Suffixes that the ISIC 2018 Task-1 release appends to the mask filename.
#: The image is ``ISIC_0001234.jpg`` but its mask is
#: ``ISIC_0001234_segmentation.png``, so the stems only match after the
#: suffix is removed.
_MASK_SUFFIXES_TO_STRIP = (
    "_segmentation",
    "_segmentation_mask",
    "_mask",
    "_gt",
)


def _normalise_stem(stem: str) -> str:
    """Return the comparison key for an image/mask filename stem.

    Strips the known ISIC mask suffixes and lowercases the result so that
    ``ISIC_0001234`` (image) and ``ISIC_0001234_segmentation`` (mask) map to
    the same key.
    """
    key = stem.strip().lower()
    changed = True
    while changed:
        changed = False
        for suffix in _MASK_SUFFIXES_TO_STRIP:
            if key.endswith(suffix) and len(key) > len(suffix):
                key = key[: -len(suffix)]
                changed = True
    return key


class ISIC2018Loader(BaseDatasetLoader):
    """Loader for ISIC 2018 (Task 1 segmentation + Task 3 classification)."""

    name = "isic2018"
    default_subdir = "isic2018"
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
        self.mask_dir: Path | None = None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def _find_metadata_file(self) -> Path:
        """Return the *classification* ground-truth CSV.

        The official archive ships two ground-truth CSVs: the Task 1-2 one
        describes the segmentation masks and has **no** diagnosis column, while
        the Task 3 one carries the classification labels.  Picking the wrong
        one makes the label column undetectable, so the candidates are
        inspected and the first one that actually contains a diagnosis column
        wins.  A file without any usable label column is kept only as a last
        resort, and the loader then reports a clear error.
        """
        candidates: list[Path] = []
        for pattern in _METADATA_PATTERNS:
            candidates.extend(self.root.rglob(pattern))
        if not candidates:
            raise FileNotFoundError(
                f"{self.name}: no ground-truth CSV found under {self.root}. "
                "Download the ISIC 2018 challenge data (see DATASET_GUIDE.md)."
            )

        # Deduplicate while keeping the shallowest match first.
        seen: set[Path] = set()
        ordered: list[Path] = []
        for path in sorted(candidates, key=lambda p: (len(p.parts), str(p))):
            if path in seen:
                continue
            seen.add(path)
            ordered.append(path)

        fallback: Path | None = None
        for path in ordered:
            try:
                header = pd.read_csv(path, nrows=0)
            except Exception:
                continue
            columns = {str(c).strip() for c in header.columns}
            if columns & {"diagnosis", "dx", "label", "dx_type"}:
                return path
            if fallback is None:
                fallback = path
        if fallback is not None:
            return fallback
        raise FileNotFoundError(
            f"{self.name}: no ground-truth CSV found under {self.root}. "
            "Download the ISIC 2018 challenge data (see DATASET_GUIDE.md)."
        )

    def _find_image_dir(self) -> Path:
        for pattern in _IMAGE_DIR_PATTERNS:
            for path in self.root.rglob(pattern):
                if path.is_dir() and self._contains_images(path):
                    return path
        # Any directory containing images works as a fallback.
        for path in sorted(self.root.rglob("*")):
            if path.is_dir() and self._contains_images(path):
                return path
        raise FileNotFoundError(f"{self.name}: no image directory found under {self.root}")

    def _find_mask_dir(self) -> Path | None:
        for pattern in _MASK_DIR_PATTERNS:
            for path in self.root.rglob(pattern):
                if path.is_dir() and any(
                    p.suffix.lower() in _MASK_SUFFIXES for p in path.iterdir()
                ):
                    return path
        # Any directory whose files are mostly masks and whose name hints at
        # ground truth.  A mask folder is recognised by the ``_segmentation``
        # suffix that the official release uses.
        for path in sorted(self.root.rglob("*")):
            if not path.is_dir():
                continue
            name = path.name.lower()
            if "groundtruth" in name or "mask" in name or "segmentation" in name:
                if any(p.suffix.lower() in _MASK_SUFFIXES for p in path.iterdir()):
                    return path
        return None

    @staticmethod
    def _contains_images(path: Path) -> bool:
        return any(p.suffix.lower() in _IMAGE_SUFFIXES for p in path.iterdir())

    def _discover(self) -> pd.DataFrame:
        self.metadata_path = self._find_metadata_file()
        self.image_dir = self._find_image_dir()
        self.mask_dir = self._find_mask_dir()

        frame = pd.read_csv(self.metadata_path)
        frame.columns = [str(c).strip() for c in frame.columns]
        label_column = self._detect_label_column(frame)
        id_column = self._detect_id_column(frame)

        images = {
            _normalise_stem(p.stem): p
            for p in self.image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        }
        masks = {}
        if self.mask_dir is not None:
            masks = {
                _normalise_stem(p.stem): p
                for p in self.mask_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in _MASK_SUFFIXES
            }

        rows: list[dict[str, Any]] = []
        missing_masks = 0
        for _, row in frame.iterrows():
            key = str(row[id_column]).strip()
            lookup_key = _normalise_stem(key)
            path = images.get(lookup_key)
            if path is None:
                continue
            mask_path = masks.get(lookup_key)
            if mask_path is None:
                missing_masks += 1
            rows.append(
                {
                    "image_path": path,
                    "image_id": key,
                    "label": str(row[label_column]).strip(),
                    "mask_path": mask_path,
                    "has_mask": mask_path is not None,
                }
            )

        if not rows:
            raise RuntimeError(
                f"{self.name}: metadata rows did not match any image file in "
                f"{self.image_dir}."
            )

        self.notes = [
            f"metadata file: {self.metadata_path.name}",
            f"image dir: {self.image_dir.name}",
            f"mask dir: {self.mask_dir.name if self.mask_dir else 'NOT FOUND'}",
            f"images with a segmentation mask: "
            f"{len(rows) - missing_masks}/{len(rows)}",
        ]
        if self.mask_dir is None:
            self.notes.append(
                "LIMITATION: no segmentation ground truth found. Classification "
                "is unaffected; the U-Net segmentation stage cannot be trained "
                "on this dataset."
            )

        result = pd.DataFrame(rows)
        result.attrs["mask_dir"] = self.mask_dir
        return result

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
        for candidate in ("image", "image_id", "id", "image_name"):
            if candidate in frame.columns:
                return candidate
        raise RuntimeError(
            f"{self.name}: no image identifier column found. Available: "
            f"{list(frame.columns)}"
        )

    # ------------------------------------------------------------------
    # Segmentation pair discovery
    # ------------------------------------------------------------------
    def _find_segmentation_image_dir(self, masks: dict[str, Path]) -> Path | None:
        """Return the image folder that actually matches the given masks.

        In the official archive the segmentation images (Task 1-2) live in a
        *different* folder from the classification images (Task 3), and both
        sit next to each other under the dataset root.  Folder-name priority is
        therefore not enough: the candidate that matches the most masks wins.
        """
        candidates: list[Path] = []
        for pattern in _IMAGE_DIR_PATTERNS:
            for path in self.root.rglob(pattern):
                if path.is_dir() and self._contains_images(path):
                    candidates.append(path)
        for path in sorted(self.root.rglob("*")):
            if path.is_dir() and self._contains_images(path) and path not in candidates:
                candidates.append(path)
        if not candidates:
            return None

        best: Path | None = None
        best_score = -1
        for path in candidates:
            keys = {
                _normalise_stem(p.stem)
                for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
            }
            score = len(keys & set(masks))
            if score > best_score:
                best, best_score = path, score
        # Require at least one overlap, otherwise the pairing is meaningless.
        return best if best_score > 0 else None

    def build_segmentation_pairs(self) -> pd.DataFrame:
        """Return the image/mask pairs used to train the U-Net.

        The pairs are resolved from the mask folder and the image folder that
        actually matches it, independently of the classification ground truth,
        because the official archive keeps the segmentation images (Task 1-2)
        and the classification images (Task 3) in separate folders.

        The diagnosis label is joined in when the classification ground truth
        covers the same image; it is left empty otherwise.
        """
        self.mask_dir = self._find_mask_dir()
        if self.mask_dir is None:
            raise RuntimeError(
                f"{self.name}: no segmentation ground truth found under "
                f"{self.root}. The U-Net cannot be trained without the "
                "ISIC 2018 Task 1-2 masks (see DATASET_GUIDE.md)."
            )

        masks = {
            _normalise_stem(p.stem): p
            for p in self.mask_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in _MASK_SUFFIXES
        }
        if not masks:
            raise RuntimeError(
                f"{self.name}: the mask folder {self.mask_dir.name} holds no "
                f"readable mask files (suffixes: {sorted(_MASK_SUFFIXES)})."
            )

        segmentation_image_dir = self._find_segmentation_image_dir(masks)
        if segmentation_image_dir is None:
            raise RuntimeError(
                f"{self.name}: found the mask folder {self.mask_dir.name} but "
                "no image folder whose files match those masks. Check that the "
                "ground truth folder sits next to the image folder."
            )

        images = {
            _normalise_stem(p.stem): p
            for p in segmentation_image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        }

        # Best-effort: attach the diagnosis label from the classification GT.
        labels: dict[str, str] = {}
        try:
            classification = self._discover()
            for _, row in classification.iterrows():
                labels[_normalise_stem(str(row["image_id"]))] = str(row["label"])
        except Exception:
            labels = {}

        rows: list[dict[str, Any]] = []
        missing_images = 0
        for key, mask_path in masks.items():
            image_path = images.get(key)
            if image_path is None:
                missing_images += 1
                continue
            rows.append(
                {
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "image_id": image_path.stem,
                    "label": labels.get(key, ""),
                }
            )

        if not rows:
            raise RuntimeError(
                f"{self.name}: no image/mask pairs found. The mask folder "
                f"{self.mask_dir.name} holds {len(masks)} masks but none of "
                f"them matched an image in {segmentation_image_dir.name}."
            )

        pairs = pd.DataFrame(rows)
        self.notes.append(
            f"segmentation pairs: {len(pairs)} (image folder "
            f"{segmentation_image_dir.name}, mask folder {self.mask_dir.name})"
            + (f"; {missing_images} masks had no matching image" if missing_images else "")
        )
        return pairs[["image_path", "mask_path", "image_id", "label"]]
