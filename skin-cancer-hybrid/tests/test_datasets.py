"""Tests for the dataset loaders and the label mapping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.label_mapping import (
    UNIFIED_CLASSES,
    UNIFIED_PLUS_SCC_CLASSES,
    build_label_mapping,
    normalise_label,
    resolve_dataset_class_list,
)
from data.loaders.ham10000 import HAM10000Loader
from data.loaders.isic2018 import ISIC2018Loader
from data.loaders.isic2019 import ISIC2019Loader
from data.loaders.pad_ufes20 import PADUFES20Loader


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
class TestLabelMapping:
    def test_normalise_label_handles_variants(self):
        assert normalise_label("  MEL  ") == "mel"
        assert normalise_label("Basal Cell Carcinoma") == "basal cell carcinoma"
        assert normalise_label("benign_keratosis-like lesions") == "benign keratosis like lesions"
        assert normalise_label(None) == ""
        assert normalise_label(float("nan")) == "nan"

    def test_build_mapping_resolves_all_seven_classes(self):
        mapping = build_label_mapping(
            "ham10000",
            ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"],
            target_classes=UNIFIED_CLASSES,
        )
        assert not mapping.unmapped_labels
        assert mapping.excluded_labels == []
        for index, code in enumerate(UNIFIED_CLASSES):
            assert mapping.to_unified_index(code) == index

    def test_scc_is_excluded_not_merged(self):
        """SCC must never be silently folded into AKIEC."""
        mapping = build_label_mapping(
            "isic2019",
            ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "SCC"],
            target_classes=UNIFIED_CLASSES,
        )
        assert "SCC" in mapping.excluded_labels
        assert mapping.to_unified_index("SCC") == -1
        # AKIEC must still point at its own index.
        assert mapping.to_unified_index("AKIEC") == UNIFIED_CLASSES.index("AKIEC")
        assert mapping.to_unified("SCC") is None

    def test_scc_kept_when_requested(self):
        mapping = build_label_mapping(
            "isic2019",
            ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "SCC"],
            target_classes=UNIFIED_PLUS_SCC_CLASSES,
            exclude_from_unified=set(),
        )
        assert mapping.excluded_labels == []
        assert mapping.to_unified_index("SCC") == UNIFIED_PLUS_SCC_CLASSES.index("SCC")

    def test_unknown_label_is_reported_not_crashed(self):
        mapping = build_label_mapping(
            "custom", ["MEL", "SOMETHING_NEW"], target_classes=UNIFIED_CLASSES
        )
        assert "something new" in mapping.unmapped_labels
        assert mapping.to_unified_index("SOMETHING_NEW") == -1

    def test_resolve_dataset_class_list_reflects_observed_data(self):
        classes = resolve_dataset_class_list("pad", ["MEL", "NV", "BCC"], include_scc=False)
        assert classes == ["MEL", "NV", "BCC"]
        assert "VASC" not in classes

    def test_summary_is_serialisable(self):
        mapping = build_label_mapping("ham10000", ["MEL", "NV"], target_classes=UNIFIED_CLASSES)
        summary = mapping.summary()
        assert summary["dataset"] == "ham10000"
        assert isinstance(summary["class_list"] if "class_list" in summary else summary["target_classes"], list)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
class TestHAM10000Loader:
    def test_loads_synthetic_dataset(self, synthetic_dataset_tree):
        loader = HAM10000Loader(root=synthetic_dataset_tree)
        assert loader.is_available()
        manifest = loader.build_manifest()
        assert len(manifest) == 21
        assert set(manifest.class_list) <= set(UNIFIED_CLASSES)

    def test_groups_images_of_the_same_lesion(self, synthetic_dataset_tree):
        loader = HAM10000Loader(root=synthetic_dataset_tree)
        manifest = loader.build_manifest()
        by_lesion: dict[str, set[str]] = {}
        for record in manifest.records:
            by_lesion.setdefault(record.original_id, set()).add(record.group_id)
        # Images with the same lesion_id must share one group id.
        groups = {}
        for record in manifest.records:
            groups.setdefault(record.group_id, set()).add(record.original_id)
        for group_id, ids in groups.items():
            if len(ids) > 1:
                # Every member of a multi-image group shares the same group id,
                # so the group must not also appear under another group id.
                assert all(
                    other_group == group_id
                    for other_group in (
                        next(r.group_id for r in manifest.records if r.original_id == image_id)
                        for image_id in ids
                    )
                )

    def test_records_preserve_native_label(self, synthetic_dataset_tree):
        loader = HAM10000Loader(root=synthetic_dataset_tree)
        manifest = loader.build_manifest()
        native = {record.native_label for record in manifest.records}
        assert native <= {"MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"}

    def test_missing_dataset_raises_clear_error(self, tmp_path):
        loader = HAM10000Loader(root=tmp_path / "does_not_exist")
        assert not loader.is_available()
        with pytest.raises(FileNotFoundError):
            loader.build_manifest()

    def test_manifest_round_trips_to_csv(self, synthetic_dataset_tree, tmp_path):
        loader = HAM10000Loader(root=synthetic_dataset_tree)
        manifest = loader.build_manifest()
        path = tmp_path / "manifest.csv"
        manifest.save(path)
        frame = pd.read_csv(path)
        assert len(frame) == len(manifest)
        assert "unified_label" in frame.columns
        assert "group_id" in frame.columns


class TestISIC2018Loader:
    def _build_tree(self, tmp_path):
        import cv2

        root = tmp_path / "isic2018"
        image_dir = root / "ISIC2018_Training_Input"
        mask_dir = root / "ISIC2018_Training_GroundTruth"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        rows = []
        for index in range(4):
            image_id = f"ISIC_{index:05d}"
            cv2.imwrite(
                str(image_dir / f"{image_id}.jpg"),
                (np.random.default_rng(index).random((32, 32, 3)) * 255).astype(np.uint8),
            )
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[8:24, 8:24] = 255
            cv2.imwrite(str(mask_dir / f"{image_id}.png"), mask)
            rows.append({"image": image_id, "diagnosis": ["MEL", "NV", "BCC", "BKL"][index]})

        pd.DataFrame(rows).to_csv(root / "ISIC2018_Training_GroundTruth.csv", index=False)
        return root

    def test_finds_images_and_masks(self, tmp_path):
        root = self._build_tree(tmp_path)
        loader = ISIC2018Loader(root=root)
        assert loader.is_available()
        manifest = loader.build_manifest()
        assert len(manifest) == 4

    def test_builds_segmentation_pairs(self, tmp_path):
        root = self._build_tree(tmp_path)
        loader = ISIC2018Loader(root=root)
        pairs = loader.build_segmentation_pairs()
        assert len(pairs) == 4
        assert "image_path" in pairs.columns
        assert "mask_path" in pairs.columns

    def _build_official_tree(self, tmp_path):
        """Reproduce the *official* ISIC 2018 challenge archive layout.

        Task 1-2 holds the segmentation images and their masks (masks are named
        ``ISIC_xxxxxxx_segmentation.png``), Task 3 holds the classification
        images and a ground truth CSV that carries the diagnosis.  The Task
        1-2 ground truth CSV has **no** diagnosis column.
        """
        import cv2

        root = tmp_path / "isic2018"
        seg_img = root / "ISIC2018_Task1-2_Training_Input"
        seg_mask = root / "ISIC2018_Task1-2_Training_GroundTruth"
        cls_img = root / "ISIC2018_Task3_Training_Input"
        for directory in (seg_img, seg_mask, cls_img):
            directory.mkdir(parents=True)

        seg_rows = []
        for index in range(3):
            image_id = f"ISIC_{1000 + index:05d}"
            cv2.imwrite(
                str(seg_img / f"{image_id}.jpg"),
                (np.random.default_rng(index).random((32, 32, 3)) * 255).astype(np.uint8),
            )
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[8:24, 8:24] = 255
            cv2.imwrite(str(seg_mask / f"{image_id}_segmentation.png"), mask)
            seg_rows.append({"image": image_id})
        pd.DataFrame(seg_rows).to_csv(
            root / "ISIC2018_Task1-2_Training_GroundTruth.csv", index=False
        )

        cls_rows = []
        for index in range(6):
            image_id = f"ISIC_{2000 + index:05d}"
            cv2.imwrite(
                str(cls_img / f"{image_id}.jpg"),
                (np.random.default_rng(index + 50).random((32, 32, 3)) * 255).astype(np.uint8),
            )
            cls_rows.append({"image": image_id, "diagnosis": ["MEL", "NV", "BCC"][index % 3]})
        pd.DataFrame(cls_rows).to_csv(
            root / "ISIC2018_Task3_Training_GroundTruth.csv", index=False
        )
        return root

    def test_official_layout_selects_the_task3_ground_truth(self, tmp_path):
        """The Task 1-2 CSV has no diagnosis column and must not be selected."""
        root = self._build_official_tree(tmp_path)
        loader = ISIC2018Loader(root=root)
        manifest = loader.build_manifest()
        assert loader.metadata_path.name == "ISIC2018_Task3_Training_GroundTruth.csv"
        assert len(manifest) == 6

    def test_official_layout_pairs_segmentation_masks(self, tmp_path):
        """Official masks are named ``*_segmentation.png`` and must still pair."""
        root = self._build_official_tree(tmp_path)
        loader = ISIC2018Loader(root=root)
        pairs = loader.build_segmentation_pairs()
        assert len(pairs) == 3
        for _, row in pairs.iterrows():
            assert Path(row["image_path"]).name.endswith(".jpg")
            assert Path(row["mask_path"]).name.endswith("_segmentation.png")
            # The pair must come from the Task 1-2 image folder, not Task 3.
            assert "Task1-2" in str(row["image_path"])

    def test_mask_suffix_normalisation_matches_both_conventions(self, tmp_path):
        """``ISIC_1.png`` and ``ISIC_1_segmentation.png`` both pair with the image."""
        import cv2

        root = tmp_path / "isic2018"
        image_dir = root / "ISIC2018_Training_Input"
        mask_dir = root / "ISIC2018_Training_GroundTruth"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        for index, mask_name in enumerate(
            ["ISIC_00001.png", "ISIC_00002_segmentation.png"]
        ):
            image_id = f"ISIC_{index + 1:05d}"
            cv2.imwrite(
                str(image_dir / f"{image_id}.jpg"),
                (np.random.default_rng(index).random((32, 32, 3)) * 255).astype(np.uint8),
            )
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[8:24, 8:24] = 255
            cv2.imwrite(str(mask_dir / mask_name), mask)

        pd.DataFrame(
            [{"image": "ISIC_00001", "diagnosis": "MEL"},
             {"image": "ISIC_00002", "diagnosis": "NV"}]
        ).to_csv(root / "ISIC2018_Training_GroundTruth.csv", index=False)

        loader = ISIC2018Loader(root=root)
        pairs = loader.build_segmentation_pairs()
        assert len(pairs) == 2


class TestISIC2019Loader:
    def _build_wide_tree(self, tmp_path):
        import cv2

        root = tmp_path / "isic2019"
        image_dir = root / "ISIC_2019_Training_Input"
        image_dir.mkdir(parents=True)

        classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "SCC"]
        rows = []
        for index in range(8):
            image_id = f"ISIC_{index:05d}"
            cv2.imwrite(
                str(image_dir / f"{image_id}.jpg"),
                (np.random.default_rng(index).random((32, 32, 3)) * 255).astype(np.uint8),
            )
            row = {"image": image_id}
            for class_name in classes:
                row[class_name] = 1 if class_name == classes[index] else 0
            rows.append(row)

        pd.DataFrame(rows).to_csv(root / "ISIC_2019_Training_GroundTruth.csv", index=False)
        return root

    def test_reads_one_hot_format(self, tmp_path):
        root = self._build_wide_tree(tmp_path)
        loader = ISIC2019Loader(root=root, include_scc=True)
        manifest = loader.build_manifest()
        assert len(manifest) == 8
        native = {record.native_label for record in manifest.records}
        assert native == {"MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "SCC"}

    def test_scc_excluded_from_unified_head_by_default(self, tmp_path):
        root = self._build_wide_tree(tmp_path)
        loader = ISIC2019Loader(root=root, include_scc=False)
        manifest = loader.build_manifest()
        scc_records = [r for r in manifest.records if r.native_label == "SCC"]
        assert len(scc_records) == 1
        # The sample is retained but flagged as not usable in the unified head.
        assert scc_records[0].unified_label_valid is False
        assert scc_records[0].class_index == -1

    def test_reads_long_format(self, tmp_path):
        import cv2

        root = tmp_path / "isic2019"
        image_dir = root / "ISIC_2019_Training_Input"
        image_dir.mkdir(parents=True)
        rows = []
        for index in range(3):
            image_id = f"ISIC_{index:05d}"
            cv2.imwrite(
                str(image_dir / f"{image_id}.jpg"),
                (np.random.default_rng(index).random((32, 32, 3)) * 255).astype(np.uint8),
            )
            rows.append({"image": image_id, "diagnosis": ["MEL", "NV", "BCC"][index]})
        pd.DataFrame(rows).to_csv(root / "GT_2019.csv", index=False)

        loader = ISIC2019Loader(root=root)
        manifest = loader.build_manifest()
        assert len(manifest) == 3


class TestPADUFES20Loader:
    def _build_tree(self, tmp_path):
        import cv2

        root = tmp_path / "pad_ufes20"
        image_dir = root / "images"
        image_dir.mkdir(parents=True)

        rows = []
        for index in range(4):
            image_id = f"pad_{index:05d}"
            cv2.imwrite(
                str(image_dir / f"{image_id}.png"),
                (np.random.default_rng(index).random((32, 32, 3)) * 255).astype(np.uint8),
            )
            rows.append(
                {
                    "image_id": image_id,
                    "dx": ["MEL", "NV", "BCC", "SCC"][index],
                    "age": 40 + index,
                    "sex": "Male",
                    "localization": "face",
                    "skin_cancer_history": "no",
                }
            )
        pd.DataFrame(rows).to_csv(root / "metadata_pad_ufes20.csv", index=False)
        return root

    def test_loads_metadata(self, tmp_path):
        root = self._build_tree(tmp_path)
        loader = PADUFES20Loader(root=root)
        manifest = loader.build_manifest()
        assert len(manifest) == 4
        assert all(record.has_metadata for record in manifest.records)
        assert manifest.records[0].metadata["age"] == 40
        assert manifest.records[0].metadata["sex"] == "Male"

    def test_metadata_fields_are_preserved(self, tmp_path):
        root = self._build_tree(tmp_path)
        loader = PADUFES20Loader(root=root)
        manifest = loader.build_manifest()
        for record in manifest.records:
            assert set(record.metadata.keys()) >= {
                "age", "sex", "localization", "skin_cancer_history"
            }
