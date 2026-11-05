"""Regression tests for the defects found in the GPU-readiness audit.

Each test pins a bug that would otherwise have surfaced only during a real
GPU training run:

1. ISIC 2018 official mask naming (``*_segmentation.png``).
2. ISIC 2018 official archive layout (Task 1-2 / Task 3).
3. Samples outside the active label space (SCC in the 7-class head) must
   never reach the loss - a negative target crashes CrossEntropy / FocalLoss.
4. ``class_index`` must index into the model's class list, not the full
   target space.
5. A stale split manifest must not leave validation and test empty.
6. Grad-CAM must work inside a ``torch.no_grad()`` block.
7. Batch prediction must not raise a NameError.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


# ---------------------------------------------------------------------------
# 1 + 2. ISIC 2018 official archive
# ---------------------------------------------------------------------------
class TestISIC2018OfficialArchive:
    def _build(self, tmp_path: Path) -> Path:
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
        # The official Task 1-2 CSV has no diagnosis column.
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

    def test_classification_ground_truth_is_selected(self, tmp_path):
        from data.loaders.isic2018 import ISIC2018Loader

        loader = ISIC2018Loader(root=self._build(tmp_path))
        manifest = loader.build_manifest()
        assert loader.metadata_path.name == "ISIC2018_Task3_Training_GroundTruth.csv"
        assert len(manifest) == 6

    def test_segmentation_pairs_use_the_task1_images(self, tmp_path):
        from data.loaders.isic2018 import ISIC2018Loader

        loader = ISIC2018Loader(root=self._build(tmp_path))
        pairs = loader.build_segmentation_pairs()
        assert len(pairs) == 3
        for _, row in pairs.iterrows():
            assert Path(row["mask_path"]).name.endswith("_segmentation.png")
            assert "Task1-2" in str(row["image_path"])

    def test_official_mask_suffix_pairs_with_image(self, tmp_path):
        """The image stem and the mask stem differ by ``_segmentation``."""
        from data.loaders.isic2018 import _normalise_stem

        assert _normalise_stem("ISIC_0001234_segmentation") == _normalise_stem("ISIC_0001234")
        assert _normalise_stem("ISIC_0001234") == "isic_0001234"


# ---------------------------------------------------------------------------
# 3 + 4. Label space and class index alignment
# ---------------------------------------------------------------------------
class TestLabelSpaceHandling:
    def _manifest(self, native_labels):
        """Build a manifest frame with the harmonisation columns."""
        from data.label_mapping import UNIFIED_CLASSES, build_label_mapping

        mapping = build_label_mapping(
            dataset="isic2019",
            native_labels=native_labels,
            target_classes=UNIFIED_CLASSES,
            exclude_from_unified={"SCC"},
        )
        rows = []
        for label in native_labels:
            unified = mapping.to_unified(label)
            rows.append(
                {
                    "native_label": label,
                    "unified_label": unified or "",
                    "unified_label_valid": unified is not None,
                    "class_index": mapping.to_unified_index(label),
                }
            )
        return pd.DataFrame(rows), mapping

    def test_excluded_class_has_invalid_flag_and_negative_index(self):
        frame, _ = self._manifest(["MEL", "NV", "SCC"])
        scc = frame[frame["native_label"] == "SCC"].iloc[0]
        assert not bool(scc["unified_label_valid"])
        assert int(scc["class_index"]) == -1

    def test_scc_is_never_remapped_onto_akiec(self):
        frame, _ = self._manifest(["MEL", "NV", "SCC", "AKIEC"])
        scc = frame[frame["native_label"] == "SCC"].iloc[0]
        assert scc["unified_label"] == ""
        assert scc["unified_label"] != "AKIEC"

    def test_harmonisation_reindexes_onto_the_present_class_list(self):
        """class_index must index into class_list, not the full target space."""
        from data.dataset_harmonization import harmonise_manifests
        from data.loaders.base import DatasetManifest, SampleRecord

        records = [
            SampleRecord(
                sample_id="pad_ufes20:1",
                dataset="pad_ufes20",
                image_path="a.png",
                relative_path="a.png",
                original_id="1",
                native_label="melanoma",
                native_label_norm="melanoma",
                unified_label="MEL",
                unified_label_valid=True,
                class_index=0,
                group_id="",
                group_type="none",
                has_metadata=False,
            ),
            SampleRecord(
                sample_id="pad_ufes20:2",
                dataset="pad_ufes20",
                image_path="b.png",
                relative_path="b.png",
                original_id="2",
                native_label="squamous cell carcinoma",
                native_label_norm="squamous cell carcinoma",
                unified_label="",
                unified_label_valid=False,
                class_index=-1,
                group_id="",
                group_type="none",
                has_metadata=False,
            ),
        ]
        manifest = DatasetManifest(dataset="pad_ufes20", records=records)
        result = harmonise_manifests([manifest])

        assert result.class_list == ["MEL"]
        frame = result.frame
        valid = frame[frame["unified_label_valid"].astype(str).str.lower().isin({"true", "1"})]
        assert set(valid["class_index"]) <= set(range(len(result.class_list)))

    def test_excluded_samples_never_reach_the_loss(self):
        """A negative target crashes CrossEntropy - it must be filtered out."""
        from models.losses import build_loss

        criterion = build_loss("weighted_focal", num_classes=7)
        logits = torch.randn(2, 7)
        # A target of -1 would raise "index -1 is out of bounds".
        with pytest.raises(RuntimeError):
            criterion(logits, torch.tensor([0, -1]))

    def test_pipeline_filters_invalid_labels(self, tmp_path):
        """prepare_data must exclude samples outside the active label space."""
        import cv2

        from data.pipeline import prepare_data
        from utils.config import load_config

        root = tmp_path / "data" / "isic2019"
        image_dir = root / "ISIC_2019_Training_Input"
        image_dir.mkdir(parents=True)

        classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "SCC"]
        rows = []
        for index in range(40):
            image_id = f"ISIC_{index:05d}"
            cv2.imwrite(
                str(image_dir / f"{image_id}.jpg"),
                (np.random.default_rng(index).random((32, 32, 3)) * 255).astype(np.uint8),
            )
            row = {"image": image_id}
            for name in classes:
                row[name] = 0
            row[classes[index % 8]] = 1
            rows.append(row)
        pd.DataFrame(rows).to_csv(root / "ISIC_2019_Training_GroundTruth.csv", index=False)

        config = load_config(
            Path(__file__).resolve().parents[1] / "configs" / "default.yaml",
            root=Path(__file__).resolve().parents[1],
        )
        config.set("paths.raw_dir", str(tmp_path / "data"))
        config.set("paths.splits_dir", str(tmp_path / "splits"))

        prepared = prepare_data(
            config, datasets=["isic2019"], raw_dir=str(tmp_path / "data"),
            include_scc=False, reuse_split=False, write_reports=False,
        )

        frame = prepared.manifest
        usable = frame[frame["unified_label_valid"].astype(str).str.lower().isin({"true", "1"})]
        assert len(usable) == 35, "the 5 SCC samples must be excluded"
        assert (pd.to_numeric(usable["class_index"]) >= 0).all()

        # And the excluded samples must still be visible in the manifest.
        assert len(frame) == 40, "excluded samples must be retained in the manifest"


# ---------------------------------------------------------------------------
# 5. Split handling
# ---------------------------------------------------------------------------
class TestSplitHandling:
    def test_split_file_name_encodes_the_dataset_selection(self):
        from data.splitting import split_file_for

        assert split_file_for("data/splits", ["ham10000"]).name == "split_manifest_ham10000.csv"
        assert split_file_for("data/splits", ["ham10000", "isic2018"]).name == "split_manifest_all.csv"

    def test_a_stale_split_does_not_empty_validation_and_test(self, tmp_path):
        """A split file that covers almost none of the manifest is recomputed."""
        import cv2

        from data.pipeline import prepare_data
        from utils.config import load_config

        root = tmp_path / "data" / "ham10000"
        image_dir = root / "HAM10000_images"
        image_dir.mkdir(parents=True)

        labels = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
        rows = []
        for index in range(70):
            image_id = f"ISIC_{index:05d}"
            cv2.imwrite(
                str(image_dir / f"{image_id}.jpg"),
                (np.random.default_rng(index).random((32, 32, 3)) * 255).astype(np.uint8),
            )
            rows.append({"image_id": image_id, "lesion_id": index, "dx": labels[index % 7]})
        pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)

        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()
        # A stale manifest that knows almost none of the samples.
        pd.DataFrame(
            [{"sample_id": "ham10000:unknown", "split": "train"}]
        ).to_csv(splits_dir / "split_manifest_ham10000.csv", index=False)

        config = load_config(
            Path(__file__).resolve().parents[1] / "configs" / "default.yaml",
            root=Path(__file__).resolve().parents[1],
        )
        config.set("paths.raw_dir", str(tmp_path / "data"))
        config.set("paths.splits_dir", str(splits_dir))

        prepared = prepare_data(
            config, datasets=["ham10000"], raw_dir=str(tmp_path / "data"),
            include_scc=False, reuse_split=True, write_reports=False,
        )

        assert len(prepared.val_frame) > 0, "validation split must not be empty"
        assert len(prepared.test_frame) > 0, "test split must not be empty"
        assert len(prepared.train_frame) > 0


# ---------------------------------------------------------------------------
# 6. Grad-CAM inside torch.no_grad()
# ---------------------------------------------------------------------------
class TestGradCamInsideNoGrad:
    def test_gradcam_works_inside_no_grad(self):
        from explainability.gradcam import GradCAM
        from models.hybrid_model import HybridModelConfig, HybridSkinCancerModel

        model = HybridSkinCancerModel(
            HybridModelConfig(
                pretrained=False, use_attention=True, use_metadata=False,
                fusion_dim=16, num_classes=7, head_dims=(8,),
            )
        )
        model.eval()
        images = torch.randn(1, 3, 224, 224)

        explainer = GradCAM(model)
        try:
            with torch.no_grad():
                heatmaps, overlays, probabilities = explainer(images)
        finally:
            explainer.remove_hooks()

        assert heatmaps.shape == (1, 224, 224)
        assert overlays.shape == (1, 224, 224, 3)
        assert probabilities.shape == (1, 7)

    def test_predict_with_gradcam_succeeds(self, tmp_path):
        """The CLI path: predict() is @torch.no_grad() and calls explain()."""
        import cv2

        from inference.predict import SkinCancerPredictor
        from models.hybrid_model import (
            HybridModelConfig,
            HybridSkinCancerModel,
            model_config_to_dict,
        )

        image_path = tmp_path / "image.jpg"
        cv2.imwrite(str(image_path),
                    (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8))

        model_config = HybridModelConfig(
            pretrained=False, use_attention=True, use_metadata=False,
            fusion_dim=16, num_classes=7, head_dims=(8,),
        )
        model = HybridSkinCancerModel(model_config)

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint = checkpoint_dir / "best.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": model_config_to_dict(model_config),
                "class_names": ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"],
                "num_classes": 7,
                "metadata_dim": 0,
                "epoch": 0,
            },
            checkpoint,
        )

        predictor = SkinCancerPredictor.from_checkpoint(checkpoint, device="cpu")
        result = predictor.predict(image_path, gradcam=True)
        assert result.predicted_class in predictor.class_names
        # The Grad-CAM payload is stored under the "gradcam" key.
        assert result.get("gradcam") is not None
        assert np.asarray(result["gradcam"]["heatmap"]).shape == (224, 224)


# ---------------------------------------------------------------------------
# 7. Batch prediction must not raise a NameError
# ---------------------------------------------------------------------------
class TestBatchPredictionCli:
    def test_predict_batch_module_has_no_undefined_disclaimer(self):
        source = Path(
            Path(__file__).resolve().parents[1] / "inference" / "predict_batch.py"
        ).read_text(encoding="utf-8")
        assert "{disclaimer}" not in source, (
            "print(f\"\\n{disclaimer}\") raises NameError - the constant is "
            "DISCLAIMER"
        )
