"""Tests for the segmentation visualisation, the splitting policy and the
deduplication logic.

These fill the gaps in the required coverage list:

* segmentation visualisation,
* group-aware splitting (patient / lesion),
* duplicate and near-duplicate detection,
* cross-dataset leakage prevention,
* checkpoint save/load round-trip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Segmentation visualisation
# ---------------------------------------------------------------------------
class TestSegmentationVisualisation:
    def test_overlay_preserves_shape_and_dtype(self, synthetic_image, synthetic_mask):
        from explainability.segmentation_viz import overlay_mask

        result = overlay_mask(synthetic_image, synthetic_mask)
        assert result.shape == synthetic_image.shape
        assert result.dtype == np.uint8

    def test_overlay_only_changes_the_masked_region(self, synthetic_image):
        """Pixels outside the mask must be left untouched."""
        from explainability.segmentation_viz import overlay_mask

        mask = np.zeros(synthetic_image.shape[:2], dtype=np.uint8)
        mask[10:30, 10:30] = 255

        result = overlay_mask(synthetic_image, mask, alpha=0.5)
        outside = mask == 0
        assert np.array_equal(result[outside], synthetic_image[outside])

    def test_overlay_changes_the_masked_region(self, synthetic_image, synthetic_mask):
        from explainability.segmentation_viz import overlay_mask

        result = overlay_mask(synthetic_image, synthetic_mask, alpha=0.5)
        inside = synthetic_mask > 0
        assert not np.array_equal(result[inside], synthetic_image[inside])

    def test_overlay_rejects_a_mismatched_mask(self, synthetic_image):
        from explainability.segmentation_viz import overlay_mask

        with pytest.raises(ValueError):
            overlay_mask(synthetic_image, np.zeros((5, 5), dtype=np.uint8))

    def test_mask_contours_returns_the_right_shape(self, synthetic_mask):
        from explainability.segmentation_viz import mask_contours

        outline = mask_contours(synthetic_mask)
        assert outline.shape == synthetic_mask.shape[:2]
        assert outline.dtype == bool
        # A contour is a thin structure, so it covers far fewer pixels than
        # the filled mask.
        assert outline.sum() < (synthetic_mask > 0).sum()

    def test_save_overlay_writes_a_png(self, tmp_path, synthetic_image, synthetic_mask):
        from explainability.segmentation_viz import save_segmentation_overlay

        destination = tmp_path / "overlay.png"
        written = save_segmentation_overlay(
            synthetic_image, synthetic_mask, target_mask=synthetic_mask,
            output_path=destination, metrics={"dice": 0.87, "iou": 0.78},
        )
        assert written.exists()
        assert written.stat().st_size > 0

    def test_save_grid_writes_a_png(self, tmp_path, synthetic_image, synthetic_mask):
        from explainability.segmentation_viz import save_segmentation_grid

        destination = tmp_path / "grid.png"
        written = save_segmentation_grid(
            [synthetic_image, synthetic_image],
            [synthetic_mask, synthetic_mask],
            target_masks=[synthetic_mask, synthetic_mask],
            output_path=destination,
            metrics=[{"dice": 0.9}, {"dice": 0.5}],
        )
        assert written.exists()
        assert written.stat().st_size > 0

    def test_save_grid_rejects_mismatched_lengths(self, tmp_path, synthetic_image, synthetic_mask):
        from explainability.segmentation_viz import save_segmentation_grid

        with pytest.raises(ValueError):
            save_segmentation_grid(
                [synthetic_image, synthetic_image], [synthetic_mask],
                output_path=tmp_path / "grid.png",
            )


# ---------------------------------------------------------------------------
# Group-aware splitting
# ---------------------------------------------------------------------------
class TestGroupAwareSplitting:
    def _frame(self, groups_per_class: int = 10, classes: int = 3):
        import pandas as pd

        rows = []
        index = 0
        for class_index in range(classes):
            for group_index in range(groups_per_class):
                for member in range(2):
                    rows.append(
                        {
                            "sample_id": f"ham10000:S{index:04d}",
                            "dataset": "ham10000",
                            "unified_label": ["MEL", "NV", "BCC"][class_index],
                            "unified_label_valid": True,
                            "class_index": class_index,
                            "group_id": f"ham10000_lesion_{class_index}_{group_index}",
                            "group_type": "lesion",
                            "has_metadata": False,
                            "metadata": "{}",
                        }
                    )
                    index += 1
        return pd.DataFrame(rows)

    def test_every_group_lands_in_exactly_one_split(self, synthetic_manifest):
        """The core leakage guarantee: no group may straddle two splits."""
        from data.splitting import SplitConfig, make_splits

        frame = self._frame()
        result = make_splits(frame, SplitConfig(seed=42, group_by="auto"))

        assignments = result.frame.groupby("group_id")["split"].nunique()
        assert (assignments == 1).all(), "a group was split across two splits"

    def test_all_three_splits_are_populated(self, synthetic_manifest):
        from data.splitting import SplitConfig, make_splits

        result = make_splits(self._frame(), SplitConfig(seed=42, group_by="auto"))
        counts = result.frame["split"].value_counts()
        for name in ("train", "val", "test"):
            assert counts.get(name, 0) > 0

    def test_the_split_respects_the_requested_ratios(self, synthetic_manifest):
        from data.splitting import SplitConfig, make_splits

        result = make_splits(
            self._frame(),
            SplitConfig(seed=42, group_by="auto",
                        train_ratio=0.70, val_ratio=0.15, test_ratio=0.15),
        )
        total = len(result.frame)
        counts = result.frame["split"].value_counts()
        assert abs(counts["train"] / total - 0.70) < 0.12
        assert abs(counts["val"] / total - 0.15) < 0.12
        assert abs(counts["test"] / total - 0.15) < 0.12

    def test_splitting_is_deterministic_for_a_fixed_seed(self, synthetic_manifest):
        from data.splitting import SplitConfig, make_splits

        frame = self._frame()
        first = make_splits(frame, SplitConfig(seed=7, group_by="auto"))
        second = make_splits(frame, SplitConfig(seed=7, group_by="auto"))

        mapping = dict(zip(first.frame["sample_id"], first.frame["split"]))
        other = dict(zip(second.frame["sample_id"], second.frame["split"]))
        assert mapping == other

    def test_split_membership_is_recorded_in_the_frame(self, synthetic_manifest):
        from data.splitting import SplitConfig, make_splits

        result = make_splits(self._frame(), SplitConfig(seed=42, group_by="auto"))
        assert "split" in result.frame.columns
        assert set(result.frame["split"].unique()) <= {"train", "val", "test"}

    def test_every_sample_is_assigned(self, synthetic_manifest):
        from data.splitting import SplitConfig, make_splits

        frame = self._frame()
        result = make_splits(frame, SplitConfig(seed=42, group_by="auto"))
        assert result.frame["split"].notna().all()
        assert len(result.frame) == len(frame)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
class TestDeduplication:
    def test_exact_duplicates_are_detected(self, tmp_path):
        from data.deduplication import DedupConfig, find_duplicates

        root = tmp_path / "images"
        root.mkdir()
        payload = b"\x00\x01\x02identical bytes"
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            (root / name).write_bytes(payload)
        (root / "unique.jpg").write_bytes(b"a completely different payload")

        frame = __import__("pandas").DataFrame(
            {
                "sample_id": ["s:a", "s:b", "s:c", "s:d"],
                "image_path": [str(root / n) for n in ("a.jpg", "b.jpg", "c.jpg", "unique.jpg")],
                "dataset": ["ds"] * 4,
            }
        )

        result = find_duplicates(frame, DedupConfig(perceptual=False))
        # Three identical files form one group; the unique file stands alone.
        assert result.num_exact_duplicates == 2
        assert len(result.exact_duplicate_groups) == 1
        assert len(result.exact_duplicate_groups[0]) == 3

    def test_near_duplicates_are_detected(self, tmp_path):
        """Two visually identical images with different bytes are near-duplicates."""
        import cv2

        from data.deduplication import DedupConfig, find_duplicates

        root = tmp_path / "images"
        root.mkdir()
        base = np.full((64, 64, 3), 120, dtype=np.uint8)
        cv2.circle(base, (32, 32), 18, (200, 40, 40), -1)
        cv2.imwrite(str(root / "a.jpg"), base)
        # A visually identical copy re-encoded with different JPEG bytes.
        cv2.imwrite(str(root / "b.jpg"), base, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        frame = __import__("pandas").DataFrame(
            {
                "sample_id": ["s:a", "s:b"],
                "image_path": [str(root / "a.jpg"), str(root / "b.jpg")],
                "dataset": ["ds", "ds"],
            }
        )

        result = find_duplicates(frame, DedupConfig(exact_hash=False, perceptual=True))
        assert result.num_near_duplicates >= 1
        assert len(result.near_duplicate_groups) >= 1

    def test_distinct_images_are_kept(self, tmp_path):
        import cv2

        from data.deduplication import DedupConfig, find_duplicates

        root = tmp_path / "images"
        root.mkdir()
        rng = np.random.default_rng(0)
        paths = []
        for index in range(3):
            array = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
            path = root / f"img{index}.jpg"
            cv2.imwrite(str(path), array)
            paths.append(str(path))

        frame = __import__("pandas").DataFrame(
            {
                "sample_id": [f"s:{i}" for i in range(3)],
                "image_path": paths,
                "dataset": ["ds"] * 3,
            }
        )

        result = find_duplicates(frame, DedupConfig())
        assert result.num_exact_duplicates == 0
        assert result.num_near_duplicates == 0

    def test_cross_dataset_duplicates_are_reported(self, tmp_path):
        """The same image in two datasets must be flagged as a leakage risk."""
        from data.deduplication import DedupConfig, find_duplicates

        root = tmp_path / "images"
        root.mkdir()
        payload = b"\x00\x01\x02shared across datasets"
        (root / "a.jpg").write_bytes(payload)
        (root / "b.jpg").write_bytes(payload)

        frame = __import__("pandas").DataFrame(
            {
                "sample_id": ["ham:a", "isic:b"],
                "image_path": [str(root / "a.jpg"), str(root / "b.jpg")],
                "dataset": ["ham10000", "isic2019"],
            }
        )

        result = find_duplicates(frame, DedupConfig(perceptual=False))
        assert result.num_cross_dataset >= 1
        assert len(result.cross_dataset_groups) >= 1


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------
class TestCheckpointRoundTrip:
    def test_hybrid_checkpoint_survives_a_save_load_cycle(self, tmp_path):
        import torch

        from models.hybrid_model import (
            HybridModelConfig,
            HybridSkinCancerModel,
            model_config_to_dict,
        )

        config = HybridModelConfig(
            pretrained=False,
            use_attention=False,
            use_metadata=False,
            fusion_dim=16,
            num_classes=5,
            head_dims=(8,),
        )
        model = HybridSkinCancerModel(config)

        checkpoint = tmp_path / "best.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": model_config_to_dict(config),
                "class_names": ["MEL", "NV", "BCC", "AKIEC", "BKL"],
                "num_classes": 5,
                "metadata_dim": 0,
            },
            checkpoint,
        )

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        rebuilt_config = HybridModelConfig(**dict(payload["model_config"]))
        rebuilt = HybridSkinCancerModel(rebuilt_config)
        rebuilt.load_state_dict(payload["model_state_dict"], strict=True)

        model.eval()
        rebuilt.eval()
        sample = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            assert torch.allclose(model(sample), rebuilt(sample), atol=1e-6)

    def test_unet_checkpoint_survives_a_save_load_cycle(self, tmp_path):
        import torch

        from models.unet import UNet

        model = UNet(in_channels=3, out_channels=1, base_channels=8, depth=2)
        checkpoint = tmp_path / "unet.pth"
        torch.save({"model_state_dict": model.state_dict()}, checkpoint)

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        rebuilt = UNet(in_channels=3, out_channels=1, base_channels=8, depth=2)
        rebuilt.load_state_dict(payload["model_state_dict"], strict=True)

        model.eval()
        rebuilt.eval()
        sample = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            assert torch.allclose(model(sample), rebuilt(sample), atol=1e-6)


# ---------------------------------------------------------------------------
# Metadata dimension consistency
# ---------------------------------------------------------------------------
class TestMetadataDimensionConsistency:
    def test_encoder_output_dim_matches_the_model_input_dim(self):
        """The dimension the encoder produces must be the one the model accepts.

        The encoder's output length depends on the vocabulary it observed, so
        a mismatch is a genuine failure mode rather than a theoretical one.
        """
        from data.metadata import MetadataConfig, MetadataEncoder
        from models.hybrid_model import HybridModelConfig, HybridSkinCancerModel

        encoder = MetadataEncoder(MetadataConfig())
        encoder.fit(
            [
                {"age": 40, "sex": "Male", "localization": "face",
                 "skin_cancer_history": "no"},
                {"age": 55, "sex": "Female", "localization": "trunk",
                 "skin_cancer_history": "yes"},
            ]
        )

        config = HybridModelConfig(
            pretrained=False,
            use_metadata=True,
            metadata_dim=encoder.output_dim,
            metadata_output_dim=8,
            fusion_dim=16,
            num_classes=3,
            head_dims=(8,),
        )
        model = HybridSkinCancerModel(config)

        import torch

        metadata = torch.from_numpy(encoder.transform_batch(
            [{"age": 40, "sex": "Male", "localization": "face",
              "skin_cancer_history": "no"}]
        ))
        assert metadata.shape[1] == encoder.output_dim

        model.eval()
        with torch.no_grad():
            logits = model(torch.randn(1, 3, 224, 224), metadata, torch.ones(1, dtype=torch.bool))
        assert logits.shape == (1, 3)

    def test_predictor_rejects_a_mismatched_encoder(self, tmp_path):
        """A checkpoint/encoder mismatch must fail loudly, not cryptically."""
        import torch

        from data.metadata import MetadataConfig, MetadataEncoder
        from inference.predict import SkinCancerPredictor
        from models.hybrid_model import (
            HybridModelConfig,
            HybridSkinCancerModel,
            model_config_to_dict,
        )

        config = HybridModelConfig(
            pretrained=False,
            use_metadata=True,
            metadata_dim=11,
            metadata_output_dim=8,
            fusion_dim=16,
            num_classes=3,
            head_dims=(8,),
        )
        model = HybridSkinCancerModel(config)

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        checkpoint = checkpoint_dir / "best.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": model_config_to_dict(config),
                "class_names": ["MEL", "NV", "BCC"],
                "num_classes": 3,
                "metadata_dim": 11,
            },
            checkpoint,
        )

        # An encoder fitted on a *different* vocabulary produces a different
        # output dimension.  Giving one categorical field a second distinct
        # value adds another one-hot slot.
        encoder = MetadataEncoder(MetadataConfig())
        encoder.fit(
            [
                {"age": 1, "sex": "X", "localization": "Y", "skin_cancer_history": "Z"},
                {"age": 2, "sex": "W", "localization": "Y", "skin_cancer_history": "Z"},
            ]
        )
        encoder.save(checkpoint_dir / "metadata_encoder.pkl")
        assert encoder.output_dim != 11, "the fixture must produce a real mismatch"

        with pytest.raises(ValueError, match="dimension mismatch"):
            SkinCancerPredictor.from_checkpoint(checkpoint, device="cpu")
