"""End-to-end tests: Grad-CAM, inference, splitting and the data pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from data.deduplication import (
    DedupConfig,
    compute_file_hash,
    compute_perceptual_hash,
    find_duplicates,
    hamming_distance,
    write_leakage_audit,
)
from data.splitting import SplitConfig, group_stratified_split, make_splits
from explainability.gradcam import GradCAM, resolve_target_layer
from models.hybrid_model import HybridModelConfig, HybridSkinCancerModel


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def _tiny_model(**overrides) -> HybridSkinCancerModel:
    config = HybridModelConfig(
        pretrained=False,
        use_attention=True,
        use_metadata=False,
        fusion_dim=32,
        num_classes=5,
        head_dims=(16,),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return HybridSkinCancerModel(config)


class TestGradCAM:
    def test_produces_heatmaps_and_overlays(self):
        model = _tiny_model()
        model.eval()
        explainer = GradCAM(model)
        heatmaps, overlays, probabilities = explainer(torch.randn(2, 3, 224, 224))
        try:
            assert heatmaps.shape == (2, 224, 224)
            assert overlays.shape == (2, 224, 224, 3)
            assert probabilities.shape == (2, 5)
        finally:
            explainer.remove_hooks()

    def test_heatmap_values_are_normalised(self):
        model = _tiny_model()
        model.eval()
        explainer = GradCAM(model)
        heatmaps, _, _ = explainer(torch.randn(2, 3, 224, 224))
        try:
            assert heatmaps.min() >= -1e-5
            assert heatmaps.max() <= 1.0 + 1e-5
        finally:
            explainer.remove_hooks()

    def test_target_class_can_be_selected(self):
        model = _tiny_model()
        model.eval()
        explainer = GradCAM(model)
        heatmaps, _, _ = explainer(
            torch.randn(2, 3, 224, 224), target_classes=[0, 3]
        )
        try:
            assert heatmaps.shape[0] == 2
        finally:
            explainer.remove_hooks()

    def test_target_layer_resolution(self):
        model = _tiny_model()
        layer, name = resolve_target_layer(model)
        assert isinstance(layer, torch.nn.Module)
        assert isinstance(name, str)

    def test_explicit_target_layer_is_used(self):
        model = _tiny_model()
        target = torch.nn.Conv2d(3, 4, 3, padding=1)
        explainer = GradCAM(model, target_layer=target)
        try:
            assert explainer.target_layer is target
        finally:
            explainer.remove_hooks()

    def test_hooks_are_removed(self):
        model = _tiny_model()
        explainer = GradCAM(model)
        assert len(explainer._handles) == 2
        explainer.remove_hooks()
        assert len(explainer._handles) == 0


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
class TestDeduplication:
    def test_file_hash_detects_identical_files(self, tmp_path):
        import cv2

        array = (np.random.default_rng(0).random((20, 20, 3)) * 255).astype(np.uint8)
        first = tmp_path / "a.jpg"
        second = tmp_path / "b.jpg"
        cv2.imwrite(str(first), array)
        cv2.imwrite(str(second), array)
        assert compute_file_hash(first) == compute_file_hash(second)

    def test_perceptual_hash_detects_similar_images(self, tmp_path):
        import cv2

        array = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
        path = tmp_path / "a.jpg"
        cv2.imwrite(str(path), array)
        first = compute_perceptual_hash(path)
        second = compute_perceptual_hash(path)
        assert first is not None
        assert hamming_distance(first, second) == 0

    def test_duplicates_are_found_in_a_manifest(self, tmp_path):
        import cv2

        array = (np.random.default_rng(0).random((32, 32, 3)) * 255).astype(np.uint8)
        for name in ("a.jpg", "b.jpg"):
            cv2.imwrite(str(tmp_path / name), array)
        cv2.imwrite(
            str(tmp_path / "c.jpg"),
            (np.random.default_rng(9).random((32, 32, 3)) * 255).astype(np.uint8),
        )

        frame = pd.DataFrame(
            {
                "sample_id": ["ham:a", "ham:b", "ham:c"],
                "dataset": ["ham10000"] * 3,
                "image_path": [
                    str(tmp_path / "a.jpg"),
                    str(tmp_path / "b.jpg"),
                    str(tmp_path / "c.jpg"),
                ],
            }
        )
        result = find_duplicates(frame, DedupConfig())
        assert result.num_exact_duplicates >= 1
        assert result.frame["is_duplicate"].sum() >= 1

    def test_cross_dataset_duplicates_are_flagged(self, tmp_path):
        import cv2

        array = (np.random.default_rng(0).random((32, 32, 3)) * 255).astype(np.uint8)
        for name in ("a.jpg", "b.jpg"):
            cv2.imwrite(str(tmp_path / name), array)

        frame = pd.DataFrame(
            {
                "sample_id": ["ham:a", "isic:b"],
                "dataset": ["ham10000", "isic2018"],
                "image_path": [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")],
            }
        )
        result = find_duplicates(frame, DedupConfig())
        assert result.num_cross_dataset >= 1
        assert len(result.cross_dataset_groups) >= 1

    def test_leakage_audit_reports_a_passing_split(self, synthetic_manifest, tmp_path):
        frame = synthetic_manifest.copy()
        frame["split"] = ["train"] * 42 + ["val"] * 9 + ["test"] * 9
        json_path, md_path = write_leakage_audit(frame, output_dir=tmp_path)
        assert json_path.exists()
        assert md_path.exists()
        audit = json.loads(json_path.read_text(encoding="utf-8"))
        assert "split_counts" in audit


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
class TestSplitting:
    def test_groups_never_span_two_splits(self, synthetic_manifest):
        result = make_splits(synthetic_manifest, SplitConfig(seed=42))
        grouped = result.frame[result.frame["group_id"].astype(str).str.len() > 0]
        for _, sub in grouped.groupby("group_id"):
            assert sub["split"].nunique() == 1

    def test_split_ratios_are_approximately_respected(self, synthetic_manifest):
        result = make_splits(synthetic_manifest, SplitConfig(seed=42))
        counts = result.counts()
        total = sum(counts.values())
        assert counts["train"] / total == pytest.approx(0.70, abs=0.12)
        assert counts["val"] / total == pytest.approx(0.15, abs=0.12)
        assert counts["test"] / total == pytest.approx(0.15, abs=0.12)

    def test_the_split_is_reproducible(self, synthetic_manifest):
        first = make_splits(synthetic_manifest, SplitConfig(seed=42))
        second = make_splits(synthetic_manifest, SplitConfig(seed=42))
        assert first.assignments == second.assignments

    def test_a_different_seed_gives_a_different_split(self, synthetic_manifest):
        first = make_splits(synthetic_manifest, SplitConfig(seed=42))
        second = make_splits(synthetic_manifest, SplitConfig(seed=7))
        assert first.assignments != second.assignments

    def test_every_sample_is_assigned(self, synthetic_manifest):
        result = make_splits(synthetic_manifest, SplitConfig(seed=42))
        assert len(result.frame) == len(synthetic_manifest)
        assert result.frame["split"].isin(["train", "val", "test"]).all()

    def test_group_stratified_split_assigns_every_group(self):
        frame = pd.DataFrame(
            {
                "sample_id": [f"s{i}" for i in range(30)],
                "group_id": [f"g{i // 3}" for i in range(30)],
                "unified_label": [["A", "B", "C"][i % 3] for i in range(30)],
            }
        )
        assignments, group_assignments = group_stratified_split(
            frame, "group_id", {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1
        )
        assert len(assignments) == 30
        assert len(group_assignments) == 10

    def test_empty_manifest_is_handled(self):
        result = make_splits(pd.DataFrame(columns=["sample_id", "group_id"]), SplitConfig())
        assert result.frame.empty


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
class TestPipeline:
    def test_pipeline_builds_a_manifest_and_a_split(self, config, tmp_path, synthetic_dataset_tree):
        from data.pipeline import prepare_data

        prepared = prepare_data(
            config,
            datasets=["ham10000"],
            raw_dir=tmp_path,
            include_scc=False,
        )
        assert len(prepared.manifest) == 21
        assert len(prepared.train_frame) > 0
        assert prepared.split_path.exists()
        assert set(prepared.class_names) <= {
            "MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"
        }

    def test_class_weights_come_from_the_training_split(self, config, tmp_path, synthetic_dataset_tree):
        from data.pipeline import prepare_data

        prepared = prepare_data(config, datasets=["ham10000"], raw_dir=tmp_path)
        assert prepared.class_weights is not None
        assert len(prepared.class_weights) == prepared.num_classes

    def test_metadata_encoder_is_fitted_only_when_metadata_exists(
        self, config, tmp_path, synthetic_dataset_tree
    ):
        from data.pipeline import prepare_data

        prepared = prepare_data(config, datasets=["ham10000"], raw_dir=tmp_path)
        # The synthetic HAM10000 tree has age/sex but the loader does not
        # register them as metadata columns, so the branch stays off.
        assert prepared.metadata_encoder is None

    def test_dataloaders_are_built(self, config, tmp_path, synthetic_dataset_tree):
        from data.pipeline import build_dataloaders, prepare_data

        prepared = prepare_data(config, datasets=["ham10000"], raw_dir=tmp_path)
        train_loader, val_loader, test_loader = build_dataloaders(prepared, config)
        batch = next(iter(train_loader))
        assert batch["image"].shape[0] >= 1
        assert batch["image"].shape[1] == 3
        assert "label" in batch


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
class TestInference:
    @pytest.fixture
    def trained_predictor(self, tmp_path, synthetic_dataset_tree):
        """A predictor built from a checkpoint trained on synthetic data."""
        from data.metadata import MetadataEncoder, MetadataConfig
        from inference.predict import SkinCancerPredictor
        from models.hybrid_model import model_config_to_dict

        class_names = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

        # The encoder is fitted FIRST because its output dimension depends on
        # the vocabulary it observed (one slot per categorical value plus the
        # reserved <MISSING>/<UNKNOWN> tokens).  The model must be built with
        # that exact dimension - hard-coding it would desynchronise the two.
        encoder = MetadataEncoder(MetadataConfig())
        encoder.fit([{"age": 40, "sex": "Male", "localization": "face",
                      "skin_cancer_history": "no"}])
        metadata_dim = encoder.output_dim

        model_config = HybridModelConfig(
            pretrained=False,
            use_attention=True,
            use_metadata=True,
            metadata_dim=metadata_dim,
            metadata_output_dim=8,
            fusion_dim=32,
            num_classes=7,
            head_dims=(16,),
        )
        model = HybridSkinCancerModel(model_config)

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / "best.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": model_config_to_dict(model_config),
                "class_names": class_names,
                "num_classes": 7,
                "metadata_dim": metadata_dim,
                "epoch": 0,
            },
            checkpoint,
        )

        encoder.save(checkpoint_dir / "metadata_encoder.pkl")

        return SkinCancerPredictor.from_checkpoint(checkpoint, device="cpu")

    def test_prediction_returns_probabilities_and_top_k(self, trained_predictor, synthetic_dataset_tree):
        image = sorted((synthetic_dataset_tree / "HAM10000_images").glob("*.jpg"))[0]
        result = trained_predictor.predict(image, top_k=3)
        assert result["predicted_class"] in trained_predictor.class_names
        assert 0.0 <= result["confidence"] <= 1.0
        assert len(result["top_k_predictions"]) == 3
        assert len(result["probabilities"]) == 7
        assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)

    def test_prediction_with_metadata(self, trained_predictor, synthetic_dataset_tree):
        image = sorted((synthetic_dataset_tree / "HAM10000_images").glob("*.jpg"))[0]
        result = trained_predictor.predict(
            image, metadata={"age": 54, "sex": "Male", "localization": "face"}
        )
        assert result["metadata_used"] is True

    def test_prediction_without_metadata(self, trained_predictor, synthetic_dataset_tree):
        image = sorted((synthetic_dataset_tree / "HAM10000_images").glob("*.jpg"))[0]
        result = trained_predictor.predict(image)
        assert result["metadata_used"] is False

    def test_gradcam_explanation_is_produced(self, trained_predictor, synthetic_dataset_tree):
        image = sorted((synthetic_dataset_tree / "HAM10000_images").glob("*.jpg"))[0]
        explanation = trained_predictor.explain(image)
        assert explanation["heatmap"].shape == (224, 224)
        assert explanation["overlay"].shape == (224, 224, 3)

    def test_prediction_is_saved(self, trained_predictor, synthetic_dataset_tree, tmp_path):
        image = sorted((synthetic_dataset_tree / "HAM10000_images").glob("*.jpg"))[0]
        result = trained_predictor.predict(image)
        written = trained_predictor.save_prediction(result, tmp_path / "out")
        assert Path(written["prediction_json"]).exists()
        assert Path(written["prediction_figure"]).exists()

    def test_missing_image_raises(self, trained_predictor):
        with pytest.raises(FileNotFoundError):
            trained_predictor.predict("/nonexistent/image.jpg")

    def test_batch_prediction(self, trained_predictor, synthetic_dataset_tree, tmp_path):
        from inference.predict_batch import batch_predict

        images = sorted((synthetic_dataset_tree / "HAM10000_images").glob("*.jpg"))[:4]
        results = batch_predict(trained_predictor, images, output_dir=tmp_path)
        assert len(results) == 4
        assert all("predicted_class" in row for row in results)
