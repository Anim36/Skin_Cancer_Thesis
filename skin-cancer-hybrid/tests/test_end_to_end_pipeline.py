"""End-to-end smoke test of the full pipeline on synthetic data.

This exercises the path a real run takes - prepare -> split -> train ->
evaluate -> explain - without touching a real dataset and without a GPU.
The purpose is to catch integration breakage between the modules, not to
produce a meaningful accuracy number.  Synthetic data cannot.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def synthetic_tree(tmp_path) -> Path:
    """A tiny HAM10000-style dataset with masks, written to disk."""
    import cv2
    import pandas as pd

    root = tmp_path / "ham10000"
    image_dir = root / "HAM10000_images"
    image_dir.mkdir(parents=True)

    labels = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    records = []
    for index in range(28):
        image_id = f"ISIC_{index:05d}"
        array = (np.random.default_rng(index).random((40, 50, 3)) * 255).astype(np.uint8)
        cv2.imwrite(str(image_dir / f"{image_id}.jpg"), array)
        records.append(
            {
                "image_id": image_id,
                "dx": labels[index % len(labels)],
                "lesion_id": index // 2,
                "age": 30 + index,
                "sex": "Male" if index % 2 else "Female",
                "localization": "face",
            }
        )

    pd.DataFrame(records).to_csv(root / "metadata.csv", index=False)
    return root


class TestPipelineIntegration:
    def test_manifest_builds_and_records_provenance(self, synthetic_tree):
        """Every sample must carry its dataset, original id and group."""
        from data.loaders.ham10000 import HAM10000Loader

        loader = HAM10000Loader(synthetic_tree)
        manifest = loader.build_manifest()

        assert len(manifest) == 28
        frame = manifest.to_frame()
        for column in ("sample_id", "dataset", "original_id", "group_id", "source"):
            assert column in frame.columns
            assert frame[column].notna().all()

        # The sample id must be globally unique and dataset-prefixed.
        assert frame["sample_id"].is_unique
        assert frame["sample_id"].str.startswith("ham10000:").all()

        # Lesion grouping: consecutive pairs share a lesion id.
        groups = frame["group_id"].tolist()
        assert any(groups.count(group) > 1 for group in set(groups))

    def test_manifest_round_trips_through_csv(self, synthetic_tree, tmp_path):
        from data.loaders.ham10000 import HAM10000Loader

        manifest = HAM10000Loader(synthetic_tree).build_manifest()
        path = manifest.save(tmp_path / "manifest.csv")

        reloaded = __import__(
            "data.loaders.base", fromlist=["DatasetManifest"]
        ).DatasetManifest.load(path)

        assert len(reloaded) == len(manifest)
        assert reloaded.records[0].sample_id == manifest.records[0].sample_id

    def test_split_is_group_safe_and_persisted(self, synthetic_tree, tmp_path):
        from data.loaders.ham10000 import HAM10000Loader
        from data.splitting import SplitConfig, make_splits

        manifest = HAM10000Loader(synthetic_tree).build_manifest()
        result = make_splits(manifest.to_frame(), SplitConfig(seed=42))

        # No group may straddle two splits.
        spans = result.frame.groupby("group_id")["split"].nunique()
        assert (spans == 1).all()

        # The split membership must be recoverable from disk.
        split_path = tmp_path / "split_manifest.csv"
        result.frame.to_csv(split_path, index=False)
        from data.splitting import load_split_manifest

        reloaded = load_split_manifest(split_path)
        assert set(reloaded["split"].unique()) <= {"train", "val", "test"}
        assert len(reloaded) == len(result.frame)

    def test_a_training_epoch_runs_and_produces_metrics(self, synthetic_tree, tmp_path):
        """One epoch of the real training engine on synthetic data."""
        import torch

        from data.loaders.ham10000 import HAM10000Loader
        from data.splitting import SplitConfig, make_splits
        from models.hybrid_model import HybridModelConfig, HybridSkinCancerModel
        from models.losses import build_loss
        from training.trainer import Trainer, TrainerConfig

        manifest = HAM10000Loader(synthetic_tree).build_manifest()
        frame = make_splits(manifest.to_frame(), SplitConfig(seed=42)).frame
        train_frame = frame[frame["split"] == "train"].reset_index(drop=True)

        from data.datasets import SkinCancerDataset, collate_batch

        dataset = SkinCancerDataset(train_frame, train=False)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=4, shuffle=False, collate_fn=collate_batch
        )

        model = HybridSkinCancerModel(
            HybridModelConfig(
                pretrained=False,
                use_attention=False,
                use_metadata=False,
                fusion_dim=16,
                num_classes=len(manifest.class_list),
                head_dims=(8,),
            )
        )
        criterion = build_loss("weighted_focal", num_classes=len(manifest.class_list))

        config = TrainerConfig(
            epochs=1,
            output_dir=str(tmp_path / "run"),
            tensorboard=False,
            csv_history=True,
            save_best=True,
            save_last=True,
            device="cpu",
            amp=False,
            num_classes=len(manifest.class_list),
            class_names=manifest.class_list,
            early_stopping={"enabled": False},
        )

        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            criterion=criterion,
            config=config,
            metadata_dim=0,
            class_names=manifest.class_list,
        )
        summary = trainer.fit()

        assert summary["epochs_run"] == 1
        assert np.isfinite(summary["best_metric"])

        history = tmp_path / "run" / "training_history.csv"
        assert history.exists()
        assert "val_macro_f1" in history.read_text(encoding="utf-8")

        # Both checkpoints must exist so that the run can be resumed and the
        # best model can be evaluated.
        assert (tmp_path / "run" / "checkpoints" / "best.pth").exists()
        assert (tmp_path / "run" / "checkpoints" / "last.pth").exists()

    def test_gradcam_runs_against_the_real_model(self, synthetic_tree):
        """Grad-CAM must work against the actual hybrid model, not a stub."""
        import torch

        from explainability.gradcam import GradCAM
        from models.hybrid_model import HybridModelConfig, HybridSkinCancerModel

        model = HybridSkinCancerModel(
            HybridModelConfig(
                pretrained=False,
                use_attention=True,
                use_metadata=False,
                fusion_dim=16,
                num_classes=7,
                head_dims=(8,),
            )
        )
        model.eval()

        explainer = GradCAM(model)
        images = torch.randn(2, 3, 224, 224)

        heatmaps, overlays, probabilities = explainer(images)
        assert heatmaps.shape == (2, 224, 224)
        assert overlays.shape == (2, 224, 224, 3)
        assert probabilities.shape == (2, 7)

        # A user-selected class must be explainable too.
        heatmaps, _, _ = explainer(images, target_classes=[0, 1])
        assert heatmaps.shape == (2, 224, 224)

    def test_prediction_result_exposes_the_documented_api(self, synthetic_tree):
        """``result["top_k_predictions"]`` is part of the documented API."""
        import torch

        from inference.predict import PredictionResult

        result = PredictionResult(
            {
                "predicted_class": "MEL",
                "confidence": 0.9,
                "probabilities": {"MEL": 0.9, "NV": 0.1},
                "top_k": [{"class": "MEL", "probability": 0.9, "rank": 1}],
                "top_k_predictions": [{"class": "MEL", "probability": 0.9, "rank": 1}],
            }
        )
        assert result.top_k_predictions[0]["class"] == "MEL"
        assert result["top_k_predictions"] == result["top_k"]
        assert result.predicted_class == "MEL"
