"""Tests for the trainer, checkpointing and the evaluation metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from evaluation.classification_metrics import (
    classification_report_dict,
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_melanoma_metrics,
    safe_auc,
)
from evaluation.dataset_reports import build_external_validation_section, write_dataset_report
from evaluation.segmentation_metrics import (
    batch_segmentation_metrics,
    compute_segmentation_metrics,
    dice_coefficient,
    iou_coefficient,
    pixel_accuracy,
)
from training.trainer import HistoryTracker, Trainer, TrainerConfig


# ---------------------------------------------------------------------------
# A tiny dataset for the trainer tests
# ---------------------------------------------------------------------------
class _TinyDataset(Dataset):
    def __init__(self, n: int = 16, num_classes: int = 3) -> None:
        generator = torch.Generator().manual_seed(7)
        self.images = torch.randn(n, 3, 32, 32, generator=generator)
        self.labels = torch.randint(0, num_classes, (n,), generator=generator)
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return {
            "image": self.images[index],
            "label": self.labels[index],
            "metadata": torch.zeros(0),
            "sample_id": f"s{index}",
            "dataset": "tiny",
        }


class _TinyModel(torch.nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1)
        )
        self.head = torch.nn.Linear(8, num_classes)
        self.metadata_dim = 0

    def forward(self, images, metadata=None, has_metadata=None):
        features = self.backbone(images).flatten(1)
        return self.head(features)

    def parameter_groups(self, lr_backbone, lr_head):
        return [
            {"params": list(self.backbone.parameters()), "lr": lr_backbone, "name": "backbone"},
            {"params": list(self.head.parameters()), "lr": lr_head, "name": "head"},
        ]


@pytest.fixture
def tiny_setup(tmp_path):
    dataset = _TinyDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    config = TrainerConfig(
        epochs=2,
        learning_rate=1e-3,
        backbone_lr=1e-4,
        output_dir=str(tmp_path),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        tensorboard=False,
        csv_history=True,
        device="cpu",
        num_classes=3,
        class_names=["a", "b", "c"],
        early_stopping={"enabled": False},
    )
    return dataset, loader, config


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class TestTrainer:
    def test_training_runs_and_writes_a_history(self, tiny_setup):
        dataset, loader, config = tiny_setup
        model = _TinyModel()
        criterion = torch.nn.CrossEntropyLoss()

        trainer = Trainer(model, loader, loader, criterion, config, metadata_dim=0)
        summary = trainer.fit()

        assert summary["epochs_run"] == 2
        history_path = Path(config.output_dir) / "training_history.csv"
        assert history_path.exists()

        import pandas as pd

        frame = pd.read_csv(history_path)
        assert len(frame) == 2
        assert "train_loss" in frame.columns
        assert "val_macro_f1" in frame.columns

    def test_checkpoints_are_written(self, tiny_setup):
        dataset, loader, config = tiny_setup
        model = _TinyModel()
        trainer = Trainer(model, loader, loader, torch.nn.CrossEntropyLoss(), config)
        trainer.fit()

        checkpoint_dir = Path(config.checkpoint_dir)
        assert (checkpoint_dir / "best.pth").exists()
        assert (checkpoint_dir / "last.pth").exists()

    def test_checkpoint_contains_the_resume_state(self, tiny_setup):
        dataset, loader, config = tiny_setup
        model = _TinyModel()
        trainer = Trainer(model, loader, loader, torch.nn.CrossEntropyLoss(), config)
        trainer.fit()

        payload = torch.load(
            Path(config.checkpoint_dir) / "last.pth", map_location="cpu", weights_only=False
        )
        for key in (
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "epoch",
            "best_metric",
            "config",
            "class_names",
        ):
            assert key in payload, f"missing {key} in the checkpoint"

    def test_resume_continues_from_the_stored_epoch(self, tiny_setup):
        dataset, loader, config = tiny_setup
        model = _TinyModel()
        trainer = Trainer(model, loader, loader, torch.nn.CrossEntropyLoss(), config)
        trainer.fit()

        config2 = TrainerConfig(**{**config.__dict__, "epochs": 3})
        model2 = _TinyModel()
        trainer2 = Trainer(model2, loader, loader, torch.nn.CrossEntropyLoss(), config2)
        trainer2.fit(resume_from=Path(config.checkpoint_dir) / "last.pth")

        assert trainer2.start_epoch == 2
        assert trainer2.history.rows[0]["epoch"] == 2

    def test_early_stopping_triggers(self, tmp_path):
        dataset = _TinyDataset()
        loader = DataLoader(dataset, batch_size=4)
        config = TrainerConfig(
            epochs=10,
            output_dir=str(tmp_path),
            checkpoint_dir=str(tmp_path / "ckpt"),
            tensorboard=False,
            device="cpu",
            early_stopping={"enabled": True, "patience": 1, "monitor": "val_macro_f1", "mode": "max"},
        )
        model = _TinyModel()
        trainer = Trainer(model, loader, loader, torch.nn.CrossEntropyLoss(), config)
        summary = trainer.fit()
        assert summary["epochs_run"] < 10

    def test_history_tracker_flush(self, tmp_path):
        tracker = HistoryTracker(tmp_path / "history.csv")
        tracker.append({"epoch": 0, "train_loss": 1.0})
        tracker.flush()
        assert (tmp_path / "history.csv").exists()
        assert len(tracker.as_frame()) == 1


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------
class TestClassificationMetrics:
    def test_perfect_prediction(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        metrics = compute_classification_metrics(
            y_true, y_pred, class_names=["a", "b", "c"], num_classes=3
        )
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["balanced_accuracy"] == 1.0

    def test_known_confusion_matrix(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]
        matrix = compute_confusion_matrix(y_true, y_pred, [0, 1])
        assert matrix.tolist() == [[1, 1], [0, 2]]

    def test_per_class_entries(self):
        y_true = [0, 0, 1, 2]
        y_pred = [0, 1, 1, 2]
        metrics = compute_classification_metrics(
            y_true, y_pred, class_names=["a", "b", "c"], num_classes=3
        )
        assert set(metrics["per_class"]) == {"a", "b", "c"}
        assert metrics["per_class"]["a"]["support"] == 2
        assert metrics["per_class"]["a"]["recall"] == pytest.approx(0.5)
        assert metrics["per_class"]["c"]["recall"] == 1.0

    def test_specificity_is_computed(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        metrics = compute_classification_metrics(y_true, y_pred, num_classes=2)
        assert metrics["per_class"]["class_0"]["specificity"] == 1.0

    def test_roc_auc_with_probabilities(self):
        y_true = [0, 0, 1, 1]
        probabilities = [[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8]]
        metrics = compute_classification_metrics(y_true, y_true, probabilities, num_classes=2)
        assert metrics["per_class"]["class_1"]["roc_auc"] == 1.0

    def test_safe_auc_returns_none_for_a_single_class(self):
        assert safe_auc([0, 0, 0], [[0.5, 0.5]] * 3, 1) is None

    def test_melanoma_metrics(self):
        y_true = [0, 0, 0, 1, 1]
        y_pred = [0, 0, 1, 1, 1]
        focus = compute_melanoma_metrics(y_true, y_pred, melanoma_index=0)
        assert focus["sensitivity"] == pytest.approx(2 / 3)
        assert focus["specificity"] == 1.0
        assert focus["true_positives"] == 2
        assert focus["false_negatives"] == 1

    def test_classification_report(self):
        report = classification_report_dict([0, 1, 1], [0, 1, 0], ["a", "b"], 2)
        assert "a" in report
        assert "accuracy" in report

    def test_empty_input_does_not_crash(self):
        metrics = compute_classification_metrics([], [], num_classes=3)
        assert metrics["num_samples"] == 0


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------
class TestSegmentationMetrics:
    def test_perfect_overlap(self):
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[2:8, 2:8] = 1.0
        assert dice_coefficient(mask, mask) == pytest.approx(1.0, abs=1e-4)
        assert iou_coefficient(mask, mask) == pytest.approx(1.0, abs=1e-4)

    def test_no_overlap(self):
        a = np.zeros((10, 10), dtype=np.float32)
        b = np.zeros((10, 10), dtype=np.float32)
        a[0:5, 0:5] = 1.0
        b[5:10, 5:10] = 1.0
        assert dice_coefficient(a, b) == pytest.approx(0.0, abs=1e-4)
        assert iou_coefficient(a, b) == pytest.approx(0.0, abs=1e-4)

    def test_half_overlap(self):
        a = np.zeros((10, 10), dtype=np.float32)
        b = np.zeros((10, 10), dtype=np.float32)
        a[0:5, 0:5] = 1.0
        b[2:7, 2:7] = 1.0
        dice = dice_coefficient(a, b)
        assert 0.3 < dice < 0.8

    def test_pixel_accuracy(self):
        a = np.array([[1, 0], [0, 1]], dtype=np.float32)
        assert pixel_accuracy(a, a) == 1.0

    def test_compute_segmentation_metrics_keys(self):
        a = np.ones((8, 8), dtype=np.float32)
        result = compute_segmentation_metrics(a, a)
        assert set(result) == {"dice", "iou", "pixel_accuracy", "sensitivity", "specificity"}

    def test_batch_metrics(self):
        a = np.ones((8, 8), dtype=np.float32)
        result = batch_segmentation_metrics([a, a], [a, a])
        assert result["dice"] == pytest.approx(1.0, abs=1e-4)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            dice_coefficient(np.ones((4, 4)), np.ones((8, 8)))


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------
class TestReports:
    def test_dataset_report_is_written(self, tmp_path):
        metrics = compute_classification_metrics(
            [0, 1, 2], [0, 1, 2], class_names=["MEL", "NV", "BCC"], num_classes=3
        )
        report = {
            "summary": {"accuracy": 1.0, "num_samples": 3},
            "metrics": metrics,
            "class_names": ["MEL", "NV", "BCC"],
        }
        json_path, md_path = write_dataset_report(
            report, tmp_path, class_names=["MEL", "NV", "BCC"]
        )
        assert json_path.exists()
        assert md_path.exists()
        text = md_path.read_text(encoding="utf-8")
        assert "MEL" in text

    def test_external_validation_section(self):
        per_dataset = {
            "ham10000": {"accuracy": 0.9, "macro_f1": 0.88, "focus": {"sensitivity": 0.7}},
            "isic2019": {"accuracy": 0.8, "macro_f1": 0.75, "focus": {"sensitivity": 0.6}},
        }
        section = build_external_validation_section(per_dataset)
        assert section["available"] is True
        assert section["spread"]["accuracy_spread"] == pytest.approx(0.1, abs=1e-6)
