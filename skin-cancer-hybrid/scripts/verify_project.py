"""End-to-end verification of the project.

Usage
-----
::

    python scripts/verify_project.py
    python scripts/verify_project.py --run-tests

The script performs a set of independent checks and reports a pass/fail
summary:

1. every module imports,
2. every configuration file loads and merges,
3. the model components produce the expected output shapes,
4. the hybrid model performs a forward pass,
5. a checkpoint can be written and read back,
6. Grad-CAM produces a heatmap,
7. the dataset loaders work on synthetic data,
8. the split contains no leakage,
9. no absolute path is hardcoded in the source,
10. the requirements files are present.

This is the script to run before moving the project to a GPU machine.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class VerificationReport:
    """Collects the results of the individual checks."""

    def __init__(self) -> None:
        self.results: list[dict] = []

    def record(self, name: str, passed: bool, detail: str = "") -> None:
        self.results.append({"check": name, "passed": passed, "detail": detail})
        marker = "PASS" if passed else "FAIL"
        print(f"[{marker}] {name}" + (f" - {detail}" if detail else ""))

    @property
    def all_passed(self) -> bool:
        return all(result["passed"] for result in self.results)

    def summary(self) -> str:
        passed = sum(1 for result in self.results if result["passed"])
        total = len(self.results)
        return f"{passed}/{total} checks passed"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_imports(report: VerificationReport) -> None:
    modules = [
        "utils.config", "utils.seeding", "utils.hardware", "utils.logging_utils",
        "data.label_mapping", "data.preprocessing", "data.augmentation",
        "data.metadata", "data.loaders", "data.dataset_harmonization",
        "data.deduplication", "data.splitting", "data.datasets",
        "data.segmentation_dataset", "data.pipeline",
        "models.unet", "models.efficientnetv2", "models.attention",
        "models.metadata_encoder", "models.fusion", "models.classifier",
        "models.hybrid_model", "models.losses",
        "training.trainer", "training.cli_utils", "training.train_unet",
        "training.train_hybrid", "training.train_baseline",
        "training.train_multimodal", "training.ablation_study",
        "training.hyperparameter_search",
        "evaluation.classification_metrics", "evaluation.segmentation_metrics",
        "evaluation.confusion_matrix", "evaluation.roc_analysis",
        "evaluation.dataset_reports", "evaluation.evaluate",
        "explainability.gradcam", "explainability.visualization",
        "inference.predict", "inference.predict_batch",
    ]
    failures = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    report.record(
        "all modules import",
        not failures,
        f"{len(modules) - len(failures)}/{len(modules)}" if not failures else "; ".join(failures[:3]),
    )


def check_configuration(report: VerificationReport) -> None:
    from utils.config import load_config

    try:
        config = load_config(ROOT / "configs" / "default.yaml", root=ROOT)
        assert config.get("training.epochs") is not None
        # The backbone must be an EfficientNetV2 variant.  The exact timm
        # name differs between the PyTorch port (no pretrained weights) and
        # the TensorFlow port (with ImageNet weights), so the check accepts
        # either rather than pinning a string that the config may change.
        backbone_name = str(config.get("backbone.name") or "")
        assert "efficientnetv2" in backbone_name.lower(), backbone_name
        assert config.get("preprocessing.image_size") == 224
        assert config.get("attention.enabled") is True
        assert config.get("metadata_encoder.enabled") is True
        report.record("configuration loads and merges", True, f"{len(config.data)} sections")
    except Exception as exc:
        report.record("configuration loads and merges", False, str(exc))


def check_model_shapes(report: VerificationReport) -> None:
    import torch

    from models.attention import CBAM, ChannelAttention, SpatialAttention
    from models.fusion import FusionLayer
    from models.unet import UNet

    problems = []
    try:
        unet = UNet(in_channels=3, out_channels=1, base_channels=8, depth=2)
        output = unet(torch.randn(1, 3, 64, 64))
        if output.shape != (1, 1, 64, 64):
            problems.append(f"U-Net output {tuple(output.shape)}")
    except Exception as exc:
        problems.append(f"U-Net: {exc}")

    try:
        cbam = CBAM(16)
        out = cbam(torch.randn(2, 16, 8, 8))
        if out.shape != (2, 16, 8, 8):
            problems.append(f"CBAM output {tuple(out.shape)}")
        channel = ChannelAttention(16)(torch.randn(2, 16, 8, 8))
        if channel.shape != (2, 16, 8, 8):
            problems.append(f"ChannelAttention output {tuple(channel.shape)}")
        spatial = SpatialAttention(7)(torch.randn(2, 16, 8, 8))
        if spatial.shape != (2, 16, 8, 8):
            problems.append(f"SpatialAttention output {tuple(spatial.shape)}")
    except Exception as exc:
        problems.append(f"attention: {exc}")

    try:
        fusion = FusionLayer(32, 16, 64)
        out = fusion(torch.randn(4, 32), torch.randn(4, 16))
        if out.shape != (4, 64):
            problems.append(f"fusion output {tuple(out.shape)}")
    except Exception as exc:
        problems.append(f"fusion: {exc}")

    report.record(
        "model component shapes",
        not problems,
        "; ".join(problems) if problems else "U-Net, CBAM, fusion OK",
    )


def check_hybrid_forward(report: VerificationReport) -> None:
    import torch

    from models.hybrid_model import HybridModelConfig, HybridSkinCancerModel

    problems = []
    try:
        config = HybridModelConfig(
            pretrained=False,
            backbone_name="efficientnetv2_s",
            use_attention=True,
            use_metadata=True,
            metadata_dim=12,
            metadata_output_dim=8,
            fusion_dim=32,
            num_classes=7,
            head_dims=(16,),
        )
        model = HybridSkinCancerModel(config)
        model.eval()
        images = torch.randn(2, 3, 224, 224)
        metadata = torch.randn(2, 12)
        with torch.no_grad():
            logits = model(images, metadata, torch.ones(2, dtype=torch.bool))
        if logits.shape != (2, 7):
            problems.append(f"hybrid logits {tuple(logits.shape)}")
        with torch.no_grad():
            probabilities = model.predict_proba(images, metadata)
        if not torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-4):
            problems.append("softmax probabilities do not sum to 1")
    except Exception as exc:
        problems.append(f"{type(exc).__name__}: {exc}")

    report.record(
        "hybrid model forward pass",
        not problems,
        "; ".join(problems) if problems else "forward + softmax OK",
    )


def check_checkpoint_roundtrip(report: VerificationReport) -> None:
    import tempfile

    import torch

    from models.hybrid_model import HybridModelConfig, HybridSkinCancerModel

    problems = []
    try:
        config = HybridModelConfig(
            pretrained=False, use_attention=False, use_metadata=False,
            fusion_dim=32, num_classes=5, head_dims=(16,),
        )
        model = HybridSkinCancerModel(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": ["A", "B", "C", "D", "E"],
                    "num_classes": 5,
                    "metadata_dim": 0,
                    "epoch": 3,
                },
                path,
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            restored = HybridSkinCancerModel(config)
            restored.load_state_dict(payload["model_state_dict"], strict=True)
            model.eval()
            restored.eval()
            images = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                a = model(images)
                b = restored(images)
            if not torch.allclose(a, b, atol=1e-5):
                problems.append("restored model produces different logits")
    except Exception as exc:
        problems.append(f"{type(exc).__name__}: {exc}")

    report.record(
        "checkpoint save / load round-trip",
        not problems,
        "; ".join(problems) if problems else "state dict restored exactly",
    )


def check_gradcam(report: VerificationReport) -> None:
    import tempfile

    import torch

    from explainability.gradcam import GradCAM
    from models.hybrid_model import HybridModelConfig, HybridSkinCancerModel

    problems = []
    try:
        config = HybridModelConfig(
            pretrained=False, use_attention=True, use_metadata=False,
            fusion_dim=32, num_classes=4, head_dims=(16,),
        )
        model = HybridSkinCancerModel(config)
        model.eval()
        explainer = GradCAM(model)
        images = torch.randn(2, 3, 224, 224)
        heatmaps, overlays, probabilities = explainer(images)
        if heatmaps.shape != (2, 224, 224):
            problems.append(f"heatmap shape {tuple(heatmaps.shape)}")
        if overlays.shape != (2, 224, 224, 3):
            problems.append(f"overlay shape {tuple(overlays.shape)}")
        if heatmaps.min() < -1e-5 or heatmaps.max() > 1.0 + 1e-5:
            problems.append("heatmap values outside [0, 1]")
        explainer.remove_hooks()
    except Exception as exc:
        problems.append(f"{type(exc).__name__}: {exc}")

    report.record(
        "Grad-CAM produces a heatmap",
        not problems,
        "; ".join(problems) if problems else "heatmaps + overlays OK",
    )


def check_losses(report: VerificationReport) -> None:
    import torch

    from models.losses import (
        WeightedCrossEntropyLoss,
        WeightedFocalLoss,
        compute_class_weights,
    )

    problems = []
    try:
        weights = compute_class_weights([0] * 10 + [1] * 2 + [2] * 1, num_classes=3)
        if weights.shape != (3,):
            problems.append(f"weights shape {tuple(weights.shape)}")
        if float(weights[2]) <= float(weights[0]):
            problems.append("the rare class did not receive a larger weight")

        logits = torch.randn(13, 3)
        target = torch.tensor([0] * 10 + [1] * 2 + [2])
        focal = WeightedFocalLoss(gamma=2.0, class_weights=weights)(logits, target)
        if not torch.isfinite(focal):
            problems.append("focal loss is not finite")
        ce = WeightedCrossEntropyLoss(class_weights=weights)(logits, target)
        if not torch.isfinite(ce):
            problems.append("weighted CE is not finite")

        # With gamma=0 the focal loss must equal the weighted CE.
        focal_zero = WeightedFocalLoss(gamma=0.0, class_weights=weights)(logits, target)
        if not torch.allclose(focal_zero, ce, atol=1e-5):
            problems.append(
                "focal loss with gamma=0 does not reduce to weighted cross entropy "
                f"({float(focal_zero):.6f} vs {float(ce):.6f})"
            )
    except Exception as exc:
        problems.append(f"{type(exc).__name__}: {exc}")

    report.record(
        "weighted focal loss behaves correctly",
        not problems,
        "; ".join(problems) if problems else "gamma=0 reduces to weighted CE",
    )


def check_metadata_encoder(report: VerificationReport) -> None:
    from data.metadata import MetadataConfig, MetadataEncoder

    problems = []
    try:
        encoder = MetadataEncoder(MetadataConfig())
        records = [
            {"age": 40, "sex": "Male", "localization": "face", "skin_cancer_history": "no"},
            {"age": None, "sex": "Female", "localization": None, "skin_cancer_history": "yes"},
        ]
        encoder.fit(records)
        vectors = encoder.transform_batch(records)
        if vectors.shape[0] != 2:
            problems.append(f"vector shape {vectors.shape}")
        missing = encoder.transform(None)
        if missing.shape != vectors.shape[1:]:
            problems.append("the no-metadata vector has a different length")

        # The target-leakage guard must reject a label column.
        try:
            encoder.fit([{"age": 40, "dx": "mel"}])
            problems.append("the target-leakage guard did not reject a label column")
        except ValueError:
            pass
    except Exception as exc:
        problems.append(f"{type(exc).__name__}: {exc}")

    report.record(
        "metadata encoding + missing handling",
        not problems,
        "; ".join(problems) if problems else "fit / transform / guard OK",
    )


def check_label_mapping(report: VerificationReport) -> None:
    from data.label_mapping import build_label_mapping

    problems = []
    try:
        mapping = build_label_mapping(
            "isic2019",
            ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "SCC"],
            target_classes=["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"],
        )
        if "SCC" not in mapping.excluded_labels:
            problems.append("SCC was not excluded from the 7-class head")
        if mapping.to_unified_index("SCC") != -1:
            problems.append("SCC did not map to the excluded index -1")
        if mapping.to_unified_index("mel") != 0:
            problems.append("MEL did not map to index 0")
        if mapping.unmapped_labels:
            problems.append(f"unexpected unmapped labels: {mapping.unmapped_labels}")
    except Exception as exc:
        problems.append(f"{type(exc).__name__}: {exc}")

    report.record(
        "label mapping excludes SCC correctly",
        not problems,
        "; ".join(problems) if problems else "SCC excluded, others mapped",
    )


def check_no_hardcoded_paths(report: VerificationReport) -> None:
    """Verify that no absolute local path is hardcoded in the source."""
    import re

    patterns = [
        re.compile(r"[A-Za-z]:\\\\?Users\\\\?"),
        re.compile(r"/home/[a-z]+/"),
        re.compile(r"/Users/[A-Za-z]+/"),
    ]
    offenders: list[str] = []
    for path in ROOT.rglob("*.py"):
        if ".venv" in path.parts or "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if line.strip().startswith("#"):
                continue
            for pattern in patterns:
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}:{number}")
                    break

    report.record(
        "no hardcoded absolute paths",
        not offenders,
        "; ".join(offenders[:5]) if offenders else "source is path-portable",
    )


def check_requirements(report: VerificationReport) -> None:
    required = [
        "requirements.txt",
        "requirements-colab.txt",
        "requirements-kaggle.txt",
        "requirements-dev.txt",
    ]
    missing = [name for name in required if not (ROOT / name).exists()]
    report.record(
        "requirements files present",
        not missing,
        "; ".join(missing) if missing else ", ".join(required),
    )


def check_documentation(report: VerificationReport) -> None:
    required = [
        "README.md", "SETUP.md", "DATASET_GUIDE.md", "TRAINING_GUIDE.md",
        "EVALUATION_GUIDE.md", "MODEL_ARCHITECTURE.md", "LICENSES_AND_ATTRIBUTION.md",
    ]
    missing = [name for name in required if not (ROOT / name).exists()]
    report.record(
        "documentation present",
        not missing,
        "; ".join(missing) if missing else f"{len(required)} documents",
    )


def check_synthetic_pipeline(report: VerificationReport) -> None:
    """Run the loaders on a tiny synthetic dataset tree."""
    import tempfile

    import numpy as np
    import pandas as pd

    from data.loaders.ham10000 import HAM10000Loader

    problems = []
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ham10000"
            image_dir = root / "HAM10000_images"
            image_dir.mkdir(parents=True)

            import cv2

            records = []
            for index in range(6):
                image_id = f"ISIC_{index:05d}"
                array = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
                cv2.imwrite(str(image_dir / f"{image_id}.jpg"), array)
                records.append(
                    {
                        "image_id": image_id,
                        "dx": ["MEL", "NV", "BCC"][index % 3],
                        "lesion_id": index // 2,
                    }
                )
            pd.DataFrame(records).to_csv(root / "metadata.csv", index=False)

            loader = HAM10000Loader(root=root)
            manifest = loader.build_manifest()
            if len(manifest) != 6:
                problems.append(f"expected 6 samples, got {len(manifest)}")
            if manifest.class_distribution().get("MEL") != 2:
                problems.append("unexpected class distribution")
            # Two images share lesion_id 0, so they must share a group.
            groups = [record.group_id for record in manifest.records]
            if len(set(groups[:2])) != 1:
                problems.append("images of the same lesion did not share a group id")
    except Exception as exc:
        problems.append(f"{type(exc).__name__}: {exc}")

    report.record(
        "synthetic dataset loader works",
        not problems,
        "; ".join(problems) if problems else "6 samples, lesion grouping OK",
    )


def check_split_leakage(report: VerificationReport) -> None:
    """Verify that the grouping logic keeps groups inside one split."""
    import pandas as pd

    from data.splitting import SplitConfig, make_splits

    problems = []
    try:
        frame = pd.DataFrame(
            {
                "sample_id": [f"d:{i}" for i in range(100)],
                "dataset": ["test"] * 100,
                "group_id": [f"g{i // 3}" for i in range(100)],
                "group_type": ["lesion"] * 100,
                "unified_label": [["MEL", "NV", "BCC"][i % 3] for i in range(100)],
                "unified_label_valid": [True] * 100,
                "class_index": [i % 3 for i in range(100)],
            }
        )
        result = make_splits(frame, SplitConfig(seed=42))
        grouped = result.frame[result.frame["group_id"].astype(str).str.len() > 0]
        for group_id, sub in grouped.groupby("group_id"):
            if sub["split"].nunique() > 1:
                problems.append(f"group {group_id} spans multiple splits")
                break
        counts = result.counts()
        if not counts.get("train"):
            problems.append("the train split is empty")
    except Exception as exc:
        problems.append(f"{type(exc).__name__}: {exc}")

    report.record(
        "split keeps groups together",
        not problems,
        "; ".join(problems) if problems else "no group spans two splits",
    )


def run_tests(report: VerificationReport) -> None:
    """Run the pytest suite."""
    import subprocess

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q", "--no-header"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        passed = result.returncode == 0
        tail = (result.stdout or "").strip().splitlines()[-1:] if result.stdout else []
        report.record(
            "pytest suite passes",
            passed,
            tail[0] if tail else (result.stderr or "")[-200:],
        )
    except Exception as exc:
        report.record("pytest suite passes", False, str(exc))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the whole project.")
    parser.add_argument("--run-tests", action="store_true",
                        help="Also run the pytest suite (slower).")
    parser.add_argument("--output", default=None, help="Write the report as JSON.")
    args = parser.parse_args()

    print("=" * 78)
    print("PROJECT VERIFICATION")
    print("=" * 78)
    print()

    report = VerificationReport()

    check_imports(report)
    check_configuration(report)
    check_model_shapes(report)
    check_hybrid_forward(report)
    check_checkpoint_roundtrip(report)
    check_gradcam(report)
    check_losses(report)
    check_metadata_encoder(report)
    check_label_mapping(report)
    check_synthetic_pipeline(report)
    check_split_leakage(report)
    check_no_hardcoded_paths(report)
    check_requirements(report)
    check_documentation(report)

    if args.run_tests:
        run_tests(report)

    print()
    print("=" * 78)
    print(f"RESULT: {report.summary()}")
    print("=" * 78)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump({"results": report.results, "summary": report.summary()}, handle, indent=2)
        print(f"report -> {path}")

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
