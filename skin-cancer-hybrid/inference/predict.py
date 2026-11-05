"""Single-image inference.

Usage
-----
::

    python inference/predict.py --checkpoint runs/hybrid/checkpoints/best.pth \
        --image path/to/lesion.jpg

    python inference/predict.py --checkpoint ... --image lesion.jpg \
        --metadata '{"age": 54, "sex": "Male", "localization": "face", "skin_cancer_history": "no"}'

    python inference/predict.py --checkpoint ... --image lesion.jpg --gradcam \
        --output-dir runs/predictions

Python API
----------
::

    from inference.predict import SkinCancerPredictor

    predictor = SkinCancerPredictor.from_checkpoint("runs/hybrid/checkpoints/best.pth")
    result = predictor.predict("lesion.jpg", metadata={"age": 54})
    print(result["top_k_predictions"])

Disclaimer
----------
This is a **research prototype**.  The output is a model prediction for
research purposes; it is **not** a medical diagnosis and must not be used as
a substitute for professional medical assessment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.preprocessing import PreprocessConfig, PreprocessingPipeline
from utils.hardware import resolve_device

__all__ = ["SkinCancerPredictor", "PredictionResult", "main"]

DISCLAIMER = (
    "RESEARCH PROTOTYPE - NOT A MEDICAL DIAGNOSIS. This prediction was "
    "produced by an experimental deep learning model and must not be used "
    "as a substitute for professional medical assessment."
)


class PredictionResult(dict):
    """The prediction for a single image.

    Behaves like a dictionary and also exposes the main fields as attributes.
    """

    @property
    def predicted_class(self) -> str:
        return self["predicted_class"]

    @property
    def confidence(self) -> float:
        return self["confidence"]

    @property
    def probabilities(self) -> dict[str, float]:
        return self["probabilities"]

    @property
    def top_k_predictions(self) -> list[dict[str, Any]]:
        return self["top_k_predictions"]


class SkinCancerPredictor:
    """Loads a trained checkpoint and predicts on images.

    The predictor applies **exactly** the preprocessing that was used during
    training: the same image size, the same hair removal, the same CLAHE and
    the same normalisation.  The configuration is stored in the checkpoint so
    the inference pipeline cannot drift away from the training one.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Any,
        class_names: Sequence[str],
        metadata_encoder: Any | None = None,
        device: torch.device | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.class_names = list(class_names)
        self.metadata_encoder = metadata_encoder
        self.checkpoint_path = checkpoint_path

        if device is None:
            device, _, _ = resolve_device(
                preference=str(config.get("hardware.device", "auto")),
                allow_amp=False,
            )
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        self.pipeline = PreprocessingPipeline(PreprocessConfig.from_config(config))
        self._validate_metadata_dim()

    # ------------------------------------------------------------------
    def _validate_metadata_dim(self) -> None:
        """Check that the saved encoder matches the model's metadata input.

        The encoder's output length depends on the categorical vocabulary it
        observed during fitting, so it is only known after ``fit``.  If the
        checkpoint and the encoder disagree, the model would raise an obscure
        matrix-shape error deep inside the network; this turns it into an
        actionable message.
        """
        if self.metadata_encoder is None:
            return

        encoder_dim = int(getattr(self.metadata_encoder, "output_dim", 0))
        model_dim = int(getattr(getattr(self.model, "config", None), "metadata_dim", 0))
        if encoder_dim != model_dim:
            raise ValueError(
                f"Metadata dimension mismatch: the saved encoder produces "
                f"{encoder_dim} features but the model expects {model_dim}. "
                "The encoder was fitted on a different metadata schema than "
                "the one the checkpoint was trained with. Refit the encoder on "
                "the training split, or point the predictor at the matching "
                "checkpoint."
            )

    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config_path: str | Path | None = None,
        device: str | None = None,
    ) -> "SkinCancerPredictor":
        """Build a predictor from a trained checkpoint.

        Parameters
        ----------
        checkpoint_path:
            The ``.pth`` checkpoint produced by the training scripts.
        config_path:
            Configuration to use.  Defaults to the configuration stored in the
            checkpoint, falling back to ``configs/default.yaml``.
        device:
            ``auto`` | ``cuda`` | ``cpu``.
        """
        from utils.config import load_config

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if config_path is not None:
            config = load_config(config_path)
        elif isinstance(payload.get("config"), dict):
            from utils.config import Config

            project_root = Path(__file__).resolve().parents[1]
            config = Config(data=payload["config"], root=project_root)
        else:
            config = load_config(ROOT / "configs" / "default.yaml")

        class_names = payload.get("class_names") or []
        num_classes = int(payload.get("num_classes", len(class_names) or 7))
        metadata_dim = int(payload.get("metadata_dim", 0))

        from models.hybrid_model import (
            build_model_from_checkpoint_payload,
        )

        # The checkpoint stores the model configuration, so the exact
        # architecture is rebuilt - the weights then load without any
        # mismatch.  No pretrained download is needed.
        try:
            model, class_names, num_classes, metadata_dim = (
                build_model_from_checkpoint_payload(payload, config)
            )
        except ValueError:
            # An old checkpoint without a model configuration: fall back to
            # the configuration file.
            from models.hybrid_model import HybridModelConfig, HybridSkinCancerModel

            model_config = HybridModelConfig.from_config(config, num_classes=num_classes)
            model_config.pretrained = False
            model_config.metadata_dim = metadata_dim
            model_config.use_metadata = metadata_dim > 0
            model = HybridSkinCancerModel(model_config)
            model.load_state_dict(payload["model_state_dict"], strict=True)

        metadata_encoder = None
        encoder_path = checkpoint_path.parent / "metadata_encoder.pkl"
        if encoder_path.exists():
            from data.metadata import MetadataEncoder

            metadata_encoder = MetadataEncoder.load(encoder_path)

        return cls(
            model=model,
            config=config,
            class_names=class_names,
            metadata_encoder=metadata_encoder,
            checkpoint_path=checkpoint_path,
        )

    # ------------------------------------------------------------------
    def _prepare_image(self, image_path: str | Path) -> np.ndarray:
        """Read and preprocess an image into a batch tensor."""
        import cv2

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        processed = self.pipeline(image)
        tensor = torch.from_numpy(np.ascontiguousarray(processed.transpose(2, 0, 1)))
        return tensor.unsqueeze(0)

    def _prepare_metadata(self, metadata: Mapping[str, Any] | None) -> tuple:
        """Convert the metadata into the tensors the model expects."""
        metadata_dim = int(getattr(self.model.config, "metadata_dim", 0))
        if metadata_dim <= 0 or self.metadata_encoder is None:
            return None, None

        if not metadata:
            # No metadata supplied: use the learned "missing metadata" path.
            vector = self.metadata_encoder.transform(None)
            return (
                torch.from_numpy(np.ascontiguousarray(vector)).unsqueeze(0),
                torch.zeros(1, dtype=torch.bool),
            )

        vector = self.metadata_encoder.transform(dict(metadata))
        return (
            torch.from_numpy(np.ascontiguousarray(vector)).unsqueeze(0),
            torch.ones(1, dtype=torch.bool),
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        image_path: str | Path,
        metadata: Mapping[str, Any] | None = None,
        top_k: int = 3,
        gradcam: bool = False,
    ) -> PredictionResult:
        """Predict the class of a single image.

        Parameters
        ----------
        image_path:
            Path to the dermoscopic image.
        metadata:
            Optional clinical metadata (age, sex, localization,
            skin_cancer_history).  When omitted the model uses its
            image-only path.
        top_k:
            Number of top predictions to return.
        gradcam:
            Whether to compute a Grad-CAM explanation.

        Returns
        -------
        PredictionResult
        """
        images = self._prepare_image(image_path).to(self.device)
        metadata_tensor, has_metadata = self._prepare_metadata(metadata)
        if metadata_tensor is not None:
            metadata_tensor = metadata_tensor.to(self.device)
            has_metadata = has_metadata.to(self.device)

        logits = self.model(images, metadata_tensor, has_metadata)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

        order = np.argsort(probabilities)[::-1]
        top = [
            {
                "class": self.class_names[index] if index < len(self.class_names) else str(index),
                "probability": float(probabilities[index]),
                "rank": rank + 1,
            }
            for rank, index in enumerate(order[:top_k])
        ]

        result = PredictionResult(
            {
                "image_path": str(image_path),
                "predicted_class": top[0]["class"],
                "confidence": top[0]["probability"],
                "probabilities": {
                    (self.class_names[i] if i < len(self.class_names) else str(i)): float(p)
                    for i, p in enumerate(probabilities)
                },
                "top_k": top,
                # The same list under the name used by the public API
                # documentation, so both ``result["top_k"]`` and
                # ``result["top_k_predictions"]`` work.
                "top_k_predictions": top,
                "metadata_used": bool(metadata) if metadata is not None else False,
                "gradcam": None,
                "disclaimer": DISCLAIMER,
            }
        )

        if gradcam:
            result["gradcam"] = self.explain(image_path, metadata)

        return result

    def explain(
        self,
        image_path: str | Path,
        metadata: Mapping[str, Any] | None = None,
        target_class: int | None = None,
    ) -> dict[str, Any]:
        """Compute a Grad-CAM explanation for an image."""
        from explainability.gradcam import GradCAM

        images = self._prepare_image(image_path).to(self.device)
        metadata_tensor, has_metadata = self._prepare_metadata(metadata)
        if metadata_tensor is not None:
            metadata_tensor = metadata_tensor.to(self.device)
            has_metadata = has_metadata.to(self.device)

        explainer = GradCAM(self.model)
        try:
            targets = None if target_class is None else [int(target_class)]
            heatmaps, overlays, probabilities = explainer(
                images, metadata_tensor, has_metadata, target_classes=targets
            )
        finally:
            explainer.remove_hooks()

        heatmap = heatmaps[0]
        overlay = overlays[0]

        return {
            "heatmap": heatmap,
            "overlay": overlay,
            "target_class": int(target_class) if target_class is not None else None,
            "class_name": (
                self.class_names[target_class]
                if target_class is not None and target_class < len(self.class_names)
                else None
            ),
            "probabilities": probabilities[0].tolist(),
        }

    # ------------------------------------------------------------------
    def save_prediction(
        self,
        result: Mapping[str, Any],
        output_dir: str | Path,
        stem: str | None = None,
        gradcam: Mapping[str, Any] | None = None,
    ) -> dict[str, str]:
        """Save the prediction (and its explanation) to disk."""
        from explainability.visualization import (
            save_gradcam_figure,
            save_prediction_figure,
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = stem or Path(str(result["image_path"])).stem

        written: dict[str, str] = {}

        json_path = output_dir / f"{stem}_prediction.json"
        serialisable = {
            key: value
            for key, value in result.items()
            if key != "gradcam" and not isinstance(value, np.ndarray)
        }
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(serialisable, handle, indent=2, default=str)
        written["prediction_json"] = str(json_path)

        # Rebuild the original (un-normalised) image for the figure.
        image = self._original_image(result["image_path"])
        figure_path = save_prediction_figure(
            image,
            output_dir / f"{stem}_prediction.png",
            class_names=self.class_names,
            probabilities=[result["probabilities"].get(name, 0.0) for name in self.class_names],
        )
        written["prediction_figure"] = str(figure_path)

        if gradcam is not None:
            cam_path = save_gradcam_figure(
                image,
                gradcam["heatmap"],
                gradcam["overlay"],
                output_dir / f"{stem}_gradcam.png",
                class_name=gradcam.get("class_name") or result["predicted_class"],
                confidence=float(result["confidence"]),
                extra_text=(
                    "Research prototype - not a medical diagnosis."
                ),
            )
            written["gradcam_figure"] = str(cam_path)

        return written

    def _original_image(self, image_path: str | Path) -> np.ndarray:
        """Read the image without the preprocessing (for visualisation)."""
        import cv2

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict a skin-lesion class for a single image."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to the trained checkpoint.")
    parser.add_argument("--image", required=True, help="Path to the dermoscopic image.")
    parser.add_argument("--config", default=None, help="Configuration file to use.")
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument(
        "--metadata",
        default=None,
        help="Clinical metadata as JSON, e.g. "
        '\'{"age": 54, "sex": "Male", "localization": "face"}\'',
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions.")
    parser.add_argument("--gradcam", action="store_true", help="Generate a Grad-CAM overlay.")
    parser.add_argument("--target-class", type=int, default=None,
                        help="Explain a specific class index instead of the predicted one.")
    parser.add_argument("--output-dir", default=None, help="Where to save the outputs.")
    parser.add_argument("--json", action="store_true", help="Print only the JSON result.")
    args = parser.parse_args()

    predictor = SkinCancerPredictor.from_checkpoint(
        args.checkpoint, config_path=args.config, device=args.device
    )

    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as exc:
            print(f"Could not parse --metadata as JSON: {exc}", file=sys.stderr)
            return 2

    result = predictor.predict(
        args.image, metadata=metadata, top_k=args.top_k, gradcam=args.gradcam
    )

    gradcam_data = None
    if args.gradcam:
        gradcam_data = predictor.explain(
            args.image, metadata, target_class=args.target_class
        )

    if args.output_dir:
        written = predictor.save_prediction(result, args.output_dir, gradcam=gradcam_data)
        for key, value in written.items():
            print(f"{key}: {value}")

    if args.json:
        print(json.dumps(dict(result), indent=2, default=str))
    else:
        print("\n" + "=" * 70)
        print(f"Image            : {result['image_path']}")
        print(f"Predicted class  : {result['predicted_class']}")
        print(f"Confidence       : {result['confidence']:.4f}")
        print(f"Metadata used    : {result['metadata_used']}")
        print("\nTop predictions:")
        for entry in result["top_k"]:
            print(f"  {entry['rank']}. {entry['class']:<8} {entry['probability']:.4f}")
        print("\n" + DISCLAIMER)
        print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
