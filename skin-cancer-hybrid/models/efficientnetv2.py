"""EfficientNetV2 feature extractor with transfer-learning control.

The backbone comes from ``timm`` (``efficientnetv2_s`` by default) and is used
as a pure feature extractor: the ImageNet classification head is removed and
the pooled feature vector is returned.

Transfer-learning modes
=======================
``frozen`` (linear probe)
    The whole backbone is frozen and only the head is trained.  Cheap, and a
    useful lower bound in the ablation study.

``partial``
    The last ``unfreeze_last_n_blocks`` stages are unfrozen and trained with a
    smaller learning rate than the head.  This is the default and usually the
    best accuracy/compute trade-off on a single GPU.

``full``
    Every backbone parameter is trained with the (small) backbone learning
    rate.  Requires the most memory and is most prone to overfitting on a
    dataset of ~40k images.

The learning rates for the two parameter groups are configured separately in
``configs/training.yaml`` (``lr_backbone`` and ``lr_head``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

__all__ = ["EfficientNetV2Backbone", "BackboneConfig"]


@dataclass
class BackboneConfig:
    """Configuration of the EfficientNetV2 backbone."""

    name: str = "efficientnetv2_s"
    pretrained: bool = True
    in_channels: int = 3
    num_classes: int = 0
    drop_rate: float = 0.3
    drop_path_rate: float = 0.1
    global_pool: str = "avg"
    freeze_backbone: bool = False
    unfreeze_last_n_blocks: int = 2

    @classmethod
    def from_config(cls, config: Any) -> "BackboneConfig":
        def _get(key: str, default: Any) -> Any:
            if hasattr(config, "get"):
                return config.get(f"backbone.{key}", default)
            if isinstance(config, dict):
                return config.get(key, default)
            return default

        return cls(
            name=str(_get("name", "efficientnetv2_s")),
            pretrained=bool(_get("pretrained", True)),
            in_channels=int(_get("in_channels", 3)),
            num_classes=int(_get("num_classes", 0)),
            drop_rate=float(_get("drop_rate", 0.3)),
            drop_path_rate=float(_get("drop_path_rate", 0.1)),
            global_pool=str(_get("global_pool", "avg")),
            freeze_backbone=bool(_get("freeze_backbone", False)),
            unfreeze_last_n_blocks=int(_get("unfreeze_last_n_blocks", 2)),
        )


class EfficientNetV2Backbone(nn.Module):
    """EfficientNetV2 feature extractor.

    Attributes
    ----------
    feature_dim:
        Dimension of the pooled feature vector.
    model:
        The underlying ``timm`` model.
    """

    def __init__(self, config: BackboneConfig | None = None) -> None:
        super().__init__()
        self.config = config or BackboneConfig()

        try:
            import timm
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "timm is required for the EfficientNetV2 backbone. "
                "Install it with: pip install timm"
            ) from exc

        kwargs: dict[str, Any] = {
            "pretrained": self.config.pretrained,
            "num_classes": 0,
            "in_chans": self.config.in_channels,
            "drop_rate": self.config.drop_rate,
            "drop_path_rate": self.config.drop_path_rate,
            "global_pool": self.config.global_pool,
        }

        model_name = self._resolve_model_name(timm)
        try:
            self.model = timm.create_model(model_name, **kwargs)
        except TypeError:
            # Older timm versions do not accept drop_path_rate / global_pool
            # in create_model; retry with a reduced set of arguments.
            self.model = timm.create_model(
                model_name,
                pretrained=self.config.pretrained,
                num_classes=0,
                in_chans=self.config.in_channels,
            )
        except RuntimeError as exc:
            # The requested weights could not be downloaded (offline machine
            # or a name without a pretrained checkpoint).  Fall back to random
            # initialisation with a clear warning rather than crashing.
            if not self.config.pretrained:
                raise
            import warnings

            warnings.warn(
                f"Could not load pretrained weights for {model_name!r} ({exc}). "
                "Falling back to random initialisation. Transfer learning will "
                "not work until the weights are available.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.model = timm.create_model(
                model_name, pretrained=False, num_classes=0,
                in_chans=self.config.in_channels,
            )

        self.model_name = model_name
        self.feature_dim = int(self.model.num_features)

        if self.config.freeze_backbone:
            self.set_backbone_mode("frozen")
        else:
            self.set_backbone_mode("partial")

    # ------------------------------------------------------------------
    def _resolve_model_name(self, timm: Any) -> str:
        """Pick a model name that actually has pretrained weights.

        ``efficientnetv2_s`` (the original PyTorch port) ships **without**
        ImageNet weights in ``timm``; the TensorFlow port
        ``tf_efficientnetv2_s`` does.  When the configured name has no
        pretrained checkpoint, the closest available alternative is used so
        that transfer learning works out of the box.
        """
        name = self.config.name
        if not self.config.pretrained:
            return name

        try:
            available = set(timm.list_pretrained(f"*{name}*"))
        except Exception:  # pragma: no cover - defensive
            return name

        if available:
            # Prefer the exact name if it has weights.
            if name in available:
                return name
            # Otherwise prefer the ImageNet-1k variant of the same family.
            for candidate in sorted(available):
                if candidate.endswith(".in1k"):
                    return candidate
            return sorted(available)[0]

        # No weights for this family - try the TensorFlow port, which does
        # ship ImageNet weights for EfficientNetV2.
        try:
            alternatives = set(timm.list_pretrained(f"tf_{name}*"))
        except Exception:  # pragma: no cover
            alternatives = set()

        if alternatives:
            for candidate in sorted(alternatives):
                if candidate.endswith(".in1k"):
                    return candidate
            return sorted(alternatives)[0]

        return name

    # ------------------------------------------------------------------
    # Transfer-learning control
    # ------------------------------------------------------------------
    def _stage_modules(self) -> list[nn.Module]:
        """Return the model's stage-like sub-modules, deepest last."""
        for attribute in ("blocks", "layers"):
            stages = getattr(self.model, attribute, None)
            if stages is not None:
                return list(stages)
        return []

    def set_backbone_mode(self, mode: str) -> None:
        """Configure which backbone parameters are trainable.

        Parameters
        ----------
        mode:
            ``frozen`` | ``partial`` | ``full``.
        """
        mode = (mode or "frozen").lower()

        if mode == "frozen":
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            self.model.eval()
            return

        if mode == "full":
            for parameter in self.model.parameters():
                parameter.requires_grad = True
            self.model.train()
            return

        # partial: freeze everything, then unfreeze the last N stages.
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        stages = self._stage_modules()
        n = self.config.unfreeze_last_n_blocks
        if n < 0:
            n = len(stages)
        for stage in stages[len(stages) - n:] if n > 0 else []:
            for parameter in stage.parameters():
                parameter.requires_grad = True

        # The final convolution / norm layer is always trained in partial mode.
        for attribute in ("conv_head", "bn2", "head", "global_pool"):
            module = getattr(self.model, attribute, None)
            if isinstance(module, nn.Module):
                for parameter in module.parameters():
                    parameter.requires_grad = True

        self.model.train()

    def train(self, mode: bool = True) -> "EfficientNetV2Backbone":
        """Set the training mode, keeping a frozen backbone in eval mode.

        A frozen backbone must stay in eval mode so that its BatchNorm
        running statistics are not updated by the training data.
        """
        super().train(mode)
        if self.config.freeze_backbone:
            self.model.eval()
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled feature vector ``(N, feature_dim)``."""
        features = self.model(x)
        if features.dim() > 2:
            # Some timm variants return a spatial map; pool it explicitly.
            features = features.mean(dim=tuple(range(2, features.dim())))
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias of :meth:`forward_features`."""
        return self.forward_features(x)

    def get_last_conv_module(self) -> nn.Module | None:
        """Return the last convolutional module, used as a Grad-CAM target."""
        for attribute in ("conv_head", "bn2"):
            module = getattr(self.model, attribute, None)
            if isinstance(module, nn.Conv2d):
                return module
        stages = self._stage_modules()
        if stages:
            return stages[-1]
        return None

    # ------------------------------------------------------------------
    # Parameter groups
    # ------------------------------------------------------------------
    def parameter_groups(
        self, lr_backbone: float, lr_head: float
    ) -> list[dict[str, Any]]:
        """Split the parameters into backbone and head groups.

        The head group is populated by the caller (the classifier), so this
        method only returns the backbone group here.
        """
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        frozen = [p for p in self.model.parameters() if not p.requires_grad]
        groups: list[dict[str, Any]] = []
        if trainable:
            groups.append({"params": trainable, "lr": lr_backbone, "name": "backbone"})
        if frozen:
            # Frozen parameters are still handed to the optimiser with a zero
            # learning rate so that the optimiser state stays consistent
            # when the backbone is unfrozen mid-training.
            groups.append({"params": frozen, "lr": 0.0, "name": "backbone_frozen"})
        return groups
