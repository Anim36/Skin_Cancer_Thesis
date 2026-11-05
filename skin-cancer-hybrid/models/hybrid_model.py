"""The complete hybrid classification model.

Data flow
=========
::

    input image (N, 3, 224, 224)
        |
        |  [optional] U-Net lesion segmentation
        v
    segmented image (N, 3, 224, 224)
        |
        v
    EfficientNetV2-S feature extractor  ->  image features (N, F)
        |
        |  [optional] CBAM attention over the final feature map
        v
    image feature vector (N, F')
        |
        +---------------------------+
        |                           |
        v                           v
    clinical metadata      image feature vector
        |                           |
        v                           |
    MetadataEncoderNetwork          |
        |                           |
        v                           |
    metadata features (N, M)        |
        |                           |
        +-------------+-------------+
                      |
                      v
                FusionLayer  ->  fused features (N, D)
                      |
                      v
              ClassificationHead  ->  logits (N, num_classes)

Segmentation gate
=================
``use_segmentation`` can only be enabled when a **trained** U-Net checkpoint
is supplied.  A randomly initialised segmentor would produce meaningless
masks and would silently degrade the pipeline, so the model raises an error
instead of pretending.  This is deliberately strict: the thesis requires that
ablation results are trustworthy.

Gradient flow
=============
The Grad-CAM implementation needs gradients to flow through the backbone.
When ``use_segmentation`` is enabled the U-Net sits *before* the backbone, so
the gradient reaches it too.  The U-Net parameters are frozen by default
(``freeze_segmentation=True``) because it is trained separately - this saves
memory and keeps the two training stages independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import CBAM, build_attention
from models.classifier import ClassificationHead, ClassifierConfig
from models.efficientnetv2 import BackboneConfig, EfficientNetV2Backbone
from models.fusion import FusionLayer
from models.metadata_encoder import MetadataEncoderNetwork, MetadataNetworkConfig
from models.unet import UNet

__all__ = [
    "HybridSkinCancerModel",
    "HybridModelConfig",
    "build_hybrid_model",
    "model_config_to_dict",
    "build_model_from_checkpoint_payload",
]


@dataclass
class HybridModelConfig:
    """Flat configuration of the complete hybrid model."""

    # Backbone
    backbone_name: str = "efficientnetv2_s"
    pretrained: bool = True
    in_channels: int = 3
    drop_rate: float = 0.3
    drop_path_rate: float = 0.1
    global_pool: str = "avg"
    freeze_backbone: bool = False
    unfreeze_last_n_blocks: int = 2

    # Attention
    use_attention: bool = True
    attention_type: str = "cbam"
    attention_reduction_ratio: int = 16
    attention_spatial_kernel_size: int = 7
    attention_apply_on: str = "features"

    # Metadata
    use_metadata: bool = False
    metadata_dim: int = 0
    metadata_hidden_dims: Sequence[int] = (128, 64)
    metadata_output_dim: int = 64
    metadata_dropout: float = 0.2
    metadata_batch_norm: bool = True
    metadata_activation: str = "gelu"

    # Fusion
    fusion_type: str = "concat"
    fusion_dim: int = 256
    fusion_dropout: float = 0.3

    # Classifier
    num_classes: int = 7
    head_dims: Sequence[int] = (256, 128)
    head_dropout: float = 0.3
    head_batch_norm: bool = True
    head_activation: str = "gelu"

    # Segmentation
    use_segmentation: bool = False
    unet_checkpoint: str | None = None
    freeze_segmentation: bool = True
    unet_base_channels: int = 32
    unet_depth: int = 4

    @classmethod
    def from_config(cls, config: Any, num_classes: int | None = None) -> "HybridModelConfig":
        """Build the flat configuration from the project configuration."""

        def _get(path: str, default: Any) -> Any:
            if hasattr(config, "get"):
                return config.get(path, default)
            if isinstance(config, dict):
                return config.get(path, default)
            return default

        metadata_settings = MetadataNetworkConfig.from_config(config)
        classifier_settings = ClassifierConfig.from_config(config)

        resolved_num_classes = num_classes
        if resolved_num_classes is None:
            resolved_num_classes = int(_get("model.num_classes", 7))

        return cls(
            backbone_name=str(_get("backbone.name", "efficientnetv2_s")),
            pretrained=bool(_get("backbone.pretrained", True)),
            in_channels=int(_get("backbone.in_channels", 3)),
            drop_rate=float(_get("backbone.drop_rate", 0.3)),
            drop_path_rate=float(_get("backbone.drop_path_rate", 0.1)),
            global_pool=str(_get("backbone.global_pool", "avg")),
            freeze_backbone=bool(_get("backbone.freeze_backbone", False)),
            unfreeze_last_n_blocks=int(_get("backbone.unfreeze_last_n_blocks", 2)),
            use_attention=bool(_get("attention.enabled", True)),
            attention_type=str(_get("attention.type", "cbam")),
            attention_reduction_ratio=int(_get("attention.reduction_ratio", 16)),
            attention_spatial_kernel_size=int(_get("attention.spatial_kernel_size", 7)),
            attention_apply_on=str(_get("attention.apply_on", "features")),
            use_metadata=bool(_get("metadata_encoder.enabled", True)),
            metadata_dim=int(_get("metadata_dim", 0)),
            metadata_hidden_dims=tuple(metadata_settings["hidden_dims"]),
            metadata_output_dim=int(metadata_settings["output_dim"]),
            metadata_dropout=float(metadata_settings["dropout"]),
            metadata_batch_norm=bool(metadata_settings["batch_norm"]),
            metadata_activation=str(metadata_settings["activation"]),
            fusion_type=str(_get("fusion.type", "concat")),
            fusion_dim=int(_get("fusion.fusion_dim", 256)),
            fusion_dropout=float(_get("fusion.dropout", 0.3)),
            num_classes=int(resolved_num_classes),
            head_dims=tuple(classifier_settings["head_dims"]),
            head_dropout=float(classifier_settings["dropout"]),
            head_batch_norm=bool(classifier_settings["batch_norm"]),
            head_activation=str(classifier_settings["activation"]),
            use_segmentation=bool(_get("hybrid.use_segmentation", False)),
            unet_checkpoint=_get("hybrid.unet_checkpoint", None),
            freeze_segmentation=True,
            unet_base_channels=int(_get("segmentation.base_channels", 32)),
            unet_depth=int(_get("segmentation.depth", 4)),
        )


class HybridSkinCancerModel(nn.Module):
    """The complete explainable hybrid classifier.

    Parameters
    ----------
    config:
        The flat :class:`HybridModelConfig`.
    """

    def __init__(self, config: HybridModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or HybridModelConfig()

        # -- backbone ------------------------------------------------------
        self.backbone = EfficientNetV2Backbone(
            BackboneConfig(
                name=self.config.backbone_name,
                pretrained=self.config.pretrained,
                in_channels=self.config.in_channels,
                num_classes=0,
                drop_rate=self.config.drop_rate,
                drop_path_rate=self.config.drop_path_rate,
                global_pool=self.config.global_pool,
                freeze_backbone=self.config.freeze_backbone,
                unfreeze_last_n_blocks=self.config.unfreeze_last_n_blocks,
            )
        )
        self.feature_dim = self.backbone.feature_dim

        # -- attention -----------------------------------------------------
        self.attention: nn.Module | None = None
        if self.config.use_attention:
            self.attention = build_attention(
                self.feature_dim,
                attention_type=self.config.attention_type,
                reduction_ratio=self.config.attention_reduction_ratio,
                spatial_kernel_size=self.config.attention_spatial_kernel_size,
            )
            # A channel-attention-only variant operates on the pooled vector,
            # so the feature dimension is unchanged.
            self.attention_feature_dim = self.feature_dim

        # -- metadata encoder ---------------------------------------------
        self.metadata_encoder: MetadataEncoderNetwork | None = None
        metadata_dim = 0
        if self.config.use_metadata and self.config.metadata_dim > 0:
            self.metadata_encoder = MetadataEncoderNetwork(
                input_dim=self.config.metadata_dim,
                hidden_dims=self.config.metadata_hidden_dims,
                output_dim=self.config.metadata_output_dim,
                dropout=self.config.metadata_dropout,
                batch_norm=self.config.metadata_batch_norm,
                activation=self.config.metadata_activation,
            )
            metadata_dim = self.config.metadata_output_dim
        self.metadata_feature_dim = metadata_dim

        # -- fusion --------------------------------------------------------
        self.fusion = FusionLayer(
            image_dim=self.feature_dim,
            metadata_dim=metadata_dim,
            fusion_dim=self.config.fusion_dim,
            fusion_type=self.config.fusion_type,
            dropout=self.config.fusion_dropout,
        )

        # -- classifier ----------------------------------------------------
        self.classifier = ClassificationHead(
            in_dim=self.config.fusion_dim,
            num_classes=self.config.num_classes,
            hidden_dims=self.config.head_dims,
            dropout=self.config.head_dropout,
            batch_norm=self.config.head_batch_norm,
            activation=self.config.head_activation,
        )

        # -- segmentation --------------------------------------------------
        self.segmenter: UNet | None = None
        if self.config.use_segmentation:
            self.segmenter = self._build_segmenter()

    # ------------------------------------------------------------------
    def _build_segmenter(self) -> UNet:
        """Instantiate the U-Net and load a trained checkpoint.

        Raises
        ------
        RuntimeError
            When no checkpoint is configured or the checkpoint is missing.  A
            randomly initialised segmentor would produce meaningless masks,
            so the model refuses to run instead of silently degrading.
        """
        checkpoint = self.config.unet_checkpoint
        if not checkpoint:
            raise RuntimeError(
                "use_segmentation=True requires a trained U-Net checkpoint "
                "(hybrid.unet_checkpoint). Train one with "
                "python training/train_unet.py first, or disable segmentation "
                "for a controlled ablation."
            )

        path = Path(checkpoint)
        if not path.exists():
            raise RuntimeError(
                f"U-Net checkpoint not found: {path}. Train the segmentation "
                "model first with: python training/train_unet.py"
            )

        unet = UNet(
            in_channels=self.config.in_channels,
            out_channels=1,
            base_channels=self.config.unet_base_channels,
            depth=self.config.unet_depth,
        )

        payload = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = payload.get("model_state_dict", payload.get("state_dict", payload))
        if not isinstance(state_dict, dict):
            raise RuntimeError(
                f"Checkpoint {path} does not contain a recognisable state dict."
            )
        unet.load_state_dict(state_dict, strict=True)

        if self.config.freeze_segmentation:
            for parameter in unet.parameters():
                parameter.requires_grad = False
            unet.eval()

        return unet

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def segment(self, images: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Run the U-Net segmentor and return a binary lesion mask.

        Parameters
        ----------
        images:
            ``(N, 3, H, W)`` input images.
        threshold:
            Probability threshold for the mask.

        Returns
        -------
        torch.Tensor
            ``(N, 1, H, W)`` binary mask.
        """
        if self.segmenter is None:
            raise RuntimeError(
                "Segmentation is not enabled on this model. Build it with "
                "use_segmentation=True and a trained U-Net checkpoint."
            )
        was_training = self.segmenter.training
        self.segmenter.eval()
        with torch.no_grad():
            logits = self.segmenter(images)
        if was_training:
            self.segmenter.train()
        if isinstance(logits, list):
            logits = logits[0]
        return (torch.sigmoid(logits) >= threshold).float()

    def apply_segmentation(
        self, images: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Blend the lesion region with the background.

        Parameters
        ----------
        images:
            ``(N, 3, H, W)`` input images.
        mask:
            Optional precomputed binary mask.  When ``None`` the U-Net is run.

        Returns
        -------
        torch.Tensor
            The masked images.  Pixels outside the lesion are dimmed rather
            than zeroed, because a hard zero border creates an artificial
            edge that the backbone can latch onto.
        """
        if self.segmenter is None and mask is None:
            return images
        if mask is None:
            mask = self.segment(images)
        mask = mask.to(images.dtype)
        # Dim the background to 20% instead of removing it entirely.
        background = torch.full_like(images, 0.2)
        return images * mask + background * (1.0 - mask)

    def extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Run the backbone (and optional attention) to obtain image features."""
        if self.attention is not None and self.config.attention_apply_on == "features":
            # Attention over the pooled feature vector (channel attention only
            # is meaningful here; the spatial branch needs a 4-D map).
            features = self.backbone.forward_features(images)
            if isinstance(self.attention, CBAM):
                # CBAM needs a 4-D map; apply it to the backbone's last
                # convolutional output instead of the pooled vector.
                spatial = self.backbone.model.forward_features(images)
                if spatial.dim() == 4:
                    spatial = self.attention(spatial)
                    features = spatial.mean(dim=tuple(range(2, spatial.dim())))
                else:
                    features = self.attention(features)
            else:
                features = self.attention(features)
            return features

        features = self.backbone.forward_features(images)
        if self.attention is not None and self.config.attention_apply_on == "pooled":
            features = self.attention(features)
        return features

    def forward(
        self,
        images: torch.Tensor,
        metadata: torch.Tensor | None = None,
        has_metadata: torch.Tensor | None = None,
        apply_segmentation: bool | None = None,
    ) -> torch.Tensor:
        """Run the full pipeline and return the class logits.

        Parameters
        ----------
        images:
            ``(N, 3, H, W)`` input images.
        metadata:
            ``(N, metadata_dim)`` clinical metadata (optional).
        has_metadata:
            Optional boolean mask marking the samples that really have
            clinical metadata.
        apply_segmentation:
            Force the segmentation stage on or off.  ``None`` uses the model
            configuration.

        Returns
        -------
        torch.Tensor
            ``(N, num_classes)`` logits.
        """
        use_segmentation = (
            self.config.use_segmentation if apply_segmentation is None else apply_segmentation
        )
        if use_segmentation and self.segmenter is not None:
            images = self.apply_segmentation(images)

        image_features = self.extract_image_features(images)

        metadata_features = None
        if self.metadata_encoder is not None:
            if metadata is None:
                metadata = torch.zeros(
                    images.shape[0],
                    self.config.metadata_dim,
                    dtype=images.dtype,
                    device=images.device,
                )
            metadata_features = self.metadata_encoder(metadata, has_metadata)

        fused = self.fusion(image_features, metadata_features)
        return self.classifier(fused)

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(
        self,
        images: torch.Tensor,
        metadata: torch.Tensor | None = None,
        has_metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the softmax class probabilities."""
        logits = self.forward(images, metadata, has_metadata)
        return F.softmax(logits, dim=1)

    def parameter_groups(self, lr_backbone: float, lr_head: float) -> list[dict[str, Any]]:
        """Split the trainable parameters into backbone and head groups.

        The segmentation module is excluded because it is trained separately
        and frozen in the hybrid model.
        """
        groups = self.backbone.parameter_groups(lr_backbone, lr_head)
        head_parameters: list[nn.Parameter] = []
        for module in (self.fusion, self.classifier, self.metadata_encoder):
            if module is None:
                continue
            head_parameters.extend(p for p in module.parameters() if p.requires_grad)
        if head_parameters:
            groups.append({"params": head_parameters, "lr": lr_head, "name": "head"})
        return groups

    def gradcam_target_layer(self) -> nn.Module | None:
        """Return the recommended Grad-CAM target layer."""
        if self.attention is not None and self.config.attention_apply_on == "features":
            # The attention output is the most informative activation for
            # Grad-CAM because it is the last tensor before pooling.
            return self.attention
        return self.backbone.get_last_conv_module()


def build_hybrid_model(
    config: Any,
    num_classes: int,
    overrides: Mapping[str, Any] | None = None,
) -> HybridSkinCancerModel:
    """Build the hybrid model from the project configuration.

    Parameters
    ----------
    config:
        The project configuration object.
    num_classes:
        Number of classes in the training label space.
    overrides:
        Optional field overrides, used by the ablation study to toggle
        individual components.
    """
    model_config = HybridModelConfig.from_config(config, num_classes=num_classes)
    if overrides:
        for key, value in overrides.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
            else:
                raise KeyError(f"Unknown model configuration field {key!r}")
    return HybridSkinCancerModel(model_config)


def model_config_to_dict(model_config: HybridModelConfig) -> dict[str, Any]:
    """Serialise a :class:`HybridModelConfig` to a plain dictionary."""
    from dataclasses import asdict

    data = asdict(model_config)
    # The checkpoint path is machine specific - it is re-resolved at load time.
    data["unet_checkpoint"] = None
    return data


def build_model_from_checkpoint_payload(
    payload: Mapping[str, Any],
    config: Any | None = None,
) -> tuple[HybridSkinCancerModel, list[str], int, int]:
    """Rebuild the exact model that a checkpoint was trained with.

    A checkpoint stores the model configuration (``model_config``), so the
    architecture is reconstructed exactly - feature dimensions, head sizes,
    attention configuration and metadata branch included.  This is essential:
    rebuilding the model from the *current* YAML configuration would silently
    produce a different architecture and the weights would not load.

    Returns
    -------
    tuple
        ``(model, class_names, num_classes, metadata_dim)``.
    """
    class_names = list(payload.get("class_names") or [])
    num_classes = int(payload.get("num_classes", len(class_names) or 7))
    metadata_dim = int(payload.get("metadata_dim", 0))

    stored = payload.get("model_config")
    if isinstance(stored, Mapping):
        model_config = HybridModelConfig(**dict(stored))
        # The class count comes from the checkpoint, not from the config.
        model_config.num_classes = num_classes
        model_config.metadata_dim = metadata_dim
        model_config.use_metadata = metadata_dim > 0
        # The trained weights replace the ImageNet initialisation, so the
        # pretrained download is skipped (inference works offline).
        model_config.pretrained = False
    else:
        if config is None:
            raise ValueError(
                "The checkpoint does not contain a 'model_config' entry, so the "
                "architecture cannot be rebuilt. Pass the training "
                "configuration explicitly."
            )
        model_config = HybridModelConfig.from_config(config, num_classes=num_classes)
        model_config.metadata_dim = metadata_dim
        model_config.use_metadata = metadata_dim > 0
        model_config.pretrained = False

    model = HybridSkinCancerModel(model_config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, class_names, num_classes, metadata_dim
