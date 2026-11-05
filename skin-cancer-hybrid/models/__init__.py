"""Model components: U-Net, EfficientNetV2, attention, fusion, classifier, losses."""

from models.unet import UNet, UNetConfig
from models.efficientnetv2 import EfficientNetV2Backbone, BackboneConfig
from models.attention import (
    CBAM,
    ChannelAttention,
    SpatialAttention,
    build_attention,
)
from models.metadata_encoder import MetadataEncoderNetwork
from models.fusion import FusionLayer, build_fusion
from models.classifier import ClassificationHead, build_classifier_head
from models.hybrid_model import (
    HybridSkinCancerModel,
    HybridModelConfig,
    build_hybrid_model,
)
from models.losses import (
    WeightedFocalLoss,
    WeightedCrossEntropyLoss,
    CrossEntropyLossWrapper,
    compute_class_weights,
    build_loss,
)

__all__ = [
    "UNet",
    "UNetConfig",
    "EfficientNetV2Backbone",
    "BackboneConfig",
    "CBAM",
    "ChannelAttention",
    "SpatialAttention",
    "build_attention",
    "MetadataEncoderNetwork",
    "FusionLayer",
    "build_fusion",
    "ClassificationHead",
    "build_classifier_head",
    "HybridSkinCancerModel",
    "HybridModelConfig",
    "build_hybrid_model",
    "WeightedFocalLoss",
    "WeightedCrossEntropyLoss",
    "CrossEntropyLossWrapper",
    "compute_class_weights",
    "build_loss",
]
