"""Tests for the model components: U-Net, attention, fusion, classifier, hybrid."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.attention import (
    CBAM,
    ChannelAttention,
    SpatialAttention,
    build_attention,
)
from models.classifier import ClassificationHead
from models.efficientnetv2 import BackboneConfig, EfficientNetV2Backbone
from models.fusion import FusionLayer
from models.hybrid_model import HybridModelConfig, HybridSkinCancerModel
from models.metadata_encoder import MetadataEncoderNetwork
from models.unet import UNet


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------
class TestUNet:
    @pytest.mark.parametrize("depth", [2, 3, 4])
    def test_output_shape_matches_input(self, depth):
        unet = UNet(in_channels=3, out_channels=1, base_channels=8, depth=depth)
        output = unet(torch.randn(2, 3, 64, 64))
        assert output.shape == (2, 1, 64, 64)

    def test_attention_gates_preserve_the_output_shape(self):
        unet = UNet(base_channels=8, depth=3, attention=True)
        output = unet(torch.randn(1, 3, 64, 64))
        assert output.shape == (1, 1, 64, 64)

    def test_without_attention_gates(self):
        unet = UNet(base_channels=8, depth=3, attention=False)
        output = unet(torch.randn(1, 3, 64, 64))
        assert output.shape == (1, 1, 64, 64)

    def test_deep_supervision_returns_auxiliary_outputs(self):
        unet = UNet(base_channels=8, depth=3, deep_supervision=True)
        outputs = unet(torch.randn(1, 3, 64, 64))
        assert isinstance(outputs, list)
        assert len(outputs) >= 2
        assert outputs[0].shape == (1, 1, 64, 64)
        for auxiliary in outputs[1:]:
            assert auxiliary.shape[0] == 1
            assert auxiliary.shape[1] == 1

    def test_predict_mask_returns_binary_values(self):
        unet = UNet(base_channels=8, depth=2)
        mask = unet.predict_mask(torch.randn(2, 3, 32, 32))
        assert mask.shape == (2, 1, 32, 32)
        assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}

    def test_configurable_channels(self):
        unet = UNet(in_channels=1, out_channels=2, base_channels=8, depth=2)
        output = unet(torch.randn(2, 1, 32, 32))
        assert output.shape == (2, 2, 32, 32)

    def test_transposed_convolution_variant(self):
        unet = UNet(base_channels=8, depth=3, bilinear=False)
        output = unet(torch.randn(1, 3, 64, 64))
        assert output.shape == (1, 1, 64, 64)

    def test_gradients_flow(self):
        unet = UNet(base_channels=8, depth=2)
        output = unet(torch.randn(1, 3, 32, 32))
        output.mean().backward()
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in unet.parameters()
        )


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
class TestAttention:
    def test_channel_attention_shape(self):
        module = ChannelAttention(64, reduction_ratio=16)
        output = module(torch.randn(2, 64, 8, 8))
        assert output.shape == (2, 64, 8, 8)

    def test_channel_attention_returns_weights(self):
        module = ChannelAttention(32)
        output, weights = module(torch.randn(2, 32, 8, 8), return_weights=True)
        assert output.shape == (2, 32, 8, 8)
        assert weights.shape == (2, 32, 1, 1)
        # The weights are a sigmoid, so they live in (0, 1).
        assert float(weights.min()) >= 0.0 and float(weights.max()) <= 1.0

    def test_spatial_attention_shape_and_weights(self):
        module = SpatialAttention(7)
        output, weights = module(torch.randn(2, 16, 8, 8), return_weights=True)
        assert output.shape == (2, 16, 8, 8)
        assert weights.shape == (2, 1, 8, 8)

    def test_spatial_attention_rejects_even_kernel(self):
        with pytest.raises(ValueError):
            SpatialAttention(6)

    def test_cbam_shape_and_weights(self):
        module = CBAM(32, reduction_ratio=8)
        output, channel_weights, spatial_weights = module(
            torch.randn(2, 32, 8, 8), return_weights=True
        )
        assert output.shape == (2, 32, 8, 8)
        assert channel_weights.shape == (2, 32, 1, 1)
        assert spatial_weights.shape == (2, 1, 8, 8)

    def test_cbam_reduces_the_channel_count(self):
        module = CBAM(64, reduction_ratio=16)
        assert module.channel_attention.hidden == 4

    @pytest.mark.parametrize("kind", ["cbam", "channel", "spatial"])
    def test_build_attention_factory(self, kind):
        module = build_attention(32, attention_type=kind)
        assert module is not None

    def test_build_attention_none(self):
        assert build_attention(32, attention_type="none") is None

    def test_build_attention_rejects_unknown(self):
        with pytest.raises(ValueError):
            build_attention(32, attention_type="unknown")


# ---------------------------------------------------------------------------
# Fusion / classifier / metadata network
# ---------------------------------------------------------------------------
class TestFusion:
    @pytest.mark.parametrize("kind", ["concat", "gated", "attention", "concat_gated"])
    def test_output_shape(self, kind):
        fusion = FusionLayer(128, 32, 64, fusion_type=kind)
        output = fusion(torch.randn(4, 128), torch.randn(4, 32))
        assert output.shape == (4, 64)

    def test_without_metadata(self):
        fusion = FusionLayer(128, 0, 64)
        output = fusion(torch.randn(4, 128))
        assert output.shape == (4, 64)

    def test_rejects_unknown_type(self):
        with pytest.raises(ValueError):
            FusionLayer(128, 32, 64, fusion_type="nonsense")

    def test_rejects_mismatched_batch_sizes(self):
        fusion = FusionLayer(128, 32, 64)
        with pytest.raises(ValueError):
            fusion(torch.randn(4, 128), torch.randn(3, 32))


class TestClassifierHead:
    def test_output_shape(self):
        head = ClassificationHead(64, 7, hidden_dims=[32, 16])
        output = head(torch.randn(8, 64))
        assert output.shape == (8, 7)

    def test_works_with_a_batch_of_one(self):
        head = ClassificationHead(64, 7, hidden_dims=[32])
        head.train()
        output = head(torch.randn(1, 64))
        assert output.shape == (1, 7)


# ---------------------------------------------------------------------------
# Hybrid model
# ---------------------------------------------------------------------------
def _tiny_config(**overrides) -> HybridModelConfig:
    config = HybridModelConfig(
        pretrained=False,
        use_attention=True,
        use_metadata=True,
        metadata_dim=12,
        metadata_output_dim=8,
        fusion_dim=32,
        num_classes=7,
        head_dims=(16,),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class TestHybridModel:
    def test_forward_pass_with_metadata(self):
        model = HybridSkinCancerModel(_tiny_config())
        model.eval()
        with torch.no_grad():
            logits = model(torch.randn(2, 3, 224, 224), torch.randn(2, 12), torch.ones(2, dtype=torch.bool))
        assert logits.shape == (2, 7)

    def test_forward_pass_without_metadata(self):
        model = HybridSkinCancerModel(_tiny_config(use_metadata=False, metadata_dim=0))
        model.eval()
        with torch.no_grad():
            logits = model(torch.randn(2, 3, 224, 224))
        assert logits.shape == (2, 7)

    def test_predict_proba_sums_to_one(self):
        model = HybridSkinCancerModel(_tiny_config())
        model.eval()
        with torch.no_grad():
            probabilities = model.predict_proba(torch.randn(4, 3, 224, 224), torch.randn(4, 12))
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(4), atol=1e-5)
        assert (probabilities >= 0).all()

    def test_gradients_flow_to_the_backbone(self):
        model = HybridSkinCancerModel(_tiny_config())
        model.train()
        logits = model(torch.randn(2, 3, 224, 224), torch.randn(2, 12), torch.ones(2, dtype=torch.bool))
        loss = logits.mean()
        loss.backward()
        trainable = [
            p for p in model.backbone.model.parameters() if p.requires_grad
        ]
        assert trainable, "no trainable backbone parameters"
        assert any(p.grad is not None for p in trainable)

    def test_segmentation_requires_a_checkpoint(self):
        config = _tiny_config(use_segmentation=True, unet_checkpoint=None)
        with pytest.raises(RuntimeError, match="checkpoint"):
            HybridSkinCancerModel(config)

    def test_segmentation_rejects_a_missing_checkpoint_file(self, tmp_path):
        config = _tiny_config(
            use_segmentation=True, unet_checkpoint=str(tmp_path / "missing.pth")
        )
        with pytest.raises(RuntimeError, match="not found"):
            HybridSkinCancerModel(config)

    def test_segmentation_loads_a_trained_checkpoint(self, tmp_path):
        checkpoint = tmp_path / "unet.pth"
        unet = UNet(base_channels=8, depth=2)
        torch.save({"model_state_dict": unet.state_dict()}, checkpoint)

        config = _tiny_config(
            use_segmentation=True,
            unet_checkpoint=str(checkpoint),
            unet_base_channels=8,
            unet_depth=2,
        )
        model = HybridSkinCancerModel(config)
        model.eval()
        with torch.no_grad():
            logits = model(torch.randn(1, 3, 224, 224))
        assert logits.shape == (1, 7)

    def test_parameter_groups_split_backbone_and_head(self):
        model = HybridSkinCancerModel(_tiny_config())
        groups = model.parameter_groups(1e-5, 1e-4)
        names = {group.get("name") for group in groups}
        assert "backbone" in names
        assert "head" in names

    def test_gradcam_target_layer_is_resolvable(self):
        model = HybridSkinCancerModel(_tiny_config())
        layer = model.gradcam_target_layer()
        assert isinstance(layer, torch.nn.Module)


class TestBackbone:
    def test_feature_extraction_shape(self):
        backbone = EfficientNetV2Backbone(
            BackboneConfig(pretrained=False, drop_path_rate=0.0)
        )
        features = backbone.forward_features(torch.randn(1, 3, 224, 224))
        assert features.shape[0] == 1
        assert features.shape[1] == backbone.feature_dim

    def test_frozen_mode_disables_gradients(self):
        backbone = EfficientNetV2Backbone(
            BackboneConfig(pretrained=False, freeze_backbone=True, drop_path_rate=0.0)
        )
        assert all(not p.requires_grad for p in backbone.model.parameters())

    def test_full_mode_enables_gradients(self):
        backbone = EfficientNetV2Backbone(
            BackboneConfig(pretrained=False, freeze_backbone=False, unfreeze_last_n_blocks=-1,
                           drop_path_rate=0.0)
        )
        assert any(p.requires_grad for p in backbone.model.parameters())

    def test_partial_mode_unfreezes_only_some_blocks(self):
        backbone = EfficientNetV2Backbone(
            BackboneConfig(pretrained=False, unfreeze_last_n_blocks=1, drop_path_rate=0.0)
        )
        trainable = [p for p in backbone.model.parameters() if p.requires_grad]
        frozen = [p for p in backbone.model.parameters() if not p.requires_grad]
        assert trainable and frozen

    def test_frozen_backbone_stays_in_eval_mode(self):
        backbone = EfficientNetV2Backbone(
            BackboneConfig(pretrained=False, freeze_backbone=True, drop_path_rate=0.0)
        )
        backbone.train()
        assert not backbone.model.training
