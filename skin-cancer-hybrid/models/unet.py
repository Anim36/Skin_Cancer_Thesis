"""U-Net for dermoscopic lesion segmentation.

Architecture
============
The classic U-Net (Ronneberger et al., 2015) with an optional attention gate on
the skip connections:

    Encoder  : double-conv blocks at 4 resolutions, each halving H/W and
               doubling the channel count.
    Bottleneck: the deepest (lowest resolution) double-conv block.
    Decoder  : up-convolution + skip concatenation + double-conv block.
    Skip     : 1x1 convolution "attention gate" that filters the encoder
               features using the coarser decoder signal, so irrelevant
               background activations are suppressed before concatenation.
    Output   : 1x1 convolution to a single channel, then a sigmoid.

Design notes
============
* The output is a **logit** map; the sigmoid is applied inside the loss so
  that the BCE part is numerically stable (``BCEWithLogitsLoss``).
* ``Deep supervision`` is available as an option: auxiliary heads are placed
  on the decoder stages and their losses are summed, which stabilises early
  training and improves the gradient flow to the encoder.
* Group normalisation is used instead of batch normalisation by default.
  Segmentation batches are small and batch statistics are unstable for
  single-lesion images, so BatchNorm is a poor fit here.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["UNet", "UNetConfig", "double_conv", "AttentionGate", "build_unet"]


def _norm_layer(norm_type: str, channels: int, groups: int = 8) -> nn.Module:
    """Build a normalisation layer.

    ``GroupNorm`` is the default because segmentation batches are small.
    """
    norm_type = (norm_type or "group").lower()
    if norm_type == "batch":
        return nn.BatchNorm2d(channels)
    if norm_type == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    # GroupNorm requires the channel count to be divisible by the group count.
    groups = min(groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(num_groups=max(1, groups), num_channels=channels)


class DoubleConv(nn.Module):
    """Two consecutive (conv -> norm -> activation) blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        norm_type: str = "group",
    ) -> None:
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            _norm_layer(norm_type, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm_layer(norm_type, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Attention gate for the U-Net skip connections.

    The gate computes an attention map from the *coarser* decoder feature
    ``g`` and the *finer* encoder feature ``x``:

    .. math::
        \\alpha = \\sigma\\big(W_f \\cdot \\psi(W_g \\cdot g) + W_x \\cdot x\\big)

    where :math:`\\psi` is a 1x1 convolution followed by an interpolation to
    the resolution of ``x``.  The encoder feature is then scaled element-wise
    by :math:`\\alpha` before it is concatenated with the decoder feature.
    """

    def __init__(
        self,
        gate_channels: int,
        in_channels: int,
        inter_channels: int | None = None,
    ) -> None:
        super().__init__()
        inter_channels = inter_channels or max(1, gate_channels // 2)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.x_conv = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=True)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Filter ``x`` (encoder feature) using the coarser ``g`` (decoder feature)."""
        theta_x = self.x_conv(x)
        phi_g = self.gate_conv(g)
        if phi_g.shape[-2:] != theta_x.shape[-2:]:
            phi_g = F.interpolate(
                phi_g, size=theta_x.shape[-2:], mode="bilinear", align_corners=False
            )
        alpha = self.psi(theta_x + phi_g)
        return x * alpha


class UNet(nn.Module):
    """U-Net segmentation model.

    Parameters
    ----------
    in_channels:
        Number of input image channels (3 for RGB).
    out_channels:
        Number of output channels (1 for a binary lesion mask).
    base_channels:
        Channel count of the first encoder stage.  32 is a good compromise
        between accuracy and memory on a single consumer GPU.
    depth:
        Number of encoder stages.  ``depth=4`` gives a 1/16 bottleneck.
    bilinear:
        Use bilinear upsampling (``True``) or transposed convolution
        (``False``) in the decoder.
    attention:
        Whether to place attention gates on the skip connections.
    deep_supervision:
        Whether to return auxiliary logits from the decoder stages.
    norm_type:
        ``group`` (default), ``batch`` or ``instance``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        bilinear: bool = True,
        attention: bool = True,
        deep_supervision: bool = False,
        norm_type: str = "group",
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.bilinear = bilinear
        self.attention = attention
        self.deep_supervision = deep_supervision

        # -- encoder ------------------------------------------------------
        self.encoders = nn.ModuleList()
        channels = [in_channels]
        current = base_channels
        for _ in range(depth):
            self.encoders.append(
                DoubleConv(channels[-1], current, norm_type=norm_type)
            )
            channels.append(current)
            current *= 2
        self.bottleneck = DoubleConv(channels[-1], current, norm_type=norm_type)
        self.pool = nn.MaxPool2d(2, 2)

        # -- decoder ------------------------------------------------------
        # The decoder mirrors the encoder: each stage halves the channel
        # count and doubles the spatial resolution.  The skip connection at
        # stage ``i`` carries ``channels[-(i + 1)]`` feature channels.
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        decoder_channels: list[int] = []
        current = current  # bottleneck channels
        for index in range(depth):
            stage_channels = current // 2
            if bilinear:
                # ``nn.Upsample`` does not change the channel count, so an
                # explicit convolution performs the channel reduction.
                up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(current, stage_channels, kernel_size=3, padding=1, bias=False),
                )
            else:
                up = nn.ConvTranspose2d(
                    current, stage_channels, kernel_size=2, stride=2
                )
            self.upconvs.append(up)

            skip_channels = channels[-(index + 1)]
            self.attention_gates.append(
                AttentionGate(gate_channels=stage_channels, in_channels=skip_channels)
                if attention
                else nn.Identity()
            )
            self.decoders.append(
                DoubleConv(stage_channels + skip_channels, stage_channels, norm_type=norm_type)
            )
            decoder_channels.append(stage_channels)
            current = stage_channels

        self.final_conv = nn.Conv2d(current, out_channels, kernel_size=1)

        # -- deep supervision heads --------------------------------------
        # One auxiliary head per decoder stage except the last, each matching
        # the channel count of the decoder feature it is attached to.
        self.aux_heads = nn.ModuleList()
        if deep_supervision:
            for stage_channels in decoder_channels[:-1]:
                self.aux_heads.append(
                    nn.Conv2d(stage_channels, out_channels, kernel_size=1)
                )

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: Any) -> "UNet":
        """Build a U-Net from the project configuration."""
        def _get(key: str, default: Any) -> Any:
            if hasattr(config, "get"):
                return config.get(f"segmentation.{key}", default)
            if isinstance(config, dict):
                return config.get(key, default)
            return default

        encoder = str(_get("encoder", "unet"))
        if encoder == "none":
            raise ValueError("segmentation.encoder is 'none' - no U-Net requested.")
        return cls(
            in_channels=int(_get("in_channels", 3)),
            out_channels=int(_get("out_channels", 1)),
            base_channels=int(_get("base_channels", 32)),
            depth=int(_get("depth", 4)),
            bilinear=bool(_get("bilinear", True)),
            attention=("attention" in encoder),
            deep_supervision=bool(_get("deep_supervision", False)),
            norm_type=str(_get("norm_type", "group")),
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """Run the U-Net.

        Parameters
        ----------
        x:
            Input tensor of shape ``(N, C, H, W)``.

        Returns
        -------
        torch.Tensor | list[torch.Tensor]
            The main logit map ``(N, out_channels, H, W)`` and, when deep
            supervision is enabled, the auxiliary logit maps.
        """
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        aux_outputs: list[torch.Tensor] = []
        for index, (upconv, decoder, gate) in enumerate(
            zip(self.upconvs, self.decoders, self.attention_gates)
        ):
            x = upconv(x)
            skip = skips[-(index + 1)]

            if skip.shape[-2:] != x.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            if self.attention and not isinstance(gate, nn.Identity):
                skip = gate(skip, x)

            x = torch.cat([skip, x], dim=1)
            x = decoder(x)
            if self.deep_supervision and index < len(self.aux_heads):
                aux_outputs.append(self.aux_heads[index](x))

        logits = self.final_conv(x)

        if self.deep_supervision and aux_outputs:
            return [logits] + aux_outputs
        return logits

    # ------------------------------------------------------------------
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return a binary mask for the input batch."""
        logits = self.forward(x)
        if isinstance(logits, list):
            logits = logits[0]
        return (torch.sigmoid(logits) >= threshold).float()


def build_unet(config: Any) -> UNet:
    """Factory used by the training scripts."""
    return UNet.from_config(config)


#: Alias kept for a uniform configuration API across the model modules.
UNetConfig = UNet
