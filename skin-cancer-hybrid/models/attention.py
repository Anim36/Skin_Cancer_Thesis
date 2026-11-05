"""Attention mechanisms: channel attention, spatial attention and CBAM.

Mathematical background
=======================
**Channel attention (squeeze-and-excitation style)**

The input feature map is :math:`F \\in \\mathbb{R}^{C \\times H \\times W}`.
Squeezing the spatial dimensions produces the channel descriptor

.. math::
    z_c = \\frac{1}{HW} \\sum_{i=1}^{HW} F_{c,i}

The descriptor is passed through two shared fully-connected layers with a
reduction ratio :math:`r`:

.. math::
    s = \\sigma\\big(W_2 \\,\\delta(W_1 z)\\big), \\qquad
    W_1 \\in \\mathbb{R}^{C/r \\times C},\\; W_2 \\in \\mathbb{R}^{C \\times C/r}

and the output is the channel-reweighted feature
:math:`\\tilde{F}_c = s_c \\cdot F_c`.  The channel weights answer *"which
feature channels matter?"*.

**Spatial attention**

A channel-average followed by a channel-max pooling compresses the channel
dimension and yields a 2-D spatial map

.. math::
    A_s = \\sigma\\big(f^7 \\times [\\mathrm{AvgPool}(F);\\ \\mathrm{MaxPool}(F)]\\big)

The output is :math:`\\tilde{F}_{i,j} = A_{s,i,j} \\cdot F_{i,j}`.  The spatial
weights answer *"which locations matter?"*.

**CBAM** (Woo et al., 2018)

Channel attention is applied first and spatial attention second, both
multiplicatively.  Because the two mechanisms are complementary and cheap
(their parameter count is negligible next to the backbone), CBAM improves
the representation at almost no cost.

For dermoscopy this is meaningful: channel attention can emphasise
pigment-network versus vascular-pattern channels, and spatial attention can
suppress the surrounding skin, ruler markings and hair artefacts.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "build_attention",
]


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation channel attention.

    Parameters
    ----------
    channels:
        Number of feature channels.
    reduction_ratio:
        The bottleneck ratio :math:`r`.  A larger ratio means fewer
        parameters.  ``r=16`` is the value from the original CBAM paper.
    """

    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        if reduction_ratio <= 0:
            raise ValueError("reduction_ratio must be positive")
        # Guard against a reduction that would collapse to zero channels.
        hidden = max(1, channels // reduction_ratio)
        self.channels = channels
        self.hidden = hidden
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        """Reweight the channels of ``x``.

        Parameters
        ----------
        x:
            Feature map of shape ``(N, C, H, W)``.
        return_weights:
            When ``True`` also return the channel attention map ``(N, C, 1, 1)``.
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        weights = torch.sigmoid(avg_out + max_out)
        out = x * weights
        if return_weights:
            return out, weights
        return out


class SpatialAttention(nn.Module):
    """Spatial attention over the channel dimension.

    Parameters
    ----------
    kernel_size:
        Size of the 7x7 convolution used to build the spatial map.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so that the padding is exact")
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False
        )

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        """Reweight the spatial positions of ``x``."""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        weights = torch.sigmoid(self.conv(combined))
        out = x * weights
        if return_weights:
            return out, weights
        return out


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Applies channel attention followed by spatial attention.  When
    ``residual`` is enabled the module returns ``x + attention(x)`` instead of
    the pure attention output, which makes it safe to insert into a
    pretrained backbone.
    """

    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 7,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)
        self.residual = residual

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        """Apply CBAM to ``x`` of shape ``(N, C, H, W)``."""
        out, channel_weights = self.channel_attention(x, return_weights=True)
        out, spatial_weights = self.spatial_attention(out, return_weights=True)
        if self.residual:
            out = x + out
        if return_weights:
            return out, channel_weights, spatial_weights
        return out


def build_attention(
    channels: int,
    attention_type: str = "cbam",
    reduction_ratio: int = 16,
    spatial_kernel_size: int = 7,
) -> nn.Module | None:
    """Factory that builds the requested attention module.

    Parameters
    ----------
    channels:
        Number of feature channels the attention operates on.
    attention_type:
        ``cbam`` | ``channel`` | ``spatial`` | ``none``.

    Returns
    -------
    torch.nn.Module | None
        ``None`` when attention is disabled.
    """
    attention_type = (attention_type or "none").lower()
    if attention_type == "none":
        return None
    if attention_type == "cbam":
        return CBAM(
            channels,
            reduction_ratio=reduction_ratio,
            spatial_kernel_size=spatial_kernel_size,
        )
    if attention_type == "channel":
        return ChannelAttention(channels, reduction_ratio)
    if attention_type == "spatial":
        return SpatialAttention(spatial_kernel_size)
    raise ValueError(
        f"Unknown attention type {attention_type!r}. "
        "Expected one of: cbam, channel, spatial, none."
    )
