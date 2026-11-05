"""Fusion of image features and clinical metadata features.

Four fusion strategies are implemented so that the fusion design itself can be
ablated:

``concat`` (default)
    .. math:: z = [\\,f_{img} \\;\\Vert\\; f_{meta}\\,]

    The simplest and usually the strongest option for this kind of task.
    The concatenated vector is projected back to ``fusion_dim``.

``gated``
    A learned sigmoid gate per image feature dimension modulates how much of
    the image signal passes into the fused representation:

    .. math:: g = \\sigma(W_g f_{img}), \\quad z = g \\odot f_{img} + f_{meta}

    Useful when the metadata can be noisy or missing.

``attention``
    The metadata vector produces a query that attends over the image feature
    dimensions, letting the clinical information decide which visual features
    matter.  Implemented as a single-head, single-query cross attention.

``concat_gated``
    Concatenation followed by a learned gate on the concatenated vector.

Every strategy projects into a common ``fusion_dim`` so that the downstream
classifier is identical regardless of the fusion type.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

__all__ = ["FusionLayer", "build_fusion"]


class FusionLayer(nn.Module):
    """Fuse an image feature vector with a metadata feature vector.

    Parameters
    ----------
    image_dim:
        Dimension of the image feature.
    metadata_dim:
        Dimension of the metadata feature (0 disables the metadata branch).
    fusion_dim:
        Dimension of the fused representation.
    fusion_type:
        ``concat`` | ``gated`` | ``attention`` | ``concat_gated``.
    dropout:
        Dropout applied after the fusion projection.
    """

    def __init__(
        self,
        image_dim: int,
        metadata_dim: int = 0,
        fusion_dim: int = 256,
        fusion_type: str = "concat",
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.image_dim = int(image_dim)
        self.metadata_dim = int(metadata_dim)
        self.fusion_dim = int(fusion_dim)
        self.fusion_type = (fusion_type or "concat").lower()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if self.metadata_dim <= 0:
            # Metadata-free path: project the image features and stop.
            self.projection = nn.Linear(self.image_dim, self.fusion_dim)
            self.metadata_projection = None
            self.gate = None
            self.attention = None
            return

        if self.fusion_type == "concat":
            self.projection = nn.Linear(self.image_dim + self.metadata_dim, self.fusion_dim)
            self.metadata_projection = None
            self.gate = None
            self.attention = None
        elif self.fusion_type == "gated":
            self.metadata_projection = nn.Linear(self.metadata_dim, self.image_dim)
            self.gate = nn.Linear(self.image_dim, self.image_dim)
            self.projection = nn.Linear(self.image_dim, self.fusion_dim)
            self.attention = None
        elif self.fusion_type == "attention":
            self.metadata_projection = nn.Linear(self.metadata_dim, self.image_dim)
            # A single query (from the metadata) attends over the image
            # feature dimensions.
            self.query = nn.Linear(self.image_dim, self.image_dim)
            self.key = nn.Linear(self.image_dim, self.image_dim)
            self.value = nn.Linear(self.image_dim, self.image_dim)
            self.attention = nn.MultiheadAttention(
                embed_dim=self.image_dim, num_heads=1, batch_first=True
            )
            self.projection = nn.Linear(self.image_dim, self.fusion_dim)
        elif self.fusion_type == "concat_gated":
            combined_dim = self.image_dim + self.metadata_dim
            self.projection = nn.Linear(combined_dim, self.fusion_dim)
            self.gate = nn.Sequential(
                nn.Linear(combined_dim, combined_dim),
                nn.Sigmoid(),
            )
            self.metadata_projection = None
            self.attention = None
        else:
            raise ValueError(
                f"Unknown fusion type {fusion_type!r}. Expected one of: "
                "concat, gated, attention, concat_gated."
            )

    # ------------------------------------------------------------------
    def forward(
        self,
        image_features: torch.Tensor,
        metadata_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse the two feature sources.

        Parameters
        ----------
        image_features:
            ``(N, image_dim)`` image features.
        metadata_features:
            ``(N, metadata_dim)`` metadata features.  May be ``None`` when the
            metadata branch is disabled.

        Returns
        -------
        torch.Tensor
            ``(N, fusion_dim)`` fused features.
        """
        if self.metadata_dim <= 0 or metadata_features is None:
            return self.dropout(self.projection(image_features))

        if metadata_features.shape[0] != image_features.shape[0]:
            raise ValueError(
                "Batch size mismatch between image features "
                f"({image_features.shape[0]}) and metadata features "
                f"({metadata_features.shape[0]})."
            )

        if self.fusion_type == "concat":
            combined = torch.cat([image_features, metadata_features], dim=1)
            return self.dropout(self.projection(combined))

        if self.fusion_type == "gated":
            projected = self.metadata_projection(metadata_features)
            gate = torch.sigmoid(self.gate(image_features))
            fused = gate * image_features + projected
            return self.dropout(self.projection(fused))

        if self.fusion_type == "attention":
            projected = self.metadata_projection(metadata_features).unsqueeze(1)
            query = self.query(projected)
            key = self.key(image_features).unsqueeze(1)
            value = self.value(image_features).unsqueeze(1)
            attended, _ = self.attention(query, key, value)
            fused = image_features + attended.squeeze(1)
            return self.dropout(self.projection(fused))

        # concat_gated
        combined = torch.cat([image_features, metadata_features], dim=1)
        gate = self.gate(combined)
        return self.dropout(self.projection(gate * combined))


def build_fusion(
    image_dim: int,
    metadata_dim: int = 0,
    fusion_type: str = "concat",
    fusion_dim: int = 256,
    dropout: float = 0.3,
) -> FusionLayer:
    """Factory used by the hybrid model."""
    return FusionLayer(
        image_dim=image_dim,
        metadata_dim=metadata_dim,
        fusion_dim=fusion_dim,
        fusion_type=fusion_type,
        dropout=dropout,
    )
