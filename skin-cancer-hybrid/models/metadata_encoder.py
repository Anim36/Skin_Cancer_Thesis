"""Neural encoder for clinical metadata.

The encoder turns the dense metadata vector produced by
:class:`data.metadata.MetadataEncoder` into a learned embedding that can be
fused with the image features.

Structure
=========
::

    input (D)
      -> Linear(D, H1) -> BatchNorm1d -> GELU -> Dropout
      -> Linear(H1, H2) -> BatchNorm1d -> GELU -> Dropout
      -> Linear(H2, output_dim)

Notes
=====
* ``BatchNorm1d`` is switched off automatically when the batch size is 1,
  because batch statistics are undefined in that case.  This matters for
  inference on a single image.
* A dedicated **missing-metadata embedding** is learned.  When a sample has no
  clinical metadata at all (all datasets except PAD-UFES-20), the encoder
  adds this learned vector instead of the imputed one.  The network therefore
  learns what "no clinical information" means rather than being fed an
  artificial age/sex that could bias the prediction.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

__all__ = ["MetadataEncoderNetwork", "MetadataNetworkConfig"]


def _activation(name: str) -> nn.Module:
    name = (name or "gelu").lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation {name!r}")


class MetadataEncoderNetwork(nn.Module):
    """MLP encoder for clinical metadata.

    Parameters
    ----------
    input_dim:
        Length of the metadata vector (the fitted encoder's ``output_dim``).
    hidden_dims:
        Sizes of the hidden layers.
    output_dim:
        Dimension of the produced embedding.
    dropout:
        Dropout probability applied after each hidden layer.
    batch_norm:
        Whether to use BatchNorm1d on the hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        output_dim: int = 64,
        dropout: float = 0.2,
        batch_norm: bool = True,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.use_batch_norm = bool(batch_norm)

        layers: list[nn.Module] = []
        previous = self.input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(previous, int(hidden)))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(int(hidden)))
            layers.append(_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = int(hidden)

        self.feature_layers = nn.Sequential(*layers)
        self.projection = nn.Linear(previous, self.output_dim)

        # Learned embedding for "no clinical metadata available".
        self.missing_embedding = nn.Parameter(torch.zeros(self.output_dim))
        nn.init.normal_(self.missing_embedding, std=0.02)

    # ------------------------------------------------------------------
    def forward(
        self,
        metadata: torch.Tensor,
        has_metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode the metadata.

        Parameters
        ----------
        metadata:
            Metadata tensor of shape ``(N, input_dim)``.
        has_metadata:
            Optional boolean tensor of shape ``(N,)`` marking which samples
            actually have clinical metadata.  When it is ``None`` the presence
            of metadata is inferred from the vector itself (all-zero rows are
            treated as missing).

        Returns
        -------
        torch.Tensor
            The metadata embedding of shape ``(N, output_dim)``.
        """
        if metadata.dim() == 1:
            metadata = metadata.unsqueeze(0)

        if self.training and self.use_batch_norm:
            # BatchNorm1d cannot compute statistics for a batch of one.
            if metadata.shape[0] == 1:
                self._disable_batch_norm_stats()

        embedding = self.projection(self.feature_layers(metadata))

        if has_metadata is None:
            # A metadata vector is considered absent when it carries no
            # information at all (all zeros).
            has_metadata = metadata.abs().sum(dim=1) > 0

        # Keep the mask boolean until after the inversion, then cast.
        has_metadata = has_metadata.reshape(-1).to(torch.bool)
        missing = (~has_metadata).to(embedding.dtype).reshape(-1, 1)
        return embedding + missing * self.missing_embedding

    def _disable_batch_norm_stats(self) -> None:
        """Temporarily switch BatchNorm layers into evaluation mode."""
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()


class MetadataNetworkConfig:
    """Helper that reads the metadata-encoder settings from the config."""

    @staticmethod
    def from_config(config: Any) -> dict[str, Any]:
        def _get(key: str, default: Any) -> Any:
            if hasattr(config, "get"):
                return config.get(f"metadata_encoder.{key}", default)
            if isinstance(config, dict):
                return config.get(key, default)
            return default

        return {
            "hidden_dims": list(_get("hidden_dims", [128, 64])),
            "dropout": float(_get("dropout", 0.2)),
            "output_dim": int(_get("output_dim", 64)),
            "batch_norm": bool(_get("batch_norm", True)),
            "activation": str(_get("activation", "gelu")),
            "enabled": bool(_get("enabled", True)),
        }
