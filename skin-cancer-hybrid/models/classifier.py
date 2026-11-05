"""Multiclass classifier head.

The head is a small MLP applied to the fused representation and produces
mutually exclusive class logits (a softmax is applied by the loss / at
prediction time).

The model returns **logits** during training (numerically stable for
cross-entropy) and the caller applies the softmax.  The prediction API applies
it explicitly so that both probabilities and the predicted class are
available.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

__all__ = ["ClassificationHead", "ClassifierConfig", "build_classifier_head"]


def _activation(name: str) -> nn.Module:
    name = (name or "gelu").lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unknown activation {name!r}")


class ClassificationHead(nn.Module):
    """Fully connected classification head.

    Parameters
    ----------
    in_dim:
        Dimension of the incoming feature vector.
    num_classes:
        Number of output classes.
    hidden_dims:
        Sizes of the hidden layers.  An empty list gives a linear head.
    dropout:
        Dropout probability.
    batch_norm:
        Whether to use BatchNorm1d on the hidden layers.
    activation:
        Activation function name.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.3,
        batch_norm: bool = True,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.use_batch_norm = bool(batch_norm)

        layers: list[nn.Module] = []
        previous = self.in_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(previous, int(hidden)))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(int(hidden)))
            layers.append(_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = int(hidden)

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(previous, self.num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return the class logits ``(N, num_classes)``."""
        if features.dim() > 2:
            features = features.flatten(1)
        if self.training and self.use_batch_norm and features.shape[0] == 1:
            for module in self.hidden:
                if isinstance(module, nn.BatchNorm1d):
                    module.eval()
        return self.output(self.hidden(features))


class ClassifierConfig:
    """Helper that reads the classifier settings from the project config."""

    @staticmethod
    def from_config(config: Any) -> dict[str, Any]:
        def _get(key: str, default: Any) -> Any:
            if hasattr(config, "get"):
                return config.get(f"classifier.{key}", default)
            if isinstance(config, dict):
                return config.get(key, default)
            return default

        return {
            "head_dims": list(_get("head_dims", [256, 128])),
            "dropout": float(_get("dropout", 0.3)),
            "batch_norm": bool(_get("batch_norm", True)),
            "activation": str(_get("activation", "gelu")),
        }


def build_classifier_head(
    in_dim: int,
    num_classes: int,
    config: Any | None = None,
) -> ClassificationHead:
    """Factory used by the hybrid model and the baseline trainer."""
    settings = ClassifierConfig.from_config(config) if config is not None else {
        "head_dims": [256, 128],
        "dropout": 0.3,
        "batch_norm": True,
        "activation": "gelu",
    }
    return ClassificationHead(
        in_dim=in_dim,
        num_classes=num_classes,
        hidden_dims=settings["head_dims"],
        dropout=settings["dropout"],
        batch_norm=settings["batch_norm"],
        activation=settings["activation"],
    )
