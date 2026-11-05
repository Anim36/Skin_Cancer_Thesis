"""Loss functions for imbalanced multiclass skin-lesion classification.

The three losses that the thesis compares are implemented here:

* :class:`CrossEntropyLossWrapper` - ordinary (unweighted) cross entropy.
* :class:`WeightedCrossEntropyLoss` - cross entropy with per-class weights.
* :class:`WeightedFocalLoss` - the focal loss with per-class weights.

Class weights
=============
Skin-lesion datasets are strongly imbalanced: in HAM10000, ``NV`` accounts
for roughly two thirds of the samples while ``VASC`` and ``DF`` are rare.  A
model trained with unweighted cross entropy therefore optimises the majority
class and can reach a high accuracy while completely failing to detect
melanoma - which is the clinically important class.

Two weighting schemes are provided:

``inverse_frequency``
    :math:`w_c = N / (C \\cdot n_c)`

``effective_number`` (Cui et al., 2019)
    :math:`w_c = (1 - \\beta) / (1 - \\beta^{n_c})`, which is smoother than
    the raw inverse frequency and is less affected by extremely small
    classes.

**The weights are always computed from the training split only.**  Using
validation or test labels to derive them would leak information and inflate
the reported metrics.  :func:`compute_class_weights` takes the training
labels as an explicit argument so that this cannot happen by accident.

Focal loss
==========
The focal loss (Lin et al., 2017) down-weights well-classified examples:

.. math::
    FL(p_t) = -\\alpha_{y} (1 - p_t)^{\\gamma} \\log(p_t)

With :math:`\\gamma = 0` it reduces exactly to weighted cross entropy, which
makes the comparison between the two losses a controlled one.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "compute_class_weights",
    "WeightedFocalLoss",
    "WeightedCrossEntropyLoss",
    "CrossEntropyLossWrapper",
    "build_loss",
]


def compute_class_weights(
    labels: Sequence[int] | np.ndarray,
    num_classes: int,
    strategy: str = "effective_number",
    beta: float = 0.999,
    cap: float = 10.0,
    normalize: bool = True,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute per-class loss weights from **training** labels.

    Parameters
    ----------
    labels:
        The class indices of the training split.
    num_classes:
        Total number of classes in the label space.
    strategy:
        ``effective_number`` | ``inverse_frequency`` | ``none``.
    beta:
        The ``beta`` of the effective-number scheme.
    cap:
        Maximum ratio allowed between the largest and the smallest class
        weight, so that a class with very few samples cannot dominate the
        loss.  ``None`` disables the cap.  The cap bounds the *ratio*: a cap
        on the raw values would be undone by the normalisation that follows.
    normalize:
        Whether to rescale the weights so that their weighted mean equals 1
        over the training distribution.  This keeps the overall loss scale
        comparable across experiments.
    device:
        Device on which the tensor is created.

    Returns
    -------
    torch.Tensor
        A tensor of shape ``(num_classes,)``.
    """
    if strategy in (None, "none"):
        weights = torch.ones(num_classes, dtype=torch.float32)
        return weights.to(device) if device is not None else weights

    array = np.asarray(labels, dtype=np.int64).reshape(-1)
    array = array[(array >= 0) & (array < num_classes)]
    counts = np.bincount(array, minlength=num_classes).astype(np.float64)

    if counts.sum() == 0:
        weights = torch.ones(num_classes, dtype=torch.float32)
        return weights.to(device) if device is not None else weights

    # Classes with no training sample get a neutral weight rather than an
    # infinite one; the classifier head simply never receives them.
    present = counts > 0
    raw = np.zeros(num_classes, dtype=np.float64)

    if strategy == "inverse_frequency":
        with np.errstate(divide="ignore", invalid="ignore"):
            values = len(array) / (num_classes * np.maximum(counts, 1))
        raw[present] = values[present]
    elif strategy == "effective_number":
        values = (1.0 - beta) / (1.0 - np.power(beta, np.maximum(counts, 1)))
        raw[present] = values[present]
    else:
        raise ValueError(
            f"Unknown class weighting strategy {strategy!r}. Expected one of: "
            "effective_number, inverse_frequency, none."
        )

    # The cap bounds the *ratio* between the rarest and the most common
    # class, which is what actually controls how strongly the rare class
    # dominates the loss.  Bounding the raw values before the normalisation
    # would have no effect, because the normalisation rescales them anyway.
    #
    # The excess above the minimum is compressed linearly rather than clipped:
    #
    #     w_i' = w_min + (w_i - w_min) * s
    #
    # with ``s`` chosen so that the largest weight becomes ``w_min * cap``.
    # Clipping (``w_i = min(w_i, w_min * cap)``) would collapse every class
    # above the threshold onto the same value and destroy the ordering between
    # them, whereas the linear compression keeps the ordering strictly
    # monotonic while still guaranteeing ``max(w) / min(w) <= cap``.
    if cap is not None and cap > 0 and raw[present].min() > 0:
        maximum = float(raw[present].max())
        minimum = float(raw[present].min())
        if maximum / minimum > cap:
            spread = maximum - minimum
            if spread > 0:
                scale = minimum * (cap - 1.0) / spread
                raw = np.where(
                    present,
                    minimum + (raw - minimum) * scale,
                    raw,
                )

    if normalize and raw[present].sum() > 0:
        # Normalise so the weighted average over the training distribution
        # is 1.0 - this keeps the loss magnitude comparable across runs.
        sample_weight = raw[array]
        mean_weight = sample_weight.mean()
        if mean_weight > 0:
            raw = raw / mean_weight

    weights = torch.tensor(raw, dtype=torch.float32)
    return weights.to(device) if device is not None else weights


class WeightedFocalLoss(nn.Module):
    """Multiclass focal loss with optional per-class weights.

    Parameters
    ----------
    gamma:
        The focusing parameter.  ``gamma=0`` reduces the loss to weighted
        cross entropy; ``gamma=2`` is the value from the original paper.
    alpha:
        Either a scalar multiplier or a per-class tensor of weights.  When
        ``None`` the ``class_weights`` argument is used.
    class_weights:
        Per-class weights, typically computed by
        :func:`compute_class_weights` from the training split.
    reduction:
        ``mean`` | ``sum`` | ``none``.
    label_smoothing:
        Label smoothing factor in ``[0, 1)``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        if alpha is None:
            self.register_buffer("alpha_tensor", torch.tensor(1.0))
            self.alpha_is_vector = False
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha_tensor", alpha.float())
            self.alpha_is_vector = alpha.numel() > 1
        else:
            self.register_buffer("alpha_tensor", torch.tensor(float(alpha)))
            self.alpha_is_vector = False

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Parameters
        ----------
        logits:
            ``(N, C)`` unnormalised class scores.
        target:
            ``(N,)`` integer class indices.

        Returns
        -------
        torch.Tensor
            The reduced loss (or the unreduced per-sample loss when
            ``reduction='none'``).
        """
        if logits.dim() != 2:
            raise ValueError(f"Expected 2-D logits, got shape {tuple(logits.shape)}")

        num_classes = logits.shape[1]
        log_probs = F.log_softmax(logits, dim=1)

        if self.label_smoothing > 0:
            # PyTorch's own label-smoothing implementation is numerically
            # stable, so the smoothed targets are built here and the focal
            # modulation is applied to the resulting per-sample probability.
            smooth = self.label_smoothing / max(1, num_classes - 1)
            confidence = 1.0 - self.label_smoothing
            one_hot = F.one_hot(target, num_classes).to(log_probs.dtype)
            smooth_targets = one_hot * confidence + smooth
            log_pt = (smooth_targets * log_probs).sum(dim=1)
            pt = log_pt.exp()
        else:
            log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
            pt = log_pt.exp()

        # Focal modulation: (1 - p_t)^gamma down-weights easy examples.
        focal_weight = (1.0 - pt).clamp(min=0.0, max=1.0).pow(self.gamma)

        alpha = self._alpha_for(target, num_classes, logits.dtype)
        if self.class_weights is not None:
            weights = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            alpha = alpha * weights[target]

        loss = -alpha * focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unknown reduction {self.reduction!r}")

    def _alpha_for(
        self, target: torch.Tensor, num_classes: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return the per-sample alpha factor."""
        alpha = self.alpha_tensor.to(device=target.device, dtype=dtype)
        if self.alpha_is_vector and alpha.numel() >= num_classes:
            return alpha[target]
        return alpha.expand(target.shape[0])


class WeightedCrossEntropyLoss(nn.Module):
    """Cross entropy with per-class weights.

    This is the control condition for the focal-loss comparison: with
    ``gamma=0`` the focal loss reduces to exactly this loss.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cross entropy with the fitted per-class weights."""
        if self.class_weights is not None:
            weights = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        else:
            weights = None
        return F.cross_entropy(
            logits,
            target,
            weight=weights,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )


class CrossEntropyLossWrapper(nn.Module):
    """Ordinary unweighted cross entropy (the baseline loss)."""

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits, target, label_smoothing=self.label_smoothing
        )


def build_loss(
    name: str,
    num_classes: int,
    class_weights: torch.Tensor | None = None,
    gamma: float = 2.0,
    alpha: float | torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> nn.Module:
    """Factory that builds the configured loss function.

    Parameters
    ----------
    name:
        ``weighted_focal`` | ``weighted_ce`` | ``ce``.
    num_classes:
        Number of classes (used for validation of the weight vector).
    class_weights:
        Per-class weights computed from the **training** split.
    """
    name = (name or "weighted_focal").lower()
    if name == "weighted_focal":
        return WeightedFocalLoss(
            gamma=gamma,
            alpha=alpha,
            class_weights=class_weights,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
    if name in {"weighted_ce", "weighted_cross_entropy"}:
        return WeightedCrossEntropyLoss(
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )
    if name in {"ce", "cross_entropy"}:
        return CrossEntropyLossWrapper(label_smoothing=label_smoothing)
    raise ValueError(
        f"Unknown loss {name!r}. Expected one of: weighted_focal, weighted_ce, ce."
    )
