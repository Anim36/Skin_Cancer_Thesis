"""Grad-CAM explainability for the hybrid classifier.

Implementation
==============
Grad-CAM (Selvaraju et al., 2017) produces a class-discriminative heatmap by
weighting the spatial activations of a target convolutional layer with the
global average of their gradients:

.. math::
    \\alpha^{c}_k = \\frac{1}{Z}\\sum_{i}\\frac{\\partial Y^c}{\\partial A^k_{ij}},
    \\qquad
    L^c_{\\mathrm{Grad-CAM}} = \\mathrm{ReLU}\\Big(\\sum_k \\alpha^c_k A^k\\Big)

where :math:`Y^c` is the score of class ``c`` for the sample, :math:`A^k` is
the activation map of channel ``k`` at the target layer and :math:`Z` is the
number of spatial positions.

The implementation in this module is written directly against PyTorch, so it
works with the project's own hybrid model (backbone + attention + fusion +
head) without any dependency on an external Grad-CAM package.

Target layer selection
======================
The target layer must be a convolutional layer whose activations still carry
spatial information.  For the hybrid model the preferred target is the
attention module (or the last convolution of the backbone before pooling).
:func:`resolve_target_layer` selects it automatically and falls back
gracefully.

Limitations (important for the thesis)
======================================
* Grad-CAM produces a **coarse** localisation: the resolution of the heatmap
  is that of the target layer, not that of the input image.
* The heatmap is not a segmentation and must not be read as one.
* The method explains the model's *response*, not the true causal
  contribution of the highlighted region to the diagnosis.
* In dermoscopy, attention maps frequently overlap with the lesion simply
  because the lesion is the most salient object in the image, so a
  "correct-looking" heatmap is weak evidence of genuine reasoning.
* The classifier head is trained on masked or unmasked images depending on
  the configuration; the heatmap therefore explains whatever input the model
  actually received.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "GradCAM",
    "GradCAMResult",
    "resolve_target_layer",
    "overlay_heatmap",
]


@dataclass
class GradCAMResult:
    """The outcome of a Grad-CAM computation for a single sample."""

    heatmap: np.ndarray          # normalised to [0, 1], shape (H, W)
    overlay: np.ndarray          # RGB uint8 image with the heatmap blended in
    target_class: int
    class_name: str
    confidence: float
    class_index: int = -1

    def save(self, directory: str | Path, stem: str) -> tuple[Path, Path]:
        """Write the heatmap and the overlay as PNG files."""
        from explainability.visualization import save_heatmap, save_overlay

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        heatmap_path = save_heatmap(self.heatmap, directory / f"{stem}_heatmap.png")
        overlay_path = save_overlay(self.overlay, directory / f"{stem}_overlay.png")
        return heatmap_path, overlay_path


def resolve_target_layer(
    model: nn.Module,
    preferred_names: Sequence[str] = (),
) -> tuple[nn.Module, str]:
    """Find the Grad-CAM target layer of a model.

    The resolution order is:

    1. an explicit target-layer hook on the model (``gradcam_target_layer``),
    2. a module whose attribute path matches one of ``preferred_names``,
    3. the last convolutional layer of the model.

    Returns
    -------
    tuple
        ``(module, name)``.
    """
    # 1. The model may expose its own recommendation.
    if hasattr(model, "gradcam_target_layer"):
        try:
            module = model.gradcam_target_layer()
            if isinstance(module, nn.Module):
                return module, type(module).__name__
        except Exception:  # pragma: no cover - defensive
            pass

    # 2. Named attributes.
    for name in preferred_names:
        candidate: Any = model
        path = name
        for part in name.split("."):
            if not hasattr(candidate, part):
                candidate = None
                break
            candidate = getattr(candidate, part)
        if isinstance(candidate, nn.Module):
            return candidate, path

    # 3. The last convolutional layer in the module tree.
    last_conv: nn.Module | None = None
    last_name = ""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
            last_name = name
    if last_conv is None:
        raise RuntimeError(
            "No convolutional layer was found, so Grad-CAM cannot be computed "
            "for this model."
        )
    return last_conv, last_name or "last_conv"


class GradCAM:
    """Grad-CAM explainer for the hybrid classifier.

    Parameters
    ----------
    model:
        The model to explain.  It must return logits.
    target_layer:
        The layer whose activations are used.  Resolved automatically when
        omitted.
    target_layer_names:
        Preferred attribute paths, tried before the automatic fallback.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
        target_layer_names: Sequence[str] = (),
    ) -> None:
        self.model = model
        self.model.eval()
        self.target_layer, self.target_layer_name = (
            (target_layer, "explicit")
            if target_layer is not None
            else resolve_target_layer(model, target_layer_names)
        )
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list[Any] = []
        self._register_hooks()

    # ------------------------------------------------------------------
    def _register_hooks(self) -> None:
        def forward_hook(_module, _inputs, output):
            self._activations = output.detach()

        def backward_hook(_module, _grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        """Remove the registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.remove_hooks()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def __call__(
        self,
        images: torch.Tensor,
        metadata: torch.Tensor | None = None,
        has_metadata: torch.Tensor | None = None,
        target_classes: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the Grad-CAM heatmaps for a batch.

        Parameters
        ----------
        images:
            ``(N, 3, H, W)`` input batch.
        metadata:
            Optional clinical metadata batch.
        has_metadata:
            Optional metadata presence mask.
        target_classes:
            The class to explain for each sample.  ``None`` explains the
            predicted class of every sample.

        Returns
        -------
        tuple
            ``(heatmaps, overlays, probabilities)`` where ``heatmaps`` has
            shape ``(N, H, W)``, ``overlays`` has shape ``(N, H, W, 3)`` and
            ``probabilities`` has shape ``(N, num_classes)``.
        """
        device = next(self.model.parameters()).device
        images = images.to(device)

        # Grad-CAM needs an autograd graph.  The caller may already be inside
        # a ``torch.no_grad()`` block (for example ``SkinCancerPredictor.predict``
        # is decorated with ``@torch.no_grad()`` and calls ``explain``), so the
        # gradient mode is enabled explicitly here instead of relying on the
        # ambient context.
        with torch.enable_grad():
            if target_classes is None:
                with torch.no_grad():
                    logits = self.model(images, metadata, has_metadata)
                target_classes = logits.argmax(dim=1).detach().cpu().tolist()

            # The forward pass must build the graph so the gradients can be
            # obtained from the backward pass.
            self.model.zero_grad(set_to_none=True)
            logits = self.model(images, metadata, has_metadata)

            selected = logits.gather(
                1, torch.tensor(target_classes, device=device).long().view(-1, 1)
            ).sum()
            self.model.zero_grad(set_to_none=True)
            selected.backward(retain_graph=True)

            heatmaps = self._compute_heatmaps(images)

        probabilities = torch.softmax(logits.detach(), dim=1).cpu().numpy()
        overlays = [
            overlay_heatmap(
                _denormalize_image(image), heatmap, alpha=0.5
            )
            for image, heatmap in zip(images.cpu().numpy(), heatmaps)
        ]
        return heatmaps, np.stack(overlays), probabilities

    def _compute_heatmaps(self, images: torch.Tensor) -> np.ndarray:
        """Turn the captured activations and gradients into heatmaps."""
        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "No activations or gradients were captured. Make sure the "
                "target layer is part of the forward pass."
            )

        activations = self._activations
        gradients = self._gradients

        # Global average pooling of the gradients gives the channel weights.
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        if activations.dim() == 4:
            weighted = (weights * activations).sum(dim=1)
        elif activations.dim() == 2:
            # A pooled feature vector has no spatial dimension, so the
            # heatmap is uniform.  This happens when attention is applied to
            # the pooled features; the caller should then choose a
            # convolutional target layer instead.
            weighted = activations.sum(dim=1, keepdim=True)
        else:
            raise RuntimeError(
                f"Unexpected activation shape {tuple(activations.shape)} for Grad-CAM."
            )

        heatmap = torch.relu(weighted)
        heatmap = heatmap - heatmap.min(dim=1, keepdim=True).values
        heatmap = heatmap / (heatmap.max(dim=1, keepdim=True).values + 1e-8)

        # Upsample to the input resolution.
        heatmap = torch.nn.functional.interpolate(
            heatmap.unsqueeze(1),
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        return heatmap.cpu().numpy()


def _denormalize_image(image: np.ndarray) -> np.ndarray:
    """Convert a normalised CHW tensor back to an RGB uint8 image."""
    from data.preprocessing import IMAGENET_MEAN, IMAGENET_STD

    array = image.transpose(1, 2, 0).astype(np.float32)
    array = array * IMAGENET_STD + IMAGENET_MEAN
    array = np.clip(array, 0.0, 1.0)
    return (array * 255).astype(np.uint8)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """Blend a heatmap onto an RGB image.

    Parameters
    ----------
    image:
        RGB ``uint8`` image of shape ``(H, W, 3)``.
    heatmap:
        Heatmap of shape ``(H, W)`` with values in ``[0, 1]``.
    alpha:
        Blending weight of the heatmap.
    colormap:
        OpenCV colormap name.

    Returns
    -------
    numpy.ndarray
        The blended RGB ``uint8`` image.
    """
    import cv2

    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(
            heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
        )

    scaled = np.clip(heatmap, 0.0, 1.0)
    colored = cv2.applyColorMap(
        np.uint8(255 * scaled), getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_JET)
    )
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    blended = (alpha * colored + (1.0 - alpha) * image).astype(np.uint8)
    return blended
