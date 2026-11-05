"""Dermoscopic image preprocessing and augmentation.

Pipeline applied to every image (train / val / test alike)::

    load -> validate -> RGB -> resize(224) -> hair removal -> CLAHE
         -> [denoise] -> normalise

Random augmentation is applied **only** during training and is kept
deliberately conservative: colour, asymmetry and texture carry the diagnostic
signal in dermoscopy, so aggressive hue jitter or elastic warping is avoided.

Hair removal
------------
Two algorithms are implemented:

``dullrazor``
    The classic DullRazor algorithm: morphological closing with a black
    kernel, grayscale thresholding of the *hair map* (not the image),
    inpainting the detected hairs.  Faithful to the published method.

``morphological``
    A lighter, faster variant that uses a per-image grayscale threshold plus
    a median/open morphological filter to suppress dark hair structures.  It
    is less aggressive and preserves skin texture better on low-contrast
    images where DullRazor can over-segment vessels.

Both operate on the *original* image; originals are never overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

__all__ = [
    "PreprocessConfig",
    "PreprocessingPipeline",
    "remove_hair",
    "remove_hair_dullrazor",
    "remove_hair_morphological",
    "apply_clahe",
    "denoise_image",
    "normalize_image",
    "validate_image",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

#: Interpolation constants keyed by name so that OpenCV stays behind one API.
_INTERPOLATIONS = {
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
    "nearest": cv2.INTER_NEAREST,
    "area": cv2.INTER_AREA,
}


@dataclass
class PreprocessConfig:
    """Configuration for the deterministic preprocessing stage."""

    image_size: int = 224
    color_mode: str = "RGB"
    interpolation: str = "bilinear"
    hair_removal: bool = True
    hair_removal_method: str = "dullrazor"   # dullrazor | morphological | none
    clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    denoise: bool = False
    denoise_strength: int = 5
    normalize: str = "imagenet"               # imagenet | unit | none

    @classmethod
    def from_config(cls, config: Any) -> "PreprocessConfig":
        """Build from a project :class:`~utils.config.Config` or plain dict."""
        def _get(key: str, default: Any) -> Any:
            if hasattr(config, "get"):
                return config.get(f"preprocessing.{key}", default)
            if isinstance(config, dict):
                return config.get(key, default)
            return default

        return cls(
            image_size=int(_get("image_size", 224)),
            color_mode=str(_get("color_mode", "RGB")),
            interpolation=str(_get("interpolation", "bilinear")),
            hair_removal=bool(_get("hair_removal", True)),
            hair_removal_method=str(_get("hair_removal_method", "dullrazor")),
            clahe=bool(_get("clahe", True)),
            clahe_clip_limit=float(_get("clahe_clip_limit", 2.0)),
            clahe_grid_size=int(_get("clahe_grid_size", 8)),
            denoise=bool(_get("denoise", False)),
            denoise_strength=int(_get("denoise_strength", 5)),
            normalize=str(_get("normalize", "imagenet")),
        )


# ---------------------------------------------------------------------------
# Low level operations
# ---------------------------------------------------------------------------
def validate_image(image: np.ndarray | None) -> bool:
    """Return ``True`` when ``image`` is a usable 2-D or 3-D array."""
    if image is None or not isinstance(image, np.ndarray):
        return False
    if image.size == 0:
        return False
    if image.ndim not in (2, 3):
        return False
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        return False
    return bool(np.isfinite(image).all())


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert any decoded image array to a contiguous 3-channel RGB uint8 array."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:
        # OpenCV decodes BGR; images loaded with PIL come in RGB.  The loader
        # normalises this by always reading with OpenCV (BGR) and converting.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def remove_hair_dullrazor(
    image: np.ndarray,
    kernel_size: int = 17,
    inpaint_radius: int = 3,
) -> np.ndarray:
    """Remove hair with the DullRazor algorithm.

    Steps
    -----
    1. Grayscale copy of the image.
    2. Morphological closing with a black structuring element to connect hair
       pixels into elongated structures.
    3. Threshold the closed image against the original grayscale to build a
       *hair map* (dark structures that were not present in the original).
    4. Inpaint the hair map region from the surrounding skin.

    Parameters
    ----------
    image:
        RGB ``uint8`` image.
    kernel_size:
        Side length of the black structuring element.  Larger values remove
        thicker hair but risk erasing fine texture; 17 is the value commonly
        used in the literature.
    inpaint_radius:
        Radius of the OpenCV inpainting neighbourhood.

    Returns
    -------
    numpy.ndarray
        Hair-free RGB image (a new array; the input is not modified).
    """
    if image.size == 0:
        return image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 1. Morphological closing with a black kernel links hair segments.
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # 2. Difference against the original highlights removed dark structures.
    diff = cv2.subtract(closed, gray)

    # 3. Threshold into a clean hair mask.
    _, hair_mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

    # 4. Dilate slightly to cover hair edges.
    dilate_kernel = np.ones((3, 3), np.uint8)
    hair_mask = cv2.dilate(hair_mask, dilate_kernel, iterations=1)

    if hair_mask.max() == 0:
        return image.copy()

    # 5. Inpaint the hair regions from surrounding skin.
    return cv2.inpaint(image, hair_mask, inpaint_radius, cv2.INPAINT_TELEA)


def remove_hair_morphological(
    image: np.ndarray,
    kernel_size: int = 9,
    threshold_offset: int = 12,
) -> np.ndarray:
    """Remove hair with a lighter morphological filter.

    Dark thin structures are suppressed using a grayscale median background
    estimate and a blackhat transform.  This variant is gentler than
    DullRazor and preserves fine skin texture on low-contrast images.
    """
    if image.size == 0:
        return image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    background = cv2.medianBlur(gray, kernel_size)
    blackhat = cv2.subtract(background, gray)

    _, hair_mask = cv2.threshold(blackhat, threshold_offset, 255, cv2.THRESH_BINARY)
    hair_mask = cv2.morphologyEx(
        hair_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
    )
    hair_mask = cv2.dilate(hair_mask, np.ones((3, 3), np.uint8), iterations=1)

    if hair_mask.max() == 0:
        return image.copy()

    return cv2.inpaint(image, hair_mask, 3, cv2.INPAINT_TELEA)


def remove_hair(image: np.ndarray, method: str = "dullrazor") -> np.ndarray:
    """Dispatch to the configured hair removal algorithm."""
    method = (method or "none").lower()
    if method == "dullrazor":
        return remove_hair_dullrazor(image)
    if method in {"morphological", "morph"}:
        return remove_hair_morphological(image)
    return image.copy()


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: int = 8,
) -> np.ndarray:
    """Apply CLAHE to the luminance channel, preserving colour fidelity.

    CLAHE is applied on the LAB L channel rather than on a per-channel
    grayscale copy so that chroma (an important diagnostic cue in dermoscopy)
    is left untouched.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(grid_size, grid_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def denoise_image(image: np.ndarray, strength: int = 5) -> np.ndarray:
    """Apply non-local means denoising (optional, off by default)."""
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)


def normalize_image(
    image: np.ndarray,
    mode: str = "imagenet",
) -> np.ndarray:
    """Normalise a ``uint8`` RGB image to ``float32`` in ``[0, 1]``.

    ``imagenet`` applies the standard ImageNet channel statistics used by the
    pretrained backbone; ``unit`` only scales to ``[0, 1]``; ``none`` performs
    no scaling.
    """
    image = image.astype(np.float32)
    if mode == "none":
        return image
    image = image / 255.0
    if mode == "imagenet":
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
@dataclass
class PreprocessingPipeline:
    """Deterministic preprocessing pipeline shared by all datasets.

    The pipeline is stateless with respect to the image content and can be
    reused safely across DataLoader worker processes.
    """

    config: PreprocessConfig = field(default_factory=PreprocessConfig)

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    # -- helpers -----------------------------------------------------------
    def _resize(self, image: np.ndarray) -> np.ndarray:
        size = self.config.image_size
        interp = _INTERPOLATIONS.get(self.config.interpolation, cv2.INTER_LINEAR)
        if image.shape[0] == size and image.shape[1] == size:
            return image
        return cv2.resize(image, (size, size), interpolation=interp)

    # -- public API --------------------------------------------------------
    def __call__(self, image: np.ndarray, apply_hair_removal: bool | None = None) -> np.ndarray:
        """Run the full preprocessing pipeline on a decoded image.

        Parameters
        ----------
        image:
            Decoded image as ``uint8`` (any channel count).
        apply_hair_removal:
            Override the configured hair removal behaviour.  Useful to compare
            the effect of hair removal in an ablation study.

        Returns
        -------
        numpy.ndarray
            ``float32`` image of shape ``(H, W, 3)``.
        """
        if not validate_image(image):
            raise ValueError("Invalid image passed to the preprocessing pipeline")

        working = to_rgb(image)

        # Hair removal runs at the original resolution, where hair structures
        # are still high-frequency detail; resizing first would blur them and
        # make them harder to detect.
        use_hair = self.config.hair_removal
        if apply_hair_removal is not None:
            use_hair = apply_hair_removal
        if use_hair:
            working = remove_hair(working, self.config.hair_removal_method)

        working = self._resize(working)

        if self.config.clahe:
            working = apply_clahe(
                working,
                clip_limit=self.config.clahe_clip_limit,
                grid_size=self.config.clahe_grid_size,
            )

        if self.config.denoise:
            working = denoise_image(working, self.config.denoise_strength)

        return normalize_image(working, self.config.normalize)

    def load_and_process(
        self,
        path: str,
        apply_hair_removal: bool | None = None,
    ) -> np.ndarray:
        """Read an image from disk (BGR via OpenCV) and preprocess it."""
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return self(image, apply_hair_removal=apply_hair_removal)

    def preprocess_mask(
        self,
        mask: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Resize and binarise a segmentation mask.

        Nearest-neighbour interpolation is mandatory for masks - bilinear
        would create intermediate grey values that corrupt the ground truth.
        """
        if mask.ndim == 3:
            mask = mask[..., 0]
        size = self.config.image_size
        interp = _INTERPOLATIONS["nearest"]
        if mask.shape[0] != size or mask.shape[1] != size:
            mask = cv2.resize(mask, (size, size), interpolation=interp)
        binary = (mask > threshold).astype(np.float32) if mask.dtype != np.uint8 else (
            mask > 127
        ).astype(np.float32)
        return binary


def build_preprocessing_pipeline(config: Any) -> PreprocessingPipeline:
    """Factory used by loaders and training scripts."""
    return PreprocessingPipeline(PreprocessConfig.from_config(config))
