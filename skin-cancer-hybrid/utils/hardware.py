"""Hardware detection, device selection and mixed-precision helpers.

The project must run on three very different targets:

* a CPU-only development machine (this repository's home),
* a local NVIDIA GPU workstation (Windows or Linux),
* a hosted GPU (Google Colab, Kaggle Notebooks, cloud VMs).

Every entry point resolves its device through :func:`resolve_device` so that
no script hardcodes ``"cuda"``.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "HardwareInfo",
    "detect_hardware",
    "resolve_device",
    "autocast_context",
    "GradScaler",
    "channels_last_if_needed",
    "describe_hardware",
]


@dataclass
class HardwareInfo:
    """Snapshot of the compute environment."""

    torch_version: str = ""
    python_version: str = ""
    cuda_available: bool = False
    cuda_version: str = ""
    cudnn_version: str = ""
    device_count: int = 0
    device_names: list[str] = field(default_factory=list)
    mps_available: bool = False
    amp_supported: bool = False
    total_memory_gb: float | None = None
    cpu_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "torch_version": self.torch_version,
            "python_version": self.python_version,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "cudnn_version": self.cudnn_version,
            "device_count": self.device_count,
            "device_names": self.device_names,
            "mps_available": self.mps_available,
            "amp_supported": self.amp_supported,
            "total_memory_gb": self.total_memory_gb,
            "cpu_count": self.cpu_count,
        }


def detect_hardware() -> HardwareInfo:
    """Probe the current machine for available accelerators."""
    import platform

    info = HardwareInfo()
    info.python_version = platform.python_version()
    info.cpu_count = os.cpu_count() or 1

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required. Install it with: pip install torch torchvision"
        ) from exc

    info.torch_version = torch.__version__
    info.cuda_available = bool(torch.cuda.is_available())
    info.cuda_version = torch.version.cuda or ""
    info.cudnn_version = str(torch.backends.cudnn.version() or "")

    if info.cuda_available:
        info.device_count = torch.cuda.device_count()
        info.device_names = [
            torch.cuda.get_device_name(i) for i in range(info.device_count)
        ]
        # AMP with float16 requires compute capability >= 6.0 (Pascal).
        try:
            major, _ = torch.cuda.get_device_capability(0)
            info.amp_supported = major >= 6
        except Exception:  # pragma: no cover - defensive
            info.amp_supported = False
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            info.total_memory_gb = round(total / (1024**3), 2)
        except Exception:  # pragma: no cover - defensive
            info.total_memory_gb = None

    if not info.cuda_available:
        # Apple Silicon MPS is a nice-to-have fallback for local development.
        try:
            info.mps_available = bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            )
        except Exception:  # pragma: no cover - defensive
            info.mps_available = False
        info.amp_supported = False

    return info


def resolve_device(
    preference: str = "auto",
    allow_amp: bool = True,
    amp_dtype: str = "float16",
) -> tuple[Any, bool, str]:
    """Resolve the compute device and whether to use mixed precision.

    Parameters
    ----------
    preference:
        ``"auto"``, ``"cuda"``, ``"cpu"`` or ``"mps"``.  With ``"auto"`` the
        best available accelerator is chosen and CPU is the final fallback.
    allow_amp:
        Whether mixed precision may be enabled at all.
    amp_dtype:
        ``"float16"`` or ``"bfloat16"``.

    Returns
    -------
    tuple
        ``(device, use_amp, amp_dtype_name)``.
    """
    import torch

    preference = (preference or "auto").lower()
    info = detect_hardware()

    if preference == "cuda":
        if not info.cuda_available:
            raise RuntimeError(
                "device='cuda' was requested but no CUDA GPU is available. "
                "Use device='auto' for automatic fallback."
            )
        device = torch.device("cuda")
    elif preference == "cpu":
        device = torch.device("cpu")
    elif preference == "mps":
        if not info.mps_available:
            raise RuntimeError("device='mps' requested but MPS is unavailable.")
        device = torch.device("mps")
    else:  # auto
        if info.cuda_available:
            device = torch.device("cuda")
        elif info.mps_available:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    use_amp = bool(allow_amp and device.type == "cuda" and info.amp_supported)
    if device.type == "cpu":
        # bfloat16 on CPU is technically possible but slow; keep float32.
        use_amp = False
    dtype_name = "float16"
    if use_amp and amp_dtype == "bfloat16":
        dtype_name = "bfloat16"

    return device, use_amp, dtype_name


def _amp_dtype(name: str):
    import torch

    if name == "bfloat16":
        return torch.bfloat16
    return torch.float16


@contextlib.contextmanager
def autocast_context(device: Any, enabled: bool, dtype_name: str = "float16"):
    """Context manager yielding an autocast region appropriate for the device.

    On CPU (or when disabled) this is a no-op so the same training code path
    works everywhere.
    """
    import torch

    if not enabled or getattr(device, "type", "cpu") != "cuda":
        yield
        return

    with torch.autocast(device_type="cuda", dtype=_amp_dtype(dtype_name)):
        yield


def GradScaler(enabled: bool):  # noqa: N802 - mirrors torch.amp API naming
    """Return a GradScaler, or a disabled shim when AMP is off."""
    import torch

    if enabled:
        try:
            return torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):  # pragma: no cover - older torch
            return torch.cuda.amp.GradScaler()
    return torch.amp.GradScaler("cuda", enabled=False)


def channels_last_if_needed(model: Any, device: Any, enabled: bool = True):
    """Move a model to ``device`` and optionally use channels-last memory format.

    Channels-last is a meaningful speedup for convolutional networks on modern
    GPUs with Tensor Cores; it is disabled on CPU where it gives no benefit.
    """
    import torch

    model = model.to(device)
    if enabled and getattr(device, "type", "cpu") == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


def describe_hardware() -> str:
    """Human-readable summary of the compute environment."""
    info = detect_hardware()
    lines = [
        f"Python              : {info.python_version}",
        f"PyTorch             : {info.torch_version}",
        f"CUDA available      : {info.cuda_available}",
        f"CUDA version        : {info.cuda_version or 'n/a'}",
        f"cuDNN version       : {info.cudnn_version or 'n/a'}",
        f"GPU count           : {info.device_count}",
        f"GPU names           : {', '.join(info.device_names) or 'n/a'}",
        f"GPU memory (GB)     : {info.total_memory_gb if info.total_memory_gb else 'n/a'}",
        f"MPS available       : {info.mps_available}",
        f"AMP supported       : {info.amp_supported}",
        f"CPU cores           : {info.cpu_count}",
    ]
    return "\n".join(lines)
