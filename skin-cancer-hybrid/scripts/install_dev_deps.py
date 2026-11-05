"""Bootstrap the development environment (helper for the development machine).

This script installs the remaining scientific Python dependencies into the
active interpreter.  It exists because the interactive shell on the
development machine occasionally mangles long ``pip install`` command lines;
running the same call from a script file avoids the quoting problem.

Usage::

    python scripts/install_dev_deps.py
    python scripts/install_dev_deps.py --pip-args="--no-cache-dir"

On a GPU machine, install PyTorch first with the CUDA build::

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
"""

from __future__ import annotations

import argparse
import subprocess
import sys

PACKAGES = [
    "pandas",
    "scikit-learn",
    "scipy",
    "opencv-python",
    "albumentations",
    "matplotlib",
    "seaborn",
    "PyYAML",
    "tqdm",
    "tensorboard",
    "optuna",
    "timm",
    "pytest",
    "imagehash",
    "joblib",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pip-args",
        default="--no-cache-dir --timeout 300 --retries 8",
        help="Extra arguments forwarded to pip.",
    )
    parser.add_argument(
        "--packages",
        nargs="*",
        default=None,
        help="Override the package list.",
    )
    args = parser.parse_args()

    packages = args.packages or PACKAGES
    command = [sys.executable, "-m", "pip", "install", *args.pip_args.split(), *packages]

    print("Running:", " ".join(command), flush=True)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(f"pip exited with code {result.returncode}", file=sys.stderr)
        return result.returncode

    print("\nVerifying imports...", flush=True)
    verify = [
        sys.executable,
        "-c",
        "import pandas, sklearn, cv2, albumentations, matplotlib, seaborn, yaml, "
        "tqdm, timm, optuna, torch, torchvision; "
        "print('OK  torch', torch.__version__, '| timm', timm.__version__)",
    ]
    return subprocess.run(verify, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
