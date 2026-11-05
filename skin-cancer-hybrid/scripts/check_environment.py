"""Check the development environment.

Usage
-----
::

    python scripts/check_environment.py
    python scripts/check_environment.py --config configs/default.yaml --data-dir data/raw

The script reports the Python version, the installed package versions, the
available compute hardware and whether the project can run on this machine.
It never fails because a GPU is missing - a CPU-only machine is a perfectly
valid development environment for this project.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# (import name, expected version).  The import name is what
# ``importlib.import_module`` needs; it is not always the same as the PyPI
# distribution name (``scikit-learn`` -> ``sklearn``,
# ``opencv-python`` -> ``cv2``, ``PyYAML`` -> ``yaml``).
REQUIRED_PACKAGES = [
    ("torch", "2.4.1"),
    ("torchvision", "0.19.1"),
    ("timm", "1.0.9"),
    ("numpy", "1.26.4"),
    ("pandas", "2.2.3"),
    ("sklearn", "1.5.2"),
    ("cv2", "4.10.0.84"),
    ("PIL", "10.4.0"),
    ("albumentations", "1.4.21"),
    ("matplotlib", "3.9.2"),
    ("seaborn", "0.13.2"),
    ("yaml", "6.0.2"),
    ("tqdm", "4.66.5"),
    ("tensorboard", "2.18.0"),
    ("optuna", "3.6.2"),
]

#: Packages that are nice to have but do not block the project.
OPTIONAL_PACKAGES = [
    ("pytest", None),
    ("imagehash", None),
    ("scikit-image", None),
]


def check_packages() -> tuple[list[tuple[str, str, str]], list[str]]:
    """Report the installed version of every required package."""
    rows: list[tuple[str, str, str]] = []
    missing: list[str] = []

    for name, expected in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(name)
            installed = getattr(module, "__version__", "unknown")
            rows.append((name, expected, str(installed)))
        except ImportError:
            rows.append((name, expected, "NOT INSTALLED"))
            missing.append(name)

    for name, _ in OPTIONAL_PACKAGES:
        try:
            module = importlib.import_module(name)
            rows.append((name, "optional", str(getattr(module, "__version__", "installed"))))
        except ImportError:
            rows.append((name, "optional", "not installed (optional)"))

    return rows, missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the development environment.")
    parser.add_argument("--config", default=None, help="Configuration file to validate.")
    parser.add_argument("--data-dir", default=None, help="Dataset root to inspect.")
    args = parser.parse_args()

    print("=" * 78)
    print("ENVIRONMENT CHECK")
    print("=" * 78)
    print(f"Python      : {platform.python_version()} ({sys.executable})")
    print(f"Platform    : {platform.platform()}")
    print(f"Project root: {ROOT}")
    print()

    # -- packages ----------------------------------------------------------
    print("-" * 78)
    print("PACKAGES")
    print("-" * 78)
    rows, missing = check_packages()
    print(f"{'package':<20}{'expected':<14}{'installed'}")
    for name, expected, installed in rows:
        marker = " " if installed != "NOT INSTALLED" else "!"
        print(f"{marker}{name:<19}{expected:<14}{installed}")
    print()

    # -- hardware ----------------------------------------------------------
    print("-" * 78)
    print("HARDWARE")
    print("-" * 78)
    try:
        from utils.hardware import describe_hardware

        print(describe_hardware())
    except Exception as exc:
        print(f"Hardware detection failed: {exc}")
    print()

    # -- configuration -----------------------------------------------------
    print("-" * 78)
    print("CONFIGURATION")
    print("-" * 78)
    try:
        from utils.config import load_config

        config_path = Path(args.config) if args.config else ROOT / "configs" / "default.yaml"
        config = load_config(config_path, root=ROOT)
        print(f"Loaded {config_path.name}: {len(config.data)} top-level sections")
        for section in sorted(config.data.keys()):
            print(f"  - {section}")
    except Exception as exc:
        print(f"Configuration check FAILED: {exc}")
        missing.append("configuration")
    print()

    # -- datasets ----------------------------------------------------------
    print("-" * 78)
    print("DATASETS")
    print("-" * 78)
    from data.pipeline import dataset_roots_from_config

    try:
        config = load_config(
            Path(args.config) if args.config else ROOT / "configs" / "default.yaml",
            root=ROOT,
        )
    except Exception:
        config = None

    if config is not None:
        roots = dataset_roots_from_config(config, args.data_dir)
        found_any = False
        for name, root in roots.items():
            exists = root.exists()
            found_any = found_any or exists
            status = "present" if exists else "not found"
            extra = ""
            if exists:
                try:
                    count = sum(1 for p in root.rglob("*") if p.is_file())
                    extra = f" ({count} files)"
                except OSError:
                    pass
            print(f"  {name:<12} {status:<12} {root}{extra}")
        if not found_any:
            print("\n  No dataset found yet. See DATASET_GUIDE.md for the download")
            print("  and placement instructions. The project can be developed and")
            print("  tested without the datasets.")
    print()

    # -- summary -----------------------------------------------------------
    print("=" * 78)
    if missing:
        print(f"STATUS: {len(missing)} issue(s) found:")
        for item in missing:
            print(f"  - {item}")
        print("\nInstall the missing packages with:")
        print("  pip install -r requirements.txt")
        return 1

    print("STATUS: environment is ready.")
    print("A CUDA GPU is NOT required for development, unit tests or the")
    print("synthetic forward-pass checks. It is only needed for full training.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
