"""Shared pytest fixtures.

All tests run on synthetic data only.  Synthetic data is **not** research
data and never produces a meaningful accuracy number - it exists purely to
exercise the code paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """The project root directory."""
    return ROOT


@pytest.fixture(scope="session")
def config():
    """The merged project configuration."""
    from utils.config import load_config

    return load_config(ROOT / "configs" / "default.yaml", root=ROOT)


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded random generator."""
    return np.random.default_rng(1234)


@pytest.fixture
def synthetic_image(rng) -> np.ndarray:
    """A random RGB uint8 image."""
    return (rng.random((96, 128, 3)) * 255).astype(np.uint8)


@pytest.fixture
def synthetic_mask(rng) -> np.ndarray:
    """A random binary mask with a central blob."""
    mask = np.zeros((96, 128), dtype=np.uint8)
    mask[20:70, 30:90] = 255
    return mask


@pytest.fixture
def synthetic_dataset_tree(tmp_path) -> Path:
    """A tiny synthetic HAM10000-style dataset tree."""
    import cv2
    import pandas as pd

    root = tmp_path / "ham10000"
    image_dir = root / "HAM10000_images"
    image_dir.mkdir(parents=True)

    labels = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    records = []
    for index in range(21):
        image_id = f"ISIC_{index:05d}"
        array = (np.random.default_rng(index).random((40, 50, 3)) * 255).astype(np.uint8)
        cv2.imwrite(str(image_dir / f"{image_id}.jpg"), array)
        records.append(
            {
                "image_id": image_id,
                "dx": labels[index % len(labels)],
                "lesion_id": index // 2,
                "age": 30 + index,
                "sex": "Male" if index % 2 else "Female",
                "localization": "face",
            }
        )

    pd.DataFrame(records).to_csv(root / "metadata.csv", index=False)
    return root


@pytest.fixture
def synthetic_manifest():
    """A synthetic harmonised manifest."""
    import pandas as pd

    rows = []
    for index in range(60):
        rows.append(
            {
                "sample_id": f"ham10000:ISIC_{index:05d}",
                "dataset": "ham10000",
                "image_path": f"/tmp/ISIC_{index:05d}.jpg",
                "relative_path": f"ISIC_{index:05d}.jpg",
                "original_id": f"ISIC_{index:05d}",
                "native_label": ["MEL", "NV", "BCC"][index % 3],
                "native_label_norm": ["mel", "nv", "bcc"][index % 3],
                "unified_label": ["MEL", "NV", "BCC"][index % 3],
                "unified_label_valid": True,
                "class_index": index % 3,
                "group_id": f"ham10000_lesion_{index // 3}",
                "group_type": "lesion",
                "has_metadata": False,
                "metadata": "{}",
                "source": f"ham10000:ISIC_{index:05d}.jpg",
            }
        )
    return pd.DataFrame(rows)
