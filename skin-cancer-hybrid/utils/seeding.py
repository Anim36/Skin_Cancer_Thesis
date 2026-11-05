"""Reproducible seeding and deterministic execution helpers."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

__all__ = ["seed_everything", "worker_init_fn", "seed_worker", "get_generator"]


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Seed Python, NumPy and PyTorch RNGs.

    Parameters
    ----------
    seed:
        The base random seed.
    deterministic:
        When true, cuDNN is put into deterministic mode and benchmarking is
        disabled.  This trades a small amount of throughput for reproducibility
        across runs of the same experiment.

    Returns
    -------
    int
        The seed that was applied, for convenient logging.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.benchmark = True

    return seed


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker so each worker has a distinct, reproducible RNG.

    PyTorch calls this automatically when it is passed as ``worker_init_fn``.
    Without it, every worker inherits the parent's RNG state and produces
    identical augmentation sequences.
    """
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


#: Alias kept for readability at call sites.
worker_init_fn = seed_worker


def get_generator(seed: int) -> Any:
    """Return a seeded ``torch.Generator`` for DataLoader shuffling."""
    import torch

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator
