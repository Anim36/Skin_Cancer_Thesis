"""Lightweight logging helpers with console + rotating-file output."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

__all__ = ["setup_logging", "get_logger"]

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED: set[str] = set()


def setup_logging(
    name: str = "skin-cancer-hybrid",
    log_dir: str | Path | None = None,
    level: int = logging.INFO,
    filename: str = "run.log",
) -> logging.Logger:
    """Configure and return a logger writing to stdout and (optionally) a file.

    Repeated calls with the same ``name`` reconfigure the handlers instead of
    duplicating them, which keeps notebook re-runs clean.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if name in _CONFIGURED:
        return logger

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _CONFIGURED.add(name)
    return logger


def get_logger(name: str = "skin-cancer-hybrid") -> logging.Logger:
    """Return a logger, configuring the default one on first use."""
    if name not in _CONFIGURED:
        return setup_logging(name)
    return logging.getLogger(name)


def log_config(logger: logging.Logger, config: Any, keys: list[str] | None = None) -> None:
    """Log a compact view of the configuration for run provenance."""
    data = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    if keys:
        data = {k: data.get(k) for k in keys}
    logger.info("Configuration snapshot:")
    for key, value in data.items():
        logger.info("  %-28s %s", key, value)
