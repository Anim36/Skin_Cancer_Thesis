"""Shared utilities: configuration, seeding, hardware detection, logging."""

from utils.config import Config, load_config, merge_dicts, apply_overrides, parse_value
from utils.seeding import seed_everything, seed_worker, get_generator
from utils.hardware import (
    HardwareInfo,
    detect_hardware,
    resolve_device,
    autocast_context,
    describe_hardware,
)
from utils.logging_utils import setup_logging, get_logger

__all__ = [
    "Config",
    "load_config",
    "merge_dicts",
    "apply_overrides",
    "parse_value",
    "seed_everything",
    "seed_worker",
    "get_generator",
    "HardwareInfo",
    "detect_hardware",
    "resolve_device",
    "autocast_context",
    "describe_hardware",
    "setup_logging",
    "get_logger",
]
