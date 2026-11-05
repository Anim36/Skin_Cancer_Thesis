"""Shared command-line helpers for the training and evaluation scripts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "add_common_arguments",
    "parse_args",
    "build_config_from_args",
    "make_run_directory",
    "print_banner",
]


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the arguments that every script in this project accepts."""
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the main YAML configuration file "
        "(default: configs/default.yaml next to the project root).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override configuration entries, e.g. "
        "--set training.batch_size=8 hardware.device=cuda",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory that contains the four dataset folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for checkpoints, logs and reports of this run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "cpu", "mps"],
        help="Compute device. 'auto' picks the best available accelerator.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override the number of DataLoader workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        choices=["ham10000", "isic2018", "isic2019", "pad_ufes20"],
        help="Restrict the run to a subset of the datasets.",
    )
    parser.add_argument(
        "--include-scc",
        action="store_true",
        help="Use the 8-class label space so that SCC samples are used too.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Restrict the run to the named experiments (ablation study).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of Optuna trials (hyperparameter search).",
    )
    return parser


def parse_args(
    description: str,
    extra_arguments: Sequence[Any] | None = None,
) -> argparse.Namespace:
    """Build the parser, add the common arguments and parse."""
    parser = argparse.ArgumentParser(description=description)
    add_common_arguments(parser)
    for argument in extra_arguments or []:
        parser.add_argument(*argument[0], **argument[1])
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> Any:
    """Load the configuration and apply the command-line overrides."""
    from utils.config import load_config

    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config) if args.config else project_root / "configs" / "default.yaml"

    overrides = list(args.overrides or [])
    if args.data_dir:
        overrides.append(f"paths.raw_dir={args.data_dir!r}")
    if args.output_dir:
        overrides.append(f"project.output_dir={args.output_dir!r}")
    if args.device:
        overrides.append(f"hardware.device={args.device!r}")
    if args.epochs is not None:
        overrides.append(f"training.epochs={args.epochs}")
    if args.batch_size is not None:
        overrides.append(f"training.batch_size={args.batch_size}")
    if args.num_workers is not None:
        overrides.append(f"hardware.num_workers={args.num_workers}")
    if args.seed is not None:
        overrides.append(f"project.seed={args.seed}")

    config = load_config(config_path, overrides=overrides, root=project_root)
    return config


def make_run_directory(config: Any, name: str) -> Path:
    """Create (and return) the output directory of a run.

    The directory layout is::

        <output_dir>/<name>/
            checkpoints/   best.pth, last.pth
            tensorboard/
            training_history.csv
            training_summary.json
            config.yaml
    """
    output_root = Path(config.get("project.output_dir", "runs"))
    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    config.save(run_dir / "config.yaml")
    return run_dir


def print_banner(title: str) -> None:
    """Print a section banner to stdout."""
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}", flush=True)
