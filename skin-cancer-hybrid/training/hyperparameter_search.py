"""Optuna-based hyperparameter optimisation.

Usage
-----
::

    python training/hyperparameter_search.py --data-dir data/raw --n-trials 20
    python training/hyperparameter_search.py --data-dir data/raw --n-trials 50 \
        --set hyperparameter_search.storage="sqlite:///experiments/optuna.db"

Protocol
========
The objective is the **validation** macro F1.  The test split is never
touched during the search - tuning on the test set would invalidate every
number the thesis reports.

Each trial:
1. Samples a hyperparameter configuration from the search space.
2. Rebuilds the model and the loss with those values.
3. Trains for the configured number of epochs.
4. Returns the best validation macro F1.

The best configuration is written to
``experiments/hyperparameter_search/best_config.yaml`` and can be applied
directly with::

    python training/train_hybrid.py --config experiments/hyperparameter_search/best_config.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.pipeline import build_dataloaders, prepare_data
from models.hybrid_model import (
    HybridModelConfig,
    HybridSkinCancerModel,
    model_config_to_dict,
)
from models.losses import build_loss
from training.cli_utils import build_config_from_args, make_run_directory, parse_args, print_banner
from training.trainer import Trainer
from training.train_hybrid import build_trainer_config
from utils.logging_utils import setup_logging
from utils.seeding import seed_everything


def suggest(trial, params: dict[str, Any]) -> dict[str, Any]:
    """Translate the configured search space into Optuna suggestions."""
    suggested: dict[str, Any] = {}
    for name, spec in (params or {}).items():
        if not isinstance(spec, dict):
            continue
        kind = str(spec.get("type", "float"))
        if kind == "float":
            suggested[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif kind == "int":
            suggested[name] = trial.suggest_int(
                name, int(spec["low"]), int(spec["high"]), log=bool(spec.get("log", False))
            )
        elif kind == "categorical":
            suggested[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ValueError(f"Unknown search-space type {kind!r} for parameter {name!r}")
    return suggested


def main() -> int:
    print_banner("Optuna hyperparameter search")
    args = parse_args("Run the Optuna hyperparameter search.")
    config = build_config_from_args(args)

    try:
        import optuna
    except ImportError:
        print("Optuna is not installed. Install it with: pip install optuna")
        return 1

    search_config = config.get("hyperparameter_search", {}) or {}
    if not search_config.get("enabled", False):
        print(
            "Hyperparameter search is disabled in the configuration. Enable it "
            "with: --set hyperparameter_search.enabled=True"
        )
        return 1

    output_dir = Path(config.get("paths.experiments_dir", "experiments")) / "hyperparameter_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        "skin-cancer-hybrid", output_dir, filename="hyperparameter_search.log"
    )

    seed = int(config.get("project.seed", 42))
    seed_everything(seed, deterministic=bool(config.get("project.deterministic", True)))

    prepared = prepare_data(
        config,
        datasets=args.datasets,
        raw_dir=args.data_dir,
        include_scc=args.include_scc,
    )
    logger.info(
        "data ready | train=%d val=%d | classes=%s",
        len(prepared.train_frame),
        len(prepared.val_frame),
        prepared.class_names,
    )

    if len(prepared.train_frame) == 0:
        logger.error("The training split is empty.")
        return 1

    metadata_dim = prepared.metadata_encoder.output_dim if prepared.metadata_encoder else 0
    train_loader, val_loader, _ = build_dataloaders(prepared, config)

    n_trials = int(args.n_trials or search_config.get("n_trials", 20))
    metric_name = str(search_config.get("metric", "val_macro_f1"))
    direction = str(search_config.get("direction", "maximize"))

    def objective(trial) -> float:
        """Train one trial and return its validation metric."""
        values = suggest(trial, search_config.get("params", {}))

        trial_config = config.to_dict()
        trial_config["training"] = dict(trial_config.get("training", {}))
        if "lr_head" in values:
            trial_config["training"]["lr_head"] = values["lr_head"]
        if "lr_backbone" in values:
            trial_config["training"]["lr_backbone"] = values["lr_backbone"]
        if "weight_decay" in values:
            trial_config["training"]["weight_decay"] = values["weight_decay"]
        if "focal_gamma" in values:
            trial_config["training"].setdefault("focal", {})
            trial_config["training"]["focal"]["gamma"] = values["focal_gamma"]
        if "head_hidden" in values:
            trial_config["classifier"] = dict(trial_config.get("classifier", {}))
            trial_config["classifier"]["head_dims"] = [int(values["head_hidden"])]
        if "attention_reduction" in values:
            trial_config["attention"] = dict(trial_config.get("attention", {}))
            trial_config["attention"]["reduction_ratio"] = int(values["attention_reduction"])
        if "dropout" in values:
            trial_config["classifier"] = dict(trial_config.get("classifier", {}))
            trial_config["classifier"]["dropout"] = values["dropout"]

        from utils.config import Config

        trial_obj = Config(data=trial_config, root=config.root)

        model_config = HybridModelConfig.from_config(
            trial_obj, num_classes=prepared.num_classes
        )
        model_config.metadata_dim = metadata_dim
        model_config.use_metadata = bool(metadata_dim > 0)
        model = HybridSkinCancerModel(model_config)

        focal = trial_obj.get("training.focal", {}) or {}
        criterion = build_loss(
            name=str(trial_obj.get("training.loss", "weighted_focal")),
            num_classes=prepared.num_classes,
            class_weights=prepared.class_weights,
            gamma=float(focal.get("gamma", 2.0)),
            alpha=focal.get("alpha", None),
            label_smoothing=float(focal.get("label_smoothing", 0.0)),
        )

        trial_run_dir = output_dir / f"trial_{trial.number}"
        trial_run_dir.mkdir(parents=True, exist_ok=True)

        trainer_config = build_trainer_config(trial_obj, trial_run_dir, args)
        trainer_config.num_classes = prepared.num_classes
        trainer_config.class_names = prepared.class_names
        # The search uses a short schedule; the best configuration is retrained
        # with the full schedule afterwards.
        trainer_config.epochs = int(args.epochs or trial_config["training"].get("epochs", 50))
        trainer_config.early_stopping = {
            "enabled": True,
            "patience": max(3, trainer_config.epochs // 4),
            "monitor": metric_name,
            "mode": direction,
        }

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            config=trainer_config,
            metadata_dim=model_config.metadata_dim,
            class_names=prepared.class_names,
            project_config=trial_obj,
            model_config=model_config_to_dict(model_config),
        )
        summary = trainer.fit()

        value = float(summary["best_metric"])
        trial.set_user_attr("best_epoch", summary["best_epoch"])
        logger.info(
            "trial %d finished | %s=%.4f | params=%s",
            trial.number,
            metric_name,
            value,
            values,
        )
        return value

    storage = search_config.get("storage", None)
    study = optuna.create_study(
        direction=direction,
        study_name=str(search_config.get("study_name", "skin-cancer-hybrid")),
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=seed)
        if str(search_config.get("sampler", "tpe")) == "tpe"
        else None,
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=search_config.get("timeout", None),
        gc_after_trial=True,
    )

    best = study.best_trial
    logger.info("best trial %d | %s=%.4f", best.number, metric_name, best.value)
    logger.info("best params: %s", best.params)

    # -- persist the best configuration -------------------------------------
    best_config = config.to_dict()
    training = best_config.setdefault("training", {})
    if "lr_head" in best.params:
        training["lr_head"] = best.params["lr_head"]
    if "lr_backbone" in best.params:
        training["lr_backbone"] = best.params["lr_backbone"]
    if "weight_decay" in best.params:
        training["weight_decay"] = best.params["weight_decay"]
    if "focal_gamma" in best.params:
        training.setdefault("focal", {})["gamma"] = best.params["focal_gamma"]
    if "head_hidden" in best.params:
        best_config.setdefault("classifier", {})["head_dims"] = [int(best.params["head_hidden"])]
    if "attention_reduction" in best.params:
        best_config.setdefault("attention", {})["reduction_ratio"] = int(
            best.params["attention_reduction"]
        )
    if "dropout" in best.params:
        best_config.setdefault("classifier", {})["dropout"] = best.params["dropout"]

    best_path = output_dir / "best_config.yaml"
    with best_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(best_config, handle, sort_keys=False)

    summary = {
        "best_trial_number": best.number,
        "best_value": best.value,
        "metric": metric_name,
        "direction": direction,
        "best_params": best.params,
        "n_trials": n_trials,
        "best_config_path": str(best_path),
    }
    with (output_dir / "search_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    logger.info("best configuration -> %s", best_path)
    logger.info(
        "Retrain with the full schedule using: "
        "python training/train_hybrid.py --config %s",
        best_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
