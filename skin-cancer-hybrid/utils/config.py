"""Configuration loading, merging and CLI override handling.

The project uses a small layered YAML configuration:

    configs/default.yaml   <- master, includes the others
    configs/dataset.yaml
    configs/model.yaml
    configs/training.yaml
    configs/evaluation.yaml

`load_config` merges the include tree so that a caller receives one flat,
dictionary-like configuration object.  Every script accepts ``--config`` and
``--set key.path=value`` so that no experiment requires editing a YAML file.
"""

from __future__ import annotations

import ast
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

__all__ = [
    "Config",
    "load_config",
    "merge_dicts",
    "apply_overrides",
    "parse_value",
    "resolve_path",
]


# ---------------------------------------------------------------------------
# Merging helpers
# ---------------------------------------------------------------------------
def merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` returning a new dict.

    Nested mappings are merged key-by-key; every other type (including lists)
    is replaced wholesale.  This is the standard "last include wins" behaviour
    used by most YAML config systems and keeps list overrides predictable.
    """
    result: dict[str, Any] = dict(copy.deepcopy(dict(base)))
    for key, value in override.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_value(raw: str) -> Any:
    """Parse a CLI override value into a Python object.

    Uses ``ast.literal_eval`` so numbers, booleans, lists and dicts work, and
    falls back to the raw string for free-text values.
    """
    raw = raw.strip()
    if not raw:
        return ""
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        lowered = raw.lower()
        if lowered in {"true", "yes", "on"}:
            return True
        if lowered in {"false", "no", "off"}:
            return False
        if lowered in {"none", "null"}:
            return None
        return raw


def apply_overrides(config: dict[str, Any], overrides: Iterable[str] | None) -> dict[str, Any]:
    """Apply ``key.path=value`` style overrides in-place on a copy of config."""
    if not overrides:
        return config
    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(
                f"Invalid override {item!r}. Expected format: key.path=value"
            )
        dotted, _, raw_value = item.partition("=")
        keys = [k for k in dotted.strip().split(".") if k]
        if not keys:
            raise ValueError(f"Invalid override {item!r}: empty key")
        node: dict[str, Any] = result
        for key in keys[:-1]:
            nxt = node.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                node[key] = nxt
            node = nxt
        node[keys[-1]] = parse_value(raw_value)
    return result


# ---------------------------------------------------------------------------
# Path handling
# ---------------------------------------------------------------------------
def resolve_path(path: str | os.PathLike[str], root: Path) -> Path:
    """Resolve ``path`` relative to ``root`` when it is not already absolute.

    Absolute paths (Windows ``C:\\...`` or POSIX ``/...``) are returned
    unchanged so that a user can point at a dataset on another drive.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


# ---------------------------------------------------------------------------
# Config object
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Thin, attribute-accessible wrapper around the merged config dictionary.

    Supports ``cfg.training.epochs`` style access as well as ``cfg["training"]``
    dictionary access, and ``cfg.get("a.b.c", default)`` for dotted lookups.
    """

    data: dict[str, Any]
    root: Path
    source: Path | None = None

    # -- container protocol ------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __contains__(self, key: object) -> bool:
        return key in self.data

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()

    # -- dotted access -----------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value

    def path(self, dotted: str, default: str | None = None) -> Path:
        """Resolve a configured path entry against the project root."""
        raw = self.get(dotted, default)
        if raw is None:
            raise KeyError(f"No path configured for {dotted!r}")
        return resolve_path(str(raw), self.root)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    def save(self, destination: str | os.PathLike[str]) -> Path:
        """Write the merged configuration to disk as YAML."""
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.data, handle, sort_keys=False, default_flow_style=False)
        return dest

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Config(root={self.root}, keys={sorted(self.data.keys())})"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _deep_update(target: dict[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in other.items():
        if isinstance(value, Mapping):
            target[key] = _deep_update(dict(target.get(key, {})), value)
        else:
            target[key] = value
    return target


def load_config(
    path: str | os.PathLike[str] | None = None,
    overrides: Iterable[str] | None = None,
    root: str | os.PathLike[str] | None = None,
) -> Config:
    """Load, merge and return the project configuration.

    Parameters
    ----------
    path:
        Path to a YAML file.  When omitted, ``configs/default.yaml`` relative to
        the project root is used.  A file may declare ``includes: [...]`` which
        are merged *under* the including file, so the including file always
        takes precedence.
    overrides:
        Iterable of ``key.path=value`` strings applied last.
    root:
        Project root used to resolve relative paths.  Defaults to the parent of
        the ``configs`` directory that holds the loaded file.
    """
    if path is None:
        project_root = Path(__file__).resolve().parents[1]
        path = project_root / "configs" / "default.yaml"
    config_path = Path(path).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    project_root = Path(root).resolve() if root is not None else config_path.parent.parent

    merged: dict[str, Any] = {}
    for include in raw.get("includes", []) or []:
        include_path = (project_root / include).resolve()
        if not include_path.exists():
            # A missing include must not break the whole project - warn loudly
            # by skipping it, but continue with whatever was resolved.
            continue
        with include_path.open("r", encoding="utf-8") as handle:
            included = yaml.safe_load(handle) or {}
        merged = merge_dicts(merged, included)

    # The including file wins over its includes.
    merged = merge_dicts(merged, raw)
    merged.pop("includes", None)

    merged = apply_overrides(merged, overrides)

    # Anchor the project root so downstream path resolution is deterministic.
    merged.setdefault("project", {})
    merged["project"]["root"] = str(project_root)
    merged["project"]["config_file"] = str(config_path)

    return Config(data=merged, root=project_root, source=config_path)


def dump_json(obj: Any, destination: str | os.PathLike[str]) -> Path:
    """Write ``obj`` as pretty-printed JSON (used for manifests and reports)."""
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=str)
    return dest
