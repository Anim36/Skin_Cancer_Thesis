"""Clinical metadata encoding for multimodal (image + metadata) classification.

Source of the metadata
======================
Only **PAD-UFES-20** ships real clinical metadata in this project:

* ``age``                - numerical, in years
* ``sex``                - categorical
* ``localization``       - categorical (anatomical lesion site)
* ``skin_cancer_history``- categorical (yes / no)

The other three datasets (HAM10000, ISIC 2018, ISIC 2019) ship **no** clinical
metadata.  The project therefore never *invents* metadata for them: their
rows carry an explicit "metadata unavailable" flag, and the encoder produces a
learned missing-metadata embedding for those samples.

Anti-leakage guarantees
=======================
1. The target label is **never** an input feature.  The encoder's input
   schema is fixed to the four PAD-UFES-20 fields and is validated at fit
   time.
2. All statistics (median, scaling, category vocabularies) are fitted on the
   **training split only** via :meth:`MetadataEncoder.fit`.
3. The fitted encoder is pickled and stored alongside the model checkpoint so
   that inference applies exactly the same transformation.
4. ``fit`` refuses to run on a split that contains any target column.

Missing-value strategy
======================
* **Numerical** - median imputation computed on the training split, plus a
  binary "was missing" indicator feature so the model can distinguish
  "imputed from typical" from "genuinely observed".
* **Categorical** - a dedicated ``<MISSING>`` token, so an absent anatomical
  site is never conflated with a real category.

Both strategies are documented in ``MODEL_ARCHITECTURE.md``.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "MetadataEncoder",
    "MetadataConfig",
    "encode_metadata_row",
    "MISSING_TOKEN",
]

MISSING_TOKEN = "<MISSING>"
UNKNOWN_TOKEN = "<UNKNOWN>"

#: Columns that must never be used as metadata features.
FORBIDDEN_FEATURE_NAMES: set[str] = {
    "label", "target", "dx", "diagnosis", "class", "y", "label_id",
    "unified_label", "native_label", "target_label", "ground_truth",
}


@dataclass
class MetadataConfig:
    """Which metadata fields to use and how."""

    numerical: list[str] = field(default_factory=lambda: ["age"])
    categorical: list[str] = field(
        default_factory=lambda: ["sex", "localization", "skin_cancer_history"]
    )
    missing_strategy_numerical: str = "median_then_indicator"
    missing_strategy_categorical: str = "special_token"
    normalize_numerical: bool = True
    max_categories: int = 64

    @classmethod
    def from_config(cls, config: Any) -> "MetadataConfig":
        def _get(key: str, default: Any) -> Any:
            if hasattr(config, "get"):
                return config.get(f"metadata.{key}", default)
            if isinstance(config, dict):
                return config.get(key, default)
            return default

        strategy = _get("missing_strategy", {}) or {}
        if not isinstance(strategy, Mapping):
            strategy = {}

        return cls(
            numerical=list(_get("numerical", ["age"])),
            categorical=list(_get("categorical", ["sex", "localization",
                                                 "skin_cancer_history"])),
            missing_strategy_numerical=str(
                strategy.get("numerical", "median_then_indicator")
            ),
            missing_strategy_categorical=str(
                strategy.get("categorical", "special_token")
            ),
            normalize_numerical=bool(_get("normalize_numerical", True)),
            max_categories=int(_get("max_categories", 64)),
        )


def _clean_string(value: Any) -> str:
    """Normalise a categorical value to a trimmed string."""
    if value is None:
        return MISSING_TOKEN
    if isinstance(value, float) and np.isnan(value):
        return MISSING_TOKEN
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a", "unknown"}:
        return MISSING_TOKEN
    return text


def _clean_numeric(value: Any) -> float:
    """Convert a value to a float, or NaN when it is missing/invalid."""
    if value is None:
        return float("nan")
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
            return float("nan")
        try:
            return float(text)
        except ValueError:
            return float("nan")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return float("nan") if np.isnan(result) else result


def encode_metadata_row(
    row: Mapping[str, Any] | None,
    numerical: Sequence[str],
    categorical: Sequence[str],
) -> dict[str, Any]:
    """Extract and clean the metadata fields from a single record.

    Returns a dict with one entry per requested field.  A ``None`` ``row``
    yields an all-missing record, which the encoder maps to the dedicated
    "no metadata" vector.
    """
    out: dict[str, Any] = {}
    source = row or {}
    for name in numerical:
        out[name] = _clean_numeric(source.get(name))
    for name in categorical:
        out[name] = _clean_string(source.get(name))
    return out


class MetadataEncoder:
    """Fitted transformer for clinical metadata.

    The encoder converts a raw metadata record into a dense float32 vector of
    fixed length.  The length depends on the fitted vocabulary, so it is only
    known after :meth:`fit`.

    Attributes
    ----------
    numerical_medians:
        Per-field median used for imputation (fitted on train only).
    numerical_means / numerical_stds:
        Standardisation statistics (fitted on train only).
    categorical_vocabularies:
        Per-field ``category -> index`` mapping including the reserved
        ``<MISSING>`` and ``<UNKNOWN>`` tokens.
    output_dim:
        Length of the produced vector.
    """

    def __init__(self, config: MetadataConfig | None = None) -> None:
        self.config = config or MetadataConfig()
        self.numerical_medians: dict[str, float] = {}
        self.numerical_means: dict[str, float] = {}
        self.numerical_stds: dict[str, float] = {}
        self.categorical_vocabularies: dict[str, dict[str, int]] = {}
        self.output_dim: int = 0
        self._fitted = False
        # A dedicated slot for "no metadata at all" (datasets without clinical
        # fields).  It is a separate learned embedding in the network, but the
        # encoder still needs a fixed-length vector.
        self._no_metadata_vector: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def _check_no_target_leakage(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Raise if a forbidden target column is present in the metadata."""
        for record in records:
            for key in record.keys():
                if str(key).strip().lower() in FORBIDDEN_FEATURE_NAMES:
                    raise ValueError(
                        f"Refusing to fit metadata encoder: column {key!r} looks "
                        "like a target label. Metadata features must never "
                        "include the prediction target (anti-leakage guard)."
                    )

    def fit(self, records: Sequence[Mapping[str, Any]]) -> "MetadataEncoder":
        """Fit the encoder on the **training split** metadata.

        Parameters
        ----------
        records:
            One mapping per training sample.  Each mapping may contain only
            the declared metadata fields; extra keys are ignored except for
            the target-leakage guard.

        Returns
        -------
        MetadataEncoder
            ``self``, for chaining.
        """
        self._check_no_target_leakage(records)

        # -- numerical --------------------------------------------------
        for name in self.config.numerical:
            values = np.array(
                [_clean_numeric(r.get(name)) for r in records], dtype=np.float64
            )
            finite = values[np.isfinite(values)]
            if finite.size:
                median = float(np.median(finite))
                filled = np.where(np.isfinite(values), values, median)
                mean = float(filled.mean())
                std = float(filled.std())
                if std < 1e-8:
                    std = 1.0
            else:
                # Every value is missing - fall back to a neutral constant and
                # record that the field carries no information.
                median, mean, std = 0.0, 0.0, 1.0
            self.numerical_medians[name] = median
            self.numerical_means[name] = mean
            self.numerical_stds[name] = std

        # -- categorical ------------------------------------------------
        for name in self.config.categorical:
            values = [_clean_string(r.get(name)) for r in records]
            counts: dict[str, int] = {}
            for value in values:
                counts[value] = counts.get(value, 0) + 1
            # Reserve index 0 for <MISSING> and 1 for <UNKNOWN>; the rest are
            # ordered by descending frequency so the encoding is deterministic
            # and reflects the training distribution.
            ordered = sorted(
                (v for v in counts if v != MISSING_TOKEN),
                key=lambda v: (-counts[v], v),
            )
            ordered = ordered[: max(0, self.config.max_categories - 2)]
            vocabulary = {MISSING_TOKEN: 0, UNKNOWN_TOKEN: 1}
            for offset, value in enumerate(ordered):
                vocabulary[value] = offset + 2
            self.categorical_vocabularies[name] = vocabulary

        self._compute_output_dim()
        self._build_no_metadata_vector()
        self._fitted = True
        return self

    def _compute_output_dim(self) -> None:
        dim = 0
        for name in self.config.numerical:
            dim += 1  # imputed value
            if self.config.missing_strategy_numerical == "median_then_indicator":
                dim += 1  # missing indicator
        for name in self.config.categorical:
            dim += len(self.categorical_vocabularies.get(name, {MISSING_TOKEN: 0,
                                                                 UNKNOWN_TOKEN: 1}))
        self.output_dim = dim

    def _build_no_metadata_vector(self) -> None:
        """Build the vector used when no clinical metadata is available."""
        vector = np.zeros(self.output_dim, dtype=np.float32)
        offset = 0
        for name in self.config.numerical:
            median = self.numerical_medians.get(name, 0.0)
            mean = self.numerical_means.get(name, 0.0)
            std = self.numerical_stds.get(name, 1.0)
            value = median
            if self.config.normalize_numerical:
                value = (value - mean) / std
            vector[offset] = value
            offset += 1
            if self.config.missing_strategy_numerical == "median_then_indicator":
                vector[offset] = 1.0  # marked as missing
                offset += 1
        for name in self.config.categorical:
            vocabulary = self.categorical_vocabularies.get(name, {})
            index = vocabulary.get(MISSING_TOKEN, 0)
            vector[offset + index] = 1.0
            offset += len(vocabulary)
        self._no_metadata_vector = vector

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------
    def transform(self, record: Mapping[str, Any] | None) -> np.ndarray:
        """Transform one metadata record into a dense float32 vector.

        Passing ``None`` (or an empty mapping) returns the "no metadata"
        vector, which is how datasets without clinical fields are handled.
        """
        self._require_fitted()
        if not record:
            return (self._no_metadata_vector if self._no_metadata_vector is not None
                    else np.zeros(self.output_dim, dtype=np.float32))

        values = encode_metadata_row(
            record, self.config.numerical, self.config.categorical
        )
        vector = np.zeros(self.output_dim, dtype=np.float32)
        offset = 0

        for name in self.config.numerical:
            raw = values[name]
            is_missing = not np.isfinite(raw)
            filled = self.numerical_medians.get(name, 0.0) if is_missing else raw
            if self.config.normalize_numerical:
                mean = self.numerical_means.get(name, 0.0)
                std = self.numerical_stds.get(name, 1.0)
                filled = (filled - mean) / std
            vector[offset] = filled
            offset += 1
            if self.config.missing_strategy_numerical == "median_then_indicator":
                vector[offset] = 1.0 if is_missing else 0.0
                offset += 1

        for name in self.config.categorical:
            vocabulary = self.categorical_vocabularies.get(name, {})
            token = values[name]
            if token not in vocabulary:
                token = UNKNOWN_TOKEN
            index = vocabulary[token]
            vector[offset + index] = 1.0
            offset += len(vocabulary)

        return vector

    def transform_batch(self, records: Sequence[Mapping[str, Any] | None]) -> np.ndarray:
        """Transform a sequence of records into a ``(N, output_dim)`` array."""
        if not records:
            return np.zeros((0, self.output_dim), dtype=np.float32)
        return np.stack([self.transform(r) for r in records], axis=0)

    def fit_transform(
        self, records: Sequence[Mapping[str, Any]]
    ) -> np.ndarray:
        """Fit on the training records and return the transformed matrix."""
        self.fit(records)
        return self.transform_batch(records)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Pickle the fitted encoder."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "MetadataEncoder":
        """Load a previously fitted encoder."""
        with Path(path).open("rb") as handle:
            encoder = pickle.load(handle)
        if not isinstance(encoder, MetadataEncoder):
            raise TypeError(f"File {path} does not contain a MetadataEncoder")
        return encoder

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "MetadataEncoder has not been fitted. Call fit() on the "
                "training split before transform()."
            )

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def feature_names(self) -> list[str]:
        """Human-readable names of the encoded features (for reports)."""
        names: list[str] = []
        for name in self.config.numerical:
            names.append(f"{name}__value")
            if self.config.missing_strategy_numerical == "median_then_indicator":
                names.append(f"{name}__missing")
        for name in self.config.categorical:
            vocabulary = self.categorical_vocabularies.get(name, {})
            for token in vocabulary:
                names.append(f"{name}__{token}")
        return names

    def summary(self) -> dict[str, Any]:
        """JSON-serialisable summary for the dataset report."""
        return {
            "numerical_fields": list(self.config.numerical),
            "categorical_fields": list(self.config.categorical),
            "output_dim": self.output_dim,
            "numerical_medians": dict(self.numerical_medians),
            "categorical_sizes": {
                k: len(v) for k, v in self.categorical_vocabularies.items()
            },
            "missing_strategy_numerical": self.config.missing_strategy_numerical,
            "missing_strategy_categorical": self.config.missing_strategy_categorical,
        }
