"""Unified label vocabulary and the mapping rules between native and unified labels.

Design rationale
================
The four datasets use different taxonomies:

=============  ======  =================================================
Dataset        Classes Native codes
=============  ======  =================================================
HAM10000       7       MEL, NV, BCC, AKIEC, BKL, DF, VASC
ISIC 2018      7       MEL, NV, BCC, AKIEC, BKL, DF, VASC
ISIC 2019      8       MEL, NV, BCC, AKIEC, BKL, DF, VASC, SCC
PAD-UFES-20    6       MEL, NV, BCC, AKIEC, BKL, SCC
=============  ======  =================================================

The clinically critical distinction is **SCC (squamous cell carcinoma)**.
It is present in ISIC 2019 and PAD-UFES-20 but absent from HAM10000 and
ISIC 2018.  Silently folding SCC into AKIEC would be *medically wrong*:

* AKIEC is a *pre-invasive* actinic keratosis / intraepithelial carcinoma.
* SCC is an *invasive* squamous cell carcinoma.

Merging them would destroy the model's ability to detect invasive disease and
would corrupt the reported per-class metrics.

Therefore the project uses three coexisting label spaces:

1. ``native`` - the dataset's own taxonomy, always preserved.
2. ``unified`` - the 7-class taxonomy used for cross-dataset experiments.
   SCC samples are **excluded** (``unified_label = -1``), never remapped.
3. ``unified_plus_scc`` - 8 classes; used when ISIC 2019 participates in
   training, so no ISIC 2019 data is discarded.

Every sample therefore always has a home, and the "combined" experiments can be
run either with SCC excluded (pure 7-class comparison) or included (8-class,
maximum data use).  Both are supported by the same code path.

All mappings are *derived from the real dataset metadata at runtime* by
:func:`build_label_mapping`; the fallback aliases in the YAML config only
provide a sane default when a dataset's label column is missing.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "CANONICAL_CLASSES",
    "UNIFIED_CLASSES",
    "UNIFIED_PLUS_SCC_CLASSES",
    "CLASS_DESCRIPTIONS",
    "LabelMapping",
    "build_label_mapping",
    "normalise_label",
    "resolve_dataset_class_list",
]

#: Full canonical vocabulary including SCC.  Index in this list is *not* the
#: class index used by any model - indices come from the per-dataset label maps.
CANONICAL_CLASSES: list[str] = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "SCC"]

#: The classic seven-class taxonomy (no SCC).
UNIFIED_CLASSES: list[str] = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

#: Eight-class taxonomy used when SCC participates in training.
UNIFIED_PLUS_SCC_CLASSES: list[str] = [
    "MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "SCC",
]

CLASS_DESCRIPTIONS: dict[str, str] = {
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus / nevi",
    "BCC": "Basal cell carcinoma",
    "AKIEC": "Actinic keratosis / intraepithelial carcinoma",
    "BKL": "Benign keratosis-like lesions",
    "DF": "Dermatofibroma",
    "VASC": "Vascular lesions",
    "SCC": "Squamous cell carcinoma",
}

#: Classes that are safe to combine into the shared 7-class head.
UNIFIED_HEAD_EXCLUDES: set[str] = {"SCC"}

#: Alias table used to normalise native label strings to canonical codes.
#: Keys are canonical codes; values are lowercase alias strings.
DEFAULT_ALIASES: dict[str, list[str]] = {
    "MEL": ["mel", "melanoma"],
    "NV": ["nv", "melanocytic nevus", "melanocytic nevi", "nevus", "nevi",
           "melanocytic_nevus", "melanocytic_nevi"],
    "BCC": ["bcc", "basal cell carcinoma", "basal_cell_carcinoma"],
    "AKIEC": ["akiec", "akiiec", "actinic keratosis", "actinic keratoses",
              "intraepithelial carcinoma", "bowen disease",
              "actinic keratosis / intraepithelial carcinoma",
              "actinic keratosis-intraepithelial carcinoma"],
    "BKL": ["bkl", "benign keratosis", "benign keratosis-like lesions",
            "benign keratosis-like lesion", "solar lentigo", "lentigo",
            "benign keratosis-like lesions"],
    "DF": ["df", "dermatofibroma"],
    "VASC": ["vasc", "vascular lesions", "vascular lesion"],
    "SCC": ["scc", "squamous cell carcinoma", "squamous_cell_carcinoma"],
}

#: Reverse lookup built once: normalised alias -> canonical code.
_ALIAS_LOOKUP: dict[str, str] = {}
for _code, _aliases in DEFAULT_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_LOOKUP[_alias] = _code


def normalise_label(raw: Any) -> str:
    """Normalise a native label string to a lowercase, comparable form.

    Handles unicode accents, punctuation, whitespace and case so that
    ``"Basal Cell Carcinoma"``, ``"bcc"`` and ``"BCC "`` all collapse to the
    same key.
    """
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = text.replace("_", " ").replace("-", " ").replace("/", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class LabelMapping:
    """Maps native dataset labels onto a target label space.

    Attributes
    ----------
    dataset:
        Name of the dataset this mapping was built for.
    target_classes:
        Ordered list of target class codes.  The position of a class in this
        list *is* its integer index used by the models.
    native_to_unified:
        ``native label (normalised) -> unified class code``.
    unified_to_index:
        ``unified class code -> integer index``.
    native_classes:
        The dataset's own class list (always preserved).
    native_to_index:
        ``native class code -> integer index in the native head``.
    excluded_labels:
        Native labels that exist in the dataset but have no counterpart in the
        unified head (currently only SCC).  These are *retained in the
        manifest* and simply excluded from unified-head training.
    unmapped_labels:
        Native labels that could not be resolved at all.  Reported loudly.
    """

    dataset: str
    target_classes: list[str]
    native_to_unified: dict[str, str] = field(default_factory=dict)
    unified_to_index: dict[str, int] = field(default_factory=dict)
    native_classes: list[str] = field(default_factory=list)
    native_to_index: dict[str, int] = field(default_factory=dict)
    excluded_labels: list[str] = field(default_factory=list)
    unmapped_labels: list[str] = field(default_factory=list)

    # -- lookups -----------------------------------------------------------
    def to_unified(self, native_label: Any) -> str | None:
        """Return the unified code for a native label, or ``None`` if excluded.

        A label that was deliberately excluded from this mapping (currently
        only SCC) always returns ``None`` - it must never be silently
        remapped onto a different, medically different class.
        """
        key = normalise_label(native_label)
        if key in self.native_to_unified:
            return self.native_to_unified[key]
        # A native label that was not present in the training metadata may
        # still be resolvable through the canonical alias table - but only
        # when that class is part of this mapping's target space.
        canonical = _ALIAS_LOOKUP.get(key)
        if canonical is not None and canonical in self.unified_to_index:
            return canonical
        return None

    def to_unified_index(self, native_label: Any) -> int:
        """Return the unified integer index, or ``-1`` when the label is excluded."""
        code = self.to_unified(native_label)
        if code is None:
            return -1
        return self.unified_to_index.get(code, -1)

    def to_native_index(self, native_label: Any) -> int:
        """Return the native integer index, or ``-1`` when unknown."""
        key = normalise_label(native_label)
        return self.native_to_index.get(key, -1)

    def native_index_to_code(self, index: int) -> str:
        """Inverse of :meth:`to_native_index`."""
        for code, idx in self.native_to_index.items():
            if idx == index:
                return code
        return "UNKNOWN"

    def class_names(self) -> list[str]:
        """Human-readable class names for the target label space."""
        return [CLASS_DESCRIPTIONS.get(code, code) for code in self.target_classes]

    def summary(self) -> dict[str, Any]:
        """JSON-serialisable summary used by the compatibility report."""
        return {
            "dataset": self.dataset,
            "target_classes": list(self.target_classes),
            "num_target_classes": len(self.target_classes),
            "native_classes": list(self.native_classes),
            "num_native_classes": len(self.native_classes),
            "native_to_unified": dict(self.native_to_unified),
            "excluded_labels": list(self.excluded_labels),
            "unmapped_labels": list(self.unmapped_labels),
            "compatible": not self.unmapped_labels,
        }


def _resolve_aliases(
    raw_label: str,
    alias_table: Mapping[str, Sequence[str]],
) -> str | None:
    """Resolve a native label to a canonical code using an alias table."""
    key = normalise_label(raw_label)
    if not key:
        return None
    if key in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[key]
    for code, aliases in alias_table.items():
        for alias in aliases:
            if normalise_label(alias) == key:
                return code
    return None


def build_label_mapping(
    dataset: str,
    native_labels: Iterable[Any],
    target_classes: Sequence[str] | None = None,
    alias_table: Mapping[str, Sequence[str]] | None = None,
    exclude_from_unified: Iterable[str] | None = None,
) -> LabelMapping:
    """Build a :class:`LabelMapping` from the *observed* native labels.

    Parameters
    ----------
    dataset:
        Dataset name, e.g. ``"ham10000"``.
    native_labels:
        The label values actually found in the dataset metadata.  The function
        does **not** assume a fixed set of classes - it derives the native
        taxonomy from this input.
    target_classes:
        Ordered target label space.  Defaults to :data:`UNIFIED_CLASSES`.
    alias_table:
        Optional override/extension of the canonical alias table.
    exclude_from_unified:
        Canonical codes that must not enter the unified head.  Defaults to
        ``{"SCC"}``.

    Returns
    -------
    LabelMapping
    """
    target_classes = list(target_classes or UNIFIED_CLASSES)
    aliases = dict(DEFAULT_ALIASES)
    if alias_table:
        for code, extra in alias_table.items():
            key = code.upper()
            aliases.setdefault(key, [])
            aliases[key] = list(aliases[key]) + list(extra)

    # NOTE: an empty set is a *meaningful* value (nothing is excluded), so the
    # default must only apply when the argument is None - a truthiness check
    # would wrongly re-enable the SCC exclusion.
    excludes = (
        set(UNIFIED_HEAD_EXCLUDES)
        if exclude_from_unified is None
        else set(exclude_from_unified)
    )

    # Preserve the order in which native labels are first seen, but sort by
    # canonical order when the label resolves to a known class so the native
    # head is stable across runs regardless of file order.
    observed: list[str] = []
    seen: set[str] = set()
    for raw in native_labels:
        key = normalise_label(raw)
        if key and key not in seen:
            seen.add(key)
            observed.append(key)

    resolved: dict[str, str | None] = {}
    for key in observed:
        resolved[key] = _resolve_aliases(key, aliases)

    # Order native classes: known canonical codes first (canonical order),
    # then any genuinely unknown labels in alphabetical order.
    canonical_rank = {code: i for i, code in enumerate(CANONICAL_CLASSES)}
    known = [k for k in observed if resolved[k] in canonical_rank]
    unknown = [k for k in observed if resolved[k] not in canonical_rank]
    known.sort(key=lambda k: canonical_rank[resolved[k]])  # type: ignore[index]
    unknown.sort()

    native_classes = known + unknown
    native_to_index = {code: idx for idx, code in enumerate(native_classes)}

    native_to_unified: dict[str, str] = {}
    excluded: list[str] = []
    unmapped: list[str] = []

    for key in native_classes:
        code = resolved[key]
        if code is None:
            unmapped.append(key)
            continue
        if code in excludes:
            excluded.append(code)
            continue
        native_to_unified[key] = code

    unified_to_index = {code: idx for idx, code in enumerate(target_classes)}

    return LabelMapping(
        dataset=dataset,
        target_classes=target_classes,
        native_to_unified=native_to_unified,
        unified_to_index=unified_to_index,
        native_classes=native_classes,
        native_to_index=native_to_index,
        excluded_labels=sorted(set(excluded)),
        unmapped_labels=unmapped,
    )


def resolve_dataset_class_list(
    dataset: str,
    observed_labels: Iterable[Any],
    include_scc: bool = False,
    alias_table: Mapping[str, Sequence[str]] | None = None,
) -> list[str]:
    """Return the ordered class list a dataset should train on.

    ``include_scc=True`` produces the 8-class space for datasets that contain
    SCC; ``False`` produces the shared 7-class space with SCC excluded.
    """
    target = UNIFIED_PLUS_SCC_CLASSES if include_scc else UNIFIED_CLASSES
    mapping = build_label_mapping(
        dataset=dataset,
        native_labels=observed_labels,
        target_classes=target,
        alias_table=alias_table,
        exclude_from_unified=set() if include_scc else UNIFIED_HEAD_EXCLUDES,
    )
    # Only keep target classes that the dataset actually contains.
    present = set(mapping.native_to_unified.values())
    return [code for code in target if code in present]
