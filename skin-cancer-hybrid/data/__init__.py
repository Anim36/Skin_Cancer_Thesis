"""Data pipeline: loaders, harmonisation, preprocessing, splitting, datasets."""

from data.label_mapping import (
    CANONICAL_CLASSES,
    UNIFIED_CLASSES,
    UNIFIED_PLUS_SCC_CLASSES,
    CLASS_DESCRIPTIONS,
    LabelMapping,
    build_label_mapping,
    normalise_label,
)
from data.preprocessing import (
    PreprocessConfig,
    PreprocessingPipeline,
    remove_hair,
    remove_hair_dullrazor,
    apply_clahe,
    normalize_image,
)
from data.augmentation import (
    AugmentationConfig,
    build_train_augmentations,
    build_val_augmentations,
)
from data.metadata import MetadataEncoder, MetadataConfig
from data.loaders import (
    LOADERS,
    HAM10000Loader,
    ISIC2018Loader,
    ISIC2019Loader,
    PADUFES20Loader,
    DatasetManifest,
)
from data.dataset_harmonization import (
    harmonise_manifests,
    build_compatibility_report,
    write_compatibility_report,
)
from data.deduplication import DedupConfig, find_duplicates
from data.splitting import SplitConfig, make_splits
from data.datasets import SkinCancerDataset, SegmentationDataset

__all__ = [
    "CANONICAL_CLASSES",
    "UNIFIED_CLASSES",
    "UNIFIED_PLUS_SCC_CLASSES",
    "CLASS_DESCRIPTIONS",
    "LabelMapping",
    "build_label_mapping",
    "normalise_label",
    "PreprocessConfig",
    "PreprocessingPipeline",
    "remove_hair",
    "remove_hair_dullrazor",
    "apply_clahe",
    "normalize_image",
    "AugmentationConfig",
    "build_train_augmentations",
    "build_val_augmentations",
    "MetadataEncoder",
    "MetadataConfig",
    "LOADERS",
    "HAM10000Loader",
    "ISIC2018Loader",
    "ISIC2019Loader",
    "PADUFES20Loader",
    "DatasetManifest",
    "harmonise_manifests",
    "build_compatibility_report",
    "write_compatibility_report",
    "DedupConfig",
    "find_duplicates",
    "SplitConfig",
    "make_splits",
    "SkinCancerDataset",
    "SegmentationDataset",
]
