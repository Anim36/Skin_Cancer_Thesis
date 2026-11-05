"""Dataset loaders, harmonisation and the manifest pipeline."""

from data.loaders.base import (
    BaseDatasetLoader,
    DatasetManifest,
    SampleRecord,
)
from data.loaders.ham10000 import HAM10000Loader
from data.loaders.isic2018 import ISIC2018Loader
from data.loaders.isic2019 import ISIC2019Loader
from data.loaders.pad_ufes20 import PADUFES20Loader

#: Registry mapping the dataset name to its loader class.
LOADERS = {
    "ham10000": HAM10000Loader,
    "isic2018": ISIC2018Loader,
    "isic2019": ISIC2019Loader,
    "pad_ufes20": PADUFES20Loader,
}

__all__ = [
    "BaseDatasetLoader",
    "DatasetManifest",
    "SampleRecord",
    "HAM10000Loader",
    "ISIC2018Loader",
    "ISIC2019Loader",
    "PADUFES20Loader",
    "LOADERS",
]
