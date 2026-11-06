# Dataset Compatibility Report

Generated automatically by `scripts/inspect_datasets.py` from the **actual** dataset metadata found on disk. All numbers below are measured, not assumed.

- Unified label space: `MEL, NV, BCC, AKIEC, BKL, DF, VASC`
- Total samples discovered: **24464**
- Samples usable in the unified head: **23836**

## Per-dataset compatibility

### isic2019

- Samples: **24464**
- Native classes: `mel, nv, bcc, bkl, df, vasc, scc`
- Mapped into unified head: `BCC, BKL, DF, MEL, NV, VASC`
- **Excluded from unified head**: `SCC`
- Clinical metadata: no

| Class | Samples |
| --- | ---: |
| NV | 12875 |
| MEL | 4522 |
| BCC | 3323 |
| BKL | 2624 |
| VASC | 253 |
| DF | 239 |

| Excluded native label | Samples |
| --- | ---: |
| SCC | 628 |

> metadata file: ISIC_2019_Training_GroundTruth.csv
> image dir: ISIC_2019_Training_Input
> one-hot class columns detected: ['MEL', 'NV', 'BCC', 'BKL', 'DF', 'VASC', 'SCC']
> ISIC 2019 contains the 8th class SCC (squamous cell carcinoma). SCC is NOT merged into AKIEC: it is either kept as a separate class (include_scc=True) or excluded from the shared 7-class head (include_scc=False).

## Class coverage across datasets

| Class | Description | Datasets contributing |
| --- | --- | --- |
| MEL | Melanoma | isic2019 |
| NV | Melanocytic nevus / nevi | isic2019 |
| BCC | Basal cell carcinoma | isic2019 |
| AKIEC | Actinic keratosis / intraepithelial carcinoma | - |
| BKL | Benign keratosis-like lesions | isic2019 |
| DF | Dermatofibroma | isic2019 |
| VASC | Vascular lesions | isic2019 |
| SCC | Squamous cell carcinoma | - |

## Warnings

- 628 sample(s) carry class_index=-1 because their class is not part of the active label space; they are excluded from training but retained in the manifest.

## Documented decisions

- SCC (squamous cell carcinoma) is an invasive carcinoma and is NOT merged with AKIEC (a pre-invasive actinic keratosis). It is either excluded from the shared 7-class head or trained as its own 8th class.
- Every sample keeps its native label in the manifest; nothing is silently discarded. Samples outside the unified head carry unified_label_valid=False.
- Metadata is used only where the dataset actually provides it (PAD-UFES-20). Missing metadata is never invented.
- The final test split is never used for tuning, threshold selection or hyperparameter search.
