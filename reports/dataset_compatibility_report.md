# Dataset Compatibility Report

Generated automatically by `scripts/inspect_datasets.py` from the **actual** dataset metadata found on disk. All numbers below are measured, not assumed.

- Unified label space: `MEL, NV, BCC, AKIEC, BKL, DF, VASC`
- Total samples discovered: **21**
- Samples usable in the unified head: **21**

## Per-dataset compatibility

### ham10000

- Samples: **21**
- Native classes: `mel, nv, bcc, akiec, bkl, df, vasc`
- Mapped into unified head: `AKIEC, BCC, BKL, DF, MEL, NV, VASC`
- Clinical metadata: no

| Class | Samples |
| --- | ---: |
| MEL | 3 |
| NV | 3 |
| BCC | 3 |
| AKIEC | 3 |
| BKL | 3 |
| DF | 3 |
| VASC | 3 |

> metadata file: metadata.csv
> metadata columns: image_id, dx, lesion_id, age, sex, localization
> lesion-level grouping available: 10 lesions have more than one image (these are kept inside a single split)
> NOTE: HAM10000 age/sex/localization are present in some redistributions but are NOT used as metadata features by default - only PAD-UFES-20 metadata is used, to keep the multimodal comparison controlled.

## Class coverage across datasets

| Class | Description | Datasets contributing |
| --- | --- | --- |
| MEL | Melanoma | ham10000 |
| NV | Melanocytic nevus / nevi | ham10000 |
| BCC | Basal cell carcinoma | ham10000 |
| AKIEC | Actinic keratosis / intraepithelial carcinoma | ham10000 |
| BKL | Benign keratosis-like lesions | ham10000 |
| DF | Dermatofibroma | ham10000 |
| VASC | Vascular lesions | ham10000 |
| SCC | Squamous cell carcinoma | - |

## Documented decisions

- SCC (squamous cell carcinoma) is an invasive carcinoma and is NOT merged with AKIEC (a pre-invasive actinic keratosis). It is either excluded from the shared 7-class head or trained as its own 8th class.
- Every sample keeps its native label in the manifest; nothing is silently discarded. Samples outside the unified head carry unified_label_valid=False.
- Metadata is used only where the dataset actually provides it (PAD-UFES-20). Missing metadata is never invented.
- The final test split is never used for tuning, threshold selection or hyperparameter search.
