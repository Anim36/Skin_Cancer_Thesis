# Dataset Guide

Expected layouts, label harmonisation, metadata handling and leakage policy
for the four datasets.

---

## 1. The four datasets

| # | Dataset | Images | Native classes | Masks | Clinical metadata |
|---|---|---:|---:|---|---|
| 1 | **HAM10000** | 10,015 | 7 | no | no |
| 2 | **ISIC 2018** | 10,015 | 7 | **yes** | no |
| 3 | **ISIC 2019** | 25,331 | 8 | no | no |
| 4 | **PAD-UFES-20** | 2,298 | 6 | no | **yes** |

All four are used. None is permanently excluded. The framework supports
dataset-specific training and evaluation as well as a combined pipeline.

---

## 2. Directory layout

Place the datasets under `data/raw/` (or anywhere, via `--data-dir`):

```
data/raw/
├── ham10000/
│   ├── HAM10000_images/            # ISIC_xxxxxxx.jpg
│   └── metadata.csv                # image_id, dx, lesion_id, ...
├── isic2018/
│   ├── ISIC2018_Task1-2_Training_Input/       # segmentation images
│   ├── ISIC2018_Task1-2_Training_GroundTruth/  # ISIC_xxxxxxx_segmentation.png
│   ├── ISIC2018_Task1-2_Training_GroundTruth.csv   # image (no diagnosis column)
│   ├── ISIC2018_Task3_Training_Input/        # classification images
│   └── ISIC2018_Task3_Training_GroundTruth.csv   # image, diagnosis
├── isic2019/
│   ├── ISIC_2019_Training_Input/   # ISIC_xxxxxxx.jpg
│   ├── ISIC_2019_Training_GroundTruth/
│   └── ISIC_2019_Training_GroundTruth.csv
└── pad_ufes20/
    ├── images/                     # <image_id>.png
    └── metadata.csv                # image_id, dx, age, sex, localization,
                                    # skin_cancer_history
```

The official ISIC 2018 challenge archive (Task 1-2 = segmentation, Task 3 =
classification) is the layout shown above. The loader also accepts the
flattened redistribution `ISIC2018_Training_Input/` +
`ISIC2018_Training_GroundTruth/` + `ISIC2018_Training_GroundTruth.csv`.

Two details matter when preparing the data by hand:

* **Mask filenames.** The official masks are `ISIC_xxxxxxx_segmentation.png`
  while the images are `ISIC_xxxxxxx.jpg`, so the stems differ. The loader
  strips the `_segmentation` suffix before matching and therefore pairs the
  two correctly; a mask named `ISIC_xxxxxxx.png` is matched as well.
* **Which ground-truth CSV.** The archive ships two. The Task 1-2 one
  describes the masks and has **no** `diagnosis` column; the Task 3 one
  carries the classification labels. The loader inspects the candidate CSVs
  and picks the one that actually contains a diagnosis column, so the
  segmentation images are never confused with the classification images.

The loaders are tolerant of the common variations: subfolder names are
matched case-insensitively, the metadata CSV may be nested, and image files
may be `.jpg`, `.jpeg` or `.png`. When a layout is ambiguous, run:

```bash
python scripts/inspect_datasets.py --data-dir data/raw
```

It prints what was discovered per dataset and writes
`reports/dataset_compatibility_report.md`.

---

## 3. Label taxonomies and the unified space

### 3.1 Native taxonomies

| Class | Meaning | HAM | ISIC 2018 | ISIC 2019 | PAD-UFES-20 |
|---|---|:-:|:-:|:-:|:-:|
| MEL | Melanoma | ✓ | ✓ | ✓ | ✓ |
| NV | Melanocytic nevus / nevi | ✓ | ✓ | ✓ | ✓ |
| BCC | Basal cell carcinoma | ✓ | ✓ | ✓ | ✓ |
| AKIEC | Actinic keratosis / intraepithelial carcinoma | ✓ | ✓ | ✓ | ✓ |
| BKL | Benign keratosis-like lesions | ✓ | ✓ | ✓ | ✓ |
| DF | Dermatofibroma | ✓ | ✓ | ✓ | — |
| VASC | Vascular lesions | ✓ | ✓ | ✓ | — |
| SCC | Squamous cell carcinoma | — | — | ✓ | ✓ |

### 3.2 Why SCC is never merged into AKIEC

AKIEC is a **pre-invasive** actinic keratosis / intraepithelial carcinoma.
SCC is an **invasive** squamous cell carcinoma. They are different diseases
with different management.

Merging them would:

- destroy the model's ability to distinguish invasive disease,
- corrupt the AKIEC and SCC per-class metrics,
- make the reported melanoma-vs-non-melanoma screening numbers misleading.

The framework therefore maintains **three coexisting label spaces**:

| Space | Classes | When it is used |
|---|---|---|
| `native` | The dataset's own taxonomy | Always preserved in the manifest |
| `unified` | MEL, NV, BCC, AKIEC, BKL, DF, VASC | Cross-dataset experiments; SCC samples get `unified_label = -1` |
| `unified_plus_scc` | The 8 classes above | When ISIC 2019 participates, so none of its data is discarded |

A sample excluded from the unified head is **not deleted**. It stays in the
manifest with `unified_label_valid = False`, and it is still evaluated under
its native head.

### 3.3 How the mapping is derived

The mapping is **not** hard-coded from assumptions. `data/label_mapping.py`
inspects the actual label values found in each metadata file and resolves them
through an alias table:

```
"Basal Cell Carcinoma" -> bcc -> BCC
"bcc"                 -> bcc -> BCC
"NV"                  -> nv  -> NV
```

Native labels that resolve to nothing are recorded in `unmapped_labels` and
reported loudly rather than being force-fitted.

---

## 4. Per-dataset notes

### 4.1 HAM10000

- 10,015 images across 7 classes.
- `metadata.csv` contains `image_id`, `dx` and `lesion_id`. Multiple images may
  share a `lesion_id`; those images are **grouped** so they can never be
  separated across splits.
- HAM10000 does **not** have a reliable patient identifier. Splitting falls
  back to lesion-level grouping plus image-level deduplication. This
  limitation is reported in the leakage audit.
- Some redistributions include `age`, `sex` and `localization`. They are **not**
  used as metadata features by default (see §6).

### 4.2 ISIC 2018

- 10,015 images across 7 classes, identical taxonomy to HAM10000.
- Ships expert-annotated segmentation masks
  (`ISIC2018_Training_GroundTruth/*.png`), used to train the U-Net.
- The segmentation split is derived from the **classification** split manifest
  so that the segmentor and the classifier see the same training images.
- Task-1 images are used for segmentation; Task-2/3 images are used for
  classification only.

### 4.3 ISIC 2019

- 25,331 images across **8** classes — the extra class is SCC.
- Its ground truth covers Task 1 (segmentation), Task 2 (classification) and
  Task 3 (classification). The loader reads the classification ground truth.
- Largest dataset; the primary source for large-scale and cross-dataset
  generalisation experiments.

### 4.4 PAD-UFES-20

- 2,298 images across 6 classes: MEL, NV, BCC, AKIEC, BKL, SCC.
- The **only** dataset in this framework with validated clinical metadata:
  `age`, `sex`, `localization`, `skin_cancer_history`.
- Because it has no DF or VASC, a PAD-UFES-20-only run uses a 6-class head.
- It is the primary dataset for the multimodal experiments.

---

## 5. Metadata handling

### 5.1 Only PAD-UFES-20 contributes metadata

`configs/dataset.yaml` sets `metadata.source_datasets: ["pad_ufes20"]`.

HAM10000, ISIC 2018 and ISIC 2019 redistribute partial clinical fields in
some mirrors, but those fields are **not** comparable with PAD-UFES-20 and
their provenance is uncertain. Including them would make the multimodal
comparison measure the effect of a *mix of sources* rather than the effect of
real clinical metadata. Enable them only if you have verified provenance in
your copy.

### 5.2 Nothing is invented

Samples without metadata are marked `has_metadata = False`. The model does not
receive a fabricated age or sex. Instead the metadata encoder adds a **learned
"missing" embedding**, so the network can learn that "no metadata" is a
distinct situation rather than "age 0, sex unknown".

### 5.3 Preprocessing

| Field type | Strategy | Fitted on |
|---|---|---|
| Numerical (`age`) | Median imputation **+ binary missing indicator**, then z-score | **Training split only** |
| Categorical (`sex`, `localization`, `skin_cancer_history`) | One-hot with a reserved `<MISSING>` token and an `<UNKNOWN>` fallback | **Training split only** |

The fitted encoder is pickled next to the model checkpoint
(`metadata_encoder.pkl`), so inference applies exactly the training-time
transformation.

### 5.4 Anti-leakage guarantees

- The target label is never an input feature. `MetadataEncoder.fit()` raises
  if any record contains a target-like column (`label`, `dx`, `diagnosis`, …).
- All statistics are fitted on the training split only.
- The same fitted encoder is applied to validation and test data.
- Image-only inference is fully supported.

---

## 6. Splitting policy

### 6.1 Proportions and grouping

Default proportions are **70 / 15 / 15** (train / val / test).

Splitting is **group-aware**. Every sample carries a `group_id`:

| `group_type` | Meaning | Source |
|---|---|---|
| `patient` | All images of one patient | A patient id column, when the dataset has one |
| `lesion` | All images of one lesion | A lesion id column |
| `none` | No identifier available | Fallback: one group per image |

All images of a group always land in the **same** split. Splitting by lesion
but not by patient would still leak, because images of the same patient share
clinical risk factors; patient grouping is preferred when available.

### 6.2 Stratification

The splitter assigns groups greedily, scoring each candidate split by how far
it is below its target **in sample count** and **in per-class count**. The
label deficit breaks ties, which keeps the class distribution balanced across
splits rather than merely proportional.

### 6.3 Deduplication

Two complementary mechanisms run before splitting:

| Mechanism | Method | Config |
|---|---|---|
| Exact duplicates | SHA-1 of the raw file bytes | `deduplication.exact_hash` |
| Near-duplicates | Perceptual hash (dHash), Hamming distance threshold | `deduplication.perceptual`, `phash_max_distance` |

Near-duplicate clusters are merged: the first member is kept and the rest are
dropped, so the same lesion cannot appear twice within a split.

Crucially, **duplicates are detected across datasets too**. ISIC 2019 overlaps
with HAM10000 and ISIC 2018 in content. When
`splitting.unify_duplicates_across_datasets` is enabled (the default), a
duplicate group spanning two datasets is forced into a single split, which
closes a serious cross-dataset leakage channel.

### 6.4 Leakage audit

```bash
python scripts/audit_leakage.py --data-dir data/raw
```

The report states, for every dataset:

- exact and near-duplicate counts,
- cross-dataset duplicate pairs,
- the grouping identifier actually available,
- whether patient-level grouping was possible,
- any limitation that could not be resolved.

---

## 7. The harmonised manifest

Every loader emits the same CSV schema (`data/loaders/base.py`):

| Column | Meaning |
|---|---|
| `sample_id` | Globally unique: `"<dataset>:<original_id>"` |
| `dataset` | Source dataset name |
| `image_path` | Absolute path |
| `relative_path` | Path relative to the dataset root |
| `original_id` | Identifier as given by the source dataset |
| `native_label` | Label exactly as found in the source metadata |
| `native_label_norm` | Normalised native label |
| `unified_label` | Canonical code, or empty when excluded |
| `unified_label_valid` | Whether the sample is usable in the unified head |
| `class_index` | Integer index in the training label space |
| `group_id` | Patient/lesion group for leakage-safe splitting |
| `group_type` | `patient` \| `lesion` \| `none` |
| `has_metadata` | Whether clinical metadata is present |
| `metadata` | JSON string with the clinical fields |
| `source` | Provenance string (dataset + file reference) |

Keeping the source dataset and the original identifier for **every** sample
makes any run reproducible and any result auditable.

---

## 8. Reproducing the preparation

```bash
python scripts/prepare_datasets.py --data-dir data/raw
```

This writes, per dataset:

- `data/processed/<dataset>/manifest.csv` — the harmonised manifest,
- `data/processed/<dataset>/summary.json` — counts and notes.

```bash
python scripts/validate_datasets.py --data-dir data/raw
```

Checks that every referenced image exists and is readable, that labels
resolve, that masks match their images, and reports anything broken.

---

## 9. Adding a fifth dataset

1. Subclass `data.loaders.base.BaseDatasetLoader`.
2. Declare `name`, `default_subdir`, `label_columns`, and the
   `patient_id_columns` / `lesion_id_columns` / `metadata_columns` your
   dataset provides.
3. Implement `_discover()` to return a frame with an image path column and a
   label column.
4. Register the loader in `data/loaders/__init__.py` and add it to
   `data/pipeline.py`.
5. Re-run `scripts/inspect_datasets.py` to regenerate the compatibility
   report.

The label mapping, splitting, deduplication and evaluation machinery work
unchanged for the new dataset.
