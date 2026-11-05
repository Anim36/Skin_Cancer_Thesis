# Explainable Hybrid Deep Learning Framework for Early Skin Cancer Detection

A research-grade, modular PyTorch framework for multiclass dermoscopic skin
lesion classification that combines lesion segmentation, deep visual features,
attention, clinical metadata fusion, imbalance-aware loss and post-hoc
explanability.

> **Research prototype — not a medical device.** Every prediction produced by
> this software is an experimental model output. It is **not** a diagnosis and
> must never be used as a substitute for professional medical assessment. See
> [Limitations](#limitations) for the details.

---

## Table of contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Datasets and label spaces](#datasets-and-label-spaces)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Project layout](#project-layout)
- [Training](#training)
- [Evaluation](#evaluation)
- [Explainability](#explainability)
- [Inference](#inference)
- [Ablation study](#ablation-study)
- [Hyperparameter optimisation](#hyperparameter-optimisation)
- [Testing](#testing)
- [Portability](#portability)
- [Limitations](#limitations)
- [Documentation index](#documentation-index)

---

## Overview

The framework answers a single question — *what is this skin lesion, and what
made the model think so?* — through five cooperating stages:

| Stage | Module | Purpose |
|---|---|---|
| 1. Preprocessing | `data/preprocessing.py` | Hair removal, CLAHE, normalisation, train-only augmentation |
| 2. Segmentation | `models/unet.py` | Isolate the lesion so the classifier is not distracted by skin |
| 3. Backbone | `models/efficientnetv2.py` | EfficientNetV2-S ImageNet features |
| 4. Attention | `models/attention.py` | CBAM re-weighting of informative channels and locations |
| 5. Metadata fusion | `models/metadata_encoder.py`, `models/fusion.py` | Inject real clinical variables (PAD-UFES-20) |
| 6. Classification | `models/classifier.py` | Multiclass softmax over the target taxonomy |
| 7. Explanation | `explainability/gradcam.py` | Class-specific Grad-CAM heatmaps |

Design commitments that shape the whole codebase:

- **No fabricated results.** The framework computes and stores metrics; it
  never asserts an accuracy it has not measured.
- **No silent data loss.** Every sample of every dataset is accounted for in
  the harmonised manifest, including samples excluded from a particular head.
- **No medical mis-mapping.** SCC is never folded into AKIEC — see below.
- **No leakage.** Splitting is group-aware, deduplication runs across
  datasets, and the test split is never used for tuning or model selection.
- **No impossible claims.** A randomly initialised U-Net is refused as a
  segmentor rather than silently degrading the pipeline.

---

## Architecture

```
                     dermoscopic image  (224x224x3)
                                |
                                v
                    +---------------------------+
                    |  Image preprocessing      |
                    |  hair removal / CLAHE /   |
                    |  normalisation             |
                    +---------------------------+
                                |
                                v
                    +---------------------------+
                    |  U-Net lesion segmentation|  <-- optional, needs a TRAINED checkpoint
                    |  (frozen, separately      |
                    |   trained on ISIC 2018)   |
                    +---------------------------+
                                |
                    segmented image
                                |
                                v
                    +---------------------------+
                    |  EfficientNetV2-S         |
                    |  (ImageNet pretrained)    |
                    +---------------------------+
                                |
                    image feature map (spatial)
                                |
                                v
                    +---------------------------+
                    |  CBAM attention           |
                    |  channel + spatial        |
                    +---------------------------+
                                |
                    image feature vector
                                |
             +------------------+------------------+
             |                                     |
             v                                     v
    +-------------------+              +-----------------------+
    | Metadata encoder  |              |  (image-only path:    |
    | (MLP, PAD-UFES-20)|              |   learned fallback)   |
    +-------------------+              +-----------------------+
             |                                     |
             v                                     |
    metadata features -----------------------------+
             |                                     |
             +------------------+------------------+
                                |
                                v
                    +---------------------------+
                    |  Fusion layer             |
                    |  concat / gated / attention|
                    +---------------------------+
                                |
                                v
                    +---------------------------+
                    |  FC classification head   |
                    +---------------------------+
                                |
                                v
                    softmax class probabilities
                                |
                                v
                    +---------------------------+
                    |  Grad-CAM explanation     |
                    +---------------------------+
```

The mathematical details of every block are documented in
[MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).

---

## Datasets and label spaces

All four datasets are first-class citizens of the pipeline. They do **not**
share a taxonomy, and the framework never pretends they do:

| Dataset | Images | Native classes | Clinical metadata | Role |
|---|---|---|---|---|
| HAM10000 | 10,015 | 7 | no | Classification baseline |
| ISIC 2018 | 10,015 | 7 | no | Segmentation training + classification |
| ISIC 2019 | 25,331 | 8 | no | Large-scale classification, generalisation |
| PAD-UFES-20 | 2,298 | 6 | **yes** | Multimodal image + metadata fusion |

### The SCC decision

ISIC 2019 adds **SCC** (squamous cell carcinoma) and PAD-UFES-20 contains it
too. SCC is *invasive* squamous cell carcinoma; **AKIEC** is *pre-invasive*
actinic keratosis / intraepithelial carcinoma. Merging them would be
medically wrong and would corrupt every per-class metric.

The framework therefore maintains three coexisting label spaces:

1. **`native`** — the dataset's own taxonomy. Always preserved.
2. **`unified`** — the classic 7-class space. SCC samples are **excluded**
   (`unified_label = -1`), never remapped.
3. **`unified_plus_scc`** — 8 classes, used when ISIC 2019 participates so
   that none of its data is discarded.

Every sample therefore always has a home. The compatibility report is
regenerated from the real metadata files by `scripts/inspect_datasets.py`.

See [DATASET_GUIDE.md](DATASET_GUIDE.md) for the expected directory layouts
and the full compatibility matrix.

---

## Installation

Python 3.11 is the supported interpreter. Full instructions, including
Colab and Kaggle variants, are in [SETUP.md](SETUP.md).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# Install PyTorch FIRST (pick the index URL for your hardware)
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
# ... or for CUDA 12.1:
# pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
pip install -r requirements-dev.txt   # pytest, coverage

python scripts/check_environment.py
```

The project runs on **CPU-only** machines. CUDA is auto-detected and used when
available; automatic mixed precision is enabled only when the hardware
supports it.

---

## Quick start

```bash
# 1. Point the project at your data
python scripts/prepare_datasets.py --data-dir /path/to/datasets

# 2. Inspect what was found and generate the compatibility report
python scripts/inspect_datasets.py --data-dir /path/to/datasets

# 3. Audit for duplicate images and split leakage
python scripts/audit_leakage.py --data-dir /path/to/datasets

# 4. Train the U-Net segmentor (ISIC 2018 masks)
python training/train_unet.py --data-dir /path/to/datasets --epochs 30

# 5. Train the full hybrid model
python training/train_hybrid.py --data-dir /path/to/datasets --epochs 50

# 6. Evaluate
python evaluation/evaluate.py --checkpoint runs/hybrid/checkpoints/best.pth

# 7. Predict with an explanation
python inference/predict.py --checkpoint runs/hybrid/checkpoints/best.pth \
    --image lesion.jpg --gradcam --output-dir runs/predictions
```

Every script accepts `--set key.path=value` to override any YAML
configuration value without editing the files.

---

## Project layout

```
skin-cancer-hybrid/
├── configs/          default, dataset, model, training, evaluation YAML
├── data/
│   ├── loaders/      one independent loader per dataset
│   ├── preprocessing.py, augmentation.py
│   ├── metadata.py, label_mapping.py
│   ├── dataset_harmonization.py, deduplication.py, splitting.py
│   ├── segmentation_dataset.py, pipeline.py
├── models/           unet, efficientnetv2, attention, metadata_encoder,
│                     fusion, classifier, hybrid_model, losses
├── training/         trainer, train_unet, train_baseline, train_hybrid,
│                     train_multimodal, ablation_study, hyperparameter_search
├── evaluation/       evaluate, classification_metrics, segmentation_metrics,
│                     confusion_matrix, roc_analysis, dataset_reports
├── explainability/   gradcam, visualization, segmentation_viz
├── inference/        predict, predict_batch
├── scripts/          environment, dataset, leakage and verification tooling
├── tests/            pytest suite
├── checkpoints/      trained weights
├── logs/             TensorBoard and CSV training logs
├── reports/          generated compatibility and leakage reports
└── experiments/      ablation and Optuna artefacts
```

---

## Training

All training scripts share one engine (`training/trainer.py`) and one
configuration surface (`configs/training.yaml`). Starting hyper-parameters:

| Hyper-parameter | Value |
|---|---|
| Image size | 224 x 224 |
| Backbone | EfficientNetV2-S (ImageNet pretrained) |
| Optimizer | AdamW, weight decay 0.01 |
| Backbone LR | 1e-5 |
| Head LR | 1e-4 |
| Batch size | 16 (configurable) |
| Epochs | 50 (configurable) |
| Dropout | 0.3 (configurable) |
| Scheduler | Cosine annealing |
| Early stopping | patience 10 on validation macro F1 |
| Loss | Weighted Focal Loss, gamma 2.0 |

These are **starting points**, not tuned optima. The engine adds CUDA
auto-detection with CPU fallback, automatic mixed precision with gradient
scaling, gradient clipping, best/last checkpointing, resume support,
TensorBoard and CSV logging, and reproducible seeding.

Details and the full command reference: [TRAINING_GUIDE.md](TRAINING_GUIDE.md).

---

## Evaluation

`evaluation/evaluate.py` produces, for each dataset **and** for the combined
pool:

- Accuracy, precision, recall, specificity, F1, macro F1, weighted F1
- ROC-AUC and precision-recall curves (plus calibration / ECE)
- Confusion matrices and per-class classification reports
- Melanoma-specific sensitivity, precision, specificity and F1
- Training / validation loss and accuracy curves
- External validation report where a genuinely independent split exists

A single pooled accuracy is **never** reported in place of the
dataset-wise breakdown — the report always states each dataset separately.
See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md).

---

## Explainability

Grad-CAM is implemented against the real trained model, hooking the last
spatial convolution of the EfficientNetV2 backbone (or the attention module
when attention is enabled):

```python
from inference.predict import SkinCancerPredictor

predictor = SkinCancerPredictor.from_checkpoint("runs/hybrid/checkpoints/best.pth")
result = predictor.predict("lesion.jpg", metadata={"age": 54}, top_k=3)

print(result["predicted_class"], result["confidence"])
print(result["top_k_predictions"])

# Class-specific explanation for a chosen class (not just the predicted one)
explanation = predictor.explain("lesion.jpg", target_class=0)
```

Limitations of Grad-CAM in medical imaging are documented in
[MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md#grad-cam-limitations).

---

## Inference

```python
from inference.predict import SkinCancerPredictor

predictor = SkinCancerPredictor.from_checkpoint(
    "runs/hybrid/checkpoints/best.pth",
    device="cuda" if available else "cpu",
)

result = predictor.predict(
    "lesion.jpg",
    metadata={"age": 54, "sex": "Male", "localization": "face",
              "skin_cancer_history": "no"},
    top_k=3,
    gradcam=True,
)
```

The predictor applies **exactly** the training preprocessing (the pipeline
configuration is stored inside the checkpoint), runs segmentation when the
model was trained with it, and accepts metadata as optional. When metadata is
absent the model uses a learned "no metadata" embedding rather than guessing
values. Batch inference is available in `inference/predict_batch.py`.

---

## Ablation study

`training/ablation_study.py` runs the six required configurations on **one
shared, valid data split**, so the differences between them are attributable
to the component under test and nothing else:

| Code | Configuration |
|---|---|
| A | EfficientNetV2-S baseline |
| B | + CBAM attention |
| C | + clinical metadata fusion |
| D | + U-Net segmentation |
| E | + U-Net + attention |
| F | Complete hybrid (all components + Weighted Focal Loss) |

Each experiment writes its configuration, metrics and plots under
`experiments/ablation/`. The script records **measured** numbers only.

---

## Hyperparameter optimisation

Optional Optuna search over learning rates, weight decay, dropout, batch
size, focal gamma, classifier head dimensions and attention configuration.
The objective is the **validation** macro F1 — the test split is never
touched. The best configuration is written to `experiments/`.

```bash
python training/hyperparameter_search.py --data-dir /path/to/datasets --n-trials 20
```

---

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
python scripts/verify_project.py     # import + structure check
```

The suite covers dataset loaders, label mapping, preprocessing, image-mask
pairing, metadata encoding, missing-metadata handling, U-Net and
EfficientNetV2 output shapes, attention dimensions, the hybrid forward pass,
loss calculation and checkpoint save/load. Everything runs on CPU with
synthetic data; no dataset download is required.

---

## Portability

The project is written to move between environments without modification:

- **Windows and Linux** — no OS-specific paths, `pathlib` throughout.
- **CPU or CUDA** — hardware is auto-detected; CPU is a first-class path.
- **Colab / Kaggle / cloud GPU** — dedicated requirements files and a
  notebook-friendly API. See [SETUP.md](SETUP.md#colab).
- **Reproducibility** — seeds are set for Python, NumPy and PyTorch, and the
  split is seeded, so a run can be reproduced exactly.

---

## Limitations

Stated plainly, because a thesis needs them:

- **Grad-CAM is a coarse, post-hoc attribution.** It highlights regions that
  moved the score, not a causal explanation, and its resolution is limited by
  the target layer. It can highlight ruler markings, bubbles or the image
  border. It is a debugging and presentation aid, not evidence of
  "reasoning".
- **The 95%+ accuracy target is a research goal, not a result.** No number in
  this repository is pre-computed. Every metric must be measured on a real
  GPU run before it is quoted.
- **Patient identifiers are limited.** HAM10000 and ISIC 2018 expose lesion
  ids but not reliable patient ids; ISIC 2019 and PAD-UFES-20 expose neither
  in a uniform way. Where patient grouping is impossible the leakage audit
  reports this explicitly and falls back to the strongest feasible
  deduplication and lesion-level grouping.
- **Only PAD-UFES-20 carries validated clinical metadata.** The other three
  datasets ship none in this framework; their metadata is never invented.
- **This is not a clinical device.** No claim of diagnostic readiness,
  regulatory approval or clinical validity is made or implied.

---

## Documentation index

| Document | Contents |
|---|---|
| [SETUP.md](SETUP.md) | Installation, environment verification, Colab/Kaggle/cloud notes |
| [DATASET_GUIDE.md](DATASET_GUIDE.md) | Expected layouts, label mapping, leakage policy |
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | All training scripts, configuration reference, resume |
| [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) | Metrics, reports, plots, external validation |
| [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) | Block-by-block mathematics, losses, Grad-CAM |
| [GPU_READINESS_REPORT.md](GPU_READINESS_REPORT.md) | Readiness audit, defects found and fixed, exact GPU commands |
| [LICENSES_AND_ATTRIBUTION.md](LICENSES_AND_ATTRIBUTION.md) | Datasets, code and dependency licences |

---

## Licence

See [LICENSES_AND_ATTRIBUTION.md](LICENSES_AND_ATTRIBUTION.md). The **datasets
are not redistributed** by this repository and carry their own terms; you are
responsible for complying with each dataset's licence and any applicable
data-use agreement.
