# Setup Guide

Installation and environment preparation for the skin-cancer hybrid framework.

---

## 1. Requirements

| Component | Version | Notes |
|---|---|---|
| Python | 3.11 | 3.10 works; 3.12+ is untested against the pinned wheels |
| PyTorch | 2.4.1 | CPU or CUDA build |
| torchvision | 0.19.1 | must match the torch version |
| timm | 1.0.9 | provides EfficientNetV2-S |
| OpenCV | 4.10.0.84 | image processing |
| Pillow | 10.4.0 | image I/O |
| Albumentations | 1.4.21 | train-only augmentation |
| NumPy | 1.26.4 | pinned: torch 2.4.1 is not compatible with numpy 2.x |
| pandas | 2.2.3 | metadata handling |
| scikit-learn | 1.5.2 | metrics |
| Matplotlib | 3.9.2 | plots |
| Seaborn | 0.13.2 | plots |
| TensorBoard | 2.18.0 | experiment tracking |
| Optuna | 3.6.2 | hyperparameter search (optional) |
| PyYAML | 6.0.2 | configuration |
| imagehash | 4.3.1 | perceptual deduplication |

> **NumPy pin is load-bearing.** PyTorch 2.4.1 was compiled against the numpy
> 1.x C ABI. Installing numpy 2.x produces binary-import errors deep inside
> torch. If you upgrade numpy, upgrade torch to match.

---

## 2. Local installation (Windows)

```powershell
cd E:\Skin_Cancer
python -m venv .venv
.venv\Scripts\Activate.ps1

# PyTorch first - CPU build
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu

# Then the rest
pip install -r skin-cancer-hybrid\requirements.txt
pip install -r skin-cancer-hybrid\requirements-dev.txt
```

## 3. Local installation (Linux)

```bash
cd ~/Skin_Cancer
python3.11 -m venv .venv
source .venv/bin/activate

pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r skin-cancer-hybrid/requirements.txt
pip install -r skin-cancer-hybrid/requirements-dev.txt
```

### Choosing the right PyTorch build

| Target | Index URL |
|---|---|
| CPU only | `https://download.pytorch.org/whl/cpu` |
| CUDA 11.8 | `https://download.pytorch.org/whl/cu118` |
| CUDA 12.1 | `https://download.pytorch.org/whl/cu121` |
| CUDA 12.4 | `https://download.pytorch.org/whl/cu124` |

The framework does not require a GPU. It auto-detects one and falls back to
CPU; mixed precision is enabled only when the device supports it.

---

## 4. Verify the installation

```bash
python scripts/check_environment.py
```

This script reports, for each dependency, whether it is installed and at which
version, whether a CUDA device is visible, the CUDA/cuDNN versions, whether
mixed precision is usable, and it flags version mismatches against the pins in
`requirements.txt`. It exits non-zero when a required package is missing.

```bash
python scripts/verify_project.py
```

This imports every project module and checks the directory structure, so a
broken import is caught before you start a long run.

```bash
python -m pytest tests/ -q
```

The test suite is CPU-only and uses synthetic data. It never downloads a
dataset.

---

## 5. Datasets

Datasets are **not** downloaded automatically. Place them under
`data/raw/<dataset_name>/` following the layouts in
[DATASET_GUIDE.md](DATASET_GUIDE.md), or point the scripts anywhere with
`--data-dir`.

```bash
python scripts/prepare_datasets.py --data-dir /path/to/datasets
python scripts/inspect_datasets.py --data-dir /path/to/datasets
python scripts/audit_leakage.py     --data-dir /path/to/datasets
```

---

## 6. Google Colab

`requirements-colab.txt` pins the same stack. In a Colab notebook:

```python
!git clone <your-repository-url> skin-cancer-hybrid
%cd skin-cancer-hybrid

# Colab ships CUDA-enabled torch already; install the remaining packages.
!pip install -r requirements-colab.txt

from google.colab import drive
drive.mount('/content/drive')

# Point at your data
DATA_DIR = "/content/drive/MyDrive/datasets"

!python scripts/inspect_datasets.py --data-dir {DATA_DIR}
!python training/train_hybrid.py --data-dir {DATA_DIR} --epochs 50
```

Notes for Colab:

- Enable **Runtime > Change runtime type > GPU** before training.
- The venv is unnecessary; Colab manages packages per-session.
- Mount Drive *before* preparing the datasets so the manifest stores durable
  absolute paths.
- Sessions are ephemeral — copy `runs/` to Drive if you want to keep it.

---

## 7. Kaggle

`requirements-kaggle.txt` mirrors the Colab pins. Add the datasets as Kaggle
Datasets in the notebook settings, then:

```python
!pip install -r requirements-kaggle.txt

DATA_DIR = "/kaggle/input/<your-dataset-folder>"
!python scripts/prepare_datasets.py --data-dir {DATA_DIR}
!python training/train_hybrid.py --data-dir {DATA_DIR} --epochs 50
```

Use the Kaggle API from the CLI if you prefer:

```bash
kaggle datasets download -d <owner>/<dataset> -p data/raw --unzip
```

---

## 8. Another NVIDIA GPU computer

The project is portable. Copy the repository (excluding `.venv/` and
`data/raw/`), then:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python scripts/check_environment.py
```

Move your trained artefacts with the run directory:

```
runs/<experiment>/            # configs, logs, CSV history, TensorBoard
runs/<experiment>/checkpoints/# best.pth, last.pth, metadata_encoder.pkl
```

Checkpoints store the full model configuration, the class names, the
metadata dimension and the preprocessing configuration, so a checkpoint
restores exactly — no re-tuning of the YAML is needed.

---

## 9. Cloud GPU server

Any Linux host with an NVIDIA driver works. The only environment-specific
value is the PyTorch index URL.

```bash
nvidia-smi                      # confirm the driver and GPU
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
nohup python training/train_hybrid.py --data-dir /data/datasets --epochs 50 \
      > logs/train.log 2>&1 &
tail -f logs/train.log
```

For multi-GPU, use `torchrun` with the distributed launcher:

```bash
torchrun --nproc_per_node=4 training/train_hybrid.py --data-dir /data/datasets
```

---

## 10. Verifying hardware capability

```python
import torch

print("torch      :", torch.__version__)
print("cuda built :", torch.version.cuda)
print("cuda avail :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device     :", torch.cuda.get_device_name(0))
    print("capability :", torch.cuda.get_device_capability(0))
    print("amp fp16   :", torch.cuda.is_bf16_supported() or True)
```

`utils/hardware.py` wraps this. `resolve_device()` returns the device, whether
AMP should be used and the recommended dtype.

---

## 11. Configuration

All behaviour is driven by YAML in `configs/`:

| File | Controls |
|---|---|
| `default.yaml` | Paths, dataset selection, hardware, logging, seeds |
| `dataset.yaml` | Preprocessing, augmentation, splitting, deduplication, metadata, label mapping |
| `model.yaml` | Backbone, attention, metadata encoder, fusion, classifier, segmentation, Grad-CAM |
| `training.yaml` | Optimiser, LR, scheduler, loss, early stopping, checkpointing, Optuna space |

Override any value from the command line without editing files:

```bash
python training/train_hybrid.py \
    --data-dir /data/datasets \
    --set training.batch_size=32 \
    --set training.epochs=100 \
    --set model.backbone.drop_rate=0.4
```

---

## 12. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `_ARRAY_API not found` on import | numpy 2.x installed with torch 2.4.1. `pip install numpy==1.26.4` |
| `No module named pip` in the venv | The venv was created without pip. `python -m ensurepip --upgrade` |
| `RuntimeError: No CUDA GPUs are available` | A CUDA build is installed but no GPU is visible. Set `hardware.device: "cpu"` |
| `Could not find a version that satisfies optuna==3.6.3` | That version was never published. The pin is now `3.6.2` |
| `FileNotFoundError` when loading a dataset | The layout does not match `DATASET_GUIDE.md`. Run `scripts/inspect_datasets.py` |
| `RuntimeError: use_segmentation=True requires a trained U-Net checkpoint` | Working as designed. Train the U-Net first or disable segmentation for the ablation |
| `Metadata dimension mismatch` | The saved encoder and the checkpoint disagree. Refit the encoder on the training split |
| Out of memory on GPU | Lower `training.batch_size`, enable `training.amp: true`, or reduce `model.backbone.drop_path_rate` |
| Albumentations update warning | Cosmetic. `export NO_ALBUMENTATIONS_UPDATE=1` |

---

## 13. Reproducibility

`project.seed` (default 42) seeds Python, NumPy and PyTorch and configures
cuDNN for deterministic execution where available. The data split is seeded
independently, so the same seed produces the same train/val/test assignment.
Exact bit-level reproducibility across different GPU architectures is not
guaranteed by PyTorch; within the same environment it holds.
