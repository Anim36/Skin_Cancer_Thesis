# Training Guide

How to train the segmentation model, the classification models, the multimodal
model, the ablation study and the hyperparameter search.

---

## 1. Common interface

Every training script shares the same command-line surface:

```bash
python training/<script>.py [options] [--set key.path=value ...]
```

| Option | Meaning |
|---|---|
| `--config PATH` | Alternative YAML configuration (default `configs/default.yaml`) |
| `--data-dir PATH` | Root directory holding the four datasets |
| `--datasets NAME ...` | Restrict to specific datasets (default: all enabled) |
| `--epochs N` | Override the number of epochs |
| `--batch-size N` | Override the batch size |
| `--num-workers N` | Override the DataLoader worker count |
| `--output-dir PATH` | Where to write the run artefacts |
| `--device {auto,cuda,cpu}` | Force the compute device |
| `--seed N` | Random seed |
| `--resume PATH` | Resume from a checkpoint |
| `--set key.path=value` | Override any configuration value (repeatable) |

Values given via `--set` are parsed as YAML, so booleans, lists and numbers all
work:

```bash
--set training.batch_size=32 \
--set model.backbone.freeze_backbone=true \
--set datasets.enabled.isic2019=false \
--set training.scheduler_config.t_max=80
```

---

## 2. The shared engine

`training/trainer.py` implements what all five scripts share.

### 2.1 Hardware

```python
device, use_amp, dtype = resolve_device(preference=config.device, allow_amp=config.amp)
```

- `auto` uses CUDA when it is available and falls back to CPU otherwise.
- AMP is enabled only on CUDA. On CPU the `GradScaler` is disabled and
  autocast is a no-op.
- `hardware.allow_amp: false` forces full precision everywhere.

### 2.2 Optimiser and parameter groups

AdamW with **two** learning rates:

| Group | Parameters | LR |
|---|---|---|
| `backbone` | EfficientNetV2 trainable parameters | 1e-5 |
| `backbone_frozen` | Frozen backbone parameters | 0.0 (kept for optimiser-state consistency) |
| `head` | Fusion, classifier, metadata encoder | 1e-4 |

The backbone LR is deliberately 10x lower than the head LR: the backbone
carries ImageNet weights that a large learning rate would destroy, while the
head is randomly initialised and needs to move faster.

### 2.3 Mixed precision and gradient clipping

```python
with autocast_context(device, use_amp, dtype):
    loss = criterion(model(inputs, metadata, has_metadata), targets)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
scaler.step(optimizer)
scaler.update()
```

Unscalling before clipping is required: clipping the *scaled* gradients would
clip at the wrong threshold.

### 2.4 Scheduler

| `training.scheduler` | Behaviour |
|---|---|
| `cosine` | `CosineAnnealingLR` over `scheduler_config.t_max` epochs, floor at `t_min` |
| `step` | `StepLR` with `step_size` and `gamma` |
| `reduce_on_plateau` | `ReduceLROnPlateau` monitoring `scheduler_config.monitor` |
| `none` | Constant learning rate |

Optional linear warmup over `training.warmup_epochs`.

### 2.5 Early stopping

```yaml
training:
  early_stopping:
    enabled: true
    patience: 10
    monitor: "val_macro_f1"
    mode: "max"
    min_delta: 0.0005
```

The monitor must be one of the logged validation metrics.

### 2.6 Checkpoints

| File | Contents |
|---|---|
| `checkpoints/best.pth` | The checkpoint with the best monitored validation metric |
| `checkpoints/last.pth` | The most recent checkpoint (for resuming) |
| `checkpoints/metadata_encoder.pkl` | The fitted metadata encoder (multimodal runs) |

A checkpoint stores the model state dict, the **full model configuration**, the
class names, the class index, the metadata dimension, the preprocessing
configuration, the epoch, the best metric and the project configuration. That
is enough to rebuild the exact architecture at load time — the YAML does not
have to match.

### 2.7 Logging

- **TensorBoard**: `runs/<name>/tensorboard/`. Launch with
  `tensorboard --logdir runs`.
- **CSV**: `runs/<name>/history.csv` with one row per epoch.
- **Console**: epoch, train/val loss, accuracy, macro F1, learning rates and
  elapsed time.

---

## 3. `train_unet.py` — lesion segmentation

Trains the U-Net on the ISIC 2018 Task-1 masks.

```bash
python training/train_unet.py --data-dir data/raw --epochs 30 --batch-size 8
```

Configuration (`configs/training.yaml`, section `unet_training`):

```yaml
unet_training:
  epochs: 30
  batch_size: 8
  lr: 1.0e-4
  weight_decay: 1.0e-4
  loss: "dice_bce"          # dice_bce | bce | dice
  dice_weight: 1.0
  bce_weight: 1.0
  scheduler: "cosine"
  early_stopping:
    enabled: true
    patience: 8
    monitor: "val_dice"
    mode: "max"
  image_size: 224
```

The segmentation split is derived from the **classification** split manifest,
so the segmentor and the classifier train on the same images.

Outputs: `runs/unet/checkpoints/best.pth`, `runs/unet/checkpoints/last.pth`, a
Dice/IoU history CSV, and per-epoch validation metrics.

---

## 4. `train_baseline.py` — image-only baseline

EfficientNetV2-S with a linear or MLP head, no attention, no segmentation and
no metadata. This is ablation experiment **A** and the reference point for
every other experiment.

```bash
python training/train_baseline.py --data-dir data/raw --epochs 50
```

To reproduce a frozen-backbone linear probe:

```bash
python training/train_baseline.py --set model.backbone.freeze_backbone=true
```

To compare the three loss functions on an identical split:

```bash
python training/train_baseline.py --set training.loss=ce            --set project.name=run_ce
python training/train_baseline.py --set training.loss=weighted_ce    --set project.name=run_wce
python training/train_baseline.py --set training.loss=weighted_focal --set project.name=run_wfl
```

---

## 5. `train_hybrid.py` — the complete hybrid model

U-Net segmentation + EfficientNetV2 + attention + metadata fusion + weighted
focal loss.

```bash
# Step 1: train the segmentor first
python training/train_unet.py --data-dir data/raw --epochs 30

# Step 2: train the hybrid classifier on top of it
python training/train_hybrid.py \
    --data-dir data/raw \
    --epochs 50 \
    --set hybrid.use_segmentation=true \
    --set hybrid.unet_checkpoint=runs/unet/checkpoints/best.pth
```

When `hybrid.use_segmentation` is true and no checkpoint is supplied, the
model **refuses to run** rather than using a randomly initialised segmentor.
That behaviour is deliberate — an untrained U-Net produces noise, and an
ablation measured against noise is meaningless.

For the hybrid model without segmentation (the standard starting point),
leave `hybrid.use_segmentation: false`.

---

## 6. `train_multimodal.py` — image + clinical metadata

Trains the fusion path in isolation, without segmentation.

```bash
python training/train_multimodal.py --data-dir data/raw --datasets pad_ufes20
```

PAD-UFES-20 is the only dataset with real clinical metadata, so the metadata
branch is only exercised when it is included. The script warns clearly when no
metadata is present and falls back to image-only training.

The fitted encoder is written to `checkpoints/metadata_encoder.pkl` next to the
model checkpoint, and is reloaded automatically at inference.

---

## 7. `ablation_study.py` — the six experiments

```bash
python training/ablation_study.py --data-dir data/raw --epochs 50
python training/ablation_study.py --only A_baseline B_attention
```

| Code | Experiment | Configuration |
|---|---|---|
| A | Baseline | EfficientNetV2-S, no attention, no segmentation, no metadata |
| B | + Attention | Adds CBAM |
| C | + Metadata fusion | Adds the clinical metadata branch |
| D | + Segmentation | Adds the U-Net-segmented input |
| E | + Segmentation + Attention | Both |
| F | Complete hybrid | All components + Weighted Focal Loss |

All experiments run on **one shared, valid data split**, so a difference in
the results is attributable to the component under test and nothing else.

Results are written to `experiments/ablation/<experiment>/`, each with its
resolved configuration, the measured metrics and the plots. **Only measured
numbers are recorded** — no result is ever hard-coded.

---

## 8. `hyperparameter_search.py` — Optuna

```bash
python training/hyperparameter_search.py --data-dir data/raw --n-trials 20
python training/hyperparameter_search.py \
    --set hyperparameter_search.storage="sqlite:///experiments/optuna.db"
```

Search space (`configs/training.yaml`):

| Parameter | Type | Range |
|---|---|---|
| `lr_head` | float (log) | 1e-5 … 5e-4 |
| `lr_backbone` | float (log) | 1e-6 … 1e-4 |
| `weight_decay` | float (log) | 1e-4 … 1e-1 |
| `dropout` | float | 0.0 … 0.6 |
| `batch_size` | categorical | 8, 16, 32 |
| `focal_gamma` | float | 0.0 … 4.0 |
| `head_hidden` | categorical | 128, 256, 512 |
| `attention_reduction` | categorical | 8, 16, 32 |

**The objective is the validation macro F1. The test split is never used for
model selection, early stopping or hyperparameter search.** The best
configuration is written to `experiments/`.

---

## 9. Output layout

```
runs/<experiment>/
├── config.yaml               # the fully resolved configuration
├── history.csv               # one row per epoch
├── tensorboard/              # TensorBoard event files
├── checkpoints/
│   ├── best.pth
│   ├── last.pth
│   └── metadata_encoder.pkl  # multimodal runs only
├── logs/
│   └── train.log
└── metrics/
    └── validation_metrics.json
```

---

## 10. Reproducibility and resuming

```bash
python training/train_hybrid.py --data-dir data/raw --resume runs/hybrid/checkpoints/last.pth
```

Resuming restores the model weights, the optimiser state, the scheduler
state, the epoch counter and the best metric, so the run continues as if it
had never been interrupted.

Set `project.seed` to fix the seed for a run. The seed controls Python, NumPy
and PyTorch, and the data split is seeded independently.

---

## 11. Practical guidance

### Batch size and memory

| GPU | Image-only | + Attention | + Segmentation |
|---|---|---|---|
| 8 GB | 32 | 24 | 12 |
| 12 GB | 48 | 32 | 16 |
| 16 GB | 64 | 48 | 24 |
| 24 GB | 96 | 64 | 32 |

If you run out of memory: lower the batch size, keep `training.amp: true`, or
reduce `model.backbone.drop_path_rate`.

### Fine-tuning strategy

A frozen backbone for the first few epochs, then unfreezing, often converges
faster than fine-tuning everything from the start:

```bash
--set training.freeze_backbone_epochs=3 --set training.unfreeze_at_epoch=3
```

### Class imbalance

The class weights are computed from the **training split only**. If a class is
over-represented in the validation split, that is a splitting artefact to
report, not something to fix by reweighting against validation data.

---

## 12. On this development machine

The current machine is for **code development only**. Full training must not
be run here. The following are safe and fast:

```bash
python scripts/check_environment.py
python scripts/verify_project.py
python -m pytest tests/ -q
python scripts/prepare_datasets.py --data-dir data/raw    # manifest only
python training/train_hybrid.py --epochs 1 --batch-size 2  # smoke test only
```

Move the repository to a GPU machine for the real runs. See
[SETUP.md](SETUP.md) for Colab, Kaggle and cloud instructions.
