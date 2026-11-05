# Evaluation Guide

Metrics, reports, plots and the rules the framework follows when reporting
results.

---

## 1. Running an evaluation

```bash
python evaluation/evaluate.py \
    --checkpoint runs/hybrid/checkpoints/best.pth \
    --split test \
    --output-dir reports/evaluation
```

| Option | Meaning |
|---|---|
| `--checkpoint PATH` | The trained checkpoint (required) |
| `--split {train,val,test}` | Which split to evaluate (default `test`) |
| `--datasets NAME ...` | Restrict to specific datasets |
| `--include-scc` | Use the 8-class space instead of the 7-class space |
| `--batch-size N` | Inference batch size |
| `--no-plots` | Skip figure generation |
| `--device {auto,cuda,cpu,mps}` | Force the compute device |

The checkpoint supplies its own model configuration, class names and metadata
dimension, so the evaluation uses exactly the architecture that was trained.

---

## 2. Metrics

### 2.1 Multiclass

| Metric | Definition | Notes |
|---|---|---|
| Accuracy | Correct / total | Misleading under imbalance; always reported with macro F1 |
| Balanced accuracy | Mean of per-class recall | The primary headline metric under imbalance |
| Precision (macro) | Mean of per-class precision | |
| Recall (macro) | Mean of per-class recall | |
| Specificity (macro) | Mean of per-class specificity | |
| F1 (macro) | Harmonic mean of macro precision and recall | **Primary model-selection metric** |
| F1 (weighted) | F1 weighted by class support | |
| ROC-AUC (OvR) | One-vs-rest ROC-AUC per class | `None` when a class has no positives in the split |
| Balanced accuracy | | |
| Confusion matrix | `(N, N)` integer counts | |

### 2.2 Melanoma-specific (clinical focus)

The multiclass prediction is collapsed into the binary decision *"is this
melanoma?"*, which is the operating characteristic the dermatology literature
reports:

| Metric | Meaning |
|---|---|
| Sensitivity (recall) | Fraction of melanomas correctly flagged |
| Specificity | Fraction of non-melanomas correctly dismissed |
| Precision (PPV) | Fraction of flagged lesions that are truly melanoma |
| NPV | Fraction of dismissed lesions that are truly benign |
| F1 | Harmonic mean of precision and recall |
| Prevalence | Melanoma share of the evaluated split |

These are **screening-oriented** operating characteristics. They describe the
model's behaviour on a curated image set, not its performance in a clinical
population, where prevalence differs enormously.

### 2.3 Calibration

| Metric | Meaning |
|---|---|
| ECE | Expected Calibration Error — mean gap between confidence and accuracy across bins |
| Brier score | Mean squared error of the predicted probabilities |
| Reliability curve | Confidence-vs-accuracy diagram |

A well-calibrated model is one whose reported confidence matches its observed
accuracy. A model can have high accuracy and still be badly calibrated, which
makes the probabilities useless for triage.

### 2.4 Segmentation

| Metric | Definition |
|---|---|
| Dice | :math:`2|P \cap G| / (|P| + |G|)` |
| IoU | :math:`|P \cap G| / |P \cup G|` |
| Pixel accuracy | Fraction of pixels classified correctly |
| Pixel sensitivity | Recall over lesion pixels |
| Pixel specificity | Recall over background pixels |

---

## 3. Report structure

### 3.1 Dataset-wise evaluation

Each dataset is evaluated **separately**. A pooled accuracy that hides
dataset-specific performance is never reported on its own.

```
reports/evaluation/
├── evaluation_test.json          # metrics + per-dataset breakdown
├── evaluation_report_test.md     # human-readable report
├── confusion_matrix_test.png
├── per_class_performance_test.png
├── curves_test_roc.png
└── curves_test_pr.png
```

The JSON contains a `per_dataset` block with the full metric suite computed
independently for each dataset, plus a compact `per_dataset_summary`.

The Markdown report contains:

1. The overall summary.
2. A per-dataset table (accuracy, balanced accuracy, macro F1, melanoma
   sensitivity, sample count).
3. The full per-class classification report.
4. The confusion matrix.
5. The melanoma-specific block.
6. The external validation section (when available).
7. The disclaimer.

### 3.2 External validation

An **external validation** section appears when a genuinely independent split
is available — that is, when the evaluation is run on a dataset that was not
part of the training pool.

```bash
# Train on HAM10000 + ISIC 2018 only
python training/train_hybrid.py --datasets ham10000 isic2018 ...

# Evaluate on ISIC 2019, which the model has never seen
python evaluation/evaluate.py --checkpoint runs/hybrid/checkpoints/best.pth \
    --datasets isic2019 --split test
```

A dataset that participated in training is **not** external validation, and the
report labels it accordingly rather than presenting it as generalisation.

---

## 4. Plots

| Plot | Function | Contents |
|---|---|---|
| Confusion matrix | `save_confusion_matrix_plot` | Row-normalised and raw counts |
| Per-class performance | `save_per_class_plot` | Precision, recall, F1 per class |
| ROC curves | `save_roc_plot` | One-vs-rest ROC per class with the macro average |
| Precision-recall curves | `save_pr_plot` | PR per class |
| Calibration | `save_calibration_plot` | Reliability diagram |
| Training curves | `save_training_curves` | Loss and accuracy vs epoch |

Training curves are produced from the run's `history.csv`:

```bash
python evaluation/evaluate.py --checkpoint ... --output-dir reports/evaluation
```

The curves for the training and validation loss and the accuracy curves are
written from the recorded history, so they reflect the actual run.

---

## 5. Ablation comparison

The ablation study writes one directory per experiment. To compare them:

```bash
python training/ablation_study.py --data-dir data/raw --epochs 50
```

Each experiment directory contains its resolved configuration and the measured
metrics. A comparison table is written to
`experiments/ablation/summary.md`.

**All experiments use the same data split.** This is what makes the comparison
scientifically meaningful: the only thing that changes between experiments is
the component under test.

---

## 6. Rules the framework follows

1. **No fabricated results.** Every number in a report was measured by the
   code. There are no placeholder accuracies anywhere in this repository.
2. **The test split is never used for selection.** Early stopping,
   hyperparameter search and checkpoint selection all use validation metrics.
3. **No single pooled accuracy.** Dataset-specific results are always
   reported alongside the pooled ones.
4. **Undefined metrics are `None`, not 0.0.** A ROC-AUC for a class with no
   positive examples is undefined; reporting 0.0 would be misleading.
5. **No clinical-readiness claims.** Every report carries the disclaimer that
   the work is a research prototype and not a medical device.
6. **SCC is never silently merged.** Its exclusion from the 7-class head is
   documented in the report rather than hidden.

---

## 7. Interpreting the numbers

### Accuracy vs macro F1

With a class distribution like HAM10000's (NV is roughly 67% of the samples),
a model that never predicts anything but NV and BKL already scores high
accuracy. **Macro F1 and balanced accuracy** are the honest metrics here.
Always quote them together with the per-class report.

### The 95% target

95%+ accuracy is a **research target, not a guarantee**. It is reachable on
some of these datasets with the right configuration, but it is not something
this framework asserts. Run the experiments, then report what you measured.

### Generalisation gaps

A model that scores well on the combined test set but poorly on one dataset
has learned dataset-specific features (imaging device, preprocessing,
population) rather than lesion morphology. The per-dataset breakdown exists to
surface exactly this, and it is often the most interesting table in the
report.

---

## 8. Reproducing a report

```bash
# 1. Prepare the data (deterministic, seeded)
python scripts/prepare_datasets.py --data-dir data/raw

# 2. Train
python training/train_hybrid.py --data-dir data/raw --epochs 50 --seed 42

# 3. Evaluate
python evaluation/evaluate.py \
    --checkpoint runs/hybrid/checkpoints/best.pth \
    --split test --output-dir reports/evaluation
```

The split is seeded, the training is seeded and the checkpoint stores its own
configuration, so the report is reproducible from the recorded commands.
