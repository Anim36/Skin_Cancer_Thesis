# Model Architecture

Block-by-block description of the framework, with the mathematics behind each
component.

---

## 1. Overview

The model is a **hybrid** pipeline: a convolutional image branch, a
tabular-metadata branch, a fusion layer and a classification head, with an
optional U-Net segmentor in front and Grad-CAM behind.

```
image ─► preprocessing ─► [U-Net segmentation] ─► EfficientNetV2-S ─► CBAM ─┐
                                                                          ├─► fusion ─► FC head ─► softmax
clinical metadata ─► metadata encoder (MLP) ───────────────────────────────┘
```

Every block is an independent, reusable module:

| Module | File | Class |
|---|---|---|
| Segmentation | `models/unet.py` | `UNet` |
| Backbone | `models/efficientnetv2.py` | `EfficientNetV2Backbone` |
| Attention | `models/attention.py` | `ChannelAttention`, `SpatialAttention`, `CBAM` |
| Metadata | `models/metadata_encoder.py` | `MetadataEncoderNetwork` |
| Fusion | `models/fusion.py` | `FusionLayer` |
| Classifier | `models/classifier.py` | `ClassificationHead` |
| Assembly | `models/hybrid_model.py` | `HybridSkinCancerModel` |
| Losses | `models/losses.py` | `WeightedFocalLoss`, `WeightedCrossEntropyLoss` |

---

## 2. Image preprocessing

Defined in `data/preprocessing.py`, configured in `configs/dataset.yaml`.

### 2.1 Order of operations

1. **Load** the image (OpenCV / Pillow) and **validate** it (readable, non-empty,
   sufficient resolution).
2. **Convert to RGB.** Dermoscopic images are often encoded with a 4-channel
   PNG or in BGR; both are converted explicitly.
3. **Resize** to 224 x 224 with bilinear interpolation (area interpolation
   when downscaling a large image).
4. **Remove hair** (optional, on by default).
5. **CLAHE** contrast enhancement on the L channel of LAB.
6. **Optional denoising** (`fastNlMeansDenoising`, off by default).
7. **Normalise** with the ImageNet mean/std (the statistics EfficientNetV2-S
   was pretrained with).

Originals are never modified: preprocessing produces an in-memory array, and
`preprocessing.preserve_originals` documents that the source files are
read-only inputs.

### 2.2 Hair removal (DullRazor-style)

Hair is a common artefact in dermoscopy and can be mistaken for a pigmented
structure.

1. Grayscale the image.
2. Apply a black-hat morphological filter with an elongated kernel:
   :math:`\mathrm{BH}(I) = \max(I - I \ominus K, 0)`, where :math:`K` is a
   kernel oriented along the hair direction.
3. Threshold the black-hat response to obtain the hair mask.
4. Inpaint the masked region with OpenCV's `INPAINT_TELEA`.

An alternative morphological (closing-based) method is available via
`preprocessing.hair_removal_method`.

### 2.3 CLAHE

Contrast Limited Adaptive Histogram Equalisation is applied to the **L**
channel of the LAB image with a clip limit of 2.0 and an 8 x 8 grid. This
equalises illumination without shifting hue, which matters because pigment
colour is diagnostic.

### 2.4 Augmentation (training only)

Albumentations pipeline, applied **only** to the training split:

| Transform | Default | Rationale |
|---|---|---|
| Horizontal flip | p=0.5 | Dermoscopic orientation is not canonical |
| Vertical flip | p=0.0 | Disabled by default: it is unusual in practice |
| Rotation | ±30° | Moderate; larger angles can distort the lesion |
| Shift / scale | ±5%, ±15% | Mild geometric jitter |
| Random crop | p=1.0, 0.9 area | Mild zoom, retains the lesion border |
| Brightness / contrast | ±20% | Simulates illumination variation |
| Hue / saturation | ±0.02 / ±0.15 | Kept **mild**; colour is diagnostic |
| Elastic transform | off | Distorts lesion morphology unnaturally |

Validation and test data receive **no** random augmentation — only the
deterministic preprocessing above. The same conventions are applied across
all four datasets.

---

## 3. U-Net lesion segmentation

`models/unet.py`.

### 3.1 Structure

- **Encoder** — `depth` stages of `DoubleConv` (conv 3x3 → norm → ReLU →
  conv 3x3 → norm → ReLU). Each stage halves H/W and doubles the channels
  (32 → 64 → 128 → 256 → 512 at `depth=4`).
- **Bottleneck** — the deepest double-conv block, at 1/16 resolution.
- **Decoder** — bilinear upsample + 1x1 projection of the skip, concatenate,
  double-conv. Transposed convolution is available instead.
- **Skip connections** — encoder features are optionally passed through an
  **attention gate**.
- **Output** — 1x1 convolution to a single channel, producing **logits**.

### 3.2 Attention gate

On the skip connections, the coarser decoder signal gates the finer encoder
feature:

$$\alpha = \sigma\left(W_f \cdot \psi(W_g \cdot g) + W_x \cdot x\right)$$

where :math:`g` is the decoder feature, :math:`x` the encoder feature,
:math:`\psi` a 1x1 convolution followed by interpolation to the resolution of
:math:`x`, and :math:`\sigma` a sigmoid. The skip is scaled element-wise by
:math:`\alpha` before concatenation, suppressing background activations.

### 3.3 Normalisation

**GroupNorm** by default, not BatchNorm. Segmentation batches are small and
BatchNorm statistics computed over a batch of single lesions are unstable;
GroupNorm is batch-size independent.

### 3.4 Losses

In `training/train_unet.py`:

- **BCE** with logits (`BCEWithLogitsLoss`) — numerically stable.
- **Dice loss** — :math:`1 - \frac{2|P \cap G|}{|P| + |G|}`, added with a
  small smoothing term so that two empty masks score 1 rather than dividing
  by zero.
- **Combined** `dice_bce` (default): `dice_weight * dice + bce_weight * bce`.

### 3.5 Metrics

`evaluation/segmentation_metrics.py` provides Dice, IoU, pixel accuracy, pixel
sensitivity and pixel specificity.

### 3.6 Deep supervision

When enabled, auxiliary heads are placed on the decoder stages and their
losses are summed with the main loss. This improves gradient flow to the
encoder in early training.

### 3.7 Integration with the classifier

The hybrid model loads a **trained** U-Net checkpoint and freezes it
(`freeze_segmentation=True`). A randomly initialised segmentor is **refused**
— the model raises a `RuntimeError` rather than silently degrading the
pipeline. This matters scientifically: an untrained U-Net produces noise, and
reporting an ablation with a noise segmentor would be meaningless.

The predicted mask is applied by **dimming the background to 20%** rather
than zeroing it:

$$I_{masked} = I \odot M + 0.2 \cdot (1 - M)$$

A hard zero border creates an artificial edge that the backbone latches onto.

`hybrid.use_segmentation: false` bypasses segmentation entirely, which is the
controlled ablation baseline.

---

## 4. EfficientNetV2-S feature extraction

`models/efficientnetv2.py`, backed by `timm`.

### 4.1 Model resolution

`timm`'s original PyTorch port `efficientnetv2_s` ships **without** ImageNet
weights; the TensorFlow port `tf_efficientnetv2_s` has them. The loader
resolves the configured name to a pretrained-capable variant of the same
family automatically, and `configs/model.yaml` names
`tf_efficientnetv2_s` explicitly to avoid that resolution step.

### 4.2 Head removal

The model is instantiated with `num_classes=0`, which removes the ImageNet
classifier and turns the network into a feature extractor. The pooled
feature dimension is read from `model.num_features` (1280 for
EfficientNetV2-S).

### 4.3 Fine-tuning modes

| Mode | Behaviour |
|---|---|
| `frozen` | All backbone parameters frozen; the backbone stays in `eval()` mode so BatchNorm statistics are not updated |
| `partial` | The last *N* stages are unfrozen (`unfreeze_last_n_blocks`), plus the head normalisation and pooling |
| `full` | Everything trainable |

The frozen-parameter group is still handed to the optimiser with a zero
learning rate, so the optimiser state stays consistent if the backbone is
unfrozen mid-training.

### 4.4 Stochastic depth and dropout

`drop_rate` (0.3) is timm's classifier dropout; `drop_path_rate` (0.1) is
stochastic depth, which regularises the deeper stages.

---

## 5. Attention (CBAM)

`models/attention.py`. Implements Convolutional Block Attention Module (Woo
et al., 2018) as two sequential sub-modules.

### 5.1 Channel attention

$$\hat{M}_c(x) = \sigma\left(f_{MLP}\left(\mathrm{AvgPool}(x)\right) +
f_{MLP}\left(\mathrm{MaxPool}(x)\right)\right)$$

where the spatial extent is squeezed out by average and max pooling, the
same 1x1 MLP is applied to both descriptors, and the results are summed and
squashed by a sigmoid. The channel descriptor is

$$M_c(x) = F_{MLP}(\mathrm{AvgPool}(x)) + F_{MLP}(\mathrm{MaxPool}(x))$$

with $F_{MLP}$ implemented as two 1x1 convolutions with a ReLU in between and
a bottleneck ratio $r = 16$.

### 5.2 Spatial attention

$$\hat{M}_s(x) = \sigma\left(f^{7\times7}\left(
\left[\mathrm{AvgPool}_c(x);\, \mathrm{MaxPool}_c(x)\right]\right)\right)$$

where the channel dimension is squeezed by averaging and max pooling to
produce two single-channel maps, which are concatenated and passed through a
7x7 convolution and a sigmoid.

### 5.3 Sequential application

CBAM applies channel attention first, then spatial attention on the result:

$$x'' = M_s(M_c(x) \odot x) \odot M_c(x)$$

### 5.4 Placement

Attention is applied to the **spatial feature map** of the backbone, before
global pooling, because the spatial branch requires a 4-D tensor. The
attended map is then pooled to produce the image feature vector.

---

## 6. Clinical metadata encoder

`models/metadata_encoder.py` + `data/metadata.py`.

### 6.1 Input encoding

See [DATASET_GUIDE.md §5](DATASET_GUIDE.md#5-metadata-handling) for the field
strategies. The encoder receives a dense float vector:

- numerical field → imputed value + missing indicator,
- categorical field → one-hot over the fitted vocabulary.

### 6.2 Network

$$\mathbf{e} = W_2 \,\mathrm{GELU}\left(\mathrm{BN}(W_1 \mathbf{m} + \mathbf{b}_1)\right)$$

with an optional dropout between layers, hidden sizes defaulting to
(128, 64) and an output dimension of 64.

### 6.3 Missing-metadata handling

A learned parameter $\mathbf{e}_{miss}$ is added to the embedding when the
sample has no metadata:

$$\mathbf{e}' = \mathbf{e} + (1 - \mathbf{m}_{has}) \odot \mathbf{e}_{miss}$$

This is preferable to zeroing the metadata, because "no metadata" is a
meaningful state that the network should be able to distinguish from
"metadata with all-zero values".

### 6.4 Image-only inference

When metadata is absent, the image branch is projected into the fusion space
with a learned fallback so the model degrades gracefully instead of
crashing. The fusion layer supports four strategies:

| Strategy | Behaviour |
|---|---|
| `concat` | `[image ; metadata]` |
| `gated` | A learned gate blends the two branches |
| `attention` | Learned attention weights over the concatenated vector |
| `concat_gated` | Concatenation followed by a gate |

---

## 7. Classification head

`models/classifier.py`.

A stack of fully connected layers with optional BatchNorm, GELU and dropout,
ending in a linear layer to `num_classes` logits. Softmax is applied outside
the module (in the trainer and at inference) so that the loss can use the
numerically stable log-softmax.

Default: `head_dims = [256, 128]`, `dropout = 0.3`, `batch_norm = true`,
`activation = "gelu"`.

---

## 8. Weighted Focal Loss

`models/losses.py`.

### 8.1 Formulation

The multiclass focal loss with class weights:

$$L = -\frac{1}{N}\sum_{i=1}^{N} \alpha_{y_i}\,(1 - p_{y_i})^{\gamma}
\log(p_{y_i})$$

where $p_{y_i}$ is the predicted probability of the true class and
$\alpha_{y_i}$ is the weight of that class.

$\gamma$ controls how strongly easy examples are down-weighted. $\gamma = 0$
degenerates to weighted cross-entropy, which the test suite asserts.

### 8.2 Numerical stability

The implementation computes `log_softmax` once and gathers the target log
probability, rather than taking `log(softmax(...))`. This avoids a
`log(0)` and keeps extreme logits finite.

### 8.3 Class weights

Computed **from the training labels only** — never from validation or test.

Two strategies:

- `effective_number` — the Cui et al. (2019) formula
  :math:`w_c = \frac{1 - \beta}{1 - \beta^{n_c}}` with $\beta = 0.999$.
- `inverse_frequency` — :math:`w_c = \frac{N}{C \, n_c}`.

Classes with no training sample get a weight of 0 (they are never presented
to the head).

### 8.4 The cap

`training.class_weighting.cap` bounds the **ratio** between the largest and
smallest class weight, so a class with a handful of samples cannot dominate
the gradient. The excess above the minimum is compressed linearly:

$$w'_c = w_{min} + (w_c - w_{min}) \cdot s, \qquad
s = \frac{w_{min}\,(\text{cap} - 1)}{w_{max} - w_{min}}$$

This guarantees :math:`\max(w') / \min(w') \leq \text{cap}` while keeping the
ordering strictly monotonic. (Clipping at a threshold would collapse every
class above it onto the same value and destroy the ordering.)

### 8.5 Comparison

`build_loss()` provides `ce`, `weighted_ce` and `weighted_focal`, so the three
can be compared on the same split:

```bash
python training/train_baseline.py --set training.loss=ce
python training/train_baseline.py --set training.loss=weighted_ce
python training/train_baseline.py --set training.loss=weighted_focal
```

---

## 9. Training mechanics

### 9.1 Parameter groups

The backbone trains at a much lower learning rate than the new head:

| Group | Learning rate | Why |
|---|---|---|
| Backbone | 1e-5 | Pre-trained weights; a large LR destroys them |
| Head / fusion / metadata encoder | 1e-4 | Randomly initialised; needs a faster LR |

### 9.2 Mixed precision

On CUDA, `torch.autocast` runs the forward pass in float16 and a
`GradScaler` rescales the loss before `backward()` to avoid underflowing
gradients. On CPU the scaler is disabled and autocast is a no-op.

### 9.3 Gradient clipping

Gradients are clipped to a maximum norm (default 5.0) after unscaling. This
is essential with focal loss, which can produce large gradients on rare
classes.

### 9.4 Scheduler

Cosine annealing over the configured number of epochs, with optional linear
warmup. `reduce_on_plateau` and `step` are also available.

### 9.5 Early stopping

Monitors validation macro F1 (configurable to any logged metric) with a
patience of 10 epochs and a minimum improvement of 5e-4.

---

## 10. Grad-CAM

`explainability/gradcam.py`.

### 10.1 Method

For a target class $c$, Grad-CAM computes the gradients of the score
$y^c$ with respect to the activations $A^k$ of a chosen target layer:

$$\alpha_k^c = \frac{1}{Z}\sum_{i,j}\frac{\partial y^c}{\partial A_{ij}^k}$$

and forms the class-discriminative localisation map

$$L^c = \mathrm{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

which is upsampled to the input resolution and normalised to $[0, 1]$.

### 10.2 Target layer

The default is the last spatial convolution of the EfficientNetV2 backbone.
When attention is enabled, the attention module is used instead, because it
is the last tensor before pooling and therefore carries the most
class-discriminative signal. The layer can also be set explicitly by name in
`configs/model.yaml`.

### 10.3 Usage

- **Predicted class** — the default; explains why the model made the
  prediction it did.
- **User-selected class** — `explain(image, target_class=0)` explains an
  arbitrary class, which is how a false negative is investigated.
- **Batch** — the implementation handles a batch and returns per-sample
  heatmaps.

### 10.4 Output

For each sample the visualiser produces a figure with the original image, the
heat map, the overlay, a colour bar with the activation scale, the predicted
class and the confidence score.

---

## 11. Grad-CAM limitations

Stated explicitly, because Grad-CAM is often over-claimed:

1. **It is post-hoc and coarse.** The resolution of the explanation is
   bounded by the stride of the target layer (7x7 for EfficientNetV2-S at
   224x224). It cannot localise a structure finer than ~32 input pixels.
2. **It is not causal.** The heat map shows regions correlated with the
   score, not regions the model "reasoned about". It does not prove the model
   used the information.
3. **It is class-discriminative, not exhaustive.** A region can be important
   for class A and unremarkable for class B; a single heat map cannot show
   the contrast.
4. **It is sensitive to the target layer and to the normalisation.** A
   different layer produces a different-looking map, and the ReLU + min-max
   normalisation is a display convention, not a calibrated quantity.
5. **It can highlight artefacts.** Ruler markings, hair, bubbles, the
   dermatoscope border and skin markings routinely appear in heat maps. A
   strong response on such a region is a finding about the *data*, not
   evidence of diagnostic reasoning.
6. **It says nothing about causality of the decision.** Two models with
   identical accuracy can have completely different explanations; Grad-CAM
   is not a faithfulness guarantee.
7. **It does not validate the model.** A convincing heat map is not evidence
   of clinical correctness.

For these reasons the framework treats Grad-CAM as a **debugging and
presentation aid**. Segmentation overlays
(`explainability/segmentation_viz.py`) are provided for the same purpose on
the U-Net side.

---

## 12. Reproducibility

- `project.seed` seeds Python, NumPy and PyTorch.
- cuDNN is configured for deterministic execution where available.
- The data split is seeded independently of the model seed.
- Every checkpoint stores the full model configuration, the class names, the
  metadata dimension and the preprocessing configuration, so a checkpoint
  restores the exact architecture it was trained with.
