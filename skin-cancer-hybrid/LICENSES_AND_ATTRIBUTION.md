# Licences and Attribution

This document records the provenance, licences and required attribution for
the datasets, the third-party code and the dependencies used by this project.

---

## 1. Important notice about the datasets

**This repository does not redistribute any dataset.** No image, mask or
metadata file from HAM10000, ISIC 2018, ISIC 2019 or PAD-UFES-20 is included
in the source tree. The `data/raw/` directory is git-ignored.

You are responsible for:

- obtaining each dataset from its official source,
- accepting the dataset's licence and any data-use agreement,
- complying with the terms under which the data is redistributed,
- citing the original publications and data sources in your thesis.

The `data/raw/` path is ignored by git precisely so that no licensed data can
be committed by accident.

---

## 2. Datasets

### 2.1 HAM10000

| | |
|---|---|
| **Name** | Human Against Machine Identification of Skin Cancer in the Hamamatsu Medical Center |
| **Images** | ~10,015 dermoscopic images |
| **Classes** | MEL, NV, BCC, AKIEC, BKL, DF, VASC |
| **Source** | The HAM10000 dataset, hosted by the Harvard Dataverse |
| **Licence** | CC BY-NC 4.0 (non-commercial use) |
| **Original publication** | Giotis et al. (2017), *A Multiclass Benchmark for the Cross-Skin Lesion Segmentation Challenge*, arXiv:1710.05463 |

> **Non-commercial.** The CC BY-NC licence restricts commercial use. Confirm
> the terms for your intended use before relying on the dataset.

**Citation**

> Giotis I, Liakath-Ali K, Karaman O, et al. A Multiclass Benchmark for the
> Cross-Skin Lesion Segmentation Challenge. *arXiv preprint arXiv:1710.05463*,
> 2017.

### 2.2 ISIC 2018

| | |
|---|---|
| **Name** | International Skin Imaging Collaboration 2018 Challenge |
| **Images** | ~10,015 training images (Task 1: segmentation; Task 2/3: classification) |
| **Classes** | MEL, NV, BCC, AKIEC, BKL, DF, VASC |
| **Masks** | Expert-annotated lesion segmentation masks (Task 1) |
| **Source** | The ISIC Archive, `https://isic-archive.com` |
| **Licence** | CC BY-NC 4.0 for the challenge data |
| **Original publication** | Codella et al. (2018), *Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the International Symposia on Biomedical Imaging (ISBI)* |

**Citations**

> Codella NC, Rotemberg VT, Tschandl S, et al. Skin Lesion Analysis Toward
> Melanoma Detection: A Challenge at the International Symposia on Biomedical
> Imaging (ISBI) 2018. *2018 IEEE 15th International Symposium on Biomedical
> Imaging (ISBI)*, 2018.

> Rotemberg VT, Tschandl S, Cassiano AK, et al. ISIC 2018 Challenge Task 1
> and 2: Segmentation and Classification of Skin Lesion. *arXiv preprint
> arXiv:1807.01768*, 2018.

### 2.3 ISIC 2019

| | |
|---|---|
| **Name** | International Skin Imaging Collaboration 2019 Challenge |
| **Images** | ~25,331 training images across three tasks |
| **Classes** | MEL, NV, BCC, AKIEC, BKL, DF, VASC, **SCC** |
| **Source** | The ISIC Archive, `https://isic-archive.com` |
| **Licence** | CC BY-NC 4.0 for the challenge data |
| **Original publication** | Giotis et al. (2019), *The ISIC 2019 Challenge: A Skin Lesion Segmentation and Classification Benchmark*, arXiv:1908.05723 |

**Citation**

> Giotis I, Liakath-Ali K, Tschandl S, et al. The ISIC 2019 Challenge: A Skin
> Lesion Segmentation and Classification Benchmark. *arXiv preprint
> arXiv:1908.05723*, 2019.

### 2.4 PAD-UFES-20

| | |
|---|---|
| **Name** | A new dataset for skin lesion image classification |
| **Images** | 2,298 clinical images with metadata |
| **Classes** | MEL, NV, BCC, AKIEC, BKL, SCC |
| **Metadata** | age, sex, localization, skin_cancer_history |
| **Source** | Mendeley Data, `https://data.mendeley.com/datasets/36r76d5k7m/2` |
| **Licence** | CC BY 4.0 |
| **Original publication** | Bissoto et al. (2020), *A new dataset for skin lesion image classification*, arXiv:2006.09254 |

**Citation**

> Bissoto EC, Oliveira AC, Cri-state M, et al. A new dataset for skin lesion
> image classification. *arXiv preprint arXiv:2006.09254*, 2020.

---

## 3. Reuse and attribution of the datasets

When you publish results based on these datasets, cite the original
publications above **and** acknowledge the ISIC Archive and the HAM10000
Dataverse as the data providers.

The ISIC 2018 and ISIC 2019 challenges require that results be reported with
the challenge organisers' standard metrics, and their benchmark server may be
used for an independent evaluation of your model.

---

## 4. Method references

The implementation follows published methods. Cite the originals:

| Component | Reference |
|---|---|
| U-Net | Ronneberger et al. (2015), *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI |
| Attention gate | Oktay et al. (2018), *Attention U-Net*, MIDL |
| CBAM | Woo et al. (2018), *CBAM: Convolutional Block Attention Module*, ECCV |
| EfficientNetV2 | Tan et al. (2021), *EfficientNetV2: Smaller, Faster, Better*, ICML |
| Focal loss | Lin et al. (2017), *Focal Loss for Dense Object Detection*, ICCV |
| Class-balanced loss | Cui et al. (2019), *Class-Balanced Loss Based on Effective Number of Samples*, CVPR |
| Deep supervision | Lee et al. (2018), *Deep Supervision for Segmentation*, arXiv:1906.03289 |
| CLAHE | Zuiderveld (1994), *Contrast Limited Adaptive Histogram Equalization*, Graphics Gems IV |
| DullRazor | Huang et al. (2001), *A Dull Razor for Removal of Hair from Dermoscopic Images*, CVPR |
| dHash | Google's perceptual hashing |
| Grad-CAM | Selvaraju et al. (2017), *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV |
| Optuna | Akiba et al. (2019), *Optuna: A Next-generation Hyperparameter Optimization Framework*, KDD |

---

## 5. Third-party code

This repository is an original implementation. The following components are
**re-implemented from published descriptions** rather than copied, but are
listed for full transparency:

- **U-Net** — re-implemented from Ronneberger et al.
- **CBAM** — re-implemented from Woo et al.
- **EfficientNetV2** — obtained from `timm` (Apache-2.0), not vendored here.
- **Grad-CAM** — re-implemented as a PyTorch hook-based implementation rather
  than depending on `grad-cam` / `pytorch-grad-cam`.
- **DullRazor hair removal** — re-implemented as a black-hat morphological
  filter plus inpainting.

No code has been copied from the original HAM10000 / ISIC reference
implementations.

---

## 6. Dependency licences

| Package | Licence |
|---|---|
| PyTorch | BSD-3-Clause |
| torchvision | BSD-3-Clause |
| timm | Apache-2.0 |
| NumPy | BSD-3-Clause |
| pandas | BSD-3-Clause |
| SciPy | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| scikit-image | BSD-3-Clause |
| OpenCV | Apache-2.0 |
| Pillow | MIT-CMU |
| Albumentations | MIT |
| Matplotlib | PSF-based (matplotlib licence) |
| Seaborn | BSD-3-Clause |
| PyYAML | MIT |
| tqdm | MPL-2.0 / MIT (MIT/GPLv2+ dual) |
| TensorBoard | Apache-2.0 |
| Optuna | MIT |
| joblib | BSD-3-Clause |
| imagehash | BSD-2-Clause |
| pytest | MIT |

Run `pip-licenses` or `python -m pip show <package>` for the authoritative
licence text of any dependency:

```bash
pip install pip-licenses
pip-licenses --format=markdown --with-urls --with-license-file
```

---

## 7. This project's code

Unless a file states otherwise, the code in this repository is released under
the **MIT Licence**:

```
MIT License

Copyright (c) 2026 skin-cancer-hybrid contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 8. Clinical disclaimer

This software is a **research prototype**. It is not a medical device, is not
regulatory-approved, and must not be used to make clinical decisions.

Predictions produced by this framework are experimental model outputs. They do
not constitute a diagnosis and must not substitute for assessment by a
qualified healthcare professional. Any deployment in a clinical setting
would require its own regulatory approval, clinical validation, and
prospective evaluation.

---

## 9. Attribution summary for the thesis

A suitable acknowledgements section:

> We thank the International Skin Imaging Collaboration for the ISIC 2018 and
> ISIC 2019 datasets, the HAM10000 dataset from the Harvard Dataverse, and
> the PAD-UFES-20 dataset from Mendeley Data. We acknowledge the ISIC
> Archive for hosting these datasets and making them available to the
> research community.
