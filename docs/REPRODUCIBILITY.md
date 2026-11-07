
# LiverPath AI — Reproducibility Guide

## 1. Environment

- Python and PyTorch
- torchvision (ImageNet-pretrained ResNet-18)
- scikit-learn
- Apple Silicon MPS was used for model development

```bash
git clone https://github.com/janhavj/LiverPath-AI.git
cd LiverPath-AI

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Dataset

**Source:** [HepatoBench](https://huggingface.co/datasets/xtxx/HepatoBench)

- Original patches: 87,301
- Valid, unique patches: 87,220
- Seven histopathology categories
- Image dimensions: 150 × 150 pixels
- Dataset licence: CC BY 4.0

Pinned dataset revision:

`3fd9d3697fb63a84ea0635aed97f77ad3f1c5d97`

Raw images are not included in this repository.

## 3. Experimental Workflow

| Stage | Scripts |
|---|---|
| Dataset inspection and acquisition | `01`–`06` |
| Image quality control | `07`–`09` |
| Duplicate detection | `10`–`12` |
| Stratified data splitting | `13` |
| PyTorch dataset and architecture | `14`–`15` |
| Baseline training and evaluation | `16`–`24` |
| Fine-tuning and model comparison | `25`–`30` |
| Protocol freezing | `31` |
| Final held-out test evaluation | `32` |
| Scientific figures | `33` |
| RAWGraphs preparation | `34`–`37` |

Run the scripts in numerical order, after reviewing their
input paths and command-line requirements.

## 4. Data Splitting

| Partition | Patches |
|---|---:|
| Training | 61,054 |
| Validation | 13,083 |
| Test | 13,083 |
| Total | 87,220 |

Stratified patch-level split: 70% / 15% / 15%.

Exact duplicate images were removed before splitting.

**Important:** Patient and slide identifiers were unavailable,
so these are not patient-independent partitions.

## 5. Model Development

Architecture: ImageNet-pretrained ResNet-18.

Two experimental stages:

1. Frozen-backbone baseline with a trained classifier.
2. Selective fine-tuning of `layer4` and the classifier.

Training used class-weighted cross-entropy.

Validation macro-F1 was used for model selection.

| Model | Validation Macro-F1 |
|---|---:|
| Frozen-backbone baseline | 0.8245 |
| Fine-tuned ResNet-18 | 0.9057 |

## 6. Calibration and Uncertainty

- Predictive uncertainty: Shannon entropy
- Calibration method: temperature scaling
- Temperature fitted using validation predictions
- Selected temperature: 1.0833
- Error review: rank predictions by descending entropy

## 7. Final Test Evaluation

The selected model and calibration parameter were frozen
before the held-out test evaluation.

| Metric | Result |
|---|---:|
| Test accuracy | 0.9292 |
| Balanced accuracy | 0.9134 |
| Macro-F1 | 0.9005 |
| Error-detection AUROC | 0.9079 |
| Error-detection AUPRC | 0.3928 |
| Calibrated ECE | 0.0062 |

Reviewing the 10% most uncertain test patches captured
56.8% of observed classification errors.

Detailed results are available in `results/final_test/`.

## 8. Reproduction Limitations

- Model checkpoints are not included.
- Raw and processed datasets are not included.
- Training is required to regenerate model weights.
- Some scripts may require adapting local file paths.
- Exact results may vary across hardware and software versions.
- Patch-level evaluation does not establish independent-patient
  or external-institution generalisation.

This repository is an exploratory research prototype and
is not intended for clinical use.
