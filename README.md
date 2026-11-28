# De<sup>3</sup>C: Debiased Cross-Modal Consistency Correction with Reliability-Aware Collaboration for Multimodal Medical Noisy-Label Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8-blue" alt="Python 3.8">
  <img src="https://img.shields.io/badge/PyTorch-1.13.0-red" alt="PyTorch 1.13.0">
  <img src="https://img.shields.io/badge/Task-Multimodal%20Diagnosis-orange" alt="Multimodal Diagnosis">
  <img src="https://img.shields.io/badge/Learning-Noisy%20Label-purple" alt="Noisy Label Learning">
</p>

---

## News

- **[2026.11]** The implementation of **De<sup>3</sup>C** is prepared for release.

---

## Overview

This repository provides the MM-ECG implementation of:

> **De<sup>3</sup>C: Debiased Cross-Modal Consistency Correction with Reliability-Aware Collaboration for Multimodal Medical Noisy-Label Learning**

De<sup>3</sup>C is designed for robust multimodal medical diagnosis under noisy annotations. Rather than treating sample selection and label correction as isolated processes, our framework establishes reliability-aware collaboration between them to preserve informative supervision and progressively refine noisy labels.

The main contributions include:

- **Debiased Distribution Calibration (DDC):** mitigates class-dependent selection bias through distribution calibration and complementary modality-specific evidence.
- **Cross-Modal Consistency Correction (CCC):** recovers informative supervision from unselected samples using high-confidence agreement between modality-specific and fused predictions, with reliability-weighted correction.
- **Reliability-constrained Feedback Mechanism (RFM):** incorporates highly confident and temporally stable correction evidence into subsequent sample selection.

---

## Framework

<p align="center">
  <img src="img/overall.png" width="900" alt="Overview of De3C">
</p>

The overall framework contains two key modules and a collaborative feedback mechanism:

1. **Debiased Distribution Calibration**
   - Tracks class-wise prediction statistics through an exponential moving average (EMA).
   - Calibrates fused predictions and incorporates modality-specific evidence to identify reliable observed labels.
2. **Cross-Modal Consistency Correction**
   - Exploits high-confidence agreement among modality-specific and fused predictions.
   - Refines potentially corrupted labels and weights correction supervision according to predictive reliability.
3. **Reliability-constrained Feedback**
   - Collects stable, high-confidence correction evidence.
   - Refines the distribution used for subsequent sample selection.

---

## Datasets

We evaluate De<sup>3</sup>C on three multimodal medical datasets:

| Dataset | Task | Modalities | Link |
|---|---|---|---|
| MMC-AMD | AMD grading | Fundus + OCT | [Dataset](https://github.com/li-xirong/mmc-amd) |
| Harvard30k-AMD | AMD grading | SLO fundus + OCT | [Dataset](https://yutianyt.com/projects/fairvision30k/) |
| MM-ECG | Cardiac rhythm classification | 1D ECG + 2D ECG image | [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/) / [ECGInstruct](https://huggingface.co/datasets/PULSE-ECG/ECGInstruct) |

```text
data/MM-ECG/
├── final_report_with_mapping_with_ecg_new.csv
├── image_npy/
│   └── <sample>.npy
└── ecg_npy/
    └── <sample>.npy
```

---

## Requirements

The target environment is Python 3.8 and PyTorch 1.13.0. Install the main dependencies:

```bash
pip install torch==1.13.0 torchvision==0.14.0 numpy==1.23.5 pandas==1.5.3 scikit-learn==1.2.2
```

---

## Quick Start

The main components are organized as follows:

```text
De3C/
├── De3C.py
├── main.py
├── dataloader_ECG.py
├── dependency_ECG.py
└── utils_ECG.py
```

```bash
python main.py --help
```

---

## Usage

Run the main training program using Python:

```bash
python main.py --data-root data/MM-ECG --noise-rate 0.3 --device cuda:0
```

Set `--noise-rate` to `0.4` or `0.5` for other label noise levels.

---

## Acknowledgement and Contact

We thank the authors of MMC-AMD, Harvard-FairVision, MIMIC-IV-ECG, ECGInstruct and ResNet for their valuable contributions to the research community.

For any questions, feel free to contact: [Jing.Li2@liverpool.ac.uk](mailto:Jing.Li2@liverpool.ac.uk).
