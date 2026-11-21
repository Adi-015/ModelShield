# ModelShield — An AI Firewall for Adversarial Attacks (Phase 1)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-green.svg)](https://www.cs.toronto.edu/~kriz/cifar.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent deep learning cybersecurity firewall designed to detect, quarantine, and intercept adversarial image attacks before they reach mission-critical computer vision models. Developed for the academic subject **Intelligent Model Design (IMD)**.

**Official Repository:** [https://github.com/Adi-015/ModelShield.git](https://github.com/Adi-015/ModelShield.git)

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [System Architecture](#3-system-architecture)
4. [Historical Baseline & Reproducibility](#4-historical-baseline--reproducibility)
5. [Permanent Google Drive Checkpoint Storage](#5-permanent-google-drive-checkpoint-storage)
6. [Resumable Training Pipeline](#6-resumable-training-pipeline)
7. [Controlled CNN Classifier Optimization](#7-controlled-cnn-classifier-optimization)
8. [Adversarial Detector Architecture & Training](#8-adversarial-detector-architecture--training)
9. [Integrated Firewall Pipeline & Security Logic](#9-integrated-firewall-pipeline--security-logic)
10. [Local Installation & Quickstart](#10-local-installation--quickstart)
11. [Google Colab Training Guide](#11-google-colab-training-guide)
12. [Evaluation Metrics & Defense Benchmarks](#12-evaluation-metrics--defense-benchmarks)
13. [Limitations (Phase 1)](#13-limitations-phase-1)
14. [Future Development Roadmap](#14-future-development-roadmap)

---

## 1. Project Overview

Modern deep learning classification systems are highly accurate yet notoriously fragile against adversarial manipulations. **ModelShield** introduces an architectural security layer — an **AI Firewall** — positioned in front of downstream vision models. 

Rather than deploying complex or computationally heavy transformations during runtime, ModelShield deploys a dedicated, lightweight convolutional neural network (CNN) trained as a binary adversarial detector. The detector examines input samples for perturbation signatures, assigns an adversarial confidence probability, and enforces configurable security policies (`ALLOWED` vs. `BLOCKED`).

---

## 2. Problem Statement

Standard Convolutional Neural Networks (CNNs) operate under the assumption that test inputs stem from the same natural distribution as training data. Adversaries exploit the linear properties of deep models in high-dimensional spaces by adding carefully crafted, human-imperceptible perturbations to clean images. 

These adversarial modifications shift the input across decision boundaries, causing state-of-the-art classifiers to output incorrect predictions with high statistical confidence. In sensitive domains such as autonomous driving, biometric authentication, and medical diagnostics, unmonitored adversarial vulnerability presents severe operational risks.

---

## 3. System Architecture

```
                               [ Incoming Image ]
                                       │
                                       ▼
                        [ Preprocessing & Normalization ]
                                       │
                                       ▼
                        ┌───────────────────────────────┐
                        │   Module 3: AI Detector CNN   │
                        │    (Binary Classification)    │
                        └──────────────┬────────────────┘
                                       │
                                       ▼
                        Adversarial Probability: P(adv)
                                       │
                           Is P(adv) >= Threshold?
                                      / \
                                     /   \
                             YES    /     \    NO
                                   /       \
                                  ▼         ▼
                       ┌───────────────┐   ┌───────────────────────────┐
                       │    BLOCKED    │   │          ALLOWED          │
                       │  Quarantine   │   │ Forward to Classifier CNN │
                       │ Security Alert│   └─────────────┬─────────────┘
                       └───────────────┘                 │
                                                         ▼
                                                 [ Class Prediction ]
                                                 [ Confidence Score ]
```

### Module Breakdown
- **Module 1 (Core Classifier):** Custom 3-block CNN classifying images into the 10 CIFAR-10 classes.
- **Module 2 (Attack Engine):** Fast Gradient Sign Method (FGSM) adversarial attack generation across perturbation magnitudes ($\epsilon \in [0.00, 0.01, 0.03, 0.05, 0.10]$) in unnormalized physical $[0, 1]$ pixel space.
- **Module 3 (AI Firewall Detector):** Binary CNN screening incoming inputs for adversarial artifacts.
- **Module 4 (Integrated Security Layer):** Pre-inference gateway that routes verified clean inputs to the classifier while dropping malicious inputs.

---

## 4. Historical Baseline & Reproducibility

The original ModelShield experiment was trained in Google Colab with temporary session storage. Those weights were lost when the session ended. The metrics below are recovered from historical notebook outputs and serve as our experimental benchmark.

### Previously Recorded Baseline Results

#### Classifier (15 Epochs)
- **Architecture:** Custom CIFAR10Classifier (3-Block CNN)
- **Final Train Accuracy:** 70.95%
- **Final Validation Accuracy:** 78.38%
- **Final Train Loss:** 0.8344
- **Final Validation Loss:** 0.6079
- **Best Checkpoint:** Epoch 15

#### Adversarial Detector (10 Epochs)
- **Architecture:** Custom AdversarialDetector (3-Block CNN)
- **Final Train Accuracy:** 99.52%
- **Best Validation Accuracy:** 99.92%
- **Final Validation Accuracy:** 99.46%
- **Final Train Loss:** 0.0158
- **Final Validation Loss:** 0.0173
- **Best Checkpoint:** Epoch 5

#### FGSM Attack Robustness (Historical Baseline)
| Epsilon ($\epsilon$) | Clean Accuracy | Adversarial Accuracy | Attack Success Rate (ASR) |
| :---: | :---: | :---: | :---: |
| **0.00** | 79.12% | 79.12% | 0.00% |
| **0.01** | 79.12% | 22.26% | 71.87% |
| **0.03** | 79.12% | 6.36% | 92.01% |

> **Important:** To regenerate real model weights, we rerun the reproducible training scripts. The original CNN experiment remains our fixed baseline for all comparative analysis.

---

## 5. Permanent Google Drive Checkpoint Storage

Checkpoints are permanently preserved on Google Drive to prevent data loss when Colab runtimes disconnect.

### Google Drive Storage Layout
```
/content/drive/MyDrive/ModelShield/
├── checkpoints/
│   ├── baseline_classifier_best.pth
│   ├── baseline_detector_best.pth
│   ├── optimized_classifier_best.pth
│   ├── optimized_detector_best.pth
│   ├── latest_classifier.pth
│   └── latest_detector.pth
├── outputs/
│   ├── classifier_history.json
│   ├── detector_history.json
│   ├── optimization_results.json
│   ├── evaluation_results.json
│   ├── robustness_results.json
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   ├── detector_roc_curve.png
│   └── robustness_results.png
└── logs/
    └── training.log
```

### Storage Safeguards
- Non-zero file size check (`st_size > 0`) is performed after every save.
- Explicit warnings are raised if Google Drive is expected but unmounted.
- Baseline and optimized checkpoints are strictly separated and will never overwrite each other.

---

## 6. Resumable Training Pipeline

If a Google Colab session times out or disconnects mid-training, training resumes from the last completed epoch rather than restarting from epoch 1.

### Checkpoint State Preservation
Each checkpoint stores:
- `model_state_dict`: Neural network weights.
- `optimizer_state_dict`: Adam momentum and variance states.
- `scheduler_state_dict`: Learning rate schedule progress.
- `epoch`: Current epoch index.
- `val_acc` & `best_val_acc`: Validation accuracy tracking.
- `history`: Full loss and accuracy histories across completed epochs.
- `rng_state`: PyTorch, NumPy, Python, and CUDA RNG states for deterministic resumption.

### Resumption Behavior
1. `latest_<name>.pth` is saved after every completed epoch.
2. `best_<name>.pth` is saved only when validation accuracy improves.
3. If an experiment was already completed for the target number of epochs, training logs a completion notice and loads the best weights immediately, avoiding redundant computation.

---

## 7. Controlled CNN Classifier Optimization

### Preservation of Architecture
The custom CNN classifier architecture is strictly retained. It is **not** replaced with ResNet, VGG, Vision Transformers, or pretrained backbones.

### Investigation of Baseline Accuracy Gap
In the baseline, training accuracy (70.95%) was lower than validation accuracy (78.38%).
- **Cause:** `CIFAR10Classifier` uses `Dropout2d(0.25)` across all convolutional blocks and `Dropout(0.50)` in the dense head. During training, stochastic deactivation drops 25%–50% of feature maps. During validation (`eval()`), dropout is disabled, activating full model capacity. Training also incorporates random cropping and horizontal flips.

### Controlled Optimization Experiments
| Experiment | Configuration | Goal | Checkpoint |
| :--- | :--- | :--- | :--- |
| **Baseline** | Original CNN (Adam lr=0.001, wd=1e-4, 15 epochs, standard aug) | Reproduce benchmark | `baseline_classifier_best.pth` |
| **Exp A** | Enhanced data augmentation (RandomCrop, Flip, mild ColorJitter, Rotation) | Better generalization | `exp_a_augmentation_best.pth` |
| **Exp B** | Learning Rate Scheduler (`CosineAnnealingLR`) | Fine-grained convergence | `exp_b_scheduler_best.pth` |
| **Exp C** | Regularization tuning (Label Smoothing 0.05 + tuned weight decay 3e-4) | Prevent overconfidence | `exp_c_regularization_best.pth` |
| **Exp D** | Combined Best Synergistic Configuration | **Target: 85%+ validation accuracy** | `optimized_classifier_best.pth` |

---

## 8. Adversarial Detector Architecture & Training

### Architecture
- **Conv Block 1:** Conv(3 $\to$ 32, $3 \times 3$, pad 1) $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool $\to$ Dropout(0.2)
- **Conv Block 2:** Conv(32 $\to$ 64, $3 \times 3$, pad 1) $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool $\to$ Dropout(0.2)
- **Conv Block 3:** Conv(64 $\to$ 128, $3 \times 3$, pad 1) $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool $\to$ Dropout(0.3)
- **Dense Head:** Linear(2048 $\to$ 128) $\to$ ReLU $\to$ Dropout(0.4) $\to$ Linear(128 $\to$ 1)

### Compatibility & Gradient Sourcing
Because FGSM perturbations depend directly on the classifier's gradients, the detector is trained on adversarial samples generated from the specific target classifier:
- Baseline detector checkpoint: `baseline_detector_best.pth`
- Optimized detector checkpoint: `optimized_detector_best.pth`
- Train, validation, and test splits remain completely isolated to prevent data leakage.

---

## 9. Integrated Firewall Pipeline & Security Logic

The firewall acts as a real-time gateway:
1. **Input Preprocessing:** Normalizes raw input to CIFAR-10 distribution.
2. **Adversarial Screening:** Passes image to `AdversarialDetector` to obtain probability $P(\text{adv})$.
3. **Threshold Gate:**
   - If $P(\text{adv}) \ge \text{threshold}$ (default: 0.50): **`BLOCKED`** (quarantine, log alert, bypass classifier).
   - If $P(\text{adv}) < \text{threshold}$: **`ALLOWED`** (forward to classifier, return class and confidence).

---

## 10. Local Installation & Quickstart

### Prerequisites
- Python 3.10+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Adi-015/ModelShield.git
cd ModelShield
```

### 2. Setup Virtual Environment
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Automated Tests
```bash
pytest -v tests/test_models.py
```

### 5. Run Local Pipeline Demo
```bash
python main.py
```

---

## 11. Google Colab Training Guide

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **Upload** and upload `notebooks/ModelShield_Phase1.ipynb`.
3. Set runtime to **GPU (T4)** via `Runtime` $\to$ `Change runtime type`.
4. Run the Google Drive mount cell to link `/content/drive/MyDrive/ModelShield`.
5. Execute notebook cells sequentially:
   - Cell 1–2: Drive setup & persistence directory creation.
   - Cell 3–4: CUDA diagnostics & seed initialization.
   - Cell 5–6: Repository setup and CIFAR-10 loading.
   - Cell 7–8: Baseline classifier training (15 epochs, resumable) & test evaluation.
   - Cell 9–10: Controlled optimization experiments (Exp D, target 85%+ val accuracy).
   - Cell 11–12: FGSM adversarial attack visualization & ASR curve.
   - Cell 13–14: Adversarial detector training (10 epochs, resumable) & ROC-AUC evaluation.
   - Cell 15–16: Integrated ModelShield firewall benchmark & live demo.
   - Cell 17: Storage manifest verification.

---

## 12. Evaluation Metrics & Defense Benchmarks

The project reports standard scientific metrics:
- **Classifier:** Clean Accuracy, Loss, Per-Class Accuracy, Confusion Matrix.
- **Attack Robustness:** Clean Accuracy, Adversarial Accuracy, Attack Success Rate (ASR), Degradation across $\epsilon \in [0.00, 0.01, 0.03, 0.05, 0.10]$.
- **Detector:** Accuracy, Precision, Recall, F1-Score, False Positive Rate (FPR), False Negative Rate (FNR), ROC-AUC Score, ROC Curve.
- **Firewall Defense:** Attack Threat Mitigation Rate (% attacks blocked), Clean Utility Retained (% clean images correctly classified), False Alarm Rate (% clean images blocked).

---

## 13. Limitations (Phase 1)

1. **Single Attack Distribution:** Trained primarily on FGSM perturbations; does not guarantee defense against multi-step attacks (PGD, C&W).
2. **Resolution:** CIFAR-10 resolution ($32 \times 32$ pixels).
3. **No Retraining:** Classifier weights remain frozen rather than adversarially robustified.
4. **No Purification:** Blocked images are quarantined rather than denoised.

---

## 14. Future Development Roadmap

- [ ] **Phase 2 — Multi-Attack Generation:** Projected Gradient Descent (PGD), Carlini-Wagner (C&W), and DeepFool.
- [ ] **Phase 3 — Defense Robustness:** Adversarial training and statistical feature attribution (Grad-CAM, Mahalanobis distance).
- [ ] **Phase 4 — Adversarial Purification:** Denoising autoencoders and wavelet transformations.
- [ ] **Phase 5 — Web Dashboard:** Streamlit/FastAPI interactive monitoring dashboard.
