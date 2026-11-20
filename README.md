# ModelShield — An AI Firewall for Adversarial Attacks (Phase 1)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-green.svg)](https://www.cs.toronto.edu/~kriz/cifar.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent deep learning cybersecurity firewall designed to detect and intercept adversarial image attacks before they reach mission-critical computer vision models. Developed for the academic subject **Intelligent Model Design (IMD)**.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Project Objectives](#3-project-objectives)
4. [How Adversarial Attacks Work](#4-how-adversarial-attacks-work)
5. [The Fast Gradient Sign Method (FGSM)](#5-the-fast-gradient-sign-method-fgsm)
6. [System Architecture Diagram](#6-system-architecture-diagram)
7. [Technology Stack](#7-technology-stack)
8. [Project Folder Structure](#8-project-folder-structure)
9. [Local Installation & Setup](#9-local-installation--setup)
10. [Google Colab Training Instructions](#10-google-colab-training-instructions)
11. [Executing the Firewall Locally](#11-executing-the-firewall-locally)
12. [Evaluation Metrics & Benchmarks](#12-evaluation-metrics--benchmarks)
13. [Current Limitations (Phase 1)](#13-current-limitations-phase-1)
14. [Future Development Roadmap](#14-future-development-roadmap)

---

## 1. Project Overview

Modern deep learning classification systems are highly effective yet notoriously fragile against adversarial manipulations. **ModelShield** introduces an architectural security layer — an **AI Firewall** — positioned in front of downstream vision models. 

Rather than deploying complex or computationally heavy transformations during runtime, ModelShield deploys a dedicated, lightweight convolutional neural network (CNN) trained as a binary adversarial detector. The detector examines input samples for perturbation signatures, assigns an adversarial confidence probability, and enforces configurable security policies (`ALLOWED` vs. `BLOCKED`).

---

## 2. Problem Statement

Standard Convolutional Neural Networks (CNNs) operate under the assumption that test inputs stem from the same natural distribution as training data. Adversaries exploit the linear properties of deep models in high-dimensional spaces by adding carefully crafted, human-imperceptible perturbations to clean images. 

These adversarial modifications shift the input across decision boundaries, causing state-of-the-art classifiers to output incorrect predictions with high statistical confidence. In sensitive domains such as autonomous driving, biometric authentication, and medical diagnostics, unmonitored adversarial vulnerability presents severe operational risks.

---

## 3. Project Objectives

Phase 1 establishes a foundational, verified, and reproducible adversarial defense framework:
- **Module 1 (Classifier):** Train a custom multi-layer CNN recognizing all 10 CIFAR-10 classes.
- **Module 2 (Attack Generator):** Implement the Fast Gradient Sign Method (FGSM) in consistent physical pixel space and evaluate untargeted Attack Success Rates (ASR).
- **Module 3 (Firewall Detector):** Generate balanced clean-vs-adversarial datasets without data leakage and train a dedicated binary CNN detector.
- **Module 4 (Firewall Pipeline):** Build an integrated, threshold-gated firewall pipeline providing structured executive security reporting.

---

## 4. How Adversarial Attacks Work

Deep neural networks rely on dot products $\mathbf{w}^\top \mathbf{x}$ across millions of parameters. Even if an adversary perturbs each input pixel by an infinitesimal value $\epsilon$ below human perceptual limits, the cumulative inner-product perturbation across $d$ dimensions can grow as large as $\epsilon d \|\mathbf{w}\|_1$. 

In high-dimensional datasets such as CIFAR-10 ($3 \times 32 \times 32 = 3,072$ dimensions), tiny coordinated changes across dimensions can drastically shift the final activations, tricking the classifier into predicting an unrelated class while the image appears unchanged to human observers.

---

## 5. The Fast Gradient Sign Method (FGSM)

Introduced by Goodfellow et al. (2014), FGSM is a single-step white-box adversarial attack that leverages the gradient of the model's loss function with respect to the input image.

### Mathematical Formulation
$$\mathbf{x}_{\text{adv}} = \text{clip}\left(\mathbf{x} + \epsilon \cdot \text{sign}\left(\nabla_{\mathbf{x}} \mathcal{L}(f(\mathbf{x}), y)\right), 0.0, 1.0\right)$$

Where:
- $\mathbf{x}$: Original clean image tensor in unnormalized physical pixel range $[0.0, 1.0]$.
- $y$: Ground-truth class label.
- $f(\mathbf{x})$: Classifier output logits.
- $\mathcal{L}$: Cross-Entropy loss function.
- $\nabla_{\mathbf{x}} \mathcal{L}$: Loss gradient with respect to input pixels.
- $\epsilon$: Perturbation magnitude bound (e.g., $0.01, 0.03, 0.05, 0.10$).
- $\text{clip}(\cdot, 0, 1)$: Pixel bounds clamping to prevent underflow or overflow.

### Rigorous Normalization Handling
Because PyTorch classifiers process z-score normalized inputs $\mathbf{x}_{\text{norm}} = \frac{\mathbf{x} - \mu}{\sigma}$, ModelShield executes perturbations strictly in physical $[0, 1]$ pixel space via differentiable denormalization and re-normalization. This ensures that channel standard deviations ($\sigma_r, \sigma_g, \sigma_b$) and valid pixel bounds $[0, 1]$ are mathematically preserved without clipping distortion.

---

## 6. System Architecture Diagram

```
                              [ Incoming Image ]
                                      │
                                      ▼
                       [ Preprocessing & Normalization ]
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ Module 3: AI Detector CNN │
                        │  (Binary Classification)  │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                       Adversarial Probability: P(adv)
                                      │
                          Is P(adv) >= Threshold?
                                     / \
                                    /   \
                             YES   /     \   NO
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

---

## 7. Technology Stack

- **Language:** Python 3.10+ (tested on Python 3.12)
- **Deep Learning Framework:** PyTorch & Torchvision
- **Numerical Computation:** NumPy
- **Data & Metrics:** Scikit-learn (Classification Report, Confusion Matrix, PRF)
- **Visualizations:** Matplotlib
- **Image Processing:** Pillow (PIL)
- **Progress Tracking:** tqdm
- **Automated Testing:** pytest

---

## 8. Project Folder Structure

```
ModelShield/
├── src/
│   ├── data.py          # CIFAR-10 data loaders, preprocessing, split handling
│   ├── models.py        # Custom Classifier CNN and Detector CNN architectures
│   ├── attacks.py       # FGSM attack engine, ASR calculation, visualization
│   ├── train.py         # Training loops, validation, checkpoints, metrics
│   └── firewall.py      # Integrated firewall pipeline and security reporting
├── notebooks/
│   └── ModelShield_Phase1.ipynb   # Complete, runnable Google Colab training notebook
├── outputs/
│   └── .gitkeep         # Evaluation plots, confusion matrices, demo artifacts
├── checkpoints/
│   └── .gitkeep         # Saved best model weights (.pth)
├── tests/
│   └── test_models.py   # Automated pytest unit tests (runnable on CPU)
├── config.py            # Centralized hyperparameters, paths, and random seeds
├── main.py              # CLI entry point for local pipeline testing
├── requirements.txt     # Locked production dependencies
├── .gitignore           # Git exclusions for datasets, weights, caches
└── README.md            # Comprehensive project documentation
```

---

## 9. Local Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/ModelShield.git
%cd ModelShield
```

### 2. Create and Activate Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Automated Unit Tests
Verify model shapes, preprocessing round-trips, FGSM clipping, and firewall decision logic:
```bash
pytest -v
```

---

## 10. Google Colab Training Instructions

Computationally heavy training for CIFAR-10 should be performed on a free GPU in Google Colab:

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **Upload** and upload `notebooks/ModelShield_Phase1.ipynb` (or open directly from GitHub).
3. Connect to a GPU runtime:
   - Click **Runtime** -> **Change runtime type**.
   - Select **T4 GPU** under Hardware accelerator -> Click **Save**.
4. Run the first cell to clone your repository:
   ```bash
   !git clone https://github.com/<your-username>/ModelShield.git
   %cd ModelShield
   ```
5. Run all cells sequentially. The notebook will:
   - Download CIFAR-10 automatically.
   - Train the 10-class CNN classifier.
   - Evaluate classifier accuracy and generate confusion matrices on the untouched test set.
   - Generate FGSM adversarial examples across multiple epsilons ($0.01, 0.03, 0.05, 0.10$).
   - Compute the Attack Success Rate (ASR) curve.
   - Construct balanced detector datasets (50% clean, 50% FGSM).
   - Train the binary adversarial detector CNN.
   - Evaluate detector metrics (Accuracy, Precision, Recall, F1, False Positive Rate).
   - Test the complete integrated firewall pipeline.
   - Save the best weights to `checkpoints/` and optionally mirror them to your Google Drive.
6. Download the generated `.pth` checkpoints and place them in your local `checkpoints/` folder.

---

## 11. Executing the Firewall Locally

Run the integrated firewall CLI demonstration:

```bash
python main.py
```

### Command-Line Arguments
```bash
# Run demonstration with custom detection threshold
python main.py --threshold 0.60

# Run demonstration with custom FGSM perturbation
python main.py --epsilon 0.05

# Inspect a specific local image file
python main.py --image path/to/image.png --threshold 0.50
```

### Example Firewall Output Reports

**For a Suspicious Adversarial Image:**
```text
==================================================
           MODELSHIELD SECURITY REPORT            
==================================================
Input: adversarial_sample.png
Adversarial probability: 0.9241
Threshold: 0.50
Firewall decision: BLOCKED
Reason: The detector classified the image as suspicious.
==================================================
```

**For a Clean Image:**
```text
==================================================
           MODELSHIELD SECURITY REPORT            
==================================================
Input: clean_sample.png
Adversarial probability: 0.0382
Threshold: 0.50
Firewall decision: ALLOWED
Predicted class: dog
Classification confidence: 94.20%
==================================================
```

---

## 12. Evaluation Metrics & Benchmarks

The project records and reports standard scientific metrics:

1. **Classifier Metrics (Untouched Test Set):**
   - Overall Classification Accuracy.
   - Per-Class Precision, Recall, and F1-Score.
   - Multi-class Confusion Matrix.

2. **Attack Benchmarks (FGSM):**
   - Untargeted Attack Success Rate (ASR):
     $$\text{ASR} = \frac{\text{Samples Fooled by Attack}}{\text{Samples Originally Classified Correctly}}$$
   - Adversarial Accuracy across $\epsilon \in [0.0, 0.01, 0.03, 0.05, 0.10]$.

3. **Detector Firewall Metrics:**
   - Classification Accuracy on held-out clean/adversarial pairs.
   - Precision: $\frac{\text{TP}}{\text{TP} + \text{FP}}$ (proportion of flagged images that were truly adversarial).
   - Recall / True Positive Rate: $\frac{\text{TP}}{\text{TP} + \text{FN}}$ (proportion of attacks successfully detected).
   - F1-Score: Harmonic mean of precision and recall.
   - False Positive Rate (FPR): $\frac{\text{FP}}{\text{FP} + \text{TN}}$ (clean images mistakenly blocked).
   - Binary Confusion Matrix.

---

## 13. Current Limitations (Phase 1)

1. **Single Attack Distribution:** The Phase 1 detector is trained specifically on FGSM perturbations and is not guaranteed to generalize to other attack distributions.
2. **Untargeted Attacks Only:** Phase 1 focuses exclusively on untargeted FGSM attacks.
3. **Resolution:** Bounded to the CIFAR-10 resolution ($32 \times 32$ pixels).
4. **No Adversarial Purification:** Flagged images are quarantined/blocked; Phase 1 does not attempt image restoration or denoising.
5. **No Adversarial Retraining:** The classifier weights remain fixed rather than robustly re-trained.

---

## 14. Future Development Roadmap

- [ ] **Phase 2 — Multi-Attack Generation:** Projected Gradient Descent (PGD), Carlini-Wagner (C&W), and DeepFool.
- [ ] **Phase 3 — Robust Detection:** Ensemble detectors and statistical feature attribution (e.g., Grad-CAM, Mahalanobis distance).
- [ ] **Phase 4 — Adversarial Purification & Defense:** Wavelet denoising, autoencoder purification, and adversarial training.
- [ ] **Phase 5 — Interactive UI:** Streamlit / FastAPI interactive dashboard for live firewall inspection.
