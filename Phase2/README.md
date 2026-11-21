# ModelShield Phase 2 — Advanced Adversarial AI Detection & Defense

ModelShield Phase 2 is an end-to-end adversarial defense system for deep vision classifiers on CIFAR-10. It establishes an advanced experimental framework incorporating multi-attack generation (FGSM & PGD), multi-attack detector mixture training, strict dataset isolation, validation-based dynamic threshold calibration, permanent Google Drive checkpoint persistence, and modular unit testing.

---

## Key Improvements Over Phase 1

1. **Multi-Attack Engine**: Single-step FGSM and iterative Projected Gradient Descent (PGD) with $L_\infty$ projection, random restarts, and physical $[0, 1]$ pixel clamping.
2. **Elimination of Shortcut Learning**: Trains the adversarial detector on a balanced mixture (50% Clean, 25% FGSM, 25% PGD across diverse $\epsilon \in \{0.005, 0.01, 0.02, 0.03\}$), preventing high-frequency noise fingerprinting.
3. **Rigorous Data Separation**: Deterministic 40,000 / 10,000 / 10,000 partitioning; non-shuffled validation and test loaders; final evaluation on the full 10,000 test set.
4. **Validation-Based Dynamic Calibration**: Calibrates the firewall threshold $\tau^*$ exclusively on validation data to mathematically bound False Positive Rate (FPR) $\le 1.0\%$.
5. **Permanent Google Drive Storage**: Automatic cloud mount in Colab (`/content/drive/MyDrive/ModelShield_Phase2/`), non-zero file validation, and 4-way RNG restoration for seamless training resumption.

---

## Directory Structure

```
Phase2/
├── config.py                 # Central configuration and experiment registry
├── main.py                   # Unified command-line interface
├── requirements.txt          # Python dependencies
├── .gitignore                # Checkpoint and artifact exclusion rules
├── src/                      # Core modular architecture
│   ├── models.py             # Preserved 3-block CNN Classifier & Detector CNNs
│   ├── data.py               # Deterministic CIFAR-10 data loaders & transforms
│   ├── attacks.py            # Vectorized FGSM & PGD attack engines & visualizers
│   ├── detector.py           # Multi-attack mixture dataset builder & 4-facet generalization
│   ├── firewall.py           # AI Firewall gateway & validation threshold calibration
│   ├── train.py              # Resumable training loops with RNG restoration
│   └── evaluate.py           # Comprehensive evaluation & benchmarking routines
├── tests/                    # Standalone pytest suite (37 tests, 100% pass rate)
│   ├── test_models.py
│   ├── test_attacks.py
│   ├── test_detector.py
│   ├── test_firewall.py
│   └── test_integration_repairs.py # Integration & audit regression verification
├── notebooks/                # Production Jupyter notebook
│   └── ModelShield_Phase2.ipynb
├── checkpoints/              # Checkpoint directory (.pth)
├── results/                  # Figures, confusion matrices, and ROC plots
│   ├── classifier/
│   ├── detector/
│   ├── attacks/
│   └── firewall/
├── logs/                     # Structured execution logs
└── reports/                  # Detailed engineering reports
    ├── PHASE2_EXPERIMENT_REPORT.md
    └── PHASE1_VS_PHASE2_COMPARISON.md
```

---

## Installation & Environment Setup

### 1. Clone & Enter Directory
```bash
git clone https://github.com/Adi-015/ModelShield.git
cd ModelShield/Phase2
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Command-Line Interface (CLI)

`Phase2/main.py` provides a unified entrypoint for running tests, training, and evaluation:

### Fast End-to-End Verification Demo (Synthetic CPU Smoke Test)
```bash
python main.py --demo
```

### Train Baseline Classifier (Resumable)
```bash
python main.py --train-classifier --epochs 15 --batch-size 64
```

### Train Multi-Attack Mixture Detector (Resumable)
```bash
python main.py --train-detector --epochs 10 --batch-size 64
```

### Run Attack Robustness Spectrum Sweep (FGSM & PGD)
```bash
python main.py --attack-eval
```

### Calibrate Firewall Threshold (Target FPR $\le 1\%$)
```bash
python main.py --calibrate --target-fpr 0.01
```

### Run Full Test Set Benchmark Evaluation
```bash
python main.py --evaluate
```

---

## Running Automated Tests

Run the complete 37-test suite:
```bash
pytest -v Phase2/tests/
```

All tests execute on CPU in $\approx 19$ seconds with a **100% pass rate** (37 passed, 0 failed).
The suite includes rigorous unit tests for model shapes, parameters, FGSM/PGD bounds, gradient isolation, detector datasets, firewall thresholding, vector batch filtering without downstream compute leaks, and configuration persistence.

---

## Google Colab Execution

Open `Phase2/notebooks/ModelShield_Phase2.ipynb` directly in Google Colab.

### Key Colab Capabilities:
- Automatically mounts Google Drive at `/content/drive`.
- Checkpoints and evaluation plots persist permanently to `/content/drive/MyDrive/ModelShield_Phase2/`.
- If Colab times out or disconnects, re-running resumes from the exact epoch saved.

---

## Integrity & Non-Interference Guarantee

Phase 2 strictly adheres to zero-interference constraints:
- Phase 1 source files, checkpoints, notebooks, and results in the repository root remain **100% untouched**.
- Phase 2 maintains separate configurations, output directories, and checkpoints.
