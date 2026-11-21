"""
ModelShield - Centralized Configuration & Experiment Registry
Configures paths, Google Drive integration, device selection, random seeds,
hyperparameters, historical baselines, and controlled optimization experiments.
"""

from pathlib import Path
import os
import sys
import random
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch

# -----------------------------------------------------------------------------
# Environment & Environment Detection
# -----------------------------------------------------------------------------

def is_colab() -> bool:
    """Detect whether the current runtime is Google Colab."""
    return "google.colab" in sys.modules or os.path.exists("/content")

# Local repository root (default fallback)
REPO_ROOT = Path(__file__).resolve().parent

# Default Google Drive storage root (for Colab runs)
DRIVE_MOUNT_POINT = Path("/content/drive")
DRIVE_STORAGE_ROOT = Path("/content/drive/MyDrive/ModelShield")

# Global active storage paths (defaults to local repo root)
STORAGE_ROOT = REPO_ROOT
DATA_DIR = STORAGE_ROOT / "data"
CHECKPOINT_DIR = STORAGE_ROOT / "checkpoints"
OUTPUTS_DIR = STORAGE_ROOT / "outputs"
LOGS_DIR = STORAGE_ROOT / "logs"


def setup_storage(
    use_drive: Optional[bool] = None,
    custom_root: Optional[Path] = None,
) -> Tuple[Path, Path, Path, Path]:
    """
    Configure and create the project storage directory structure.

    Priority:
    1. custom_root if explicitly provided.
    2. Google Drive storage (/content/drive/MyDrive/ModelShield) if use_drive is True or if
       running in Colab with Drive mounted.
    3. Local repository root if running outside Colab or if use_drive is False.

    Raises:
        RuntimeError: If use_drive is True but Google Drive mount is not accessible.
    """
    global STORAGE_ROOT, DATA_DIR, CHECKPOINT_DIR, OUTPUTS_DIR, LOGS_DIR

    if custom_root is not None:
        STORAGE_ROOT = Path(custom_root)
    elif use_drive is True:
        if not DRIVE_MOUNT_POINT.exists():
            raise RuntimeError(
                "Google Drive persistence was requested, but /content/drive is not mounted.\n"
                "Please mount Google Drive first using:\n"
                "  from google.colab import drive\n"
                "  drive.mount('/content/drive')\n"
                "or pass use_drive=False to explicitly use temporary local storage."
            )
        STORAGE_ROOT = DRIVE_STORAGE_ROOT
    elif use_drive is False:
        STORAGE_ROOT = REPO_ROOT
    else:
        # Auto-detect: if in Colab and Drive is mounted, prefer Drive
        if is_colab() and (DRIVE_MOUNT_POINT / "MyDrive").exists():
            STORAGE_ROOT = DRIVE_STORAGE_ROOT
        else:
            STORAGE_ROOT = REPO_ROOT

    # Subdirectories
    DATA_DIR = REPO_ROOT / "data"  # Datasets stay in repo/data for fast I/O
    CHECKPOINT_DIR = STORAGE_ROOT / "checkpoints"
    OUTPUTS_DIR = STORAGE_ROOT / "outputs"
    LOGS_DIR = STORAGE_ROOT / "logs"

    # Automatically create directory structure
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    return DATA_DIR, CHECKPOINT_DIR, OUTPUTS_DIR, LOGS_DIR


# Initialize default local directory structure
setup_storage(use_drive=False)

# -----------------------------------------------------------------------------
# Checkpoint Naming Standard
# -----------------------------------------------------------------------------

# Baseline Models (Phase 2)
BASELINE_CLASSIFIER_BEST = "baseline_classifier_best.pth"
BASELINE_CLASSIFIER_LATEST = "latest_classifier.pth"
BASELINE_DETECTOR_BEST = "baseline_detector_best.pth"
BASELINE_DETECTOR_LATEST = "latest_detector.pth"

# Optimized Models (Phase 6 & 7)
OPTIMIZED_CLASSIFIER_BEST = "optimized_classifier_best.pth"
OPTIMIZED_CLASSIFIER_LATEST = "optimized_classifier_latest.pth"
OPTIMIZED_DETECTOR_BEST = "optimized_detector_best.pth"
OPTIMIZED_DETECTOR_LATEST = "optimized_detector_latest.pth"

# Legacy compatibility aliases
CLASSIFIER_CHECKPOINT_NAME = BASELINE_CLASSIFIER_BEST
DETECTOR_CHECKPOINT_NAME = BASELINE_DETECTOR_BEST
CLASSIFIER_CHECKPOINT_PATH = CHECKPOINT_DIR / BASELINE_CLASSIFIER_BEST
DETECTOR_CHECKPOINT_PATH = CHECKPOINT_DIR / BASELINE_DETECTOR_BEST

# -----------------------------------------------------------------------------
# Device & Hardware Diagnostics
# -----------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = get_device()

# -----------------------------------------------------------------------------
# Reproducibility & Determinism
# -----------------------------------------------------------------------------

RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds across Python, NumPy, and PyTorch for deterministic execution.
    Configures cuDNN deterministic flags when CUDA is active.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# CIFAR-10 Dataset Specifications
# -----------------------------------------------------------------------------

NUM_CLASSES = 10
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
IMAGE_SHAPE = (3, 32, 32)
VAL_SPLIT_RATIO = 0.2  # 10,000 validation images from 50,000 train set (40,000 train, 10,000 val)

# -----------------------------------------------------------------------------
# Historical Baseline Records (Recovered from Previous Colab Run)
# -----------------------------------------------------------------------------

HISTORICAL_CLASSIFIER = {
    "architecture": "Custom CIFAR10Classifier (3-Block CNN)",
    "epochs": 15,
    "final_train_acc": 0.7095,
    "final_val_acc": 0.7838,
    "final_train_loss": 0.8344,
    "final_val_loss": 0.6079,
    "best_epoch": 15,
    "note": "Historical reference recovered from previous Colab output; weights expired with session.",
}

HISTORICAL_DETECTOR = {
    "architecture": "Custom AdversarialDetector (3-Block CNN)",
    "epochs": 10,
    "final_train_acc": 0.9952,
    "best_val_acc": 0.9992,
    "final_val_acc": 0.9946,
    "final_train_loss": 0.0158,
    "final_val_loss": 0.0173,
    "best_epoch": 5,
    "note": "Historical reference recovered from previous Colab output; weights expired with session.",
}

HISTORICAL_FGSM = {
    0.00: {"clean_acc": 0.7912, "adv_acc": 0.7912, "asr": 0.0000},
    0.01: {"clean_acc": 0.7912, "adv_acc": 0.2226, "asr": 0.7187},
    0.03: {"clean_acc": 0.7912, "adv_acc": 0.0636, "asr": 0.9201},
}

# -----------------------------------------------------------------------------
# Baseline Training Hyperparameters (Preserved Unchanged)
# -----------------------------------------------------------------------------

BASELINE_CLASSIFIER_CONFIG = {
    "experiment_id": "baseline",
    "architecture": "Custom CNN (CIFAR10Classifier)",
    "batch_size": 64,
    "lr": 0.001,
    "optimizer": "Adam",
    "weight_decay": 1e-4,
    "epochs": 15,
    "loss_fn": "CrossEntropyLoss",
    "scheduler": None,
    "augmentation": "standard",  # RandomCrop(32, 4) + RandomHorizontalFlip
    "checkpoint_name": BASELINE_CLASSIFIER_BEST,
    "latest_checkpoint_name": BASELINE_CLASSIFIER_LATEST,
}

# Backward compatibility aliases
CLASSIFIER_BATCH_SIZE = BASELINE_CLASSIFIER_CONFIG["batch_size"]
CLASSIFIER_LR = BASELINE_CLASSIFIER_CONFIG["lr"]
CLASSIFIER_WEIGHT_DECAY = BASELINE_CLASSIFIER_CONFIG["weight_decay"]
CLASSIFIER_EPOCHS = BASELINE_CLASSIFIER_CONFIG["epochs"]

# Adversarial Attack Evaluation Settings
FGSM_EPSILONS = [0.00, 0.01, 0.03, 0.05, 0.10]
DEFAULT_EPSILON = 0.03

# Detector Hyperparameters
BASELINE_DETECTOR_CONFIG = {
    "experiment_id": "baseline_detector",
    "architecture": "Custom CNN (AdversarialDetector)",
    "batch_size": 64,
    "lr": 0.001,
    "optimizer": "Adam",
    "weight_decay": 1e-4,
    "epochs": 10,
    "loss_fn": "BCEWithLogitsLoss",
    "threshold": 0.5,
    "checkpoint_name": BASELINE_DETECTOR_BEST,
    "latest_checkpoint_name": BASELINE_DETECTOR_LATEST,
}

DETECTOR_BATCH_SIZE = BASELINE_DETECTOR_CONFIG["batch_size"]
DETECTOR_LR = BASELINE_DETECTOR_CONFIG["lr"]
DETECTOR_WEIGHT_DECAY = BASELINE_DETECTOR_CONFIG["weight_decay"]
DETECTOR_EPOCHS = BASELINE_DETECTOR_CONFIG["epochs"]
DETECTOR_THRESHOLD = BASELINE_DETECTOR_CONFIG["threshold"]

# -----------------------------------------------------------------------------
# Controlled CNN Classifier Optimization Experiments (Phase 6)
# Target: 85%+ validation accuracy while preserving the custom CNN architecture
# -----------------------------------------------------------------------------

OPTIMIZATION_EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "experiment_a": {
        "id": "Exp_A_Augmentation",
        "description": "Enhanced data augmentation (RandomCrop, Flip, mild ColorJitter, RandomRotation)",
        "epochs": 20,
        "batch_size": 64,
        "lr": 0.001,
        "optimizer": "Adam",
        "weight_decay": 1e-4,
        "scheduler": None,
        "augmentation": "enhanced",
        "label_smoothing": 0.0,
        "checkpoint_name": "exp_a_augmentation_best.pth",
    },
    "experiment_b": {
        "id": "Exp_B_LRScheduler",
        "description": "Cosine Annealing learning rate schedule (CosineAnnealingLR) with Adam",
        "epochs": 20,
        "batch_size": 64,
        "lr": 0.001,
        "optimizer": "Adam",
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR",
        "scheduler_params": {"T_max": 20, "eta_min": 1e-5},
        "augmentation": "standard",
        "label_smoothing": 0.0,
        "checkpoint_name": "exp_b_scheduler_best.pth",
    },
    "experiment_c": {
        "id": "Exp_C_Regularization",
        "description": "Regularization tuning: Label Smoothing (0.1) and tuned weight decay (5e-4)",
        "epochs": 20,
        "batch_size": 64,
        "lr": 0.001,
        "optimizer": "Adam",
        "weight_decay": 5e-4,
        "scheduler": None,
        "augmentation": "standard",
        "label_smoothing": 0.1,
        "checkpoint_name": "exp_c_regularization_best.pth",
    },
    "experiment_d": {
        "id": "Exp_D_CombinedBest",
        "description": "Synergistic combination of validated techniques (Enhanced Aug + Cosine LR + Tuned Decay)",
        "epochs": 25,
        "batch_size": 64,
        "lr": 0.001,
        "optimizer": "Adam",
        "weight_decay": 3e-4,
        "scheduler": "CosineAnnealingLR",
        "scheduler_params": {"T_max": 25, "eta_min": 1e-5},
        "augmentation": "enhanced",
        "label_smoothing": 0.05,
        "checkpoint_name": OPTIMIZED_CLASSIFIER_BEST,
        "latest_checkpoint_name": OPTIMIZED_CLASSIFIER_LATEST,
    },
    "experiment_e": {
        "id": "Exp_E_WeightInit",
        "description": "Custom CNN with Kaiming He normal weight initialization + Combined schedule",
        "epochs": 25,
        "batch_size": 64,
        "lr": 0.001,
        "optimizer": "Adam",
        "weight_decay": 3e-4,
        "scheduler": "CosineAnnealingLR",
        "scheduler_params": {"T_max": 25, "eta_min": 1e-5},
        "augmentation": "enhanced",
        "weight_init": "kaiming_normal",
        "label_smoothing": 0.05,
        "checkpoint_name": "exp_e_weight_init_best.pth",
    },
}

# -----------------------------------------------------------------------------
# Structured Logging Configuration
# -----------------------------------------------------------------------------

def setup_logger(name: str = "ModelShield", log_file: Optional[Path] = None) -> logging.Logger:
    """
    Initialize a structured logger that writes to both stdout and a log file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        if log_file is None:
            log_file = LOGS_DIR / "training.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

LOGGER = setup_logger()
