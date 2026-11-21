"""
ModelShield Phase 2 - Centralized Configuration & Experiment Registry
Platform-independent configuration for improved adversarial evaluation, multi-attack engines (FGSM + PGD),
detector generalization experiments, and calibrated pre-inference firewall gateways.
"""

from pathlib import Path
import os
import sys
import json
import random
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import torch

# -----------------------------------------------------------------------------
# Environment Detection & Storage Hierarchy
# -----------------------------------------------------------------------------

def is_colab() -> bool:
    """Detect whether current runtime environment is Google Colab."""
    return "google.colab" in sys.modules or os.path.exists("/content")

# Root directory of Phase 2
PHASE2_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PHASE2_ROOT.parent

# Read-only reference path to Phase 1 baseline weights (if available)
PHASE1_BASELINE_CLASSIFIER_PATH = REPO_ROOT / "checkpoints" / "baseline_classifier_best.pth"
PHASE1_BASELINE_DETECTOR_PATH = REPO_ROOT / "checkpoints" / "baseline_detector_best.pth"

# Google Drive storage root for Phase 2 (strictly separate from Phase 1)
DRIVE_MOUNT_POINT = Path("/content/drive")
DRIVE_STORAGE_ROOT = Path("/content/drive/MyDrive/ModelShield_Phase2")

# Active Storage Directories (defaults to local Phase2 root)
STORAGE_ROOT = PHASE2_ROOT
BASE_DIR = STORAGE_ROOT
DATA_DIR = REPO_ROOT / "data"  # Reuse shared raw dataset directory
CHECKPOINT_DIR = STORAGE_ROOT / "checkpoints"
RESULTS_DIR = STORAGE_ROOT / "results"
LOGS_DIR = STORAGE_ROOT / "logs"

# Sub-results directories
RESULTS_CLASSIFIER = RESULTS_DIR / "classifier"
RESULTS_DETECTOR = RESULTS_DIR / "detector"
RESULTS_ATTACKS = RESULTS_DIR / "attacks"
RESULTS_FIREWALL = RESULTS_DIR / "firewall"

# Dynamic Checkpoint Paths & Log Files
CLASSIFIER_CHECKPOINT_PATH = CHECKPOINT_DIR / "phase2_classifier_baseline_best.pth"
DETECTOR_CHECKPOINT_PATH = CHECKPOINT_DIR / "phase2_detector_multi_best.pth"
CLASSIFIER_CHECKPOINT_BEST = CLASSIFIER_CHECKPOINT_PATH
DETECTOR_CHECKPOINT_BEST = DETECTOR_CHECKPOINT_PATH
FIREWALL_THRESHOLD_FILE = RESULTS_FIREWALL / "calibrated_threshold.json"
CLASSIFIER_HISTORY_FILE = LOGS_DIR / "classifier_history.json"
DETECTOR_HISTORY_FILE = LOGS_DIR / "detector_history.json"
TRAINING_LOG_FILE = LOGS_DIR / "phase2_training.log"


def setup_storage(
    use_drive: Optional[bool] = None,
    custom_root: Optional[Path] = None,
    base_dir: Optional[Path] = None,
) -> Tuple[Path, Path, Path, Path]:
    """
    Configure and create the Phase 2 project storage hierarchy.
    Dynamically refreshes all path aliases to ensure paths stay synchronized.

    Priority:
    1. custom_root or base_dir if explicitly supplied.
    2. Google Drive storage (/content/drive/MyDrive/ModelShield_Phase2) if use_drive is True
       or running in Colab with Drive mounted.
    3. Local Phase 2 directory if running outside Colab or use_drive is False.
    """
    global STORAGE_ROOT, BASE_DIR, DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR
    global RESULTS_CLASSIFIER, RESULTS_DETECTOR, RESULTS_ATTACKS, RESULTS_FIREWALL
    global CLASSIFIER_CHECKPOINT_PATH, DETECTOR_CHECKPOINT_PATH
    global CLASSIFIER_CHECKPOINT_BEST, DETECTOR_CHECKPOINT_BEST, FIREWALL_THRESHOLD_FILE
    global CLASSIFIER_HISTORY_FILE, DETECTOR_HISTORY_FILE, TRAINING_LOG_FILE

    target_root = custom_root if custom_root is not None else base_dir

    if target_root is not None:
        STORAGE_ROOT = Path(target_root)
    elif use_drive is True:
        if not DRIVE_MOUNT_POINT.exists():
            raise RuntimeError(
                "Google Drive persistence was requested for Phase 2, but /content/drive is not mounted.\n"
                "Please mount Google Drive first via:\n"
                "  from google.colab import drive\n"
                "  drive.mount('/content/drive')\n"
                "or pass use_drive=False to use temporary runtime storage."
            )
        STORAGE_ROOT = DRIVE_STORAGE_ROOT
    elif use_drive is False:
        STORAGE_ROOT = PHASE2_ROOT
    else:
        if is_colab() and (DRIVE_MOUNT_POINT / "MyDrive").exists():
            STORAGE_ROOT = DRIVE_STORAGE_ROOT
        else:
            STORAGE_ROOT = PHASE2_ROOT

    BASE_DIR = STORAGE_ROOT

    # Subdirectories
    DATA_DIR = REPO_ROOT / "data"
    CHECKPOINT_DIR = STORAGE_ROOT / "checkpoints"
    RESULTS_DIR = STORAGE_ROOT / "results"
    LOGS_DIR = STORAGE_ROOT / "logs"

    RESULTS_CLASSIFIER = RESULTS_DIR / "classifier"
    RESULTS_DETECTOR = RESULTS_DIR / "detector"
    RESULTS_ATTACKS = RESULTS_DIR / "attacks"
    RESULTS_FIREWALL = RESULTS_DIR / "firewall"

    # Refresh full checkpoint paths
    CLASSIFIER_CHECKPOINT_PATH = CHECKPOINT_DIR / "phase2_classifier_baseline_best.pth"
    DETECTOR_CHECKPOINT_PATH = CHECKPOINT_DIR / "phase2_detector_multi_best.pth"
    CLASSIFIER_CHECKPOINT_BEST = CLASSIFIER_CHECKPOINT_PATH
    DETECTOR_CHECKPOINT_BEST = DETECTOR_CHECKPOINT_PATH
    FIREWALL_THRESHOLD_FILE = RESULTS_FIREWALL / "calibrated_threshold.json"
    CLASSIFIER_HISTORY_FILE = LOGS_DIR / "classifier_history.json"
    DETECTOR_HISTORY_FILE = LOGS_DIR / "detector_history.json"
    TRAINING_LOG_FILE = LOGS_DIR / "phase2_training.log"

    # Auto-create all required directory branches
    for d in (
        DATA_DIR,
        CHECKPOINT_DIR,
        RESULTS_DIR,
        LOGS_DIR,
        RESULTS_CLASSIFIER,
        RESULTS_DETECTOR,
        RESULTS_ATTACKS,
        RESULTS_FIREWALL,
    ):
        d.mkdir(parents=True, exist_ok=True)

    return DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR


def to_json_serializable(val: Any) -> Any:
    """Recursively convert PyTorch tensors, NumPy arrays/scalars, and Paths to JSON-serializable types."""
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().tolist()
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.floating, float)):
        return float(val)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, Path):
        return val.as_posix()
    if isinstance(val, dict):
        return {str(k): to_json_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [to_json_serializable(v) for v in val]
    return val


def save_json(data: Any, filepath: Union[str, Path]) -> None:
    """Save serializable data to JSON file, automatically converting tensors/arrays and creating directories."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    serializable_data = to_json_serializable(data)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=2)


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load and return JSON dictionary from file."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    logger_name: str = "ModelShield.Phase2",
) -> logging.Logger:
    """
    Configure file-based and stream logging avoiding duplicate handlers when rerun.
    Writes persistent logs to config.LOGS_DIR / 'phase2_training.log' by default.
    """
    target_log_file = Path(log_file) if log_file is not None else TRAINING_LOG_FILE
    target_log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    has_file_handler = False
    has_stream_handler = False
    resolved_target = target_log_file.resolve()

    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            try:
                if Path(handler.baseFilename).resolve() == resolved_target:
                    has_file_handler = True
                else:
                    logger.removeHandler(handler)
            except Exception:
                logger.removeHandler(handler)
        elif isinstance(handler, logging.StreamHandler):
            has_stream_handler = True

    if not has_file_handler:
        fh = logging.FileHandler(str(target_log_file), mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if not has_stream_handler:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    logger.propagate = False
    return logger


# Initialize default local directory structure
setup_storage(use_drive=False)

# -----------------------------------------------------------------------------
# Checkpoint Naming Standard
# -----------------------------------------------------------------------------

# Classifier Checkpoints
CLASSIFIER_BASELINE_BEST = "phase2_classifier_baseline_best.pth"
CLASSIFIER_BASELINE_LATEST = "phase2_classifier_baseline_latest.pth"
CLASSIFIER_OPTIMIZED_BEST = "phase2_classifier_optimized_best.pth"
CLASSIFIER_OPTIMIZED_LATEST = "phase2_classifier_optimized_latest.pth"

# Detector Checkpoints
DETECTOR_FGSM_BEST = "phase2_detector_fgsm_best.pth"
DETECTOR_MULTI_BEST = "phase2_detector_multi_best.pth"
DETECTOR_MULTI_LATEST = "phase2_detector_multi_latest.pth"

# -----------------------------------------------------------------------------
# Device Selection & Hardware Diagnostics
# -----------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = get_device()

# -----------------------------------------------------------------------------
# Reproducibility & Deterministic Settings
# -----------------------------------------------------------------------------

RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set seeds across Python, NumPy, and PyTorch for deterministic execution.
    Configures deterministic cuDNN settings.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything = set_seed


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker initialization function to guarantee distinct,
    deterministic seeds across worker subprocesses.
    """
    worker_seed = (RANDOM_SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: int = RANDOM_SEED) -> torch.Generator:
    """Return a deterministic PyTorch Generator initialized with the specified seed."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen

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
IMAGE_CHANNELS = 3
NUM_CLASSES = 10
VAL_SPLIT_RATIO = 0.2  # 10,000 validation images from 50,000 train set (40,000 train, 10,000 val)

# -----------------------------------------------------------------------------
# Baseline Classifier Hyperparameters (Phase 1 Preserved Baseline)
# -----------------------------------------------------------------------------

CLASSIFIER_BATCH_SIZE = 64
CLASSIFIER_LR = 0.001
CLASSIFIER_WEIGHT_DECAY = 1e-4
CLASSIFIER_EPOCHS = 15

# Optimized Classifier Hyperparameters (Exp D)
OPTIMIZED_CLASSIFIER_CONFIG = {
    "experiment_id": "phase2_classifier_optimized",
    "epochs": 25,
    "batch_size": 64,
    "lr": 0.001,
    "weight_decay": 3e-4,
    "scheduler": "CosineAnnealingLR",
    "scheduler_params": {"T_max": 25, "eta_min": 1e-5},
    "augmentation": "enhanced",
    "label_smoothing": 0.05,
    "checkpoint_name": CLASSIFIER_OPTIMIZED_BEST,
    "latest_checkpoint_name": CLASSIFIER_OPTIMIZED_LATEST,
}

# -----------------------------------------------------------------------------
# Adversarial Attack Engine Configurations (Phase 2)
# -----------------------------------------------------------------------------

# Expanded Epsilon Spectrum (including clean epsilon = 0.0)
ATTACK_EPSILONS: List[float] = [0.0, 0.001, 0.005, 0.01, 0.02, 0.03]
DEFAULT_EPSILON = 0.03

# Projected Gradient Descent (PGD) Default Parameters
PGD_DEFAULT_PARAMS: Dict[str, Any] = {
    "epsilon": 0.03,
    "alpha": 0.005,      # Step size (alpha approx epsilon / 6)
    "num_iter": 10,      # Iteration count
    "random_start": True,# Uniform random initialization within L_inf ball
}

# -----------------------------------------------------------------------------
# Multi-Attack Adversarial Detector Hyperparameters (Phase 2)
# -----------------------------------------------------------------------------

DETECTOR_BATCH_SIZE = 64
DETECTOR_LR = 0.001
DETECTOR_WEIGHT_DECAY = 1e-4
DETECTOR_EPOCHS = 10
DETECTOR_DEFAULT_THRESHOLD = 0.50

# Multi-Attack Detector Training Mixture (Phase 2 core improvement)
DETECTOR_TRAIN_MIXTURE: Dict[str, Any] = {
    "clean_ratio": 0.50,            # 50% clean images (Class 0)
    "fgsm_ratio": 0.25,             # 25% FGSM images (Class 1) across varied epsilons
    "pgd_ratio": 0.25,              # 25% PGD images (Class 1) across varied epsilons
    "fgsm_epsilons": [0.005, 0.01, 0.02, 0.03],
    "pgd_epsilons": [0.005, 0.01, 0.02, 0.03],
    "pgd_iters": 7,
}

# Generalization Evaluation Benchmark Configurations
GENERALIZATION_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "exp_a_in_distribution": {
        "name": "In-Distribution Benchmark",
        "description": "Evaluation on same attack configurations seen during training",
    },
    "exp_b_unseen_epsilon": {
        "name": "Unseen Epsilon Generalization",
        "description": "Evaluation on held-out epsilon values (e.g. 0.001, 0.015, 0.025)",
        "epsilons": [0.001, 0.015, 0.025],
    },
    "exp_c_unseen_attack": {
        "name": "Unseen Attack Family Generalization",
        "description": "Evaluate detector trained only on FGSM against iterative PGD attacks",
    },
    "exp_d_clean_test_set": {
        "name": "Clean Test Set Verification",
        "description": "Evaluate True Negative Rate and FPR on the full 10,000 clean CIFAR-10 test set",
    },
}

# -----------------------------------------------------------------------------
# AI Firewall Gateway & Dynamic Threshold Calibration
# -----------------------------------------------------------------------------

FIREWALL_CONFIG: Dict[str, Any] = {
    "target_fpr": 0.01,             # Target 1.0% maximum False Positive Rate on validation data
    "default_threshold": 0.50,      # Initial fallback cutoff
    "calibrated_threshold": None,   # Dynamically populated via validation ROC analysis
}

# -----------------------------------------------------------------------------
# Structured Logger Configuration
# -----------------------------------------------------------------------------

def setup_logger(name: str = "ModelShield_Phase2", log_file: Optional[Path] = None) -> logging.Logger:
    """Initialize structured logger writing to stdout and log file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file is None:
            log_file = LOGS_DIR / "phase2_training.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

LOGGER = setup_logger()
