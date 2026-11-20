"""
ModelShield - Configuration Module
Centralized settings, paths, hyperparameters, and reproducibility utilities.
"""

from pathlib import Path
import random
import numpy as np
import torch

# Base Directory Resolution
BASE_DIR = Path(__file__).resolve().parent

# Paths
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoint File Paths
CLASSIFIER_CHECKPOINT_NAME = "best_classifier.pth"
DETECTOR_CHECKPOINT_NAME = "best_detector.pth"
CLASSIFIER_CHECKPOINT_PATH = CHECKPOINT_DIR / CLASSIFIER_CHECKPOINT_NAME
DETECTOR_CHECKPOINT_PATH = CHECKPOINT_DIR / DETECTOR_CHECKPOINT_NAME

# Device Configuration
def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = get_device()

# Reproducibility Seed
RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed across Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# CIFAR-10 Dataset Configuration
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
VAL_SPLIT_RATIO = 0.2  # 10,000 validation images from 50,000 train set

# Classifier Training Hyperparameters
CLASSIFIER_BATCH_SIZE = 64
CLASSIFIER_LR = 0.001
CLASSIFIER_WEIGHT_DECAY = 1e-4
CLASSIFIER_EPOCHS = 15

# FGSM Adversarial Attack Settings
FGSM_EPSILONS = [0.01, 0.03, 0.05, 0.10]
DEFAULT_EPSILON = 0.03

# Adversarial Detector Hyperparameters
DETECTOR_BATCH_SIZE = 64
DETECTOR_LR = 0.001
DETECTOR_WEIGHT_DECAY = 1e-4
DETECTOR_EPOCHS = 10
DETECTOR_THRESHOLD = 0.5
