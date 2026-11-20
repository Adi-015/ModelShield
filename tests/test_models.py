"""
ModelShield - Test Suite
Automated unit tests covering models, preprocessing, FGSM attacks, and firewall logic.
All tests run on CPU without requiring pre-trained weights or full dataset downloads.
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path for direct pytest invocation
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image
import numpy as np
import pytest
import torch
import torch.nn as nn

import config
from src.attacks import fgsm_attack
from src.data import denormalize_tensor, normalize_tensor
from src.firewall import ModelShieldFirewall
from src.models import AdversarialDetector, CIFAR10Classifier


@pytest.fixture
def cpu_device() -> torch.device:
    """Ensure tests run consistently on CPU."""
    return torch.device("cpu")


@pytest.fixture
def dummy_batch(cpu_device) -> torch.Tensor:
    """Generate a reproducible synthetic batch of normalized CIFAR-10 images."""
    torch.manual_seed(42)
    return torch.randn(4, 3, 32, 32, device=cpu_device)


@pytest.fixture
def dummy_labels(cpu_device) -> torch.Tensor:
    """Generate sample labels for 4 images."""
    return torch.tensor([0, 3, 5, 9], device=cpu_device)


# -------------------------------------------------------------------------
# Test 1: Classifier Output Shape & Behavior
# -------------------------------------------------------------------------

def test_classifier_output_shape(dummy_batch):
    """Verify that CIFAR10Classifier outputs raw logits of shape (B, 10)."""
    model = CIFAR10Classifier(num_classes=config.NUM_CLASSES)
    model.eval()

    with torch.no_grad():
        logits = model(dummy_batch)

    assert logits.shape == (4, 10), f"Expected shape (4, 10), got {logits.shape}"
    assert torch.is_tensor(logits)
    # Check that outputs are raw logits (unbounded real numbers, not probabilities)
    assert not torch.allclose(logits.sum(dim=1), torch.ones(4)), "Logits should not sum to 1.0"


# -------------------------------------------------------------------------
# Test 2: Detector Output Shape & Probability
# -------------------------------------------------------------------------

def test_detector_output_shape(dummy_batch):
    """Verify that AdversarialDetector outputs binary logits of shape (B, 1)."""
    detector = AdversarialDetector()
    detector.eval()

    with torch.no_grad():
        logits = detector(dummy_batch)

    assert logits.shape == (4, 1), f"Expected shape (4, 1), got {logits.shape}"
    probs = torch.sigmoid(logits)
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Sigmoid probabilities must be in [0, 1]"


# -------------------------------------------------------------------------
# Test 3: Image Preprocessing and Normalization / Denormalization
# -------------------------------------------------------------------------

def test_normalization_round_trip():
    """Verify that denormalize(normalize(x)) returns original x within [0, 1]."""
    torch.manual_seed(42)
    raw_images = torch.rand(2, 3, 32, 32)  # [0, 1] range

    normed = normalize_tensor(raw_images)
    recovered = denormalize_tensor(normed)

    assert torch.allclose(raw_images, recovered, atol=1e-5), "Round-trip normalization failed"
    assert (recovered >= 0.0).all() and (recovered <= 1.0).all(), "Recovered values must be in [0, 1]"


def test_firewall_image_preprocessing():
    """Verify that preprocess_image converts PIL images to normalized shape (1, 3, 32, 32)."""
    # Create arbitrary resolution PIL image
    pil_img = Image.new("RGB", (128, 128), color=(100, 150, 200))
    tensor = ModelShieldFirewall.preprocess_image(pil_img)

    assert tensor.shape == (1, 3, 32, 32), f"Expected shape (1, 3, 32, 32), got {tensor.shape}"
    assert tensor.dtype == torch.float32


# -------------------------------------------------------------------------
# Test 4: FGSM Adversarial Attack Output Validity
# -------------------------------------------------------------------------

def test_fgsm_output_validity(dummy_batch, dummy_labels, cpu_device):
    """Verify FGSM attack output shape, mathematical bounds, and clipping."""
    model = CIFAR10Classifier()
    epsilon = 0.03

    adv_batch = fgsm_attack(
        model=model,
        images=dummy_batch,
        labels=dummy_labels,
        epsilon=epsilon,
    )

    # Output shape must match input
    assert adv_batch.shape == dummy_batch.shape

    # Check bounds in raw physical pixel space [0, 1]
    raw_orig = denormalize_tensor(dummy_batch)
    raw_adv = denormalize_tensor(adv_batch)

    # Valid pixel range
    assert (raw_adv >= 0.0).all() and (raw_adv <= 1.0).all(), "Adversarial pixels must be in [0, 1]"

    # Perturbation magnitude must not exceed epsilon
    diff = torch.abs(raw_adv - raw_orig)
    assert diff.max().item() <= (epsilon + 1e-5), f"Perturbation {diff.max().item()} exceeded epsilon {epsilon}"


def test_fgsm_zero_epsilon(dummy_batch, dummy_labels):
    """Verify that FGSM with epsilon=0 returns identical tensor."""
    model = CIFAR10Classifier()
    adv_batch = fgsm_attack(model, dummy_batch, dummy_labels, epsilon=0.0)
    assert torch.allclose(adv_batch, dummy_batch), "Epsilon 0 must produce identical images"


# -------------------------------------------------------------------------
# Test 5: Firewall Decision Logic and Thresholding
# -------------------------------------------------------------------------

class MockDetector(nn.Module):
    """Mock detector returning a fixed adversarial probability logit."""

    def __init__(self, fixed_prob: float):
        super().__init__()
        # Invert sigmoid: logit = ln(p / (1 - p))
        fixed_prob = max(1e-6, min(1.0 - 1e-6, fixed_prob))
        self.logit_val = float(np.log(fixed_prob / (1.0 - fixed_prob)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return torch.full((batch_size, 1), self.logit_val, dtype=torch.float32, device=x.device)


def test_firewall_blocks_suspicious_input():
    """Firewall must return BLOCKED when adversarial probability >= threshold."""
    classifier = CIFAR10Classifier()
    mock_detector = MockDetector(fixed_prob=0.85)

    firewall = ModelShieldFirewall(
        classifier=classifier,
        detector=mock_detector,
        classifier_path=None,
        detector_path=None,
        threshold=0.50,
        device=torch.device("cpu"),
    )

    dummy_input = torch.randn(1, 3, 32, 32)
    report = firewall.process(dummy_input)

    assert report["decision"] == "BLOCKED"
    assert report["is_adversarial"] is True
    assert report["adversarial_probability"] >= 0.50
    assert report["predicted_class_name"] is None
    assert report["classification_confidence"] is None


def test_firewall_allows_clean_input():
    """Firewall must return ALLOWED and invoke classifier when adversarial probability < threshold."""
    classifier = CIFAR10Classifier()
    mock_detector = MockDetector(fixed_prob=0.15)

    firewall = ModelShieldFirewall(
        classifier=classifier,
        detector=mock_detector,
        classifier_path=None,
        detector_path=None,
        threshold=0.50,
        device=torch.device("cpu"),
    )

    dummy_input = torch.randn(1, 3, 32, 32)
    report = firewall.process(dummy_input)

    assert report["decision"] == "ALLOWED"
    assert report["is_adversarial"] is False
    assert report["adversarial_probability"] < 0.50
    assert report["predicted_class_name"] in config.CIFAR10_CLASSES
    assert 0.0 <= report["classification_confidence"] <= 1.0


def test_firewall_custom_threshold_override():
    """Firewall must respect custom threshold overrides during process()."""
    classifier = CIFAR10Classifier()
    mock_detector = MockDetector(fixed_prob=0.45)

    firewall = ModelShieldFirewall(
        classifier=classifier,
        detector=mock_detector,
        classifier_path=None,
        detector_path=None,
        threshold=0.50,  # Default: 0.45 < 0.50 -> ALLOWED
        device=torch.device("cpu"),
    )

    dummy_input = torch.randn(1, 3, 32, 32)

    # Default threshold -> ALLOWED
    report_default = firewall.process(dummy_input)
    assert report_default["decision"] == "ALLOWED"

    # Stricter threshold 0.40 -> BLOCKED
    report_strict = firewall.process(dummy_input, custom_threshold=0.40)
    assert report_strict["decision"] == "BLOCKED"
