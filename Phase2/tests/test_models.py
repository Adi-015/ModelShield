"""
Unit tests for ModelShield Phase 2 model architectures.
Verifies shapes, parameter bounds, weight initialization, and evaluation modes.
"""

import pytest
import torch
import torch.nn as nn
from Phase2.src.models import (
    CIFAR10Classifier,
    AdversarialDetector,
    initialize_weights,
    count_parameters
)


class TestCIFAR10Classifier:
    def test_forward_output_shape(self):
        model = CIFAR10Classifier(num_classes=10)
        model.eval()
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 10), f"Expected shape (4, 10), got {out.shape}"

    def test_single_sample_shape(self):
        model = CIFAR10Classifier(num_classes=10)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 10)

    def test_parameter_count(self):
        model = CIFAR10Classifier(num_classes=10)
        params = count_parameters(model)
        # 3-block CNN with 32, 64, 128 channels and FC 512, 10
        assert params > 100_000 and params < 5_000_000, f"Unexpected parameter count: {params}"

    def test_custom_num_classes(self):
        model = CIFAR10Classifier(num_classes=5)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 5)

    def test_initialize_weights(self):
        model = CIFAR10Classifier(num_classes=10)
        # Should execute without errors
        model.apply(initialize_weights)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)


class TestAdversarialDetector:
    def test_forward_output_shape_and_range(self):
        detector = AdversarialDetector(input_channels=3)
        detector.eval()
        x = torch.rand(4, 3, 32, 32)
        with torch.no_grad():
            out = detector(x)
            probs = detector.predict_proba(x)
        assert out.shape == (4, 1), f"Expected shape (4, 1), got {out.shape}"
        assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Detector predict_proba must be in [0, 1]"

    def test_single_sample(self):
        detector = AdversarialDetector(input_channels=3)
        detector.eval()
        x = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            out = detector(x)
        assert out.shape == (1, 1)

    def test_parameter_count(self):
        detector = AdversarialDetector(input_channels=3)
        params = count_parameters(detector)
        assert params > 100_000 and params < 5_000_000, f"Unexpected parameter count: {params}"
