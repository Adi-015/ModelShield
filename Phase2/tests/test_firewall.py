"""
Unit tests for ModelShield Phase 2 AI Firewall.
Verifies threshold calibration (FPR <= 1%), single/batch inspection, latency profiling, and fail-safe logic.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from Phase2.src.models import CIFAR10Classifier, AdversarialDetector
from Phase2.src.firewall import (
    ModelShieldFirewall,
    calibrate_threshold,
    profile_firewall_latency
)


class MockDetector(nn.Module):
    def __init__(self, fixed_score: float = 0.05):
        super().__init__()
        p = float(np.clip(fixed_score, 1e-6, 1.0 - 1e-6))
        self.fixed_logit = float(np.log(p / (1.0 - p)))

    def forward(self, x):
        return torch.full((x.shape[0], 1), self.fixed_logit, device=x.device)


class TestCalibration:
    def test_calibrate_threshold_target_fpr(self):
        np.random.seed(42)
        # 1000 clean scores in [0.0, 0.2]
        clean_scores = np.random.uniform(0.0, 0.2, 1000)
        # 1000 adversarial scores in [0.3, 1.0]
        adv_scores = np.random.uniform(0.3, 1.0, 1000)
        
        target_fpr = 0.01
        calib = calibrate_threshold(clean_scores, adv_scores, target_fpr=target_fpr)
        
        assert "optimal_threshold" in calib
        assert calib["calibrated_fpr"] <= target_fpr + 1e-4, f"FPR exceeded target: {calib['calibrated_fpr']}"
        assert calib["optimal_threshold"] > 0.0
        assert calib["optimal_threshold"] < 1.0

    def test_calibrate_threshold_fallback(self):
        # Case where all clean scores are high
        clean_scores = np.ones(100)
        adv_scores = np.ones(100)
        calib = calibrate_threshold(clean_scores, adv_scores, target_fpr=0.01)
        assert calib["optimal_threshold"] >= 0.5


class TestModelShieldFirewall:
    @pytest.fixture
    def setup_firewall(self):
        classifier = CIFAR10Classifier(num_classes=10)
        classifier.eval()
        detector = AdversarialDetector(input_channels=3)
        detector.eval()
        firewall = ModelShieldFirewall(
            classifier=classifier,
            detector=detector,
            threshold=0.5,
            device=torch.device("cpu")
        )
        return firewall, classifier, detector

    def test_single_inspect_allowed(self, setup_firewall):
        firewall, classifier, _ = setup_firewall
        # Replace detector with one returning score 0.05 (< 0.5)
        firewall.detector = MockDetector(fixed_score=0.05)
        
        img = torch.rand(1, 3, 32, 32)
        decision = firewall.inspect(img)
        
        assert decision["is_blocked"] is False
        assert decision["action"] == "ALLOWED"
        assert decision["predicted_class"] is not None
        assert decision["detector_score"] == pytest.approx(0.05, abs=1e-3)
        assert decision["latency_ms"] > 0

    def test_single_inspect_blocked(self, setup_firewall):
        firewall, _, _ = setup_firewall
        # Replace detector with one returning score 0.95 (>= 0.5)
        firewall.detector = MockDetector(fixed_score=0.95)
        
        img = torch.rand(1, 3, 32, 32)
        decision = firewall.inspect(img)
        
        assert decision["is_blocked"] is True
        assert decision["action"] == "BLOCKED"
        assert decision["predicted_class"] is None
        assert decision["detector_score"] == pytest.approx(0.95, abs=1e-3)

    def test_batch_inspect(self, setup_firewall):
        firewall, _, _ = setup_firewall
        firewall.detector = MockDetector(fixed_score=0.2)  # All allowed
        
        images = torch.rand(6, 3, 32, 32)
        batch_res = firewall.batch_inspect(images)
        
        assert batch_res["total_inspected"] == 6
        assert batch_res["passed_count"] == 6
        assert batch_res["blocked_count"] == 0
        assert len(batch_res["decisions"]) == 6

    def test_latency_profiling(self, setup_firewall):
        firewall, _, _ = setup_firewall
        profile = profile_firewall_latency(firewall, input_shape=(1, 3, 32, 32), iterations=5)
        
        assert "classifier_mean_ms" in profile
        assert "detector_mean_ms" in profile
        assert "total_mean_ms" in profile
        assert "throughput_fps" in profile
        assert profile["total_mean_ms"] > 0
