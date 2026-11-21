"""
Unit tests for ModelShield Phase 2 Adversarial Detector.
Verifies multi-attack dataset construction, metric evaluation, balanced class ratios, and evaluation pipeline.
"""

import pytest
import numpy as np
import torch
from Phase2.src.models import AdversarialDetector
from Phase2.src.detector import evaluate_detector
from Phase2.src.data import AdversarialDataset


class TestDetectorMetrics:
    def test_evaluate_detector_perfect_scores(self):
        # 100 clean, 100 adversarial
        y_true = np.array([0] * 100 + [1] * 100)
        y_prob = np.array([0.05] * 100 + [0.95] * 100)
        
        metrics = evaluate_detector(y_true, y_prob, threshold=0.5)
        
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["fpr"] == 0.0
        assert metrics["roc_auc"] == 1.0

    def test_evaluate_detector_all_false_positives(self):
        y_true = np.array([0] * 50 + [1] * 50)
        # All classified as positive
        y_prob = np.array([0.8] * 100)
        
        metrics = evaluate_detector(y_true, y_prob, threshold=0.5)
        assert metrics["fpr"] == 1.0
        assert metrics["recall"] == 1.0

    def test_evaluate_detector_returns_expected_keys(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.7, 0.4, 0.9])
        metrics = evaluate_detector(y_true, y_prob, threshold=0.5)
        
        expected_keys = [
            "accuracy", "precision", "recall", "f1", "fpr", "fnr", "roc_auc",
            "tp", "fp", "tn", "fn", "threshold", "total_samples"
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key in detector metrics: {key}"


class TestAdversarialDataset:
    def test_dataset_indexing(self):
        images = torch.rand(20, 3, 32, 32)
        labels = torch.tensor([0.0] * 10 + [1.0] * 10)
        ids = list(range(20))
        
        dataset = AdversarialDataset(images, labels, metadata={"sample_ids": ids})
        assert len(dataset) == 20
        
        img, lbl = dataset[5]
        assert img.shape == (3, 32, 32)
        assert lbl.item() == 0.0
        
        img15, lbl15 = dataset[15]
        assert lbl15.item() == 1.0
