"""
Phase 2 Integration Repairs Test Suite.
Validates the resolution of pre-training audit integration blockers:
  - Configuration interface compatibility (BASE_DIR, setup_storage kwargs, save_json/load_json)
  - Attack gradient isolation (zero parameter gradient leak)
  - Attack visualization function export
  - Firewall batch filtering without classifier leakage on blocked items
  - Classification loss metric tracking in evaluation
  - Firewall defense signature polymorphism
  - Detector generalization metric dictionary conformity
  - Checkpoint loader argument order flexibility
  - Multi-attack dataset construction balance
"""

import tempfile
import json
from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Phase2 import config
from Phase2.src.models import CIFAR10Classifier, AdversarialDetector
from Phase2.src.attacks import fgsm_attack, pgd_attack, plot_attack_visualizations
from Phase2.src.evaluate import (
    evaluate_classifier,
    evaluate_firewall_defense,
    evaluate_attack_spectrum,
)
from Phase2.src.detector import build_multi_attack_dataset, evaluate_detector_generalization
from Phase2.src.firewall import ModelShieldFirewall, calibrate_threshold
from Phase2.src.train import load_checkpoint, save_checkpoint


def test_config_interface_and_json_helpers():
    """Verify config exposes required attributes and handles setup_storage kwargs and JSON I/O."""
    assert hasattr(config, "BASE_DIR")
    assert hasattr(config, "CLASSIFIER_CHECKPOINT_BEST")
    assert hasattr(config, "DETECTOR_CHECKPOINT_BEST")
    assert hasattr(config, "FIREWALL_THRESHOLD_FILE")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test_drive"
        # Test base_dir keyword argument
        config.setup_storage(base_dir=tmp_path, use_drive=True)
        assert config.STORAGE_ROOT == tmp_path
        assert config.BASE_DIR == tmp_path
        assert config.CHECKPOINT_DIR.parent == tmp_path

        # Test save_json and load_json helpers
        test_file = tmp_path / "test.json"
        payload = {"model": "ModelShield", "version": 2, "metric": 0.99}
        config.save_json(payload, test_file)
        assert test_file.exists()

        loaded = config.load_json(test_file)
        assert loaded == payload

        # Reset storage to local
        config.setup_storage(use_drive=False)


def test_attack_parameter_gradients_isolation():
    """Verify attacks do not populate or leave gradients in classifier parameters."""
    device = torch.device("cpu")
    model = CIFAR10Classifier().to(device)
    model.eval()

    images = torch.rand(4, 3, 32, 32, device=device)
    labels = torch.tensor([0, 1, 2, 3], device=device)

    # Test FGSM
    _ = fgsm_attack(model, images, labels, epsilon=0.03)
    for name, param in model.named_parameters():
        assert param.grad is None, f"Parameter {name} retained gradients after FGSM attack!"

    # Test PGD
    _ = pgd_attack(model, images, labels, epsilon=0.03, alpha=0.005, num_iter=5)
    for name, param in model.named_parameters():
        assert param.grad is None, f"Parameter {name} retained gradients after PGD attack!"


def test_attack_visualizations_export():
    """Verify plot_attack_visualizations generates expected figure and exports cleanly."""
    device = torch.device("cpu")
    model = CIFAR10Classifier().to(device)
    clean_images = torch.rand(4, 3, 32, 32, device=device)
    adv_images = torch.clamp(clean_images + 0.03, 0.0, 1.0)
    labels = torch.tensor([0, 1, 2, 3], device=device)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "vis_test.png"
        fig = plot_attack_visualizations(
            clean_images=clean_images,
            adv_images=adv_images,
            labels=labels,
            model=model,
            sample_indices=[0, 1],
            attack_name="Test Attack",
            save_path=save_path,
        )
        assert fig is not None
        assert save_path.exists()
        assert save_path.stat().st_size > 0


def test_firewall_batch_classifier_bypassing():
    """Verify firewall process_batch never forwards blocked images to the downstream classifier."""
    device = torch.device("cpu")

    class SpyClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3 * 32 * 32, 10)
            self.forward_call_count = 0
            self.samples_processed = 0

        def forward(self, x):
            self.forward_call_count += 1
            self.samples_processed += x.size(0)
            return self.linear(x.view(x.size(0), -1))

    # Detector that flags first 2 samples as adversarial (prob 0.9) and last 2 as clean (prob 0.1)
    class FixedDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3 * 32 * 32, 1)

        def forward(self, x):
            # return large logits for idx 0, 1 and negative logits for idx 2, 3
            batch_size = x.size(0)
            logits = torch.zeros(batch_size, 1)
            for i in range(batch_size):
                logits[i, 0] = 5.0 if i < 2 else -5.0
            return logits

    spy_classifier = SpyClassifier()
    fixed_detector = FixedDetector()

    firewall = ModelShieldFirewall(
        classifier=spy_classifier,
        detector=fixed_detector,
        threshold=0.5,
        device=device,
    )

    batch = torch.rand(4, 3, 32, 32)
    reports = firewall.process_batch(batch)

    assert len(reports) == 4
    assert reports[0]["decision"] == "BLOCKED"
    assert reports[1]["decision"] == "BLOCKED"
    assert reports[2]["decision"] == "ALLOWED"
    assert reports[3]["decision"] == "ALLOWED"

    # Crucial assertion: SpyClassifier forward should only have seen the 2 ALLOWED samples
    assert spy_classifier.samples_processed == 2
    assert spy_classifier.forward_call_count == 1


def test_evaluate_classifier_returns_test_loss():
    """Verify evaluate_classifier computes and returns test_loss alongside test_accuracy."""
    device = torch.device("cpu")
    model = CIFAR10Classifier().to(device)

    images = torch.rand(16, 3, 32, 32)
    labels = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)

    results = evaluate_classifier(model, loader, device=device)
    assert "test_accuracy" in results
    assert "test_loss" in results
    assert isinstance(results["test_loss"], float)
    assert results["test_loss"] > 0.0


def test_evaluate_firewall_defense_signatures():
    """Verify evaluate_firewall_defense works with both firewall instance and individual components."""
    device = torch.device("cpu")
    clf = CIFAR10Classifier().to(device)
    det = AdversarialDetector().to(device)
    fw = ModelShieldFirewall(classifier=clf, detector=det, threshold=0.5, device=device)

    images = torch.rand(8, 3, 32, 32)
    labels = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    # Calling with firewall object
    res1 = evaluate_firewall_defense(firewall=fw, test_loader=loader, device=device, epsilon=0.03)
    assert "clean_accuracy" in res1
    assert "firewall_clean_utility" in res1

    # Calling with individual components
    res2 = evaluate_firewall_defense(clf, det, loader, device=device, epsilon=0.03)
    assert "clean_accuracy" in res2
    assert "firewall_clean_utility" in res2


def test_detector_generalization_kwargs_and_keys():
    """Verify evaluate_detector_generalization accepts test_loader and reports required keys."""
    device = torch.device("cpu")
    clf = CIFAR10Classifier().to(device)
    det = AdversarialDetector().to(device)

    images = torch.rand(8, 3, 32, 32)
    labels = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    results = evaluate_detector_generalization(
        detector=det,
        classifier=clf,
        test_loader=loader,
        device=device,
    )
    assert "clean_test_fpr" in results
    clean_metrics = results["clean_test_fpr"]
    assert "test_samples" in clean_metrics
    assert "clean_test_fpr" in clean_metrics
    assert "clean_test_tnr" in clean_metrics


def test_calibrate_threshold_calling_conventions():
    """Verify calibrate_threshold accepts both model and score array calling styles."""
    clean_scores = np.linspace(0.01, 0.20, 100)
    adv_scores = np.linspace(0.80, 0.99, 100)

    # Scores style
    res_scores = calibrate_threshold(clean_scores, adv_scores, target_fpr=0.05)
    assert "optimal_threshold" in res_scores
    assert res_scores["calibrated_fpr"] <= 0.05

    # Keyword style with detector
    device = torch.device("cpu")
    det = AdversarialDetector().to(device)
    images = torch.rand(8, 3, 32, 32)
    labels = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    res_model = calibrate_threshold(
        detector=det,
        clean_val_loader=loader,
        target_fpr=0.05,
        device=device,
    )
    assert "optimal_threshold" in res_model
    assert "calibrated_fpr" in res_model


def test_load_checkpoint_order_agnostic():
    """Verify load_checkpoint handles (filepath, model) and (model, filepath)."""
    device = torch.device("cpu")
    model1 = CIFAR10Classifier()
    model2 = CIFAR10Classifier()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "ckpt.pt"
        save_checkpoint({"epoch": 1, "model_state_dict": model1.state_dict()}, ckpt_path)

        # Standard order: (filepath, model)
        ckpt_a = load_checkpoint(ckpt_path, model1, device=device)
        assert ckpt_a["epoch"] == 1

        # Swapped order: (model, filepath)
        ckpt_b = load_checkpoint(model2, ckpt_path, device=device)
        assert ckpt_b["epoch"] == 1


def test_build_multi_attack_dataset_composition():
    """Verify build_multi_attack_dataset yields balanced binary labels 0 and 1."""
    device = torch.device("cpu")
    clf = CIFAR10Classifier().to(device)

    images = torch.rand(8, 3, 32, 32)
    labels = torch.randint(0, 10, (8,))
    clean_loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    mixture_loader = build_multi_attack_dataset(
        classifier=clf,
        clean_loader=clean_loader,
        device=device,
        batch_size=4,
    )

    all_labels = []
    for imgs, lbls in mixture_loader:
        all_labels.extend(lbls.tolist())

    assert len(all_labels) > 0
    zeros = sum(1 for y in all_labels if y == 0.0)
    ones = sum(1 for y in all_labels if y == 1.0)
    assert zeros == ones, f"Expected balanced clean and adversarial labels, got {zeros} vs {ones}"


def test_evaluate_attack_spectrum_keywords():
    """Verify evaluate_attack_spectrum accepts both test_loader and dataloader."""
    device = torch.device("cpu")
    clf = CIFAR10Classifier().to(device)
    images = torch.rand(8, 3, 32, 32)
    labels = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    # Test with dataloader keyword (as used in notebook Cell 8)
    res1 = evaluate_attack_spectrum(model=clf, dataloader=loader, epsilons=[0.01], device=device)
    assert "fgsm" in res1
    assert "pgd" in res1

    # Test with test_loader keyword
    res2 = evaluate_attack_spectrum(model=clf, test_loader=loader, epsilons=[0.01], device=device)
    assert "fgsm" in res2
    assert "pgd" in res2


def test_atomic_checkpoint_saving():
    """Verify save_checkpoint writes atomically, verifies file size, and cleans up temp files."""
    device = torch.device("cpu")
    model = CIFAR10Classifier()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_dir = Path(tmp_dir) / "checkpoints"
        latest_path = ckpt_dir / "latest_ckpt.pth"
        best_path = ckpt_dir / "best_ckpt.pth"

        state = {"epoch": 1, "model_state_dict": model.state_dict(), "val_acc": 0.85}
        save_checkpoint(state, filepath=latest_path, is_best=True, best_filepath=best_path)

        assert latest_path.exists()
        assert best_path.exists()
        assert latest_path.stat().st_size > 0
        assert best_path.stat().st_size > 0

        # Verify no orphaned temporary staging files remain
        tmp_files = list(ckpt_dir.glob(".tmp_*"))
        assert len(tmp_files) == 0, f"Found orphaned temp files: {tmp_files}"

        # Test failure rollback: unpicklable object
        bad_state = {"epoch": 2, "bad_obj": lambda x: x}
        with pytest.raises(Exception):
            save_checkpoint(bad_state, filepath=latest_path)

        # Confirm destination was not corrupted by failed save
        recovered = torch.load(latest_path, map_location="cpu", weights_only=False)
        assert recovered["epoch"] == 1
        assert len(list(ckpt_dir.glob(".tmp_*"))) == 0


def test_json_persistence_helpers_with_tensors_and_arrays():
    """Verify save_json converts tensors, numpy arrays/scalars, and Paths to JSON-serializable types."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = Path(tmp_dir) / "complex_metrics.json"

        complex_payload = {
            "epoch": np.int64(10),
            "accuracy": np.float32(0.875),
            "tensor_scores": torch.tensor([0.1, 0.4, 0.9]),
            "confusion_matrix": np.array([[10, 2], [1, 15]]),
            "save_path": Path("/content/drive/MyDrive/ModelShield_Phase2"),
            "nested": {
                "sub_metric": np.float64(0.999),
                "sub_tensor": torch.tensor(42),
            },
        }

        config.save_json(complex_payload, test_file)
        assert test_file.exists()

        loaded = config.load_json(test_file)
        assert loaded["epoch"] == 10
        assert abs(loaded["accuracy"] - 0.875) < 1e-4
        assert loaded["tensor_scores"] == [pytest.approx(0.1), pytest.approx(0.4), pytest.approx(0.9)]
        assert loaded["confusion_matrix"] == [[10, 2], [1, 15]]
        assert "/content/drive/MyDrive/ModelShield_Phase2" in loaded["save_path"]
        assert abs(loaded["nested"]["sub_metric"] - 0.999) < 1e-4
        assert loaded["nested"]["sub_tensor"] == 42


def test_threshold_persistence_and_reload():
    """Verify calibrated threshold saves to JSON and can be loaded after restart."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        thresh_file = Path(tmp_dir) / "results" / "firewall" / "calibrated_threshold.json"

        clean_scores = np.linspace(0.01, 0.15, 100)
        adv_scores = np.linspace(0.85, 0.99, 100)

        calib_res = calibrate_threshold(clean_scores, adv_scores, target_fpr=0.01, save_path=thresh_file)
        assert thresh_file.exists()
        assert thresh_file.stat().st_size > 0

        # Simulate reload after fresh runtime restart
        loaded_calib = config.load_json(thresh_file)
        assert "optimal_threshold" in loaded_calib
        assert loaded_calib["optimal_threshold"] == calib_res["optimal_threshold"]
        assert loaded_calib["target_fpr"] == 0.01


def test_persistent_training_histories_and_logging():
    """Verify setup_logging avoids duplicate handlers and creates persistent log files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = Path(tmp_dir) / "logs" / "phase2_training.log"

        logger1 = config.setup_logging(log_file=log_file, logger_name="TestLogger")
        logger1.info("Initial log message across runtime.")

        assert log_file.exists()
        with open(log_file, "r", encoding="utf-8") as f:
            content1 = f.read()
        assert "Initial log message across runtime." in content1

        # Rerun setup_logging (simulate cell rerun) and check handler count
        initial_file_handlers = [h for h in logger1.handlers if isinstance(h, config.logging.FileHandler)]
        assert len(initial_file_handlers) == 1

        logger2 = config.setup_logging(log_file=log_file, logger_name="TestLogger")
        logger2.info("Second log message.")

        rerun_file_handlers = [h for h in logger2.handlers if isinstance(h, config.logging.FileHandler)]
        assert len(rerun_file_handlers) == 1

        with open(log_file, "r", encoding="utf-8") as f:
            content2 = f.read()
        assert "Second log message." in content2

        # Clean up handlers before temporary directory exit
        for h in list(logger2.handlers):
            h.close()
            logger2.removeHandler(h)
        for h in list(logger1.handlers):
            h.close()
            logger1.removeHandler(h)


