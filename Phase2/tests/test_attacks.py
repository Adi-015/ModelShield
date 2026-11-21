"""
Unit tests for ModelShield Phase 2 Adversarial Attacks (FGSM & PGD).
Verifies Linf bounds, [0, 1] physical clamping, parameter immutability, and epsilon scaling.
"""

import pytest
import torch
import torch.nn as nn
from Phase2.src.models import CIFAR10Classifier
from Phase2.src.data import denormalize_tensor, normalize_tensor
from Phase2.src.attacks import fgsm_attack, pgd_attack, evaluate_attacks


@pytest.fixture
def dummy_model():
    model = CIFAR10Classifier(num_classes=10)
    model.eval()
    return model


@pytest.fixture
def sample_batch():
    torch.manual_seed(42)
    # Generate raw physical images in [0, 1], then normalize them as CIFAR-10 data loaders do
    raw_images = torch.rand(8, 3, 32, 32)
    images = normalize_tensor(raw_images)
    labels = torch.randint(0, 10, (8,))
    return images, labels


class TestFGSM:
    def test_fgsm_perturbation_bound(self, dummy_model, sample_batch):
        images, labels = sample_batch
        epsilon = 0.03
        adv_images = fgsm_attack(dummy_model, images, labels, epsilon=epsilon)
        
        # Check difference in physical [0, 1] pixel space
        orig_raw = denormalize_tensor(images)
        adv_raw = denormalize_tensor(adv_images)
        diff = (adv_raw - orig_raw).abs()
        max_diff = diff.max().item()
        assert max_diff <= epsilon + 1e-4, f"Perturbation exceeded epsilon: {max_diff} > {epsilon}"

    def test_fgsm_pixel_bounds(self, dummy_model, sample_batch):
        images, labels = sample_batch
        adv_images = fgsm_attack(dummy_model, images, labels, epsilon=0.03)
        adv_raw = denormalize_tensor(adv_images)
        assert adv_raw.min().item() >= -1e-6, "Pixel values dropped below 0.0"
        assert adv_raw.max().item() <= 1.0 + 1e-6, "Pixel values exceeded 1.0"

    def test_fgsm_zero_epsilon_identity(self, dummy_model, sample_batch):
        images, labels = sample_batch
        adv_images = fgsm_attack(dummy_model, images, labels, epsilon=0.0)
        assert torch.allclose(adv_images, images, atol=1e-6), "Epsilon 0 should return original images"

    def test_fgsm_model_parameters_immutable(self, dummy_model, sample_batch):
        images, labels = sample_batch
        weights_before = [p.clone() for p in dummy_model.parameters()]
        
        _ = fgsm_attack(dummy_model, images, labels, epsilon=0.03)
        
        for p_before, p_after in zip(weights_before, dummy_model.parameters()):
            assert torch.equal(p_before, p_after), "Model parameters were modified during attack generation!"


class TestPGD:
    def test_pgd_perturbation_bound(self, dummy_model, sample_batch):
        images, labels = sample_batch
        epsilon = 0.03
        adv_images = pgd_attack(
            dummy_model, images, labels, epsilon=epsilon, alpha=0.007, num_steps=10
        )
        
        orig_raw = denormalize_tensor(images)
        adv_raw = denormalize_tensor(adv_images)
        diff = (adv_raw - orig_raw).abs()
        max_diff = diff.max().item()
        assert max_diff <= epsilon + 1e-4, f"Perturbation exceeded epsilon: {max_diff} > {epsilon}"

    def test_pgd_pixel_bounds(self, dummy_model, sample_batch):
        images, labels = sample_batch
        adv_images = pgd_attack(
            dummy_model, images, labels, epsilon=0.03, alpha=0.007, num_steps=10
        )
        adv_raw = denormalize_tensor(adv_images)
        assert adv_raw.min().item() >= -1e-6, "Pixel values dropped below 0.0"
        assert adv_raw.max().item() <= 1.0 + 1e-6, "Pixel values exceeded 1.0"

    def test_pgd_zero_epsilon_identity(self, dummy_model, sample_batch):
        images, labels = sample_batch
        adv_images = pgd_attack(dummy_model, images, labels, epsilon=0.0, num_steps=5)
        assert torch.allclose(adv_images, images, atol=1e-6), "Epsilon 0 should return original images"

    def test_pgd_model_parameters_immutable(self, dummy_model, sample_batch):
        images, labels = sample_batch
        weights_before = [p.clone() for p in dummy_model.parameters()]
        
        _ = pgd_attack(dummy_model, images, labels, epsilon=0.03, alpha=0.007, num_steps=10)
        
        for p_before, p_after in zip(weights_before, dummy_model.parameters()):
            assert torch.equal(p_before, p_after), "Model parameters were modified during attack generation!"
