"""
ModelShield - Adversarial Attack Module
Implements the Fast Gradient Sign Method (FGSM) adversarial attack,
Attack Success Rate (ASR) computation, and attack visualization tools.
"""

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from src.data import denormalize_tensor, normalize_tensor


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    loss_fn: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    Generate adversarial examples using the Fast Gradient Sign Method (FGSM).

    Mathematical Formulation:
        x_adv = clip(x_raw + epsilon * sign(grad_x(Loss(f(x_norm), y))), 0.0, 1.0)
        x_norm_adv = normalize(x_adv)

    Perturbations are performed in the unnormalized [0, 1] physical image space
    to ensure exact epsilon perturbation bounds and proper clipping.

    Args:
        model: Target PyTorch classifier.
        images: Batch of normalized images of shape (B, 3, H, W).
        labels: Ground-truth class labels of shape (B,).
        epsilon: Perturbation magnitude.
        loss_fn: Loss function (defaults to CrossEntropyLoss).

    Returns:
        adv_images: Normalized adversarial image tensor of shape (B, 3, H, W).
    """
    if epsilon == 0.0:
        return images.clone()

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    model_training_mode = model.training
    model.eval()

    # Convert to unnormalized [0, 1] pixel space for consistent perturbation
    raw_images = denormalize_tensor(images).detach().clone()
    raw_images.requires_grad = True

    # Re-normalize to feed into the classifier
    norm_images = normalize_tensor(raw_images)
    outputs = model(norm_images)
    loss = loss_fn(outputs, labels)

    # Compute gradients with respect to raw pixel values
    model.zero_grad()
    loss.backward()

    # Gradient sign
    data_grad = raw_images.grad.data
    sign_data_grad = data_grad.sign()

    # Perturb in [0, 1] pixel space and clamp to valid range
    perturbed_raw = raw_images + epsilon * sign_data_grad
    perturbed_raw = torch.clamp(perturbed_raw, 0.0, 1.0)

    # Convert back to normalized distribution for model consumption
    adv_norm = normalize_tensor(perturbed_raw).detach()

    if model_training_mode:
        model.train()

    return adv_norm


def evaluate_attack_success_rate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    epsilon: float,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Calculate untargeted Attack Success Rate (ASR) for a given epsilon.

    Denominator: Samples originally correctly classified by the model.
    Numerator: Samples originally correct whose prediction changed under attack.

    Formula:
        ASR = fooled_count / originally_correct_count

    Args:
        model: Target classifier.
        dataloader: DataLoader containing clean evaluation images.
        epsilon: FGSM perturbation magnitude.
        device: Computation device (CPU or CUDA).
        max_samples: Optional cap on evaluated samples for quick testing.

    Returns:
        Dictionary containing clean_acc, adv_acc, asr, and sample counts.
    """
    model.eval()
    total_evaluated = 0
    originally_correct = 0
    fooled_count = 0
    adv_correct = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)
        if max_samples and (total_evaluated + batch_size) > max_samples:
            slice_len = max_samples - total_evaluated
            images = images[:slice_len]
            labels = labels[:slice_len]
            batch_size = slice_len

        # 1. Clean inference
        with torch.no_grad():
            clean_logits = model(images)
            clean_preds = clean_logits.argmax(dim=1)

        correct_mask = (clean_preds == labels)
        batch_originally_correct = correct_mask.sum().item()

        if batch_originally_correct > 0:
            # Generate adversarial examples for correctly classified images
            correct_images = images[correct_mask]
            correct_labels = labels[correct_mask]

            adv_images = fgsm_attack(
                model=model,
                images=correct_images,
                labels=correct_labels,
                epsilon=epsilon,
            )

            with torch.no_grad():
                adv_logits = model(adv_images)
                adv_preds = adv_logits.argmax(dim=1)

            # Attack succeeded if prediction is no longer the true label
            batch_fooled = (adv_preds != correct_labels).sum().item()
            fooled_count += batch_fooled

        # Also compute overall adversarial accuracy on the entire batch
        adv_batch = fgsm_attack(
            model=model,
            images=images,
            labels=labels,
            epsilon=epsilon,
        )
        with torch.no_grad():
            adv_batch_logits = model(adv_batch)
            adv_batch_preds = adv_batch_logits.argmax(dim=1)
            adv_correct += (adv_batch_preds == labels).sum().item()

        total_evaluated += batch_size
        originally_correct += batch_originally_correct

        if max_samples and total_evaluated >= max_samples:
            break

    clean_acc = (originally_correct / total_evaluated) if total_evaluated > 0 else 0.0
    adv_acc = (adv_correct / total_evaluated) if total_evaluated > 0 else 0.0
    asr = (fooled_count / originally_correct) if originally_correct > 0 else 0.0

    return {
        "epsilon": epsilon,
        "total_evaluated": total_evaluated,
        "originally_correct": originally_correct,
        "fooled_count": fooled_count,
        "clean_accuracy": clean_acc,
        "adversarial_accuracy": adv_acc,
        "attack_success_rate": asr,
    }


def visualize_adversarial_examples(
    model: nn.Module,
    sample_image: torch.Tensor,
    sample_label: int,
    epsilons: List[float],
    device: torch.device,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Display clean vs adversarial images side-by-side with predicted probabilities.

    Args:
        model: Target classifier.
        sample_image: Single normalized tensor of shape (3, 32, 32) or (1, 3, 32, 32).
        sample_label: Integer ground truth label.
        epsilons: List of epsilon values to evaluate.
        device: Torch device.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure object.
    """
    if sample_image.dim() == 3:
        sample_image = sample_image.unsqueeze(0)

    sample_image = sample_image.to(device)
    target_label = torch.tensor([sample_label], device=device)

    model.eval()
    with torch.no_grad():
        clean_logits = model(sample_image)
        clean_probs = F.softmax(clean_logits, dim=1).squeeze(0)
        clean_pred = clean_probs.argmax().item()
        clean_conf = clean_probs[clean_pred].item()

    clean_raw = denormalize_tensor(sample_image).squeeze(0).permute(1, 2, 0).cpu().numpy()

    num_cols = len(epsilons) + 1
    fig, axes = plt.subplots(2, num_cols, figsize=(3.5 * num_cols, 6))

    # Column 0: Clean Image
    axes[0, 0].imshow(clean_raw)
    axes[0, 0].set_title(
        f"Original Clean\nTrue: {config.CIFAR10_CLASSES[sample_label]}\nPred: {config.CIFAR10_CLASSES[clean_pred]} ({clean_conf * 100:.1f}%)",
        fontsize=10,
    )
    axes[0, 0].axis("off")

    # Clean top-3 probabilities
    top3_probs, top3_idx = torch.topk(clean_probs, 3)
    bar_labels = [config.CIFAR10_CLASSES[i] for i in top3_idx.cpu().numpy()]
    axes[1, 0].barh(bar_labels[::-1], top3_probs.cpu().numpy()[::-1], color="forestgreen")
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_title("Original Probs", fontsize=9)

    # Subsequent columns: Perturbed for each epsilon
    for col_idx, eps in enumerate(epsilons, start=1):
        adv_tensor = fgsm_attack(model, sample_image, target_label, epsilon=eps)
        with torch.no_grad():
            adv_logits = model(adv_tensor)
            adv_probs = F.softmax(adv_logits, dim=1).squeeze(0)
            adv_pred = adv_probs.argmax().item()
            adv_conf = adv_probs[adv_pred].item()

        adv_raw = denormalize_tensor(adv_tensor).squeeze(0).permute(1, 2, 0).cpu().numpy()

        is_fooled = adv_pred != sample_label
        title_color = "darkred" if is_fooled else "black"

        axes[0, col_idx].imshow(adv_raw)
        axes[0, col_idx].set_title(
            f"FGSM (eps={eps})\nPred: {config.CIFAR10_CLASSES[adv_pred]} ({adv_conf * 100:.1f}%)\nStatus: {'FOOLED' if is_fooled else 'CORRECT'}",
            fontsize=10,
            color=title_color,
        )
        axes[0, col_idx].axis("off")

        # Adversarial top-3 probabilities
        top3_adv_probs, top3_adv_idx = torch.topk(adv_probs, 3)
        adv_bar_labels = [config.CIFAR10_CLASSES[i] for i in top3_adv_idx.cpu().numpy()]
        bar_color = "firebrick" if is_fooled else "steelblue"
        axes[1, col_idx].barh(
            adv_bar_labels[::-1],
            top3_adv_probs.cpu().numpy()[::-1],
            color=bar_color,
        )
        axes[1, col_idx].set_xlim(0, 1)
        axes[1, col_idx].set_title(f"eps={eps} Probs", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
