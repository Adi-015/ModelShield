"""
ModelShield Phase 2 - Comprehensive Evaluation and Benchmarking Module
Provides structured evaluation metrics for classifiers, attack engines,
detectors, and end-to-end calibrated firewall gateways.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from Phase2 import config
except ImportError:
    import config

try:
    from Phase2.src.attacks import fgsm_attack, pgd_attack, evaluate_attacks
except ImportError:
    from src.attacks import fgsm_attack, pgd_attack, evaluate_attacks


def evaluate_classifier(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
    save_cm_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate trained classifier on the official untouched CIFAR-10 test set (10,000 images).
    """
    target_device = device or config.DEVICE
    model.eval()

    all_preds: List[int] = []
    all_labels: List[int] = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(target_device)
            labels_dev = labels.to(target_device)
            logits = model(images)
            loss = criterion(logits, labels_dev)
            total_loss += loss.item() * images.size(0)

            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    acc = float((all_preds_np == all_labels_np).mean())
    avg_loss = float(total_loss / len(all_labels_np)) if len(all_labels_np) > 0 else 0.0
    num_classes = len(config.CIFAR10_CLASSES)
    all_class_indices = list(range(num_classes))
    cls_report = classification_report(
        all_labels_np,
        all_preds_np,
        labels=all_class_indices,
        target_names=config.CIFAR10_CLASSES,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels_np, all_preds_np, labels=all_class_indices)

    per_class_acc = {}
    for i, class_name in enumerate(config.CIFAR10_CLASSES):
        total_cls = (all_labels_np == i).sum()
        correct_cls = ((all_labels_np == i) & (all_preds_np == i)).sum()
        per_class_acc[class_name] = float(correct_cls / total_cls) if total_cls > 0 else 0.0

    if save_cm_path:
        save_cm_path = Path(save_cm_path)
        save_cm_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=config.CIFAR10_CLASSES,
            yticklabels=config.CIFAR10_CLASSES,
            title="CIFAR-10 Classifier Test Confusion Matrix (Phase 2)",
            ylabel="True Label",
            xlabel="Predicted Label",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8,
                )
        plt.tight_layout()
        plt.savefig(str(save_cm_path), dpi=300, bbox_inches="tight")
        plt.close(fig)

    return {
        "test_accuracy": acc,
        "test_loss": avg_loss,
        "per_class_accuracy": per_class_acc,
        "classification_report": cls_report,
        "confusion_matrix": cm.tolist(),
        "total_test_samples": len(all_labels_np),
    }


def evaluate_attack_spectrum(
    model: nn.Module,
    test_loader: Optional[DataLoader] = None,
    attack_types: List[str] = ("fgsm", "pgd"),
    epsilons: Optional[List[float]] = None,
    device: Optional[torch.device] = None,
    save_plot_path: Optional[Path] = None,
    max_samples: Optional[int] = None,
    dataloader: Optional[DataLoader] = None,
    **kwargs: Any,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Evaluate classifier robustness against multiple attack types across the epsilon spectrum.
    Reports conditional ASR and unconditional error rates side-by-side.
    Accepts either `test_loader` or `dataloader` keyword argument.
    """
    loader = test_loader if test_loader is not None else dataloader
    if loader is None:
        raise ValueError("evaluate_attack_spectrum requires a DataLoader (test_loader or dataloader).")
    target_device = device or config.DEVICE
    eps_list = epsilons or config.ATTACK_EPSILONS

    results: Dict[str, List[Dict[str, Any]]] = {}

    for atk in attack_types:
        atk_results = []
        print(f"\n[Attack Evaluation] Benchmarking {atk.upper()} across epsilons: {eps_list}")
        for eps in eps_list:
            res = evaluate_attacks(
                model=model,
                dataloader=loader,
                attack_type=atk,
                epsilon=eps,
                device=target_device,
                max_samples=max_samples,
            )
            atk_results.append(res)
            print(
                f"  {atk.upper()} (eps={eps:.3f}) | Adv Acc: {res['adversarial_accuracy']*100:.2f}% | "
                f"Cond ASR: {res['conditional_asr']*100:.2f}% | Uncond Error: {res['unconditional_error_rate']*100:.2f}%"
            )
        results[atk] = atk_results

    if save_plot_path:
        save_plot_path = Path(save_plot_path)
        save_plot_path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        for atk in attack_types:
            eps_vals = [r["epsilon"] for r in results[atk]]
            adv_accs = [r["adversarial_accuracy"] * 100 for r in results[atk]]
            cond_asrs = [r["conditional_asr"] * 100 for r in results[atk]]

            marker = "-o" if atk == "fgsm" else "-s"
            color = "royalblue" if atk == "fgsm" else "crimson"

            ax1.plot(eps_vals, adv_accs, marker, label=f"{atk.upper()} Adv Acc (%)", color=color, linewidth=2)
            ax2.plot(eps_vals, cond_asrs, marker, label=f"{atk.upper()} Cond ASR (%)", color=color, linewidth=2)

        ax1.set_title("Adversarial Accuracy vs Epsilon", fontsize=11)
        ax1.set_xlabel("Perturbation Bound (Epsilon $\\epsilon$)", fontsize=10)
        ax1.set_ylabel("Accuracy (%)", fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.set_title("Conditional Attack Success Rate (ASR) vs Epsilon", fontsize=11)
        ax2.set_xlabel("Perturbation Bound (Epsilon $\\epsilon$)", fontsize=10)
        ax2.set_ylabel("Conditional ASR (%)", fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(str(save_plot_path), dpi=300, bbox_inches="tight")
        plt.close(fig)

    return results


def evaluate_firewall_defense(
    classifier_or_firewall: Any = None,
    detector_or_loader: Any = None,
    clean_loader: Optional[DataLoader] = None,
    attack_type: str = "fgsm",
    epsilon: float = config.DEFAULT_EPSILON,
    threshold: Optional[float] = None,
    device: Optional[torch.device] = None,
    max_samples: Optional[int] = None,
    *,
    firewall: Optional[Any] = None,
    test_loader: Optional[DataLoader] = None,
    classifier: Optional[nn.Module] = None,
    detector: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """
    Benchmark the protective security layer against an unprotected baseline:
    Simulates equal streams of clean and adversarial test inputs.

    Supports calling conventions:
    - evaluate_firewall_defense(firewall=firewall, test_loader=test_loader, ...)
    - evaluate_firewall_defense(firewall, test_loader, ...)
    - evaluate_firewall_defense(classifier, detector, clean_loader, ...)
    """
    fw = firewall if firewall is not None else (
        classifier_or_firewall if (hasattr(classifier_or_firewall, "classifier") and hasattr(classifier_or_firewall, "detector")) else None
    )

    if fw is not None:
        clf = fw.classifier
        det = fw.detector
        thresh = threshold if threshold is not None else getattr(fw, "threshold", config.DETECTOR_DEFAULT_THRESHOLD)
        target_device = device or getattr(fw, "device", config.DEVICE)
        loader = test_loader if test_loader is not None else (
            clean_loader if clean_loader is not None else (
                detector_or_loader if isinstance(detector_or_loader, DataLoader) else None
            )
        )
    else:
        clf = classifier or classifier_or_firewall
        det = detector or detector_or_loader
        thresh = threshold if threshold is not None else config.DETECTOR_DEFAULT_THRESHOLD
        target_device = device or config.DEVICE
        loader = test_loader if test_loader is not None else (
            clean_loader if clean_loader is not None else (
                detector_or_loader if isinstance(detector_or_loader, DataLoader) else None
            )
        )

    if loader is None:
        raise ValueError("A valid DataLoader must be provided to evaluate_firewall_defense.")

    clf.eval()
    det.eval()

    total_clean = 0
    total_adv = 0

    unprotected_clean_correct = 0
    unprotected_adv_correct = 0

    clean_blocked = 0
    clean_allowed = 0
    clean_allowed_correct = 0

    adv_blocked = 0
    adv_allowed = 0
    adv_allowed_correct = 0

    collected = 0
    for images, labels in loader:
        images = images.to(target_device)
        labels = labels.to(target_device)

        batch_len = images.size(0)
        if max_samples and (collected + batch_len) > max_samples:
            slice_len = max_samples - collected
            images = images[:slice_len]
            labels = labels[:slice_len]
            batch_len = slice_len

        # Generate adversarial counterpart
        if attack_type.lower() == "pgd":
            adv_images = pgd_attack(clf, images, labels, epsilon=epsilon, num_iter=10)
        else:
            adv_images = fgsm_attack(clf, images, labels, epsilon=epsilon)

        with torch.no_grad():
            clean_preds = clf(images).argmax(dim=1)
            adv_preds = clf(adv_images).argmax(dim=1)

            clean_det_probs = torch.sigmoid(det(images).squeeze(-1))
            adv_det_probs = torch.sigmoid(det(adv_images).squeeze(-1))

            clean_is_blocked = (clean_det_probs >= thresh)
            adv_is_blocked = (adv_det_probs >= thresh)

        unprotected_clean_correct += (clean_preds == labels).sum().item()
        unprotected_adv_correct += (adv_preds == labels).sum().item()
        total_clean += batch_len
        total_adv += batch_len

        clean_blocked += clean_is_blocked.sum().item()
        clean_allowed_mask = ~clean_is_blocked
        clean_allowed += clean_allowed_mask.sum().item()
        clean_allowed_correct += ((clean_preds == labels) & clean_allowed_mask).sum().item()

        adv_blocked += adv_is_blocked.sum().item()
        adv_allowed_mask = ~adv_is_blocked
        adv_allowed += adv_allowed_mask.sum().item()
        adv_allowed_correct += ((adv_preds == labels) & adv_allowed_mask).sum().item()

        collected += batch_len
        if max_samples and collected >= max_samples:
            break

    # Ratios
    unprot_clean_acc = unprotected_clean_correct / total_clean
    unprot_adv_acc = unprotected_adv_correct / total_adv

    threat_mitigation_rate = adv_blocked / total_adv
    clean_false_alarm_rate = clean_blocked / total_clean
    clean_utility_retained = clean_allowed_correct / total_clean
    attack_passthrough_rate = adv_allowed / total_adv

    total_allowed = clean_allowed + adv_allowed
    correct_allowed = clean_allowed_correct + adv_allowed_correct
    selective_accuracy = (correct_allowed / total_allowed) if total_allowed > 0 else 0.0

    return {
        "attack_type": attack_type.upper(),
        "epsilon": float(epsilon),
        "threshold": float(thresh),
        "total_clean_evaluated": total_clean,
        "total_adv_evaluated": total_adv,
        "clean_accuracy": float(unprot_clean_acc),
        "adversarial_accuracy": float(unprot_adv_acc),
        "unprotected_clean_accuracy": float(unprot_clean_acc),
        "unprotected_adversarial_accuracy": float(unprot_adv_acc),
        "threat_mitigation_rate": float(threat_mitigation_rate),
        "attack_mitigation_rate": float(threat_mitigation_rate),
        "clean_false_alarm_rate": float(clean_false_alarm_rate),
        "clean_utility_retained": float(clean_utility_retained),
        "firewall_clean_utility": float(clean_utility_retained),
        "attack_passthrough_rate": float(attack_passthrough_rate),
        "selective_accuracy_on_allowed": float(selective_accuracy),
    }
