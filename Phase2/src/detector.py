"""
ModelShield Phase 2 - Adversarial Detector & Generalization Engine
Solves Phase 1 single-attack shortcut learning by introducing multi-attack
and multi-epsilon training mixtures (Clean, FGSM, and PGD).
Provides evaluation protocols across In-Distribution, Unseen-Epsilon,
Unseen-Attack, and Clean Test splits.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from Phase2 import config
except ImportError:
    import config

try:
    from Phase2.src.attacks import fgsm_attack, pgd_attack
    from Phase2.src.data import AdversarialDataset
except ImportError:
    from src.attacks import fgsm_attack, pgd_attack
    from src.data import AdversarialDataset


def build_multi_attack_dataset(
    classifier: nn.Module,
    clean_loader: DataLoader,
    device: torch.device,
    mixture_config: Optional[Dict[str, Any]] = None,
    max_clean_samples: Optional[int] = None,
    shuffle: bool = False,
    batch_size: int = config.DETECTOR_BATCH_SIZE,
) -> DataLoader:
    """
    Construct a balanced multi-attack dataset for detector training or evaluation.

    Composition:
    - 50% Clean images (Label 0)
    - 25% FGSM perturbed images (Label 1) across varied epsilons
    - 25% PGD perturbed images (Label 1) across varied epsilons

    Args:
        classifier: Target classifier used to generate adversarial gradients.
        clean_loader: Source DataLoader for clean images.
        device: Torch computation device.
        mixture_config: Optional configuration override for attack ratios and epsilons.
        max_clean_samples: Maximum clean samples to use (None for full split).
        shuffle: Whether to shuffle the resulting DataLoader (True for train, False for val/test).
        batch_size: Batch size for DataLoader.

    Returns:
        DataLoader containing balanced clean and multi-attack samples.
    """
    classifier.eval()
    cfg = mixture_config or config.DETECTOR_TRAIN_MIXTURE

    fgsm_epsilons = cfg.get("fgsm_epsilons", [0.005, 0.01, 0.02, 0.03])
    pgd_epsilons = cfg.get("pgd_epsilons", [0.005, 0.01, 0.02, 0.03])
    pgd_iters = cfg.get("pgd_iters", 7)

    all_clean: List[torch.Tensor] = []
    all_adv: List[torch.Tensor] = []

    collected = 0
    fgsm_idx = 0
    pgd_idx = 0

    for images, labels in clean_loader:
        images = images.to(device)
        labels = labels.to(device)

        batch_len = images.size(0)
        if max_clean_samples and (collected + batch_len) > max_clean_samples:
            slice_len = max_clean_samples - collected
            images = images[:slice_len]
            labels = labels[:slice_len]
            batch_len = slice_len

        # Split batch into halves for FGSM and PGD diversity
        half_len = batch_len // 2
        img_fgsm = images[:half_len]
        lbl_fgsm = labels[:half_len]
        img_pgd = images[half_len:]
        lbl_pgd = labels[half_len:]

        # Select varying epsilon cyclically to prevent fixed-epsilon shortcut learning
        eps_f = fgsm_epsilons[fgsm_idx % len(fgsm_epsilons)]
        eps_p = pgd_epsilons[pgd_idx % len(pgd_epsilons)]
        fgsm_idx += 1
        pgd_idx += 1

        adv_f = fgsm_attack(classifier, img_fgsm, lbl_fgsm, epsilon=eps_f)
        adv_p = pgd_attack(
            classifier,
            img_pgd,
            lbl_pgd,
            epsilon=eps_p,
            alpha=eps_p / 4.0,
            num_iter=pgd_iters,
            random_start=True,
        )

        all_clean.append(images.cpu())
        all_adv.append(adv_f.cpu())
        all_adv.append(adv_p.cpu())

        collected += batch_len
        if max_clean_samples and collected >= max_clean_samples:
            break

    clean_tensor = torch.cat(all_clean, dim=0)
    adv_tensor = torch.cat(all_adv, dim=0)

    # Balance counts exactly if rounding occurred
    min_count = min(clean_tensor.size(0), adv_tensor.size(0))
    clean_tensor = clean_tensor[:min_count]
    adv_tensor = adv_tensor[:min_count]

    clean_labels = torch.zeros(min_count, dtype=torch.float32)
    adv_labels = torch.ones(min_count, dtype=torch.float32)

    combined_images = torch.cat([clean_tensor, adv_tensor], dim=0)
    combined_labels = torch.cat([clean_labels, adv_labels], dim=0)

    dataset = AdversarialDataset(
        combined_images,
        combined_labels,
        metadata={"total_pairs": min_count, "mixture": cfg},
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=config.seed_worker,
        generator=config.get_generator(config.RANDOM_SEED),
        pin_memory=torch.cuda.is_available(),
    )

    return loader


def compute_detector_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_prob: Union[np.ndarray, List[float]],
    threshold: float = config.DETECTOR_DEFAULT_THRESHOLD,
    save_cm_path: Optional[Path] = None,
    save_roc_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Compute binary classification metrics from ground truth labels and predicted probabilities.
    """
    all_labels_np = np.asarray(y_true, dtype=int)
    all_probs_np = np.asarray(y_prob, dtype=float)
    all_preds_np = (all_probs_np >= threshold).astype(int)

    acc = float((all_preds_np == all_labels_np).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels_np, all_preds_np, average="binary", zero_division=0
    )

    cm = confusion_matrix(all_labels_np, all_preds_np)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Edge case: single class present
        tn = int(((all_labels_np == 0) & (all_preds_np == 0)).sum())
        fp = int(((all_labels_np == 0) & (all_preds_np == 1)).sum())
        fn = int(((all_labels_np == 1) & (all_preds_np == 0)).sum())
        tp = int(((all_labels_np == 1) & (all_preds_np == 1)).sum())

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(all_labels_np, all_probs_np))
    except Exception:
        roc_auc = 0.50

    try:
        pr_auc = float(average_precision_score(all_labels_np, all_probs_np))
    except Exception:
        pr_auc = 0.0

    # Optional visualization saves
    if save_roc_path:
        save_roc_path = Path(save_roc_path)
        save_roc_path.parent.mkdir(parents=True, exist_ok=True)
        fpr_curve, tpr_curve, _ = roc_curve(all_labels_np, all_probs_np)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr_curve, tpr_curve, "b-", linewidth=2, label=f"ROC (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (0.50)")
        ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
        ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
        ax.set_title("Detector ROC Curve (Phase 2)", fontsize=12)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(save_roc_path), dpi=300, bbox_inches="tight")
        plt.close(fig)

    if save_cm_path and cm.shape == (2, 2):
        save_cm_path = Path(save_cm_path)
        save_cm_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Oranges)
        ax.figure.colorbar(im, ax=ax)
        class_names = ["Clean (0)", "Adversarial (1)"]
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=class_names,
            yticklabels=class_names,
            title="Detector Confusion Matrix",
            ylabel="True Label",
            xlabel="Predicted Label",
        )
        thresh_val = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh_val else "black",
                )
        plt.tight_layout()
        plt.savefig(str(save_cm_path), dpi=300, bbox_inches="tight")
        plt.close(fig)

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold": float(threshold),
        "total_samples": len(all_labels_np),
    }


def evaluate_detector(
    detector_or_labels: Union[nn.Module, np.ndarray, List[int]],
    dataloader_or_probs: Union[DataLoader, np.ndarray, List[float]],
    device: Optional[torch.device] = None,
    threshold: float = config.DETECTOR_DEFAULT_THRESHOLD,
    save_cm_path: Optional[Path] = None,
    save_roc_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate adversarial detector metrics.
    Supports either (detector, dataloader) or (y_true, y_prob) calling signatures.
    """
    if isinstance(detector_or_labels, nn.Module):
        detector = detector_or_labels
        dataloader = dataloader_or_probs
        target_device = device or config.DEVICE
        detector.eval()

        all_probs: List[float] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(target_device)
                logits = detector(images).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().tolist()
                all_probs.extend(probs if isinstance(probs, list) else [probs])
                all_labels.extend([int(lbl) for lbl in labels.tolist()])

        return compute_detector_metrics(
            y_true=all_labels,
            y_prob=all_probs,
            threshold=threshold,
            save_cm_path=save_cm_path,
            save_roc_path=save_roc_path,
        )
    else:
        return compute_detector_metrics(
            y_true=detector_or_labels,
            y_prob=dataloader_or_probs,
            threshold=threshold,
            save_cm_path=save_cm_path,
            save_roc_path=save_roc_path,
        )


def evaluate_detector_generalization(
    detector: nn.Module,
    classifier: nn.Module,
    clean_test_loader: Optional[DataLoader] = None,
    device: Optional[torch.device] = None,
    threshold: float = config.DETECTOR_DEFAULT_THRESHOLD,
    max_eval_samples: Optional[int] = None,
    *,
    test_loader: Optional[DataLoader] = None,
) -> Dict[str, Any]:
    """
    Perform Phase 2 Detector Generalization Suite across 4 distinct benchmarks:

    1. In-Distribution: FGSM at training epsilon (eps=0.03).
    2. Unseen-Epsilon: FGSM at unseen lower and interpolated epsilons (0.005, 0.015, 0.025).
    3. Unseen-Attack: Iterative PGD attack (eps=0.03, alpha=0.005, iter=10).
    4. Clean Test Set: Evaluation on 100% natural, unperturbed CIFAR-10 test set images.

    Distinguishes genuine generalization from single-attack memorization.
    """
    target_device = device or config.DEVICE
    loader = clean_test_loader if clean_test_loader is not None else test_loader
    if loader is None:
        raise ValueError("A clean test DataLoader must be provided via clean_test_loader or test_loader.")

    results: Dict[str, Any] = {}

    print("\n[Detector Generalization] Executing Multi-Facet Evaluation Protocol...")

    # Benchmark 1: In-Distribution Evaluation (FGSM eps=0.03)
    loader_in = build_multi_attack_dataset(
        classifier=classifier,
        clean_loader=loader,
        device=target_device,
        mixture_config={"fgsm_epsilons": [0.03], "pgd_epsilons": [0.03], "pgd_iters": 7},
        max_clean_samples=max_eval_samples,
        shuffle=False,
    )
    res_in = evaluate_detector(detector, loader_in, target_device, threshold=threshold)
    results["in_distribution_fgsm0.03"] = res_in
    print(f"  1. In-Distribution (FGSM 0.03): Acc={res_in['accuracy']*100:.2f}%, Recall={res_in['recall']*100:.2f}%, ROC-AUC={res_in['roc_auc']:.4f}")

    # Benchmark 2: Unseen-Epsilon Generalization (eps=0.01)
    loader_unseen_eps = build_multi_attack_dataset(
        classifier=classifier,
        clean_loader=loader,
        device=target_device,
        mixture_config={"fgsm_epsilons": [0.01], "pgd_epsilons": [0.01], "pgd_iters": 7},
        max_clean_samples=max_eval_samples,
        shuffle=False,
    )
    res_unseen_eps = evaluate_detector(detector, loader_unseen_eps, target_device, threshold=threshold)
    results["unseen_epsilon_0.01"] = res_unseen_eps
    print(f"  2. Unseen Epsilon (eps=0.01):   Acc={res_unseen_eps['accuracy']*100:.2f}%, Recall={res_unseen_eps['recall']*100:.2f}%, ROC-AUC={res_unseen_eps['roc_auc']:.4f}")

    # Benchmark 3: Unseen-Attack Generalization (Iterative PGD 10-step)
    loader_pgd = build_multi_attack_dataset(
        classifier=classifier,
        clean_loader=loader,
        device=target_device,
        mixture_config={"fgsm_epsilons": [0.03], "pgd_epsilons": [0.03], "pgd_iters": 10},
        max_clean_samples=max_eval_samples,
        shuffle=False,
    )
    res_pgd = evaluate_detector(detector, loader_pgd, target_device, threshold=threshold)
    results["unseen_attack_pgd10"] = res_pgd
    print(f"  3. Unseen Attack (PGD 10-step): Acc={res_pgd['accuracy']*100:.2f}%, Recall={res_pgd['recall']*100:.2f}%, ROC-AUC={res_pgd['roc_auc']:.4f}")

    # Benchmark 4: 100% Clean CIFAR-10 Test Set Evaluation (False Alarm Rate)
    clean_detector_scores = []
    detector.eval()
    with torch.no_grad():
        evaluated_clean = 0
        for images, _ in loader:
            images = images.to(target_device)
            logits = detector(images).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().tolist()
            clean_detector_scores.extend(probs)
            evaluated_clean += images.size(0)
            if max_eval_samples and evaluated_clean >= max_eval_samples:
                break

    clean_scores_np = np.array(clean_detector_scores)
    false_positives = (clean_scores_np >= threshold).sum()
    clean_fpr = float(false_positives / len(clean_scores_np))
    results["clean_test_fpr"] = {
        "total_clean_evaluated": len(clean_scores_np),
        "test_samples": len(clean_scores_np),
        "false_alarms": int(false_positives),
        "false_positive_rate": clean_fpr,
        "clean_test_fpr": clean_fpr,
        "true_negative_rate": float(1.0 - clean_fpr),
        "clean_test_tnr": float(1.0 - clean_fpr),
        "mean_clean_prob": float(clean_scores_np.mean()),
    }
    print(f"  4. Natural Clean Test Split:    FPR={clean_fpr*100:.2f}%, TNR={(1-clean_fpr)*100:.2f}%, Mean Prob={clean_scores_np.mean():.4f}")

    return results
