"""
ModelShield - Training, Evaluation, and Checkpoint Management Module
Implements resumable training for the CIFAR-10 classifier and the adversarial detector,
safe checkpoint saving/loading with size verification, structured experiment history serialization,
ROC-AUC computation, and end-to-end firewall defense benchmarking.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.attacks import fgsm_attack
from src.data import AdversarialDataset


# -------------------------------------------------------------------------
# Checkpoint Utilities (Phase 3 & Phase 4)
# -------------------------------------------------------------------------

def save_checkpoint(
    state: Dict[str, Any],
    filepath: Union[str, Path],
    is_best: bool = False,
    best_filepath: Optional[Union[str, Path]] = None,
) -> None:
    """
    Save training state dictionary to disk with nonzero size verification.

    Args:
        state: Checkpoint dictionary containing model/optimizer/scheduler states, epoch, etc.
        filepath: Target destination for latest checkpoint.
        is_best: If True, also save a copy to best_filepath.
        best_filepath: Destination path for the best model weights.
    """
    target_path = Path(filepath)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    torch.save(state, target_path)

    # Verification: Ensure file exists and has non-zero size
    if not target_path.exists():
        raise FileNotFoundError(f"Checkpoint save failed: File not found at {target_path}")

    file_size_bytes = target_path.stat().st_size
    if file_size_bytes == 0:
        raise IOError(f"Checkpoint save corrupted: 0 bytes written to {target_path}")

    file_size_mb = file_size_bytes / (1024 * 1024)
    epoch = state.get("epoch", "?")
    val_acc = state.get("val_acc", 0.0)

    # Best checkpoint saving
    if is_best and best_filepath is not None:
        best_path = Path(best_filepath)
        best_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, best_path)
        if not best_path.exists() or best_path.stat().st_size == 0:
            raise IOError(f"Failed to verify best checkpoint at {best_path}")
        best_size_mb = best_path.stat().st_size / (1024 * 1024)
        print(
            f"  [Checkpoint Verified] Saved NEW BEST -> {best_path.name} "
            f"({best_size_mb:.2f} MB | Val Acc: {val_acc*100:.2f}%)"
        )
    else:
        print(
            f"  [Checkpoint Verified] Saved LATEST (Epoch {epoch}) -> {target_path.name} "
            f"({file_size_mb:.2f} MB)"
        )


def load_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Safely load checkpoint weights, optimizer state, scheduler state, and metadata.

    Args:
        filepath: Path to checkpoint file.
        model: Target model architecture to load state into.
        optimizer: Optional optimizer to restore state into.
        scheduler: Optional learning rate scheduler to restore state into.
        device: Device to map weights to.

    Returns:
        Checkpoint dictionary metadata.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Cannot load checkpoint: File does not exist at {path}")

    target_device = device or config.DEVICE
    checkpoint = torch.load(path, map_location=target_device, weights_only=False)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unrecognized checkpoint format at {path}. Expected dictionary.")

    # Restore model weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Restore optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        if checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore RNG state if available for deterministic resumption
    if "rng_state" in checkpoint:
        try:
            rng_info = checkpoint["rng_state"]
            if "torch" in rng_info:
                torch.set_rng_state(rng_info["torch"])
            if "numpy" in rng_info:
                np.random.set_state(rng_info["numpy"])
            if "python" in rng_info:
                random.setstate(rng_info["python"])
            if torch.cuda.is_available() and "cuda" in rng_info and rng_info["cuda"]:
                torch.cuda.set_rng_state_all(rng_info["cuda"])
        except Exception as e:
            print(f"  [Notice] RNG state could not be completely restored: {e}")

    return checkpoint


# -------------------------------------------------------------------------
# Training History Plotting & Serialization
# -------------------------------------------------------------------------

def save_json_history(history: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save serializable training history to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def plot_training_history(
    history: Dict[str, List[float]],
    title: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot train vs validation loss and accuracy curves across epochs.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss subplot
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r--s", label="Val Loss")
    ax1.set_title(f"{title} - Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy subplot
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Acc (%)")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "r--s", label="Val Acc (%)")
    ax2.set_title(f"{title} - Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    return fig


# -------------------------------------------------------------------------
# Module 1: Resumable Classifier Training and Evaluation
# -------------------------------------------------------------------------

def train_classifier_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train the classifier for a single epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate classifier loss and accuracy on validation or test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = config.CLASSIFIER_EPOCHS,
    lr: float = config.CLASSIFIER_LR,
    weight_decay: float = config.CLASSIFIER_WEIGHT_DECAY,
    checkpoint_path: Path = config.CLASSIFIER_CHECKPOINT_PATH,
    latest_checkpoint_path: Optional[Path] = None,
    scheduler_type: Optional[str] = None,
    scheduler_params: Optional[Dict[str, Any]] = None,
    label_smoothing: float = 0.0,
    resume: bool = True,
    history_json_path: Optional[Path] = None,
    experiment_id: str = "classifier",
) -> Dict[str, List[float]]:
    """
    Train classifier with Adam optimizer, optional LR scheduler, best/latest checkpointing,
    and automatic training resumption.
    """
    checkpoint_path = Path(checkpoint_path)
    if latest_checkpoint_path is None:
        latest_checkpoint_path = checkpoint_path.parent / f"latest_{checkpoint_path.name}"
    latest_checkpoint_path = Path(latest_checkpoint_path)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Optional learning rate scheduler
    scheduler = None
    if scheduler_type == "CosineAnnealingLR":
        t_max = scheduler_params.get("T_max", epochs) if scheduler_params else epochs
        eta_min = scheduler_params.get("eta_min", 1e-5) if scheduler_params else 1e-5
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    elif scheduler_type == "StepLR":
        step_size = scheduler_params.get("step_size", 10) if scheduler_params else 10
        gamma = scheduler_params.get("gamma", 0.5) if scheduler_params else 0.5
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0
    start_epoch = 1

    # Resumable training check
    if resume and latest_checkpoint_path.exists():
        print(f"\n[Classifier Training] Found existing checkpoint at: {latest_checkpoint_path}")
        try:
            ckpt = load_checkpoint(latest_checkpoint_path, model, optimizer, scheduler, device=device)
            last_epoch = ckpt.get("epoch", 0)
            best_val_acc = ckpt.get("best_val_acc", ckpt.get("val_acc", -1.0))
            if "history" in ckpt and ckpt["history"]:
                history = ckpt["history"]

            if last_epoch >= epochs:
                print(
                    f"[Classifier Training] Experiment '{experiment_id}' already COMPLETED {last_epoch}/{epochs} epochs. "
                    f"Skipping redundant training. (Best Val Acc: {best_val_acc*100:.2f}%)"
                )
                if checkpoint_path.exists():
                    load_checkpoint(checkpoint_path, model, device=device)
                return history

            start_epoch = last_epoch + 1
            print(
                f"[Classifier Training] RESUMING from epoch {start_epoch}/{epochs}. "
                f"(Previous Best Val Acc: {best_val_acc*100:.2f}%)"
            )
        except Exception as e:
            print(f"[!] Warning: Could not resume from checkpoint ({e}). Starting fresh from epoch 1.")
            start_epoch = 1
    else:
        print(f"\n[Classifier Training] STARTING FRESH: {epochs} epochs on device: {device}")

    # Training Loop
    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = train_classifier_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_classifier(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] (lr: {current_lr:.6f}) "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
        )

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        # RNG State capture
        rng_state = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        # Checkpoint state dictionary
        state = {
            "epoch": epoch,
            "experiment_id": experiment_id,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "val_acc": float(val_acc),
            "val_loss": float(val_loss),
            "best_val_acc": float(best_val_acc),
            "history": history,
            "rng_state": rng_state,
        }

        # Save latest checkpoint after every epoch
        save_checkpoint(
            state=state,
            filepath=latest_checkpoint_path,
            is_best=is_best,
            best_filepath=checkpoint_path,
        )

        # Save history JSON
        if history_json_path:
            save_json_history(history, history_json_path)

    # Load best checkpoint into model before returning
    if checkpoint_path.exists():
        print(f"\n[Classifier Training] Loading best weights from: {checkpoint_path}")
        load_checkpoint(checkpoint_path, model, device=device)

    return history


def evaluate_classifier_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    save_cm_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate trained classifier on the untouched test set.
    Generates classification report and confusion matrix.
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    acc = float((all_preds_np == all_labels_np).mean())
    cls_report = classification_report(
        all_labels_np,
        all_preds_np,
        target_names=config.CIFAR10_CLASSES,
        digits=4,
        output_dict=True,
    )
    cm = confusion_matrix(all_labels_np, all_preds_np)

    # Per-class accuracy calculation
    per_class_acc = {}
    for i, class_name in enumerate(config.CIFAR10_CLASSES):
        class_total = (all_labels_np == i).sum()
        class_correct = ((all_labels_np == i) & (all_preds_np == i)).sum()
        per_class_acc[class_name] = float(class_correct / class_total) if class_total > 0 else 0.0

    # Plot confusion matrix
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
            title="CIFAR-10 Classifier Confusion Matrix (Test Set)",
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
        "per_class_accuracy": per_class_acc,
        "classification_report": cls_report,
        "confusion_matrix": cm.tolist(),
    }


# -------------------------------------------------------------------------
# Module 3: Adversarial Detector Dataset Generation, Training & Evaluation
# -------------------------------------------------------------------------

def build_detector_dataset(
    classifier: nn.Module,
    clean_loader: DataLoader,
    epsilon: float,
    device: torch.device,
    max_clean_samples: Optional[int] = None,
) -> DataLoader:
    """
    Build balanced clean vs adversarial dataset strictly from the provided DataLoader.
    - Class 0: Clean images
    - Class 1: FGSM adversarial images
    """
    classifier.eval()
    all_clean_images: List[torch.Tensor] = []
    all_adv_images: List[torch.Tensor] = []

    collected = 0
    for images, labels in clean_loader:
        images = images.to(device)
        labels = labels.to(device)

        batch_len = images.size(0)
        if max_clean_samples and (collected + batch_len) > max_clean_samples:
            images = images[: max_clean_samples - collected]
            labels = labels[: max_clean_samples - collected]
            batch_len = images.size(0)

        # Generate adversarial counterpart using the specified classifier
        adv_batch = fgsm_attack(classifier, images, labels, epsilon=epsilon)

        all_clean_images.append(images.cpu())
        all_adv_images.append(adv_batch.cpu())
        collected += batch_len

        if max_clean_samples and collected >= max_clean_samples:
            break

    clean_tensor = torch.cat(all_clean_images, dim=0)
    adv_tensor = torch.cat(all_adv_images, dim=0)

    clean_labels = torch.zeros(clean_tensor.size(0), dtype=torch.float32)
    adv_labels = torch.ones(adv_tensor.size(0), dtype=torch.float32)

    combined_images = torch.cat([clean_tensor, adv_tensor], dim=0)
    combined_labels = torch.cat([clean_labels, adv_labels], dim=0)

    dataset = AdversarialDataset(combined_images, combined_labels)
    loader = DataLoader(
        dataset,
        batch_size=config.DETECTOR_BATCH_SIZE,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def train_detector(
    detector: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = config.DETECTOR_EPOCHS,
    lr: float = config.DETECTOR_LR,
    weight_decay: float = config.DETECTOR_WEIGHT_DECAY,
    checkpoint_path: Path = config.DETECTOR_CHECKPOINT_PATH,
    latest_checkpoint_path: Optional[Path] = None,
    threshold: float = config.DETECTOR_THRESHOLD,
    resume: bool = True,
    history_json_path: Optional[Path] = None,
    experiment_id: str = "detector",
) -> Dict[str, List[float]]:
    """
    Train binary adversarial detector with BCEWithLogitsLoss, best/latest checkpointing,
    and automatic training resumption.
    """
    checkpoint_path = Path(checkpoint_path)
    if latest_checkpoint_path is None:
        latest_checkpoint_path = checkpoint_path.parent / f"latest_{checkpoint_path.name}"
    latest_checkpoint_path = Path(latest_checkpoint_path)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(detector.parameters(), lr=lr, weight_decay=weight_decay)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0
    start_epoch = 1

    # Resumable training check
    if resume and latest_checkpoint_path.exists():
        print(f"\n[Detector Training] Found existing checkpoint at: {latest_checkpoint_path}")
        try:
            ckpt = load_checkpoint(latest_checkpoint_path, detector, optimizer, device=device)
            last_epoch = ckpt.get("epoch", 0)
            best_val_acc = ckpt.get("best_val_acc", ckpt.get("val_acc", -1.0))
            if "history" in ckpt and ckpt["history"]:
                history = ckpt["history"]

            if last_epoch >= epochs:
                print(
                    f"[Detector Training] Experiment '{experiment_id}' already COMPLETED {last_epoch}/{epochs} epochs. "
                    f"Skipping redundant training. (Best Val Acc: {best_val_acc*100:.2f}%)"
                )
                if checkpoint_path.exists():
                    load_checkpoint(checkpoint_path, detector, device=device)
                return history

            start_epoch = last_epoch + 1
            print(
                f"[Detector Training] RESUMING from epoch {start_epoch}/{epochs}. "
                f"(Previous Best Val Acc: {best_val_acc*100:.2f}%)"
            )
        except Exception as e:
            print(f"[!] Warning: Could not resume detector checkpoint ({e}). Starting fresh.")
            start_epoch = 1
    else:
        print(f"\n[Detector Training] STARTING FRESH: {epochs} epochs on device: {device}")

    # Training Loop
    for epoch in range(start_epoch, epochs + 1):
        detector.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = detector(images).squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        detector.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = detector(images).squeeze(-1)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * images.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs >= threshold).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
        )

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        rng_state = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        state = {
            "epoch": epoch,
            "experiment_id": experiment_id,
            "model_state_dict": detector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": float(val_acc),
            "val_loss": float(val_loss),
            "best_val_acc": float(best_val_acc),
            "threshold": threshold,
            "history": history,
            "rng_state": rng_state,
        }

        save_checkpoint(
            state=state,
            filepath=latest_checkpoint_path,
            is_best=is_best,
            best_filepath=checkpoint_path,
        )

        if history_json_path:
            save_json_history(history, history_json_path)

    if checkpoint_path.exists():
        print(f"\n[Detector Training] Loading best weights from: {checkpoint_path}")
        load_checkpoint(checkpoint_path, detector, device=device)

    return history


def evaluate_detector_test_set(
    detector: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    threshold: float = config.DETECTOR_THRESHOLD,
    save_cm_path: Optional[Path] = None,
    save_roc_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate adversarial detector on test set.
    Calculates Accuracy, Precision, Recall, F1-Score, False Positive Rate (FPR),
    False Negative Rate (FNR), ROC-AUC score, and confusion matrix.
    """
    detector.eval()
    all_probs: List[float] = []
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = detector(images).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().tolist()
            preds = [(1 if p >= threshold else 0) for p in probs]

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend([int(lbl) for lbl in labels.tolist()])

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    all_probs_np = np.array(all_probs)

    acc = float((all_preds_np == all_labels_np).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels_np, all_preds_np, average="binary", zero_division=0
    )

    cm = confusion_matrix(all_labels_np, all_preds_np)
    # cm layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # ROC-AUC calculation
    try:
        roc_auc = float(roc_auc_score(all_labels_np, all_probs_np))
    except Exception:
        roc_auc = 0.0

    # Save ROC curve plot
    if save_roc_path:
        save_roc_path = Path(save_roc_path)
        save_roc_path.parent.mkdir(parents=True, exist_ok=True)
        fpr_curve, tpr_curve, _ = roc_curve(all_labels_np, all_probs_np)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr_curve, tpr_curve, "b-", linewidth=2, label=f"Detector ROC (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.50)")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
        ax.set_title("Adversarial Detector ROC Curve", fontsize=12)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(save_roc_path), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Plot confusion matrix
    if save_cm_path:
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
            title="Adversarial Detector Confusion Matrix (Test Set)",
            ylabel="True Label",
            xlabel="Predicted Label",
        )

        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11,
                )
        plt.tight_layout()
        plt.savefig(str(save_cm_path), dpi=300, bbox_inches="tight")
        plt.close(fig)

    return {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "roc_auc": roc_auc,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": cm.tolist(),
    }


# -------------------------------------------------------------------------
# Module 4: Integrated ModelShield Defense Evaluation (Phase 8)
# -------------------------------------------------------------------------

def evaluate_firewall_defense(
    classifier: nn.Module,
    detector: nn.Module,
    clean_loader: DataLoader,
    epsilon: float,
    threshold: float,
    device: torch.device,
    max_samples: Optional[int] = 1000,
) -> Dict[str, Any]:
    """
    Evaluate the end-to-end protective utility of ModelShield:
    Simulates real-world traffic of clean and FGSM adversarial images.

    Compares:
    1. Unprotected Classifier: Accuracy on clean vs under attack.
    2. Protected Pipeline:
       - Adversarial samples blocked (Attack Threat Mitigated).
       - Clean samples allowed and correctly classified (Utility Retained).
       - False alarm rate (Clean samples blocked).
       - Effective accuracy across mixed traffic.
    """
    classifier.eval()
    detector.eval()

    total_clean = 0
    total_adv = 0

    # Unprotected metrics
    unprotected_clean_correct = 0
    unprotected_adv_correct = 0

    # Protected firewall metrics
    clean_allowed = 0
    clean_blocked = 0
    clean_allowed_correct = 0

    adv_blocked = 0  # Desired defense action
    adv_allowed = 0  # Attack slipped through
    adv_allowed_correct = 0

    collected = 0
    for images, labels in clean_loader:
        images = images.to(device)
        labels = labels.to(device)

        batch_len = images.size(0)
        if max_samples and (collected + batch_len) > max_samples:
            images = images[: max_samples - collected]
            labels = labels[: max_samples - collected]
            batch_len = images.size(0)

        # 1. Generate adversarial counterparts
        adv_images = fgsm_attack(classifier, images, labels, epsilon=epsilon)

        with torch.no_grad():
            # Classifier predictions
            clean_preds = classifier(images).argmax(dim=1)
            adv_preds = classifier(adv_images).argmax(dim=1)

            # Detector predictions
            clean_det_probs = torch.sigmoid(detector(images).squeeze(-1))
            adv_det_probs = torch.sigmoid(detector(adv_images).squeeze(-1))

            clean_is_blocked = (clean_det_probs >= threshold)
            adv_is_blocked = (adv_det_probs >= threshold)

        # Update Unprotected counts
        unprotected_clean_correct += (clean_preds == labels).sum().item()
        unprotected_adv_correct += (adv_preds == labels).sum().item()
        total_clean += batch_len
        total_adv += batch_len

        # Update Protected counts for Clean images
        clean_blocked += clean_is_blocked.sum().item()
        clean_allowed_mask = ~clean_is_blocked
        clean_allowed += clean_allowed_mask.sum().item()
        clean_allowed_correct += ((clean_preds == labels) & clean_allowed_mask).sum().item()

        # Update Protected counts for Adversarial images
        adv_blocked += adv_is_blocked.sum().item()
        adv_allowed_mask = ~adv_is_blocked
        adv_allowed += adv_allowed_mask.sum().item()
        adv_allowed_correct += ((adv_preds == labels) & adv_allowed_mask).sum().item()

        collected += batch_len
        if max_samples and collected >= max_samples:
            break

    # Ratios
    unprotected_clean_acc = unprotected_clean_correct / total_clean
    unprotected_adv_acc = unprotected_adv_correct / total_adv

    threat_mitigation_rate = adv_blocked / total_adv  # % of adversarial inputs blocked
    false_alarm_rate = clean_blocked / total_clean      # % of clean inputs blocked
    clean_retention_acc = clean_allowed_correct / total_clean

    return {
        "epsilon": epsilon,
        "threshold": threshold,
        "total_evaluated_clean": total_clean,
        "total_evaluated_adversarial": total_adv,
        "unprotected_clean_accuracy": float(unprotected_clean_acc),
        "unprotected_adversarial_accuracy": float(unprotected_adv_acc),
        "attack_threat_mitigation_rate": float(threat_mitigation_rate),
        "false_alarm_rate": float(false_alarm_rate),
        "clean_utility_retained": float(clean_retention_acc),
        "clean_blocked_count": int(clean_blocked),
        "adv_blocked_count": int(adv_blocked),
    }
