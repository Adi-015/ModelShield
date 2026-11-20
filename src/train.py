"""
ModelShield - Training and Evaluation Module
Handles training, validation, checkpoint saving, metric computation,
and evaluation for both the CIFAR-10 classifier and the adversarial detector.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import config
from src.attacks import fgsm_attack
from src.data import AdversarialDataset


# -------------------------------------------------------------------------
# Training History Plotting
# -------------------------------------------------------------------------

def plot_training_history(
    history: Dict[str, List[float]],
    title: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot train vs validation loss and accuracy across epochs.
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
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# -------------------------------------------------------------------------
# Module 1: Classifier Training and Evaluation
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
    """Evaluate classifier loss and accuracy on validation/test set."""
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
) -> Dict[str, List[float]]:
    """
    Train classifier with Adam optimizer, CrossEntropyLoss, and best-checkpoint saving.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0

    print(f"[Classifier Training] Starting {epochs} epochs on device: {device}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_classifier_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate_classifier(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
        )

        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            print(f"  -> Saved new best classifier checkpoint (Val Acc: {val_acc * 100:.2f}%)")

    return history


def evaluate_classifier_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    save_cm_path: Optional[Path] = None,
) -> Dict[str, object]:
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
    )
    cm = confusion_matrix(all_labels_np, all_preds_np)

    # Plot confusion matrix
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

    # Annotate counts in cells
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
    if save_cm_path:
        plt.savefig(save_cm_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "test_accuracy": acc,
        "classification_report": cls_report,
        "confusion_matrix": cm,
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

    Guarantees no data leakage between splits.
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

        # Generate adversarial counterpart
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
    threshold: float = config.DETECTOR_THRESHOLD,
) -> Dict[str, List[float]]:
    """
    Train binary adversarial detector with BCEWithLogitsLoss.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(detector.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0

    print(f"[Detector Training] Starting {epochs} epochs on device: {device}")

    for epoch in range(1, epochs + 1):
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

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": detector.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "threshold": threshold,
                },
                checkpoint_path,
            )
            print(f"  -> Saved new best detector checkpoint (Val Acc: {val_acc * 100:.2f}%)")

    return history


def evaluate_detector_test_set(
    detector: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    threshold: float = config.DETECTOR_THRESHOLD,
    save_cm_path: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Evaluate adversarial detector on test set.
    Calculates Accuracy, Precision, Recall, F1-Score, False Positive Rate (FPR),
    and confusion matrix.
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

    acc = float((all_preds_np == all_labels_np).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels_np, all_preds_np, average="binary", zero_division=0
    )

    cm = confusion_matrix(all_labels_np, all_preds_np)
    # cm layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Plot confusion matrix
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
    if save_cm_path:
        plt.savefig(save_cm_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "false_positive_rate": fpr,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": cm,
    }
