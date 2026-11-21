"""
ModelShield Phase 2 - Training, Checkpointing, and Optimization Module
Handles resumable training for the baseline/optimized classifier and the multi-attack detector,
ensuring non-zero checkpoint file size verification, comprehensive RNG state preservation,
and structured progress serialization.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import json
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from Phase2 import config
except ImportError:
    import config


# -------------------------------------------------------------------------
# Checkpoint Saving & Loading with Validation
# -------------------------------------------------------------------------

def _atomic_torch_save(state: Dict[str, Any], destination: Union[str, Path]) -> int:
    """
    Atomically save PyTorch state dictionary to destination:
    1. Writes to a temporary staging file in the target directory.
    2. Validates that the temporary file exists and has non-zero size.
    3. Uses os.replace() to atomically replace the destination checkpoint.
    4. Guarantees cleanup of temporary staging file if saving fails.
    Returns file size in bytes.
    """
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f".tmp_{target.name}_{os.getpid()}_{random.randint(1000, 9999)}.pt"

    try:
        torch.save(state, tmp_path)
        if not tmp_path.exists():
            raise FileNotFoundError(f"Staged temporary checkpoint not found at {tmp_path}")
        size_bytes = tmp_path.stat().st_size
        if size_bytes == 0:
            raise IOError(f"Staged temporary checkpoint has 0 bytes: {tmp_path}")
        os.replace(tmp_path, target)
        return size_bytes
    except Exception as e:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise e


def save_checkpoint(
    state: Dict[str, Any],
    filepath: Union[str, Path],
    is_best: bool = False,
    best_filepath: Optional[Union[str, Path]] = None,
) -> None:
    """
    Atomically save state dictionary to disk with nonzero file size verification and staging.

    Args:
        state: Checkpoint dictionary containing model/optimizer/scheduler states, epoch, etc.
        filepath: Target destination for latest checkpoint.
        is_best: If True, also atomically save a copy to best_filepath.
        best_filepath: Destination path for the best model weights.
    """
    target_path = Path(filepath)
    file_size_bytes = _atomic_torch_save(state, target_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    epoch = state.get("epoch", "?")
    val_acc = state.get("val_acc", 0.0)

    if is_best and best_filepath is not None:
        best_path = Path(best_filepath)
        best_size_bytes = _atomic_torch_save(state, best_path)
        best_size_mb = best_size_bytes / (1024 * 1024)
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
    filepath: Union[str, Path, nn.Module],
    model: Union[nn.Module, str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Safely load model weights, optimizer state, scheduler state, and RNG states.
    Supports either (filepath, model) or (model, filepath) argument ordering.
    """
    if isinstance(filepath, nn.Module) and isinstance(model, (str, Path)):
        filepath, model = model, filepath

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

    # Restore RNG state for deterministic resumption
    if "rng_state" in checkpoint:
        try:
            rng_info = checkpoint["rng_state"]
            if "torch" in rng_info and rng_info["torch"] is not None:
                torch.set_rng_state(rng_info["torch"])
            if "numpy" in rng_info and rng_info["numpy"] is not None:
                np.random.set_state(rng_info["numpy"])
            if "python" in rng_info and rng_info["python"] is not None:
                random.setstate(rng_info["python"])
            if torch.cuda.is_available() and "cuda" in rng_info and rng_info["cuda"] is not None:
                torch.cuda.set_rng_state_all(rng_info["cuda"])
        except Exception as e:
            print(f"  [Notice] RNG state could not be restored: {e}")

    return checkpoint


def save_json_history(history: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save serializable training history to a JSON file with automatic directory creation and type conversion."""
    config.save_json(history, filepath)


def plot_training_history(
    history: Dict[str, List[float]],
    title: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot train vs validation loss and accuracy curves across epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r--s", label="Val Loss")
    ax1.set_title(f"{title} - Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

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
# Classifier Training Pipeline
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

    return running_loss / total, correct / total


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

    return running_loss / total, correct / total


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: Optional[torch.device] = None,
    epochs: int = config.CLASSIFIER_EPOCHS,
    lr: float = config.CLASSIFIER_LR,
    weight_decay: float = config.CLASSIFIER_WEIGHT_DECAY,
    checkpoint_path: Optional[Path] = None,
    latest_checkpoint_path: Optional[Path] = None,
    scheduler_type: Optional[str] = None,
    scheduler_params: Optional[Dict[str, Any]] = None,
    label_smoothing: float = 0.0,
    resume: bool = True,
    history_json_path: Optional[Path] = None,
    experiment_id: str = "phase2_classifier",
) -> Dict[str, List[float]]:
    """
    Train classifier with Adam optimizer, optional LR scheduler, best/latest checkpointing,
    and automatic training resumption.
    """
    target_device = device or config.DEVICE
    if checkpoint_path is None:
        checkpoint_path = config.CHECKPOINT_DIR / config.CLASSIFIER_BASELINE_BEST
    checkpoint_path = Path(checkpoint_path)

    if latest_checkpoint_path is None:
        latest_checkpoint_path = checkpoint_path.parent / f"latest_{checkpoint_path.name}"
    latest_checkpoint_path = Path(latest_checkpoint_path)

    if history_json_path is None:
        history_json_path = config.CLASSIFIER_HISTORY_FILE
    history_json_path = Path(history_json_path)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

    # Resumption Check
    if resume and latest_checkpoint_path.exists():
        print(f"\n[Classifier] Found existing checkpoint: {latest_checkpoint_path}")
        try:
            ckpt = load_checkpoint(latest_checkpoint_path, model, optimizer, scheduler, device=target_device)
            last_epoch = ckpt.get("epoch", 0)
            best_val_acc = ckpt.get("best_val_acc", ckpt.get("val_acc", -1.0))
            if "history" in ckpt and ckpt["history"]:
                history = ckpt["history"]

            if last_epoch >= epochs:
                print(f"[Classifier] Experiment '{experiment_id}' already COMPLETED {last_epoch}/{epochs} epochs. Skipping training.")
                if checkpoint_path.exists():
                    load_checkpoint(checkpoint_path, model, device=target_device)
                return history

            start_epoch = last_epoch + 1
            print(f"[Classifier] Resuming from epoch {start_epoch}/{epochs} (Previous Best: {best_val_acc*100:.2f}%)")
        except Exception as e:
            print(f"[!] Warning: Could not resume from checkpoint ({e}). Starting fresh from epoch 1.")
            start_epoch = 1
    else:
        print(f"\n[Classifier] Starting fresh training: {epochs} epochs on device: {target_device}")

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = train_classifier_epoch(model, train_loader, criterion, optimizer, target_device)
        val_loss, val_acc = validate_classifier(model, val_loader, criterion, target_device)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))

        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] (lr: {cur_lr:.6f}) "
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
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "val_acc": float(val_acc),
            "val_loss": float(val_loss),
            "best_val_acc": float(best_val_acc),
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
        load_checkpoint(checkpoint_path, model, device=target_device)

    return history


# -------------------------------------------------------------------------
# Adversarial Detector Training Pipeline
# -------------------------------------------------------------------------

def train_detector(
    detector: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: Optional[torch.device] = None,
    epochs: int = config.DETECTOR_EPOCHS,
    lr: float = config.DETECTOR_LR,
    weight_decay: float = config.DETECTOR_WEIGHT_DECAY,
    checkpoint_path: Optional[Path] = None,
    latest_checkpoint_path: Optional[Path] = None,
    threshold: float = config.DETECTOR_DEFAULT_THRESHOLD,
    resume: bool = True,
    history_json_path: Optional[Path] = None,
    experiment_id: str = "phase2_detector_multi",
) -> Dict[str, List[float]]:
    """
    Train binary adversarial detector with BCEWithLogitsLoss, best/latest checkpointing,
    and automatic training resumption.
    """
    target_device = device or config.DEVICE
    if checkpoint_path is None:
        checkpoint_path = config.CHECKPOINT_DIR / config.DETECTOR_MULTI_BEST
    checkpoint_path = Path(checkpoint_path)

    if latest_checkpoint_path is None:
        latest_checkpoint_path = checkpoint_path.parent / f"latest_{checkpoint_path.name}"
    latest_checkpoint_path = Path(latest_checkpoint_path)

    if history_json_path is None:
        history_json_path = config.DETECTOR_HISTORY_FILE
    history_json_path = Path(history_json_path)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(detector.parameters(), lr=lr, weight_decay=weight_decay)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0
    start_epoch = 1

    if resume and latest_checkpoint_path.exists():
        print(f"\n[Detector] Found existing checkpoint: {latest_checkpoint_path}")
        try:
            ckpt = load_checkpoint(latest_checkpoint_path, detector, optimizer, device=target_device)
            last_epoch = ckpt.get("epoch", 0)
            best_val_acc = ckpt.get("best_val_acc", ckpt.get("val_acc", -1.0))
            if "history" in ckpt and ckpt["history"]:
                history = ckpt["history"]

            if last_epoch >= epochs:
                print(f"[Detector] Experiment '{experiment_id}' already COMPLETED {last_epoch}/{epochs} epochs. Skipping training.")
                if checkpoint_path.exists():
                    load_checkpoint(checkpoint_path, detector, device=target_device)
                return history

            start_epoch = last_epoch + 1
            print(f"[Detector] Resuming from epoch {start_epoch}/{epochs} (Previous Best: {best_val_acc*100:.2f}%)")
        except Exception as e:
            print(f"[!] Warning: Could not resume detector ({e}). Starting fresh.")
            start_epoch = 1
    else:
        print(f"\n[Detector] Starting fresh training: {epochs} epochs on device: {target_device}")

    for epoch in range(start_epoch, epochs + 1):
        detector.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(target_device)
            labels = labels.to(target_device)

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
                images = images.to(target_device)
                labels = labels.to(target_device)

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
        load_checkpoint(checkpoint_path, detector, device=target_device)

    return history
