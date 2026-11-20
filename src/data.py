"""
ModelShield - Data Ingestion and Preprocessing Module
Handles CIFAR-10 downloading, train/validation/test splitting,
normalization/denormalization, DataLoaders, and adversarial dataset creation.
"""

from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
import torchvision
import torchvision.transforms as transforms

import config


# -------------------------------------------------------------------------
# Tensor Normalization and Denormalization Helpers
# -------------------------------------------------------------------------

def get_mean_std_tensors(
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return mean and std as shape (1, 3, 1, 1) tensors on target device."""
    mean = torch.tensor(config.CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(config.CIFAR10_STD, device=device).view(1, 3, 1, 1)
    return mean, std


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse CIFAR-10 normalization, mapping tensor values back to [0, 1].

    Args:
        tensor: Normalized image tensor of shape (B, 3, H, W) or (3, H, W).
    Returns:
        Tensor in range [0, 1] clamped.
    """
    is_single = tensor.dim() == 3
    if is_single:
        tensor = tensor.unsqueeze(0)

    mean, std = get_mean_std_tensors(tensor.device)
    raw = tensor * std + mean
    raw = torch.clamp(raw, 0.0, 1.0)

    if is_single:
        raw = raw.squeeze(0)
    return raw


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply CIFAR-10 normalization to a raw [0, 1] tensor.

    Args:
        tensor: Raw image tensor in range [0, 1] of shape (B, 3, H, W) or (3, H, W).
    Returns:
        Normalized tensor.
    """
    is_single = tensor.dim() == 3
    if is_single:
        tensor = tensor.unsqueeze(0)

    mean, std = get_mean_std_tensors(tensor.device)
    normed = (tensor - mean) / std

    if is_single:
        normed = normed.squeeze(0)
    return normed


# -------------------------------------------------------------------------
# Dataset Transformations and Splitting
# -------------------------------------------------------------------------

class TransformedDataset(Dataset):
    """
    Applies a specified torchvision transform to a given Dataset or Subset.
    Useful for applying different transforms (e.g. train augmentation vs val)
    after splitting the base dataset.
    """

    def __init__(self, base_dataset: Dataset, transform: Optional[transforms.Compose] = None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.base_dataset[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_cifar10_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns train and evaluation transforms for CIFAR-10.
    Train: RandomCrop + RandomHorizontalFlip + ToTensor + Normalize.
    Eval: ToTensor + Normalize.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config.CIFAR10_MEAN, config.CIFAR10_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.CIFAR10_MEAN, config.CIFAR10_STD),
    ])

    return train_transform, eval_transform


def get_cifar10_dataloaders(
    data_dir: Path = config.DATA_DIR,
    batch_size: int = config.CLASSIFIER_BATCH_SIZE,
    val_split_ratio: float = config.VAL_SPLIT_RATIO,
    random_seed: int = config.RANDOM_SEED,
    num_workers: int = 0,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Downloads CIFAR-10, partitions the training split into train and validation sets,
    and returns DataLoaders for train, validation, and untouched test splits.

    Guarantees no data leakage between splits.
    """
    train_transform, eval_transform = get_cifar10_transforms()

    # Load raw dataset without transforms so we can apply different transforms to train vs val
    raw_train_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=download,
        transform=None,
    )

    # Untouched official test set
    raw_test_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=download,
        transform=eval_transform,
    )

    # Deterministic split into train and validation sets
    total_train = len(raw_train_dataset)
    val_size = int(total_train * val_split_ratio)
    train_size = total_train - val_size

    generator = torch.Generator().manual_seed(random_seed)
    train_indices, val_indices = torch.utils.data.random_split(
        range(total_train), [train_size, val_size], generator=generator
    )

    train_subset = Subset(raw_train_dataset, train_indices.indices)
    val_subset = Subset(raw_train_dataset, val_indices.indices)

    train_dataset = TransformedDataset(train_subset, transform=train_transform)
    val_dataset = TransformedDataset(val_subset, transform=eval_transform)
    test_dataset = raw_test_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


# -------------------------------------------------------------------------
# Adversarial Dataset Container for Module 3 (Firewall Detector)
# -------------------------------------------------------------------------

class AdversarialDataset(Dataset):
    """
    Dataset storing clean and adversarial image pairs for detector training.
    Labels:
    - 0: Clean image
    - 1: Adversarial image
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        assert images.size(0) == labels.size(0), "Images and labels count must match"
        self.images = images.float()
        self.labels = labels.float()

    def __len__(self) -> int:
        return self.images.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]
