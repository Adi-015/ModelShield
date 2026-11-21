"""
ModelShield Phase 2 - Data Ingestion, Transforms & Partitioning Module
Maintains deterministic 40,000 / 10,000 / 10,000 CIFAR-10 data partitioning,
ensuring strict data isolation, reproducible DataLoader worker initialization,
and non-shuffled validation/test evaluation loaders.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

try:
    from Phase2 import config
except ImportError:
    import config


# -------------------------------------------------------------------------
# Differentiable Tensor Normalization / Denormalization Helpers
# -------------------------------------------------------------------------

def get_mean_std_tensors(
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return CIFAR-10 mean and std as shape (1, 3, 1, 1) tensors on target device."""
    mean = torch.tensor(config.CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(config.CIFAR10_STD, device=device).view(1, 3, 1, 1)
    return mean, std


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse CIFAR-10 normalization, mapping tensor values back to [0, 1] physical pixel range.

    Args:
        tensor: Normalized image tensor of shape (B, 3, H, W) or (3, H, W).
    Returns:
        Clamped tensor in [0.0, 1.0] physical pixel space.
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
    Apply standard CIFAR-10 normalization to a raw [0, 1] tensor.

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
# Dataset Wrappers & Partitions
# -------------------------------------------------------------------------

class TransformedDataset(Dataset):
    """
    Applies a specified torchvision transform to a given Dataset or Subset.
    Guarantees that train augmentations are not applied to validation or test data.
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


class IndexedDataset(Dataset):
    """
    Wraps a base dataset and returns (sample_id, image, label) tuples.
    Used for tracking individual sample identities across clean and adversarial runs.
    """

    def __init__(self, base_dataset: Dataset, offset: int = 0):
        self.base_dataset = base_dataset
        self.offset = offset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor, int]:
        sample_id = self.offset + idx
        item = self.base_dataset[idx]
        image, label = item[0], item[1]
        return sample_id, image, label


def get_cifar10_transforms(
    augmentation: str = "standard",
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns train and evaluation transforms for CIFAR-10.
    Evaluation transform strictly consists only of ToTensor + Normalize.
    """
    if augmentation == "enhanced":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(config.CIFAR10_MEAN, config.CIFAR10_STD),
        ])
    else:
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
    data_dir: Optional[Path] = None,
    batch_size: int = config.CLASSIFIER_BATCH_SIZE,
    val_split_ratio: float = config.VAL_SPLIT_RATIO,
    random_seed: int = config.RANDOM_SEED,
    augmentation: str = "standard",
    num_workers: int = 0,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Downloads CIFAR-10, partitions the 50,000 training images into 40,000 train and 10,000 val,
    and returns DataLoaders for train, validation, and untouched test splits.

    Guarantees:
    - Zero data leakage between splits.
    - Explicit generator and worker_init_fn for reproducibility.
    - shuffle=False for validation and test DataLoaders.
    """
    target_data_dir = data_dir or config.DATA_DIR
    train_transform, eval_transform = get_cifar10_transforms(augmentation=augmentation)

    # Raw training dataset without transforms to apply post-split transforms
    raw_train_dataset = torchvision.datasets.CIFAR10(
        root=str(target_data_dir),
        train=True,
        download=download,
        transform=None,
    )

    # Official untouched test set (10,000 images)
    raw_test_dataset = torchvision.datasets.CIFAR10(
        root=str(target_data_dir),
        train=False,
        download=download,
        transform=eval_transform,
    )

    # Deterministic train/validation split
    total_train = len(raw_train_dataset)
    val_size = int(total_train * val_split_ratio)
    train_size = total_train - val_size

    generator = config.get_generator(random_seed)
    train_indices, val_indices = torch.utils.data.random_split(
        range(total_train), [train_size, val_size], generator=generator
    )

    train_subset = Subset(raw_train_dataset, train_indices.indices)
    val_subset = Subset(raw_train_dataset, val_indices.indices)

    train_dataset = TransformedDataset(train_subset, transform=train_transform)
    val_dataset = TransformedDataset(val_subset, transform=eval_transform)
    test_dataset = raw_test_dataset

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=config.seed_worker if num_workers > 0 else None,
        generator=config.get_generator(random_seed),
        pin_memory=torch.cuda.is_available(),
    )

    # Non-shuffled validation DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=config.seed_worker if num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
    )

    # Non-shuffled untouched test DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=config.seed_worker if num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


# -------------------------------------------------------------------------
# Adversarial Dataset Container for Detector
# -------------------------------------------------------------------------

class AdversarialDataset(Dataset):
    """
    Dataset storing clean and adversarial image pairs for detector training/evaluation.
    Labels:
    - 0: Clean image
    - 1: Adversarially perturbed image
    """

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        assert images.size(0) == labels.size(0), "Images and labels count must match"
        self.images = images.float()
        self.labels = labels.float()
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return self.images.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]
