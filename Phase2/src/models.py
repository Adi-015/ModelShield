"""
ModelShield Phase 2 - Neural Network Architectures
Defines the preserved custom 3-block CNN classifier and the adversarial detector CNN.
Provides structured weight initialization and parameter auditing utilities.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn


def initialize_weights(model: nn.Module, method: str = "kaiming_normal") -> nn.Module:
    """
    Apply structured weight initialization across Conv2d, Linear, and BatchNorm layers.

    Args:
        model: PyTorch neural network.
        method: 'kaiming_normal' (He normal) or 'kaiming_uniform'.

    Returns:
        The initialized model.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if method == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif method == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """Return structured parameter and layer summary for logging and audit."""
    trainable_params = count_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "model_class": model.__class__.__name__,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),
    }


class CIFAR10Classifier(nn.Module):
    """
    Custom 3-Block Convolutional Neural Network for CIFAR-10 image classification.
    Preserved as the baseline architecture for Phase 2.

    Architecture:
    - Conv Block 1: Conv(3->32) -> BN -> ReLU -> Conv(32->32) -> BN -> ReLU -> MaxPool -> Dropout2d(0.25)
    - Conv Block 2: Conv(32->64) -> BN -> ReLU -> Conv(64->64) -> BN -> ReLU -> MaxPool -> Dropout2d(0.25)
    - Conv Block 3: Conv(64->128) -> BN -> ReLU -> MaxPool -> Dropout2d(0.25)
    - Dense Head: Linear(2048->256) -> BN1d -> ReLU -> Dropout(0.50) -> Linear(256->10)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

        # Block 1: input (B, 3, 32, 32) -> (B, 32, 16, 16)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),
        )

        # Block 2: (B, 32, 16, 16) -> (B, 64, 8, 8)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),
        )

        # Block 3: (B, 64, 8, 8) -> (B, 128, 4, 4)
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),
        )

        # Classification Head: 128 * 4 * 4 = 2048
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.50),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing raw logits.
        Args:
            x: Input tensor of shape (B, 3, 32, 32)
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        logits = self.classifier(out)
        return logits


class AdversarialDetector(nn.Module):
    """
    Custom 3-Block Convolutional Neural Network for binary adversarial detection.
    - Class 0: Clean image
    - Class 1: Adversarially perturbed image

    Architecture:
    - Conv Block 1: Conv(3->32) -> BN -> ReLU -> MaxPool -> Dropout2d(0.20)
    - Conv Block 2: Conv(32->64) -> BN -> ReLU -> MaxPool -> Dropout2d(0.20)
    - Conv Block 3: Conv(64->128) -> BN -> ReLU -> MaxPool -> Dropout2d(0.30)
    - Dense Head: Linear(2048->128) -> ReLU -> Dropout(0.40) -> Linear(128->1)
    """

    def __init__(self, in_channels: int = 3, input_channels: Optional[int] = None):
        super().__init__()
        channels = input_channels if input_channels is not None else in_channels

        # Block 1: (B, 3, 32, 32) -> (B, 32, 16, 16)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.20),
        )

        # Block 2: (B, 32, 16, 16) -> (B, 64, 8, 8)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.20),
        )

        # Block 3: (B, 64, 8, 8) -> (B, 128, 4, 4)
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.30),
        )

        # Binary Classification Head: 128 * 4 * 4 = 2048 -> 1
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.40),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing raw binary logit.
        Args:
            x: Input tensor of shape (B, 3, 32, 32)
        Returns:
            Logits tensor of shape (B, 1)
        """
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        logits = self.head(out)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probability in [0.0, 1.0]."""
        return torch.sigmoid(self.forward(x))
