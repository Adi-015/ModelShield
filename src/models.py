"""
ModelShield - Neural Network Architectures
Defines the CIFAR-10 image classifier and the adversarial detector CNNs.
Preserves the exact custom CNN architectures while providing initialization helpers.
"""

from typing import Optional
import torch
import torch.nn as nn


def initialize_weights(model: nn.Module, method: str = "kaiming_normal") -> nn.Module:
    """
    Apply structured weight initialization across Conv2d, Linear, and BatchNorm layers.
    Used in controlled optimization experiments without altering architecture.

    Args:
        model: PyTorch neural network.
        method: 'kaiming_normal' (default, He normal) or 'kaiming_uniform'.

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
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    return model


class CIFAR10Classifier(nn.Module):
    """
    Custom Convolutional Neural Network for 10-class CIFAR-10 image classification.

    Exact Architecture (Preserved Baseline):
    - Conv Block 1: Conv(3->32) -> BN -> ReLU -> Conv(32->32) -> BN -> ReLU -> MaxPool -> Dropout(0.25)
    - Conv Block 2: Conv(32->64) -> BN -> ReLU -> Conv(64->64) -> BN -> ReLU -> MaxPool -> Dropout(0.25)
    - Conv Block 3: Conv(64->128) -> BN -> ReLU -> MaxPool -> Dropout(0.25)
    - Fully Connected Head: Linear(2048->256) -> BN -> ReLU -> Dropout(0.5) -> Linear(256->10)

    Outputs raw unnormalized logits for use with CrossEntropyLoss.
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
            nn.Dropout(p=0.5),
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
    Custom Convolutional Neural Network for binary adversarial detection.
    - Class 0: Clean image
    - Class 1: Adversarially perturbed image (FGSM)

    Exact Architecture (Preserved Baseline):
    - Conv Block 1: Conv(3->32) -> BN -> ReLU -> MaxPool -> Dropout(0.2)
    - Conv Block 2: Conv(32->64) -> BN -> ReLU -> MaxPool -> Dropout(0.2)
    - Conv Block 3: Conv(64->128) -> BN -> ReLU -> MaxPool -> Dropout(0.3)
    - Dense Head: Linear(2048->128) -> ReLU -> Dropout(0.4) -> Linear(128->1)

    Outputs a single raw logit per image for use with BCEWithLogitsLoss.
    """

    def __init__(self):
        super().__init__()

        # Block 1: (B, 3, 32, 32) -> (B, 32, 16, 16)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),
        )

        # Block 2: (B, 32, 16, 16) -> (B, 64, 8, 8)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),
        )

        # Block 3: (B, 64, 8, 8) -> (B, 128, 4, 4)
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.3),
        )

        # Classification Head: 128 * 4 * 4 = 2048 -> 1
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
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
