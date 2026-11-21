"""
ModelShield - Integrated Firewall Pipeline Module
Inspects incoming images for adversarial perturbation before deciding whether
to forward them to the target classifier or block them as suspicious.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import config
from src.models import AdversarialDetector, CIFAR10Classifier


class ModelShieldFirewall:
    """
    ModelShield Security Layer:
    An AI Firewall that detects adversarially modified images before
    they reach a sensitive image-classification model.
    """

    def __init__(
        self,
        classifier: Optional[CIFAR10Classifier] = None,
        detector: Optional[AdversarialDetector] = None,
        classifier_path: Optional[Union[str, Path]] = config.CLASSIFIER_CHECKPOINT_PATH,
        detector_path: Optional[Union[str, Path]] = config.DETECTOR_CHECKPOINT_PATH,
        threshold: float = config.DETECTOR_THRESHOLD,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the firewall with classifier and detector models.

        Args:
            classifier: Pre-instantiated classifier (optional).
            detector: Pre-instantiated detector (optional).
            classifier_path: Path to classifier weights.
            detector_path: Path to detector weights.
            threshold: Adversarial probability cutoff (>= threshold triggers BLOCKED).
            device: Execution device (defaults to auto-detected config.DEVICE).
        """
        self.device = device or config.DEVICE
        self.threshold = threshold

        # Initialize or assign classifier
        self.classifier = classifier or CIFAR10Classifier()
        self.classifier.to(self.device)

        # Initialize or assign detector
        self.detector = detector or AdversarialDetector()
        self.detector.to(self.device)

        # Load weights if paths exist
        self.classifier_loaded = False
        self.detector_loaded = False

        if classifier_path is not None:
            c_path = Path(classifier_path)
            if c_path.exists():
                self._load_weights(self.classifier, c_path)
                self.classifier_loaded = True

        if detector_path is not None:
            d_path = Path(detector_path)
            if d_path.exists():
                self._load_weights(self.detector, d_path)
                self.detector_loaded = True

        self.classifier.eval()
        self.detector.eval()

    @staticmethod
    def _load_weights(model: nn.Module, path: Path) -> None:
        """Safely load model weights from a state_dict or training checkpoint dictionary."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Unrecognized checkpoint format at {path}")

    @staticmethod
    def preprocess_image(image: Union[Image.Image, Path, str]) -> torch.Tensor:
        """
        Preprocess a PIL Image or image file path into a CIFAR-10 normalized tensor.

        Args:
            image: PIL Image instance or file path.
        Returns:
            Tensor of shape (1, 3, 32, 32).
        """
        if isinstance(image, (str, Path)):
            img_path = Path(image)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found at {img_path}")
            pil_img = Image.open(img_path).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")

        eval_transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SHAPE[1:]),
            transforms.ToTensor(),
            transforms.Normalize(config.CIFAR10_MEAN, config.CIFAR10_STD),
        ])
        tensor = eval_transform(pil_img).unsqueeze(0)
        return tensor

    def detect(self, image_tensor: torch.Tensor) -> float:
        """
        Estimate the probability that an image contains adversarial perturbations.

        Args:
            image_tensor: Normalized image tensor of shape (1, 3, 32, 32).
        Returns:
            Adversarial probability between 0.0 and 1.0.
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)
        self.detector.eval()

        with torch.no_grad():
            logit = self.detector(image_tensor).squeeze()
            prob = torch.sigmoid(logit).item()

        return float(prob)

    def classify(self, image_tensor: torch.Tensor) -> Tuple[int, str, float]:
        """
        Classify a clean image with the core classifier.

        Args:
            image_tensor: Normalized image tensor of shape (1, 3, 32, 32).
        Returns:
            (predicted_class_id, predicted_class_name, confidence_score)
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)
        self.classifier.eval()

        with torch.no_grad():
            logits = self.classifier(image_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)
            pred_id = probs.argmax().item()
            conf = probs[pred_id].item()

        pred_name = config.CIFAR10_CLASSES[pred_id]
        return pred_id, pred_name, float(conf)

    def process(
        self,
        image_input: Union[torch.Tensor, Image.Image, Path, str],
        custom_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute the complete ModelShield Firewall pipeline on a single image:
        1. Receive and preprocess image.
        2. Evaluate adversarial probability via detector.
        3. Compare against threshold.
        4. If clean (< threshold): ALLOW and pass to classifier.
        5. If suspicious (>= threshold): BLOCK and flag.
        """
        threshold = custom_threshold if custom_threshold is not None else self.threshold

        # Preprocess
        if isinstance(image_input, torch.Tensor):
            tensor = image_input
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
        else:
            tensor = self.preprocess_image(image_input)

        # Adversarial Detection
        adv_prob = self.detect(tensor)
        is_suspicious = adv_prob >= threshold

        report: Dict[str, Any] = {
            "adversarial_probability": adv_prob,
            "threshold": threshold,
            "is_adversarial": is_suspicious,
        }

        # Firewall Decision Gate
        if is_suspicious:
            report.update({
                "decision": "BLOCKED",
                "reason": "The detector classified the image as suspicious.",
                "predicted_class_id": None,
                "predicted_class_name": None,
                "classification_confidence": None,
            })
        else:
            pred_id, pred_name, conf = self.classify(tensor)
            report.update({
                "decision": "ALLOWED",
                "reason": "Image passed adversarial verification.",
                "predicted_class_id": pred_id,
                "predicted_class_name": pred_name,
                "classification_confidence": conf,
            })

        return report

    def process_batch(
        self,
        batch_tensors: torch.Tensor,
        custom_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Vectorized evaluation of the firewall over a batch of tensors.
        Args:
            batch_tensors: Normalized tensor of shape (B, 3, 32, 32).
        Returns:
            List of inspection report dictionaries for each sample in the batch.
        """
        threshold = custom_threshold if custom_threshold is not None else self.threshold
        batch_tensors = batch_tensors.to(self.device)
        b_size = batch_tensors.size(0)

        with torch.no_grad():
            det_logits = self.detector(batch_tensors).squeeze(-1)
            det_probs = torch.sigmoid(det_logits).cpu().tolist()

            cls_logits = self.classifier(batch_tensors)
            cls_probs = F.softmax(cls_logits, dim=1).cpu()
            cls_preds = cls_probs.argmax(dim=1).tolist()
            cls_confs = [cls_probs[i, cls_preds[i]].item() for i in range(b_size)]

        reports = []
        for i in range(b_size):
            prob = det_probs[i]
            is_suspicious = prob >= threshold
            if is_suspicious:
                reports.append({
                    "sample_idx": i,
                    "adversarial_probability": float(prob),
                    "threshold": threshold,
                    "is_adversarial": True,
                    "decision": "BLOCKED",
                    "reason": "The detector classified the image as suspicious.",
                    "predicted_class_id": None,
                    "predicted_class_name": None,
                    "classification_confidence": None,
                })
            else:
                pred_id = cls_preds[i]
                reports.append({
                    "sample_idx": i,
                    "adversarial_probability": float(prob),
                    "threshold": threshold,
                    "is_adversarial": False,
                    "decision": "ALLOWED",
                    "reason": "Image passed adversarial verification.",
                    "predicted_class_id": pred_id,
                    "predicted_class_name": config.CIFAR10_CLASSES[pred_id],
                    "classification_confidence": float(cls_confs[i]),
                })
        return reports

    def format_security_report(
        self,
        result: Dict[str, Any],
        input_name: str = "test_image.png",
    ) -> str:
        """
        Format structured firewall result into an executive security report.
        """
        lines = [
            "==================================================",
            "           MODELSHIELD SECURITY REPORT            ",
            "==================================================",
            f"Input: {input_name}",
            f"Adversarial probability: {result['adversarial_probability']:.4f}",
            f"Threshold: {result['threshold']:.2f}",
            f"Firewall decision: {result['decision']}",
        ]

        if result["decision"] == "BLOCKED":
            lines.append(f"Reason: {result['reason']}")
        else:
            lines.append(f"Predicted class: {result['predicted_class_name']}")
            lines.append(f"Classification confidence: {result['classification_confidence'] * 100:.2f}%")

        lines.append("==================================================")
        return "\n".join(lines)
