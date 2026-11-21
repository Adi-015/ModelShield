"""
ModelShield Phase 2 - Integrated AI Firewall & Dynamic Threshold Calibration Module
Implements pre-inference adversarial screening, validation-based threshold calibration
under strict target False Positive Rates (FPR <= 1%), vectorized batch processing,
and end-to-end execution latency profiling.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

try:
    from Phase2 import config
except ImportError:
    import config

try:
    from Phase2.src.models import AdversarialDetector, CIFAR10Classifier
except ImportError:
    from src.models import AdversarialDetector, CIFAR10Classifier


def calibrate_threshold(
    detector_or_clean: Optional[Union[nn.Module, np.ndarray, List[float]]] = None,
    clean_loader_or_adv: Optional[Union[DataLoader, np.ndarray, List[float]]] = None,
    adv_val_loader: Optional[Union[DataLoader, np.ndarray, List[float]]] = None,
    target_fpr: float = config.FIREWALL_CONFIG["target_fpr"],
    device: Optional[torch.device] = None,
    resolution: int = 1000,
    save_path: Optional[Union[str, Path]] = None,
    detector: Optional[nn.Module] = None,
    clean_val_loader: Optional[DataLoader] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Calibrate the firewall decision threshold strictly using VALIDATION data.

    Supports two calling conventions:
    1. calibrate_threshold(detector=detector, clean_val_loader=clean_val_loader, adv_val_loader=None, target_fpr=0.01)
    2. calibrate_threshold(clean_scores_np, adv_scores_np, target_fpr=0.01)

    Objective:
        Find threshold tau* that guarantees clean False Positive Rate <= target_fpr (e.g. 1.0%),
        while maximizing adversarial detection recall on validation pairs.

    Crucial Integrity Rule:
        NEVER calibrate on test data.
    """
    if detector is not None:
        detector_or_clean = detector
    if clean_val_loader is not None:
        clean_loader_or_adv = clean_val_loader

    if detector_or_clean is None:
        raise ValueError("calibrate_threshold requires either a detector model or clean scores.")
    target_device = device or config.DEVICE

    # Check if first argument is a detector nn.Module or directly scores
    if isinstance(detector_or_clean, nn.Module):
        detector = detector_or_clean
        clean_val_loader = clean_loader_or_adv
        detector.eval()

        # 1. Collect clean validation scores
        clean_scores: List[float] = []
        with torch.no_grad():
            for images, _ in clean_val_loader:
                images = images.to(target_device)
                logits = detector(images).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().tolist()
                clean_scores.extend(probs if isinstance(probs, list) else [probs])
        clean_scores_np = np.array(clean_scores)

        # 2. Collect adversarial validation scores (if supplied)
        adv_scores_np: Optional[np.ndarray] = None
        if adv_val_loader is not None:
            adv_scores: List[float] = []
            with torch.no_grad():
                for images, labels in adv_val_loader:
                    images = images.to(target_device)
                    logits = detector(images).squeeze(-1)
                    probs = torch.sigmoid(logits).cpu().tolist()
                    adv_scores.extend(probs if isinstance(probs, list) else [probs])
            adv_scores_np = np.array(adv_scores)
    else:
        # First argument is clean scores array
        clean_scores_np = np.asarray(detector_or_clean, dtype=float)
        adv_scores_np = np.asarray(clean_loader_or_adv, dtype=float) if clean_loader_or_adv is not None else None

    # Candidate thresholds from 0.001 to 0.999
    candidates = np.linspace(0.001, 0.999, resolution)
    valid_thresholds: List[Tuple[float, float, float]] = []

    for tau in candidates:
        fpr = float((clean_scores_np >= tau).mean())
        if fpr <= target_fpr:
            tpr = float((adv_scores_np >= tau).mean()) if adv_scores_np is not None else 0.0
            valid_thresholds.append((tau, fpr, tpr))

    if not valid_thresholds:
        # Fallback to the percentile satisfying target FPR
        tau_star = float(np.percentile(clean_scores_np, (1.0 - target_fpr) * 100))
        empirical_fpr = float((clean_scores_np >= tau_star).mean())
        empirical_tpr = float((adv_scores_np >= tau_star).mean()) if adv_scores_np is not None else 0.0
    else:
        if adv_scores_np is not None:
            # Pick tau that maximizes TPR among those satisfying FPR <= target_fpr
            valid_thresholds.sort(key=lambda x: (x[2], -x[1]), reverse=True)
            tau_star, empirical_fpr, empirical_tpr = valid_thresholds[0]
        else:
            tau_star, empirical_fpr, empirical_tpr = valid_thresholds[0]

    report = {
        "optimal_threshold": float(tau_star),
        "calibrated_threshold": float(tau_star),
        "target_fpr": float(target_fpr),
        "calibrated_fpr": float(empirical_fpr),
        "calibrated_tpr": float(empirical_tpr),
        "empirical_val_fpr": float(empirical_fpr),
        "empirical_val_tpr": float(empirical_tpr),
        "clean_val_samples": len(clean_scores_np),
        "adv_val_samples": len(adv_scores_np) if adv_scores_np is not None else 0,
        "mean_clean_prob": float(clean_scores_np.mean()),
        "p99_clean_prob": float(np.percentile(clean_scores_np, 99)),
    }

    if save_path is not None:
        config.save_json(report, save_path)

    return report


class ModelShieldFirewall:
    """
    ModelShield Phase 2 Security Gateway:
    Pre-inference AI firewall that detects and filters adversarially manipulated images
    using calibrated probability thresholds before forwarding clean traffic to the classifier.
    """

    def __init__(
        self,
        classifier: Optional[CIFAR10Classifier] = None,
        detector: Optional[AdversarialDetector] = None,
        classifier_path: Optional[Union[str, Path]] = None,
        detector_path: Optional[Union[str, Path]] = None,
        threshold: float = config.DETECTOR_DEFAULT_THRESHOLD,
        device: Optional[torch.device] = None,
    ):
        self.device = device or config.DEVICE
        self.threshold = threshold

        self.classifier = classifier or CIFAR10Classifier()
        self.classifier.to(self.device)

        self.detector = detector or AdversarialDetector()
        self.detector.to(self.device)

        # Load weights if supplied
        if classifier_path is not None:
            c_path = Path(classifier_path)
            if c_path.exists():
                self._load_weights(self.classifier, c_path)

        if detector_path is not None:
            d_path = Path(detector_path)
            if d_path.exists():
                self._load_weights(self.detector, d_path)

        self.classifier.eval()
        self.detector.eval()

    @staticmethod
    def _load_weights(model: nn.Module, path: Path) -> None:
        """Safely load weights from either model_state_dict or direct dictionary."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        else:
            raise ValueError(f"Unrecognized checkpoint format at {path}")

    @staticmethod
    def preprocess_image(image: Union[Image.Image, Path, str, np.ndarray]) -> torch.Tensor:
        """
        Preprocess image input into normalized CIFAR-10 tensor of shape (1, 3, 32, 32).
        """
        if isinstance(image, (str, Path)):
            img_path = Path(image)
            if not img_path.exists():
                raise FileNotFoundError(f"Input image not found: {img_path}")
            pil_img = Image.open(img_path).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype("uint8")).convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")

        eval_transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SHAPE[1:]),
            transforms.ToTensor(),
            transforms.Normalize(config.CIFAR10_MEAN, config.CIFAR10_STD),
        ])
        return eval_transform(pil_img).unsqueeze(0)

    def detect(self, image_tensor: torch.Tensor) -> float:
        """Estimate adversarial probability using detector CNN."""
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)
        self.detector.eval()

        with torch.no_grad():
            logit = self.detector(image_tensor).squeeze()
            prob = torch.sigmoid(logit).item()

        return float(prob)

    def classify(self, image_tensor: torch.Tensor) -> Tuple[int, str, float]:
        """Classify clean image using target classifier CNN."""
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
        image_input: Union[torch.Tensor, Image.Image, Path, str, np.ndarray],
        custom_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute full ModelShield firewall pipeline on a single image:
        1. Preprocess.
        2. Screen through detector.
        3. Threshold comparison.
        4. If clean: ALLOW and classify.
        5. If suspicious: BLOCK and quarantine (classifier bypassed).
        6. Measure component and end-to-end latency with CUDA synchronization when applicable.
        """
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        threshold = custom_threshold if custom_threshold is not None else self.threshold

        # Step 1: Preprocess
        if isinstance(image_input, torch.Tensor):
            tensor = image_input if image_input.dim() == 4 else image_input.unsqueeze(0)
        else:
            tensor = self.preprocess_image(image_input)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_prep = time.perf_counter()

        # Step 2: Adversarial Screening
        adv_prob = self.detect(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_detect = time.perf_counter()
        is_suspicious = adv_prob >= threshold

        report: Dict[str, Any] = {
            "adversarial_probability": adv_prob,
            "threshold": threshold,
            "is_adversarial": is_suspicious,
            "preprocess_latency_ms": (t_prep - t0) * 1000.0,
            "detector_latency_ms": (t_detect - t_prep) * 1000.0,
            "classifier_latency_ms": 0.0,
        }

        # Step 3: Firewall Gate
        if is_suspicious:
            report.update({
                "decision": "BLOCKED",
                "reason": f"Adversarial probability ({adv_prob:.4f}) >= threshold ({threshold:.4f}).",
                "predicted_class_id": None,
                "predicted_class_name": None,
                "classification_confidence": None,
            })
        else:
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t_cls_start = time.perf_counter()
            pred_id, pred_name, conf = self.classify(tensor)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t_cls_end = time.perf_counter()
            report["classifier_latency_ms"] = (t_cls_end - t_cls_start) * 1000.0
            report.update({
                "decision": "ALLOWED",
                "reason": f"Adversarial probability ({adv_prob:.4f}) < threshold ({threshold:.4f}).",
                "predicted_class_id": pred_id,
                "predicted_class_name": pred_name,
                "classification_confidence": conf,
            })

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_total = time.perf_counter()
        report["total_latency_ms"] = (t_total - t0) * 1000.0
        return report

    def process_batch(
        self,
        batch_tensors: torch.Tensor,
        custom_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Vectorized batch evaluation of the firewall gateway over tensor batches.
        Only images that pass the detector screen are forwarded to the classifier;
        blocked images bypass downstream classification entirely.
        """
        threshold = custom_threshold if custom_threshold is not None else self.threshold
        batch_tensors = batch_tensors.to(self.device)
        b_size = batch_tensors.size(0)

        with torch.no_grad():
            det_logits = self.detector(batch_tensors).squeeze(-1)
            if det_logits.dim() == 0:
                det_logits = det_logits.unsqueeze(0)
            det_probs = torch.sigmoid(det_logits)
            det_probs_list = det_probs.cpu().tolist()
            if not isinstance(det_probs_list, list):
                det_probs_list = [det_probs_list]

            is_suspicious = det_probs >= threshold
            allowed_mask = ~is_suspicious
            allowed_indices = allowed_mask.nonzero(as_tuple=True)[0]

            cls_preds_dict: Dict[int, int] = {}
            cls_confs_dict: Dict[int, float] = {}

            if len(allowed_indices) > 0:
                allowed_tensors = batch_tensors[allowed_indices]
                cls_logits = self.classifier(allowed_tensors)
                cls_probs = F.softmax(cls_logits, dim=1)
                preds = cls_probs.argmax(dim=1).cpu().tolist()
                confs = [cls_probs[idx, preds[idx]].item() for idx in range(len(allowed_indices))]
                for i, orig_idx in enumerate(allowed_indices.cpu().tolist()):
                    cls_preds_dict[orig_idx] = preds[i]
                    cls_confs_dict[orig_idx] = confs[i]

        reports = []
        for i in range(b_size):
            prob = float(det_probs_list[i])
            if i not in cls_preds_dict:
                reports.append({
                    "sample_idx": i,
                    "adversarial_probability": prob,
                    "threshold": threshold,
                    "is_adversarial": True,
                    "decision": "BLOCKED",
                    "reason": f"Adversarial probability ({prob:.4f}) >= threshold ({threshold:.4f}).",
                    "predicted_class_id": None,
                    "predicted_class_name": None,
                    "classification_confidence": None,
                })
            else:
                pred_id = cls_preds_dict[i]
                reports.append({
                    "sample_idx": i,
                    "adversarial_probability": prob,
                    "threshold": threshold,
                    "is_adversarial": False,
                    "decision": "ALLOWED",
                    "reason": f"Adversarial probability ({prob:.4f}) < threshold ({threshold:.4f}).",
                    "predicted_class_id": pred_id,
                    "predicted_class_name": config.CIFAR10_CLASSES[pred_id],
                    "classification_confidence": float(cls_confs_dict[i]),
                })
        return reports

    def format_security_report(
        self,
        result: Dict[str, Any],
        input_name: str = "input_sample.png",
    ) -> str:
        """Format firewall result into an executive cybersecurity report."""
        lines = [
            "==================================================",
            "        MODELSHIELD PHASE 2 SECURITY REPORT       ",
            "==================================================",
            f"Input Identifier:        {input_name}",
            f"Adversarial Score:       {result['adversarial_probability']:.4f}",
            f"Calibrated Threshold:    {result['threshold']:.4f}",
            f"Firewall Decision:       {result['decision']}",
            f"Preprocess Latency:      {result.get('preprocess_latency_ms', 0):.2f} ms",
            f"Detector Latency:        {result.get('detector_latency_ms', 0):.2f} ms",
        ]

        if result["decision"] == "BLOCKED":
            lines.append(f"Security Alert:          {result['reason']}")
            lines.append("Downstream Classifier:   BYPASSED (Zero threat exposure)")
        else:
            lines.append(f"Classifier Latency:      {result.get('classifier_latency_ms', 0):.2f} ms")
            lines.append(f"Predicted Class:         {result['predicted_class_name']}")
            lines.append(f"Classification Conf:     {result['classification_confidence'] * 100:.2f}%")

        lines.append(f"Total Pipeline Latency:  {result.get('total_latency_ms', 0):.2f} ms")
        lines.append("==================================================")
        return "\n".join(lines)

    def inspect(
        self,
        image_input: Union[torch.Tensor, Image.Image, Path, str, np.ndarray],
    ) -> Dict[str, Any]:
        """Convenience method returning decision and latency metadata."""
        res = self.process(image_input)
        return {
            "is_blocked": res["decision"] == "BLOCKED",
            "action": res["decision"],
            "detector_score": res["adversarial_probability"],
            "predicted_class": res["predicted_class_id"],
            "predicted_class_name": res["predicted_class_name"],
            "threshold": res["threshold"],
            "latency_ms": res["total_latency_ms"],
            **res,
        }

    def batch_inspect(
        self,
        batch_tensors: torch.Tensor,
    ) -> Dict[str, Any]:
        """Convenience method for batch inspection with aggregated counts."""
        decisions = self.process_batch(batch_tensors)
        passed = sum(1 for r in decisions if r["decision"] == "ALLOWED")
        blocked = len(decisions) - passed
        return {
            "total_inspected": len(decisions),
            "passed_count": passed,
            "blocked_count": blocked,
            "decisions": decisions,
        }


def profile_firewall_latency(
    firewall: ModelShieldFirewall,
    input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32),
    iterations: int = 50,
) -> Dict[str, float]:
    """
    Profile execution latency of classifier, detector, and end-to-end firewall pipeline.
    """
    dummy_input = torch.rand(*input_shape, device=firewall.device)

    # Warmup
    for _ in range(5):
        _ = firewall.process(dummy_input)

    det_latencies = []
    cls_latencies = []
    total_latencies = []

    for _ in range(iterations):
        res = firewall.process(dummy_input, custom_threshold=1.0)  # force allow to measure both
        det_latencies.append(res["detector_latency_ms"])
        cls_latencies.append(res["classifier_latency_ms"])
        total_latencies.append(res["total_latency_ms"])

    det_mean = float(np.mean(det_latencies))
    cls_mean = float(np.mean(cls_latencies))
    tot_mean = float(np.mean(total_latencies))
    fps = 1000.0 / tot_mean if tot_mean > 0 else 0.0

    return {
        "detector_mean_ms": det_mean,
        "classifier_mean_ms": cls_mean,
        "total_mean_ms": tot_mean,
        "throughput_fps": fps,
        "iterations": iterations,
    }


def generate_security_report(
    result: Dict[str, Any],
    input_name: str = "sample_image.png",
) -> str:
    """Format firewall result into human-readable security report."""
    firewall = ModelShieldFirewall()
    return firewall.format_security_report(result, input_name=input_name)
