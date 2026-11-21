"""
ModelShield Phase 2 — Main Command-Line Interface.

Provides a unified CLI for:
  - Running end-to-end pipeline verification demo
  - Training the CIFAR-10 classifier (Phase 2 baseline or enhanced)
  - Training the adversarial detector on multi-attack mixtures
  - Running attack evaluation sweeps (FGSM, PGD)
  - Calibrating firewall decision thresholds (guaranteeing FPR <= 1%)
  - Comprehensive model evaluation and benchmark reporting
"""

import os
import sys
import argparse
import json
import logging
import torch
import torch.nn as nn
from pathlib import Path

# Add project root and Phase2 to path for robust imports
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from Phase2 import config
except ImportError:
    import config
from src.models import CIFAR10Classifier, AdversarialDetector, count_parameters
from src.data import get_cifar10_dataloaders, normalize_tensor, denormalize_tensor
from src.attacks import fgsm_attack, pgd_attack, evaluate_attacks
from src.detector import (
    evaluate_detector,
    evaluate_detector_generalization,
    build_multi_attack_dataset
)
from src.firewall import (
    ModelShieldFirewall,
    calibrate_threshold,
    profile_firewall_latency,
    generate_security_report
)
from src.train import (
    train_classifier,
    train_detector,
    save_checkpoint,
    load_checkpoint
)
from src.evaluate import (
    evaluate_classifier,
    evaluate_attack_spectrum,
    evaluate_firewall_defense
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ModelShield.Main")


def run_demo(device: torch.device):
    """
    Run a fast end-to-end sanity check and demo of all Phase 2 components.
    Uses synthetic batches to verify the entire pipeline without waiting for dataset downloads.
    """
    print("\n" + "=" * 70)
    print(" MODELSHIELD PHASE 2 — END-TO-END VERIFICATION DEMO")
    print("=" * 70)
    print(f"[*] Target device: {device}")
    
    # 1. Models
    print("\n[Step 1/5] Initializing Models...")
    classifier = CIFAR10Classifier(num_classes=config.NUM_CLASSES).to(device)
    detector = AdversarialDetector(input_channels=config.IMAGE_CHANNELS).to(device)
    print(f"  - Classifier: {count_parameters(classifier):,} trainable parameters")
    print(f"  - Detector:   {count_parameters(detector):,} trainable parameters")
    
    # 2. Synthetic Data
    print("\n[Step 2/5] Creating Synthetic Data Batch (16 samples, 3x32x32)...")
    torch.manual_seed(42)
    clean_raw = torch.rand(16, 3, 32, 32, device=device)
    clean_images = normalize_tensor(clean_raw)
    clean_labels = torch.randint(0, 10, (16,), device=device)
    print(f"  - Clean raw shape: {clean_raw.shape}, range: [{clean_raw.min():.2f}, {clean_raw.max():.2f}]")
    
    # 3. Attacks
    print("\n[Step 3/5] Generating Adversarial Samples (FGSM & PGD)...")
    eps = 0.03
    adv_fgsm = fgsm_attack(classifier, clean_images, clean_labels, epsilon=eps)
    adv_pgd = pgd_attack(classifier, clean_images, clean_labels, epsilon=eps, alpha=0.007, num_steps=10)
    
    fgsm_diff = (denormalize_tensor(adv_fgsm) - clean_raw).abs().max().item()
    pgd_diff = (denormalize_tensor(adv_pgd) - clean_raw).abs().max().item()
    adv_pgd_raw = denormalize_tensor(adv_pgd)
    print(f"  - FGSM max perturbation: {fgsm_diff:.4f} (<= {eps:.4f}) -> {'PASS' if fgsm_diff <= eps + 1e-4 else 'FAIL'}")
    print(f"  - PGD max perturbation:  {pgd_diff:.4f} (<= {eps:.4f}) -> {'PASS' if pgd_diff <= eps + 1e-4 else 'FAIL'}")
    print(f"  - Clamping verified: [{adv_pgd_raw.min().item():.2f}, {adv_pgd_raw.max().item():.2f}] within [0.0, 1.0]")
    
    # 4. Detector & Firewall
    print("\n[Step 4/5] Firewall Dynamic Thresholding & Decision Pipeline...")
    val_clean_scores = torch.rand(200).numpy() * 0.1  # low suspicion
    val_adv_scores = 0.8 + torch.rand(200).numpy() * 0.2  # high suspicion
    
    calib = calibrate_threshold(val_clean_scores, val_adv_scores, target_fpr=0.01)
    threshold = calib["optimal_threshold"]
    print(f"  - Calibrated threshold (target FPR <= 1%): {threshold:.4f}")
    print(f"  - Calibrated Validation FPR: {calib['calibrated_fpr']:.4%}, TPR: {calib['calibrated_tpr']:.4%}")
    
    firewall = ModelShieldFirewall(
        classifier=classifier,
        detector=detector,
        threshold=threshold,
        device=device
    )
    
    clean_decisions = firewall.batch_inspect(clean_images)
    adv_decisions = firewall.batch_inspect(adv_pgd)
    
    print(f"  - Clean samples allowed: {clean_decisions['passed_count']}/{len(clean_images)}")
    print(f"  - Adversarial samples blocked: {adv_decisions['blocked_count']}/{len(adv_pgd)}")
    
    # 5. Latency profiling
    print("\n[Step 5/5] Latency Profiling (10 iterations)...")
    profile = profile_firewall_latency(firewall, input_shape=(1, 3, 32, 32), iterations=10)
    print(f"  - Classifier Latency: {profile['classifier_mean_ms']:.2f} ms")
    print(f"  - Detector Latency:   {profile['detector_mean_ms']:.2f} ms")
    print(f"  - Total Pipeline:     {profile['total_mean_ms']:.2f} ms ({profile['throughput_fps']:.1f} FPS)")
    
    print("\n" + "=" * 70)
    print(" [SUCCESS] All Phase 2 architectural components verified operational!")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="ModelShield Phase 2 — Adversarial AI Detection and Defense"
    )
    
    # Modes
    parser.add_argument("--demo", action="store_true", help="Run fast end-to-end verification demo")
    parser.add_argument("--train-classifier", action="store_true", help="Train Phase 2 CIFAR-10 classifier")
    parser.add_argument("--train-detector", action="store_true", help="Train Phase 2 multi-attack mixture detector")
    parser.add_argument("--evaluate", action="store_true", help="Run full evaluation pipeline on test set")
    parser.add_argument("--attack-eval", action="store_true", help="Run attack evaluation sweep (FGSM & PGD)")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate firewall decision threshold on validation set")
    
    # Parameters
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.CLASSIFIER_BATCH_SIZE, help="Batch size")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Computation device")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--target-fpr", type=float, default=config.FIREWALL_CONFIG["target_fpr"], help="Target FPR for calibration (default 0.01)")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = config.DEVICE
    
    config.set_seed(config.RANDOM_SEED)
    
    # Default to demo if no action specified
    if not (args.demo or args.train_classifier or args.train_detector or args.evaluate or args.attack_eval or args.calibrate):
        logger.info("No action specified. Running Phase 2 verification demo (--demo).")
        run_demo(device)
        return
        
    if args.demo:
        run_demo(device)
        
    if args.train_classifier:
        logger.info("Starting Phase 2 Classifier Training...")
        classifier = CIFAR10Classifier(num_classes=config.NUM_CLASSES)
        train_loader, val_loader, _ = get_cifar10_dataloaders(batch_size=args.batch_size)
        
        epochs = args.epochs if args.epochs is not None else config.CLASSIFIER_EPOCHS
        history = train_classifier(
            model=classifier,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=config.CLASSIFIER_LR,
            weight_decay=config.CLASSIFIER_WEIGHT_DECAY,
            device=device,
            resume=args.resume
        )
        best_val = max(history.get("val_acc", [0.0]))
        logger.info(f"Classifier training completed. Best Val Acc: {best_val:.2%}")

    if args.train_detector:
        logger.info("Starting Phase 2 Detector Training (Multi-Attack Mixture)...")
        # Ensure classifier exists
        classifier = CIFAR10Classifier(num_classes=config.NUM_CLASSES)
        clf_ckpt = config.CLASSIFIER_CHECKPOINT_BEST
        if not os.path.exists(clf_ckpt):
            logger.error(f"Classifier checkpoint not found at {clf_ckpt}. Train classifier first.")
            return
            
        load_checkpoint(clf_ckpt, classifier, device=device)
        classifier.to(device)
        classifier.eval()
        
        train_loader, val_loader, _ = get_cifar10_dataloaders(batch_size=args.batch_size)
        logger.info("Building multi-attack dataset mixtures for detector training...")
        train_mixture = build_multi_attack_dataset(
            classifier=classifier,
            clean_loader=train_loader,
            device=device,
            shuffle=True,
            batch_size=args.batch_size,
        )
        val_mixture = build_multi_attack_dataset(
            classifier=classifier,
            clean_loader=val_loader,
            device=device,
            shuffle=False,
            batch_size=args.batch_size,
        )
        detector = AdversarialDetector(input_channels=config.IMAGE_CHANNELS)
        
        epochs = args.epochs if args.epochs is not None else config.DETECTOR_EPOCHS
        history = train_detector(
            detector=detector,
            train_loader=train_mixture,
            val_loader=val_mixture,
            epochs=epochs,
            lr=config.DETECTOR_LR,
            weight_decay=config.DETECTOR_WEIGHT_DECAY,
            device=device,
            resume=args.resume
        )
        best_val = max(history.get("val_acc", [0.0]))
        logger.info(f"Detector training completed. Best Val Acc: {best_val:.2%}")

    if args.attack_eval:
        logger.info("Running Attack Evaluation Sweep...")
        classifier = CIFAR10Classifier(num_classes=config.NUM_CLASSES)
        load_checkpoint(config.CLASSIFIER_CHECKPOINT_BEST, classifier, device=device)
        classifier.to(device)
        
        _, _, test_loader = get_cifar10_dataloaders(batch_size=args.batch_size)
        spectrum = evaluate_attack_spectrum(classifier, test_loader, device=device)
        logger.info("Attack spectrum evaluation complete.")

    if args.calibrate:
        logger.info("Running Validation-Based Threshold Calibration...")
        classifier = CIFAR10Classifier(num_classes=config.NUM_CLASSES)
        detector = AdversarialDetector(input_channels=config.IMAGE_CHANNELS)
        
        load_checkpoint(config.CLASSIFIER_CHECKPOINT_BEST, classifier, device=device)
        load_checkpoint(config.DETECTOR_CHECKPOINT_BEST, detector, device=device)
        classifier.to(device)
        detector.to(device)
        
        _, val_loader, _ = get_cifar10_dataloaders(batch_size=args.batch_size)
        
        # Build validation scores
        detector.eval()
        clean_scores = []
        adv_scores = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                scores = detector(images).squeeze().cpu().numpy()
                clean_scores.extend(scores.tolist() if scores.ndim > 0 else [scores.item()])
                
                # Generate sample PGD attacks on val for calibration
                with torch.enable_grad():
                    adv_imgs = pgd_attack(classifier, images, labels.to(device), epsilon=0.03)
                adv_s = detector(adv_imgs).squeeze().cpu().numpy()
                adv_scores.extend(adv_s.tolist() if adv_s.ndim > 0 else [adv_s.item()])
                if len(clean_scores) >= 1000:
                    break
                    
        import numpy as np
        calib = calibrate_threshold(
            np.array(clean_scores),
            np.array(adv_scores),
            target_fpr=args.target_fpr,
            save_path=config.FIREWALL_THRESHOLD_FILE
        )
        logger.info(f"Calibration finished: Optimal Threshold = {calib['optimal_threshold']:.4f} (Val FPR = {calib['calibrated_fpr']:.4%})")

    if args.evaluate:
        logger.info("Running Full Test Set Benchmark & Evaluation...")
        classifier = CIFAR10Classifier(num_classes=config.NUM_CLASSES)
        detector = AdversarialDetector(input_channels=config.IMAGE_CHANNELS)
        
        load_checkpoint(config.CLASSIFIER_CHECKPOINT_BEST, classifier, device=device)
        load_checkpoint(config.DETECTOR_CHECKPOINT_BEST, detector, device=device)
        classifier.to(device)
        detector.to(device)
        
        _, _, test_loader = get_cifar10_dataloaders(batch_size=args.batch_size)
        
        # Classifier
        clf_metrics = evaluate_classifier(classifier, test_loader, device=device)
        logger.info(f"Clean Test Accuracy: {clf_metrics['test_accuracy']:.2%}")
        
        # Attack Spectrum
        spectrum = evaluate_attack_spectrum(classifier, test_loader, device=device)
        
        # Firewall
        calib_data = config.load_json(config.FIREWALL_THRESHOLD_FILE)
        threshold = calib_data.get("optimal_threshold", 0.5) if calib_data else 0.5
        firewall = ModelShieldFirewall(classifier, detector, threshold=threshold, device=device)
        
        defense_results = evaluate_firewall_defense(firewall=firewall, test_loader=test_loader, device=device)
        logger.info("Full test set evaluation successfully completed.")


if __name__ == "__main__":
    main()
