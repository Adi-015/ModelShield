"""
ModelShield - Main Application Entry Point
Demonstrates the integrated adversarial AI firewall pipeline on clean and perturbed images.
"""

import argparse
from pathlib import Path
import sys
import torch
from PIL import Image

import config
from src.attacks import fgsm_attack
from src.data import denormalize_tensor, normalize_tensor
from src.firewall import ModelShieldFirewall
from src.models import AdversarialDetector, CIFAR10Classifier


def run_pipeline_demo(
    epsilon: float = config.DEFAULT_EPSILON,
    threshold: float = config.DETECTOR_THRESHOLD,
    save_outputs: bool = True,
) -> None:
    """
    Run an end-to-end local test of the ModelShield firewall pipeline:
    1. Loads or initializes classifier and detector models.
    2. Constructs a test sample image.
    3. Passes the clean image through the firewall (expecting ALLOWED).
    4. Generates an FGSM adversarial sample.
    5. Passes the adversarial image through the firewall (expecting BLOCKED).
    6. Displays executive security reports.
    """
    print("\n" + "=" * 60)
    print("      ModelShield: Phase 1 Adversarial AI Firewall Demo     ")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Firewall Detection Threshold: {threshold:.2f}")
    print(f"Adversarial Attack: FGSM (epsilon = {epsilon})")

    # Check for pre-trained checkpoints
    classifier_exists = config.CLASSIFIER_CHECKPOINT_PATH.exists()
    detector_exists = config.DETECTOR_CHECKPOINT_PATH.exists()

    if not (classifier_exists and detector_exists):
        print("\n[!] Checkpoint status:")
        print(f"    - Classifier ({config.CLASSIFIER_CHECKPOINT_NAME}): "
              f"{'Found' if classifier_exists else 'Not found (using initialized weights)'}")
        print(f"    - Detector ({config.DETECTOR_CHECKPOINT_NAME}): "
              f"{'Found' if detector_exists else 'Not found (using initialized weights)'}")
        print("    Note: To train production models with full CIFAR-10 accuracy,")
        print("    run 'notebooks/ModelShield_Phase1.ipynb' in Google Colab.\n")

    # Initialize firewall
    firewall = ModelShieldFirewall(
        classifier_path=config.CLASSIFIER_CHECKPOINT_PATH if classifier_exists else None,
        detector_path=config.DETECTOR_CHECKPOINT_PATH if detector_exists else None,
        threshold=threshold,
        device=config.DEVICE,
    )

    # 1. Generate or load a sample image for demonstration
    # Synthetic realistic CIFAR-10 style normalized tensor
    config.set_seed(config.RANDOM_SEED)
    sample_clean = torch.randn(1, 3, 32, 32, device=config.DEVICE)
    # Target label: class 3 ('cat')
    true_label = torch.tensor([3], device=config.DEVICE)

    # 2. Test Firewall with Clean Image
    print("\n--- [Test Case 1: Clean Input Verification] ---")
    clean_result = firewall.process(sample_clean)
    clean_report = firewall.format_security_report(clean_result, input_name="clean_sample.png")
    print(clean_report)

    # 3. Generate FGSM Adversarial Perturbation
    print("\n--- [Generating FGSM Adversarial Attack] ---")
    adv_sample = fgsm_attack(
        model=firewall.classifier,
        images=sample_clean,
        labels=true_label,
        epsilon=epsilon,
    )

    # 4. Test Firewall with Adversarial Image
    print("\n--- [Test Case 2: Adversarial Input Inspection] ---")
    adv_result = firewall.process(adv_sample)
    adv_report = firewall.format_security_report(adv_result, input_name="adversarial_sample.png")
    print(adv_report)

    # 5. Save demo output artifacts if requested
    if save_outputs:
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        clean_raw = denormalize_tensor(sample_clean).squeeze(0).permute(1, 2, 0).cpu().numpy()
        adv_raw = denormalize_tensor(adv_sample).squeeze(0).permute(1, 2, 0).cpu().numpy()

        clean_img = Image.fromarray((clean_raw * 255).astype("uint8"))
        adv_img = Image.fromarray((adv_raw * 255).astype("uint8"))

        clean_path = config.OUTPUTS_DIR / "demo_clean.png"
        adv_path = config.OUTPUTS_DIR / "demo_adversarial.png"

        clean_img.save(clean_path)
        adv_img.save(adv_path)
        print(f"\nSaved demonstration images to:")
        print(f"  - Clean: {clean_path}")
        print(f"  - Adversarial: {adv_path}")

    print("\n" + "=" * 60)
    print("                Pipeline Execution Complete                 ")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ModelShield: Phase 1 Adversarial AI Firewall CLI"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=True,
        help="Run integrated firewall pipeline demonstration (default: True)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an individual image file to inspect with the firewall",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.DETECTOR_THRESHOLD,
        help=f"Detection threshold for adversarial probability (default: {config.DETECTOR_THRESHOLD})",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=config.DEFAULT_EPSILON,
        help=f"FGSM attack perturbation magnitude (default: {config.DEFAULT_EPSILON})",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving demonstration images to outputs/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"Error: Specified image file does not exist: {img_path}")
            sys.exit(1)

        firewall = ModelShieldFirewall(threshold=args.threshold)
        result = firewall.process(img_path)
        report = firewall.format_security_report(result, input_name=img_path.name)
        print(report)
    else:
        run_pipeline_demo(
            epsilon=args.epsilon,
            threshold=args.threshold,
            save_outputs=not args.no_save,
        )


if __name__ == "__main__":
    main()
