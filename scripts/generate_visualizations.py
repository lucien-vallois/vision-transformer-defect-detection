#!/usr/bin/env python3
"""
Script to generate comprehensive visualizations for model evaluation

Usage:
    python -m scripts.generate_visualizations --model-path <model.pth> --data-dir <data_dir> --output-dir <output_dir>
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.inference import DefectDetector
from src.data.dataset import DefectDetectionDataset
from src.utils.analysis import generate_comprehensive_report
from src.utils.metrics import compute_metrics


def evaluate_model(
    model_path: str, data_dir: str, split: str = "test", batch_size: int = 32, num_workers: int = 4
):
    """Evaluate model and collect predictions"""

    # Load model
    print(f"Loading model from {model_path}...")
    detector = DefectDetector(model_path, device="auto")

    # Load dataset
    print(f"Loading {split} dataset from {data_dir}...")
    config = {
        "data": {
            "root_dir": data_dir,
            "generate_on_fly": False,
            "batch_size": batch_size,
            "num_workers": num_workers,
        },
        "model": {"num_classes": len(detector.classes)},
    }

    dataset_handler = DefectDetectionDataset(config)
    _, _, test_loader = dataset_handler.get_dataloaders(
        batch_size=batch_size, num_workers=num_workers
    )

    # Collect predictions
    print("Running inference...")
    all_preds = []
    all_probs = []
    all_labels = []

    detector.model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(test_loader):
            images = images.to(detector.device)

            # Get predictions
            outputs = detector.model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches...")

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Compute metrics
    accuracy, precision, recall, f1 = compute_metrics(all_labels, all_preds)
    print(f"\nMetrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    return all_labels, all_preds, all_probs, detector.classes


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive model visualizations")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to dataset directory")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments/visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)

    if not Path(args.data_dir).exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Evaluate model
    try:
        y_true, y_pred, y_pred_proba, class_names = evaluate_model(
            args.model_path, args.data_dir, args.split, args.batch_size, args.num_workers
        )
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Generate visualizations
    print(f"\nGenerating visualizations...")
    try:
        report = generate_comprehensive_report(
            y_true, y_pred, y_pred_proba, [c.capitalize() for c in class_names], args.output_dir
        )

        print(f"\nVisualizations saved to: {args.output_dir}")
        print("\nGenerated files:")
        for name, path in report["files"].items():
            print(f"  - {name}: {path}")

        print(f"\nError Analysis Summary:")
        print(f"  Total Errors: {report['error_analysis']['total_errors']}")
        print(f"  Error Rate: {report['error_analysis']['error_rate']:.2%}")

    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
