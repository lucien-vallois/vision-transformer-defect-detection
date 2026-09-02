#!/usr/bin/env python3
"""
Script to compare different model architectures

Usage:
    python -m scripts.compare_models --data-dir <data_dir> [--output-dir <output_dir>]
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import DefectDetectionDataset
from src.models.comparison import ModelComparator
from src.utils.analysis import generate_comprehensive_report


def main():
    parser = argparse.ArgumentParser(description="Compare different model architectures")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to dataset directory")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments/comparison",
        help="Output directory for comparison results",
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
    parser.add_argument("--vit-model", type=str, help="Path to a trained ViT checkpoint")
    parser.add_argument("--resnet-model", type=str, help="Path to a trained ResNet-50 checkpoint")
    parser.add_argument(
        "--efficientnet-model", type=str, help="Path to a trained EfficientNet-B0 checkpoint"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["vit", "resnet", "efficientnet", "all"],
        default=["vit"],
        help="Models to compare",
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.data_dir).exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Initialize comparator
    print("Initializing model comparator...")
    comparator = ModelComparator(device="auto")

    # Load dataset
    print(f"Loading {args.split} dataset from {args.data_dir}...")
    config = {
        "data": {
            "root_dir": args.data_dir,
            "generate_on_fly": False,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
        "model": {"num_classes": 5},
    }

    dataset_handler = DefectDetectionDataset(config)
    loaders = dataset_handler.get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    dataloader = dict(zip(("train", "val", "test"), loaders))[args.split]

    class_names = ["ok", "scratch", "crack", "dent", "corrosion"]

    # Load models
    models_to_load = []
    if "all" in args.models or "vit" in args.models:
        models_to_load.append("vit")
    if "all" in args.models or "resnet" in args.models:
        models_to_load.append("resnet")
    if "all" in args.models or "efficientnet" in args.models:
        models_to_load.append("efficientnet")

    checkpoint_paths = {
        "vit": args.vit_model,
        "resnet": args.resnet_model,
        "efficientnet": args.efficientnet_model,
    }
    missing = [name for name in models_to_load if not checkpoint_paths[name]]
    if missing:
        parser.error(
            "trained checkpoint required for: " + ", ".join(f"--{name}-model" for name in missing)
        )

    print("\nLoading models...")

    if "vit" in models_to_load:
        print("  Loading ViT...")
        comparator.load_vit_model(model_path=args.vit_model)

    if "resnet" in models_to_load:
        print("  Loading ResNet-50...")
        comparator.load_resnet50(model_path=args.resnet_model, num_classes=5)

    if "efficientnet" in models_to_load:
        print("  Loading EfficientNet-B0...")
        comparator.load_efficientnet_b0(model_path=args.efficientnet_model, num_classes=5)

    # Compare models
    print("\nComparing models...")
    results = comparator.compare_models(dataloader, class_names)

    # Print comparison table
    print("\n" + comparator.generate_comparison_table())

    # Get best model
    best_model_name, best_result = comparator.get_best_model("accuracy")
    print(f"\nBest model (by accuracy): {best_model_name}")
    print(f"  Accuracy: {best_result['accuracy']:.4f}")
    print(f"  F1-Score: {best_result['f1']:.4f}")
    print(f"  Latency: {best_result['mean_latency_ms']:.2f}ms")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save comparison table
    with open(output_dir / "comparison_table.txt", "w") as f:
        f.write(comparator.generate_comparison_table())

    # Generate detailed reports for each model
    print("\nGenerating detailed reports...")
    for model_name, result in results.items():
        model_output_dir = output_dir / model_name.lower().replace("-", "_")
        model_output_dir.mkdir(exist_ok=True)

        print(f"  Generating report for {model_name}...")
        generate_comprehensive_report(
            result["true_labels"],
            result["predictions"],
            result["probabilities"],
            class_names,
            str(model_output_dir),
        )

    # Save summary JSON
    import json

    summary = {
        "models": list(results.keys()),
        "best_model": best_model_name,
        "results": {
            name: {
                "accuracy": result["accuracy"],
                "f1": result["f1"],
                "mean_latency_ms": result["mean_latency_ms"],
                "total_parameters": result["total_parameters"],
            }
            for name, result in results.items()
        },
    }

    with open(output_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {args.output_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
