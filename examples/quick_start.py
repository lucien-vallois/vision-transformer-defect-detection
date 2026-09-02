#!/usr/bin/env python3
"""
Quick Start Example for Vision Transformer Defect Detection

This example verifies model construction, a forward pass, and synthetic data loading.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.model import create_vit_model
from src.data.dataset import SyntheticDefectDataset


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Vision Transformer Defect Detection - Quick Start")
    print("=" * 60)

    # 1. Create model
    print("\n1. Creating Vision Transformer model...")
    config = {
        "img_size": 64,
        "patch_size": 8,
        "in_channels": 3,
        "num_classes": 5,
        "embed_dim": 64,
        "depth": 2,
        "num_heads": 4,
        "mlp_ratio": 2.0,
    }

    model = create_vit_model(config)
    print(f"   Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # 2. Test forward pass
    print("\n2. Testing forward pass...")
    model.eval()
    dummy_input = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        output = model(dummy_input)
        probabilities = torch.softmax(output, dim=1)

    print(f"   Output shape: {output.shape}")
    print(f"   Probabilities: {probabilities[0].tolist()}")

    # 3. Create dataset
    print("\n3. Creating synthetic dataset...")
    dataset = SyntheticDefectDataset(
        root_dir="./data", split="train", generate_on_fly=True, num_samples=10, img_size=64
    )
    print(f"   Dataset created with {len(dataset)} samples")
    print(f"   Classes: {dataset.classes}")

    # 4. Test data loading
    print("\n4. Testing data loading...")
    img, label, metadata = dataset[0]
    print(f"   Image shape: {img.shape}")
    print(f"   Label: {label} ({dataset.classes[label]})")
    print(f"   Metadata: {metadata}")

    print("\n" + "=" * 60)
    print("Quick start completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Train a smoke model: python -m src.train --config configs/smoke.yaml")
    print(
        "2. Run inference: python -m src.inference --model-path <model.pth> --image-path <image.jpg>"
    )


if __name__ == "__main__":
    main()
