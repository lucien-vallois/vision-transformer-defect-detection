#!/usr/bin/env python3
"""
Synthetic Data Generation Script for Defect Detection

Generates synthetic images of metal surfaces with various defects:
- OK: No defects
- Scratch: Linear surface damage
- Crack: Structural fracture
- Dent: Impact deformation
- Corrosion: Oxidation/wear

Usage:
    python -m scripts.generate_synthetic_data --num-samples 12000 --output-dir ./data/synthetic_defects
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.data.augmentation import apply_random_defect


def _set_random_seed(seed: int) -> None:
    np.random.seed(seed % (2**32))
    cv2.setRNGSeed(seed % (2**31 - 1))


def generate_base_component(img_size: int = 224) -> np.ndarray:
    """
    Generate a base metal-surface image

    Args:
        img_size: Size of the generated image

    Returns:
        Base component image as numpy array
    """
    # Create base metallic color (aluminum-like)
    base_color = np.random.randint(160, 220, 3)  # Light gray to light blue-gray

    # Create base image
    img = np.full((img_size, img_size, 3), base_color, dtype=np.uint8)

    # Add metallic texture with multiple noise layers
    noise1 = np.random.normal(0, 15, (img_size, img_size, 3))
    noise2 = np.random.normal(0, 5, (img_size, img_size, 3))
    img = np.clip(img.astype(np.float32) + noise1 + noise2, 0, 255)

    # Add subtle gradients (lighting effects)
    gradient_x = np.linspace(0.9, 1.1, img_size).reshape(1, -1)
    gradient_y = np.linspace(0.95, 1.05, img_size).reshape(-1, 1)
    gradient = gradient_x * gradient_y

    img = np.clip(img * gradient[:, :, np.newaxis], 0, 255).astype(np.uint8)

    # Add subtle manufacturing marks (rivets, seams)
    if np.random.random() > 0.7:
        # Add rivet pattern
        margin = min(20, max(2, img_size // 4))
        rivet_centers = [
            (
                np.random.randint(margin, img_size - margin),
                np.random.randint(margin, img_size - margin),
            )
            for _ in range(np.random.randint(3, 8))
        ]

        for center in rivet_centers:
            cv2.circle(img, center, np.random.randint(3, 6), (np.random.randint(100, 140),) * 3, -1)
            # Add shadow effect
            shadow_color = tuple(int(max(0, c - 30)) for c in img[center[1], center[0]])
            cv2.circle(
                img, (center[0] + 2, center[1] + 2), np.random.randint(4, 8), shadow_color, -1
            )

    return img


def generate_component_variations(
    base_img: np.ndarray, num_variations: int = 5
) -> List[np.ndarray]:
    """
    Generate variations of a base component

    Args:
        base_img: Base component image
        num_variations: Number of variations to generate

    Returns:
        List of varied component images
    """
    variations = [base_img.copy()]

    for _ in range(num_variations - 1):
        var_img = base_img.copy()

        # Apply random variations
        if np.random.random() > 0.5:
            # Slight color variation
            color_shift = np.random.normal(0, 10, 3).astype(np.int8)
            var_img = np.clip(var_img.astype(np.int16) + color_shift, 0, 255).astype(np.uint8)

        if np.random.random() > 0.6:
            # Slight rotation
            angle = np.random.uniform(-5, 5)
            h, w = var_img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            var_img = cv2.warpAffine(var_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        if np.random.random() > 0.7:
            # Slight scaling
            scale = np.random.uniform(0.95, 1.05)
            h, w = var_img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            var_img = cv2.resize(var_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            # Crop or pad to original size
            if scale > 1.0:
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                var_img = var_img[start_h : start_h + h, start_w : start_w + w]
            else:
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                var_img = cv2.copyMakeBorder(
                    var_img, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w, cv2.BORDER_REFLECT
                )

        variations.append(var_img)

    return variations


def generate_sample(
    sample_id: int,
    defect_class: str,
    img_size: int = 224,
    base_components: List[np.ndarray] = None,
    seed: int = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Generate a single synthetic sample

    Args:
        sample_id: Unique sample ID
        defect_class: Type of defect ('ok', 'scratch', 'crack', 'dent', 'corrosion')
        img_size: Size of generated image
        base_components: Pre-generated base components to choose from
        seed: Optional per-sample random seed

    Returns:
        Tuple of (image, metadata_dict)
    """
    if seed is not None:
        _set_random_seed(seed)

    # Select or generate base component
    if base_components:
        base_img = base_components[np.random.choice(len(base_components))]
    else:
        base_img = generate_base_component(img_size)

    # Apply defect
    if defect_class == "ok":
        final_img = base_img
    else:
        final_img, _ = apply_random_defect(base_img.copy(), defect_class)

    # Create metadata
    metadata = {
        "id": sample_id,
        "class": defect_class,
        "label": ["ok", "scratch", "crack", "dent", "corrosion"].index(defect_class),
        "width": img_size,
        "height": img_size,
        "generated": True,
    }

    return final_img, metadata


_WORKER_BASE_COMPONENTS: Optional[List[np.ndarray]] = None


def _initialize_worker(base_components: List[np.ndarray]) -> None:
    global _WORKER_BASE_COMPONENTS
    _WORKER_BASE_COMPONENTS = base_components


def generate_dataset_parallel(args: Tuple[int, str, int, int]) -> Tuple[np.ndarray, Dict]:
    """Generate one sample using the base pool initialized once per worker."""
    sample_id, defect_class, img_size, seed = args
    return generate_sample(sample_id, defect_class, img_size, _WORKER_BASE_COMPONENTS, seed)


def create_class_distribution(num_samples: int, split: str = "train") -> Dict[str, int]:
    """
    Create class distribution for the dataset

    Args:
        num_samples: Total number of samples
        split: Dataset split ('train', 'val', 'test')

    Returns:
        Dictionary mapping class names to sample counts
    """
    classes = ["ok", "scratch", "crack", "dent", "corrosion"]

    if split == "train":
        # Balanced but realistic distribution
        # OK samples are more common in real scenarios
        ok_ratio = 0.4  # 40% OK samples
        defect_ratio = (1 - ok_ratio) / (len(classes) - 1)  # Equal defect distribution

        class_counts = {"ok": int(num_samples * ok_ratio)}
        for cls in classes[1:]:
            class_counts[cls] = int(num_samples * defect_ratio)

    else:
        # More realistic distribution for validation/testing
        # Higher proportion of OK samples
        ok_ratio = 0.7  # 70% OK samples
        defect_ratio = (1 - ok_ratio) / (len(classes) - 1)

        class_counts = {"ok": int(num_samples * ok_ratio)}
        for cls in classes[1:]:
            class_counts[cls] = int(num_samples * defect_ratio)

    # Adjust for rounding errors
    total_assigned = sum(class_counts.values())
    if total_assigned < num_samples:
        class_counts["ok"] += num_samples - total_assigned

    return class_counts


def generate_synthetic_dataset(
    num_samples: int = 12000,
    output_dir: str = "./data/synthetic_defects",
    img_size: int = 224,
    num_base_components: int = 50,
    num_workers: int = 1,
    seed: int = 42,
) -> Dict:
    """
    Generate complete synthetic dataset

    Args:
        num_samples: Total number of samples to generate
        output_dir: Output directory
        img_size: Image size
        num_base_components: Number of base component templates
        num_workers: Number of parallel workers (default: 1)
        seed: Random seed

    Returns:
        Dataset statistics
    """
    if not isinstance(num_samples, int) or num_samples < 7:
        raise ValueError("num_samples must be at least 7 so every split is non-empty")
    if not isinstance(img_size, int) or img_size < 8:
        raise ValueError("img_size must be at least 8")
    if not isinstance(num_base_components, int) or num_base_components < 1:
        raise ValueError("num_base_components must be at least 1")
    if not isinstance(num_workers, int) or num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    _set_random_seed(seed)

    output_path = Path(output_dir)
    images_dir = output_path / "images"

    managed_files = [
        output_path / "train_annotations.json",
        output_path / "val_annotations.json",
        output_path / "test_annotations.json",
        output_path / "dataset_stats.json",
    ]
    if (images_dir.exists() and any(images_dir.iterdir())) or any(
        path.exists() for path in managed_files
    ):
        raise FileExistsError(
            f"Output already contains generated dataset files: {output_path}. "
            "Choose an empty output directory."
        )

    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_samples} synthetic defect samples...")
    print(f"Output directory: {output_path}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Using {num_workers} workers")

    # Split dataset
    train_samples = int(num_samples * 0.8)
    val_samples = int(num_samples * 0.15)
    test_samples = num_samples - train_samples - val_samples

    splits = {"train": train_samples, "val": val_samples, "test": test_samples}

    dataset_stats = {}

    split_seed_offsets = {"train": 0, "val": 1_000_000, "test": 2_000_000}

    for split, split_samples in splits.items():
        print(f"\nGenerating {split} split ({split_samples} samples)...")

        # Use a distinct base pool per split to avoid train/evaluation leakage.
        _set_random_seed(seed + split_seed_offsets[split])
        base_components = []
        for _ in tqdm(range(num_base_components), desc=f"{split} base components"):
            base_img = generate_base_component(img_size)
            base_components.extend(generate_component_variations(base_img, 3))

        # Create class distribution
        class_counts = create_class_distribution(split_samples, split)

        print(f"Class distribution for {split}: {class_counts}")

        # Generate samples for each class
        split_annotations = []
        sample_id = 0

        executor = None
        if num_workers > 1:
            executor = ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_initialize_worker,
                initargs=(base_components,),
            )
        try:
            for class_name, count in class_counts.items():
                print(f"Generating {count} samples for class '{class_name}'...")
                if count == 0:
                    continue

                args_list = [
                    (
                        sample_id + i,
                        class_name,
                        img_size,
                        seed + split_seed_offsets[split] + sample_id + i,
                    )
                    for i in range(count)
                ]
                if executor is None:
                    generated = (
                        generate_sample(item[0], item[1], item[2], base_components, item[3])
                        for item in args_list
                    )
                else:
                    generated = executor.map(generate_dataset_parallel, args_list)

                annotations = []
                for img, metadata in generated:
                    img_filename = f"{split}_{metadata['id']:06d}_{class_name}.png"
                    img_path = images_dir / img_filename
                    if not cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
                        raise OSError(f"Failed to write generated image: {img_path}")

                    metadata["path"] = img_path.relative_to(output_path).as_posix()
                    annotations.append(metadata)
                split_annotations.extend(annotations)
                sample_id += count
        finally:
            if executor is not None:
                executor.shutdown()

        # Save annotations
        annotations_file = output_path / f"{split}_annotations.json"
        with open(annotations_file, "w", encoding="utf-8") as f:
            json.dump(split_annotations, f, indent=2)

        # Update statistics
        dataset_stats[split] = {
            "num_samples": len(split_annotations),
            "class_distribution": class_counts,
            "annotations_file": str(annotations_file),
        }

        print(f"Generated {len(split_annotations)} samples for {split} split")

    # Save overall dataset statistics
    stats_file = output_path / "dataset_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_samples": num_samples,
                "image_size": img_size,
                "classes": ["ok", "scratch", "crack", "dent", "corrosion"],
                "splits": dataset_stats,
                "generation_params": {
                    "num_base_components": num_base_components,
                    "seed": seed,
                    "num_workers": num_workers,
                },
            },
            f,
            indent=2,
        )

    print(f"\nDataset generation complete!")
    print(f"Total samples: {num_samples}")
    print(f"Images saved to: {images_dir}")
    print(f"Annotations saved to: {output_path}")
    print(f"Statistics saved to: {stats_file}")

    return dataset_stats


def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic Defect Dataset")
    parser.add_argument(
        "--num-samples", type=int, default=12000, help="Total number of samples to generate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/synthetic_defects",
        help="Output directory for dataset",
    )
    parser.add_argument("--img-size", type=int, default=224, help="Image size (square)")
    parser.add_argument(
        "--num-base-components", type=int, default=50, help="Number of base component templates"
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="Number of parallel workers (default: 1)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualization of defect types"
    )

    args = parser.parse_args()

    # Generate dataset
    stats = generate_synthetic_dataset(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        img_size=args.img_size,
        num_base_components=args.num_base_components,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Generate visualization if requested
    if args.visualize:
        print("\nGenerating visualization...")
        from src.utils.visualization import visualize_synthetic_defects

        vis_path = Path(args.output_dir) / "defect_examples.png"
        visualize_synthetic_defects(num_samples=5, save_path=str(vis_path))
        print(f"Visualization saved to: {vis_path}")


if __name__ == "__main__":
    main()
