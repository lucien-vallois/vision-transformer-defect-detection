"""
Data augmentation transforms for defect detection training
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torchvision.transforms as transforms


def get_training_augmentation(img_size: int = 224):
    """Get augmentation pipeline for training"""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_validation_augmentation(img_size: int = 224):
    """Get augmentation pipeline for validation/testing"""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_inference_transform(img_size: int = 224) -> transforms.Compose:
    """Get transform for inference"""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class DefectAugmentation:
    """Custom augmentations for defect generation"""

    @staticmethod
    def add_scratch(image: np.ndarray, intensity: float = 0.8) -> np.ndarray:
        """Add linear scratch defect"""
        img = image.copy()
        h, w = img.shape[:2]

        # Random scratch parameters
        start_point = (np.random.randint(0, w), np.random.randint(0, h))
        end_point = (np.random.randint(0, w), np.random.randint(0, h))
        thickness = np.random.randint(1, 4)
        color = (np.random.randint(50, 150),) * 3  # Dark gray

        # Draw scratch
        cv2.line(img, start_point, end_point, color, thickness)
        return cv2.addWeighted(img, 1 - intensity, image, intensity, 0)

    @staticmethod
    def add_crack(image: np.ndarray, intensity: float = 0.9) -> np.ndarray:
        """Add crack defect"""
        img = image.copy()
        h, w = img.shape[:2]

        # Create jagged crack path
        points = []
        num_segments = np.random.randint(3, 8)

        x, y = np.random.randint(0, w), np.random.randint(0, h)
        for _ in range(num_segments):
            points.append((x, y))
            x += np.random.randint(-20, 20)
            y += np.random.randint(-20, 20)
            x = np.clip(x, 0, w - 1)
            y = np.clip(y, 0, h - 1)

        # Draw crack segments
        thickness = np.random.randint(1, 3)
        color = (np.random.randint(30, 100),) * 3

        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i + 1], color, thickness)

        return cv2.addWeighted(img, 1 - intensity, image, intensity, 0)

    @staticmethod
    def add_dent(image: np.ndarray, intensity: float = 0.7) -> np.ndarray:
        """Add dent/impact defect"""
        img = image.copy()
        h, w = img.shape[:2]

        # Random dent parameters
        max_radius = max(2, min(h, w) // 6)
        radius = np.random.randint(2, max_radius + 1)
        margin = radius + 1
        center = (
            np.random.randint(margin, w - margin),
            np.random.randint(margin, h - margin),
        )
        depth_color = np.random.randint(100, 180)

        # Create circular dent
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)

        # Apply dent effect
        dent_overlay = np.full_like(img, depth_color)
        dent_overlay = cv2.GaussianBlur(dent_overlay, (radius * 2 + 1, radius * 2 + 1), 0)

        # Blend with original image
        img = cv2.seamlessClone(dent_overlay, img, mask, center, cv2.NORMAL_CLONE)
        return cv2.addWeighted(img, 1 - intensity, image, intensity, 0)

    @staticmethod
    def add_corrosion(image: np.ndarray, intensity: float = 0.6) -> np.ndarray:
        """Add corrosion/oxidation defect"""
        img = image.copy()
        h, w = img.shape[:2]

        # Create corrosion pattern
        noise = np.random.normal(0, 25, (h, w, 3))

        # Add yellowish tint for corrosion
        corrosion_tint = np.array([50, 25, 0])  # RGB
        noise = np.clip(noise + corrosion_tint, 0, 255).astype(np.uint8)

        # Apply to random regions
        mask = np.random.random((h, w)) > 0.7
        mask = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0)
        mask = (mask > 0.3).astype(np.uint8)

        # Apply corrosion
        for c in range(3):
            img[:, :, c] = np.where(
                mask, cv2.addWeighted(img[:, :, c], 0.7, noise[:, :, c], 0.3, 0), img[:, :, c]
            )

        return cv2.addWeighted(img, 1 - intensity, image, intensity, 0)

    @staticmethod
    def add_texture_variation(image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
        """Add natural texture variations"""
        img = image.copy()

        # Add subtle noise
        noise = np.random.normal(0, 10, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Add subtle blur variations
        if np.random.random() > 0.5:
            kernel_size = np.random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        return cv2.addWeighted(img, 1 - intensity, image, intensity, 0)


def apply_random_defect(
    image: np.ndarray, defect_type: Optional[str] = None
) -> Tuple[np.ndarray, Dict[str, str]]:
    """
    Apply a random defect to an image

    Args:
        image: Input image
        defect_type: Specific defect type or None for random

    Returns:
        Tuple of (augmented_image, metadata)
    """
    defect_augmenter = DefectAugmentation()

    supported_defects = ["scratch", "crack", "dent", "corrosion", "ok"]
    if defect_type is None:
        # Random defect selection
        defect_type = str(np.random.choice(supported_defects[:-1]))
    elif defect_type not in supported_defects:
        raise ValueError(
            f"Unknown defect type: {defect_type}. Expected one of: {', '.join(supported_defects)}"
        )

    if defect_type == "scratch":
        augmented = defect_augmenter.add_scratch(image)
    elif defect_type == "crack":
        augmented = defect_augmenter.add_crack(image)
    elif defect_type == "dent":
        augmented = defect_augmenter.add_dent(image)
    elif defect_type == "corrosion":
        augmented = defect_augmenter.add_corrosion(image)
    else:  # ok
        augmented = defect_augmenter.add_texture_variation(image)

    return augmented, {"type": defect_type}
