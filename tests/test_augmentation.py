"""
Unit tests for data augmentation module
"""

import sys
from pathlib import Path
from unittest.mock import patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

from src.data.augmentation import (
    get_training_augmentation,
    get_validation_augmentation,
    get_inference_transform,
    apply_random_defect,
    DefectAugmentation,
)


class TestAugmentationTransforms:
    """Test cases for augmentation transforms"""

    def test_training_augmentation(self):
        """Test training augmentation pipeline"""
        transform = get_training_augmentation(img_size=224)

        # Create dummy image
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        # Apply transform
        if hasattr(transform, "__call__"):
            try:
                # Try albumentations format
                result = transform(image=img)
                if isinstance(result, dict):
                    transformed_img = result["image"]
                else:
                    transformed_img = result
            except TypeError:
                # Torchvision format
                from PIL import Image

                pil_img = Image.fromarray(img)
                transformed_img = transform(pil_img)
        else:
            transformed_img = transform(img)

        # Check output shape
        if isinstance(transformed_img, torch.Tensor):
            assert transformed_img.shape == (3, 224, 224)
        else:
            assert transformed_img.shape[:2] == (224, 224)

    def test_validation_augmentation(self):
        """Test validation augmentation pipeline"""
        transform = get_validation_augmentation(img_size=224)

        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        try:
            result = transform(image=img)
            if isinstance(result, dict):
                transformed_img = result["image"]
            else:
                transformed_img = result
        except TypeError:
            from PIL import Image

            pil_img = Image.fromarray(img)
            transformed_img = transform(pil_img)

        if isinstance(transformed_img, torch.Tensor):
            assert transformed_img.shape == (3, 224, 224)

    def test_inference_transform(self):
        """Test inference transform"""
        transform = get_inference_transform(img_size=224)

        from PIL import Image

        img = Image.new("RGB", (256, 256), color="red")
        transformed_img = transform(img)

        assert isinstance(transformed_img, torch.Tensor)
        assert transformed_img.shape == (3, 224, 224)


class TestDefectAugmentation:
    """Test cases for defect augmentation"""

    def test_add_scratch(self):
        """Test adding scratch defect"""
        img = np.random.randint(150, 200, (224, 224, 3), dtype=np.uint8)
        result = DefectAugmentation.add_scratch(img.copy())

        assert result.shape == img.shape
        assert result.dtype == img.dtype

    def test_add_crack(self):
        """Test adding crack defect"""
        img = np.random.randint(150, 200, (224, 224, 3), dtype=np.uint8)
        result = DefectAugmentation.add_crack(img.copy())

        assert result.shape == img.shape
        assert result.dtype == img.dtype

    def test_add_dent(self):
        """Test adding dent defect"""
        img = np.random.randint(150, 200, (224, 224, 3), dtype=np.uint8)
        result = DefectAugmentation.add_dent(img.copy())

        assert result.shape == img.shape
        assert result.dtype == img.dtype

    def test_add_corrosion(self):
        """Test adding corrosion defect"""
        img = np.random.randint(150, 200, (224, 224, 3), dtype=np.uint8)
        result = DefectAugmentation.add_corrosion(img.copy())

        assert result.shape == img.shape
        assert result.dtype == img.dtype

    def test_texture_noise_can_darken_pixels(self):
        img = np.full((32, 32, 3), 180, dtype=np.uint8)
        with (
            patch("src.data.augmentation.np.random.normal", return_value=np.full(img.shape, -10.0)),
            patch("src.data.augmentation.np.random.random", return_value=0.0),
        ):
            result = DefectAugmentation.add_texture_variation(img, intensity=0)

        assert result.mean() < img.mean()


class TestRandomDefectApplication:
    """Test cases for random defect application"""

    def test_apply_random_defect_scratch(self):
        """Test applying scratch defect"""
        img = np.random.randint(150, 200, (224, 224, 3), dtype=np.uint8)
        result_img, defect_info = apply_random_defect(img.copy(), "scratch")

        assert result_img.shape == img.shape
        assert isinstance(defect_info, dict)
        assert defect_info["type"] == "scratch"

    def test_apply_random_defect_crack(self):
        """Test applying crack defect"""
        img = np.random.randint(150, 200, (224, 224, 3), dtype=np.uint8)
        result_img, defect_info = apply_random_defect(img.copy(), "crack")

        assert result_img.shape == img.shape
        assert defect_info["type"] == "crack"

    def test_apply_random_defect_dent(self):
        """Test applying dent defect"""
        img = np.random.randint(150, 200, (224, 224, 3), dtype=np.uint8)
        result_img, defect_info = apply_random_defect(img.copy(), "dent")

        assert result_img.shape == img.shape
        assert defect_info["type"] == "dent"

    def test_apply_random_defect_corrosion(self):
        """Test applying corrosion defect"""
        img = np.random.randint(150, 200, (224, 224, 3), dtype=np.uint8)
        result_img, defect_info = apply_random_defect(img.copy(), "corrosion")

        assert result_img.shape == img.shape
        assert defect_info["type"] == "corrosion"

    def test_apply_random_defect_invalid_type(self):
        """Test applying invalid defect type"""
        img = np.random.randint(150, 200, (224, 224, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Unknown defect type"):
            apply_random_defect(img.copy(), "invalid_type")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
