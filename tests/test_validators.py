"""
Unit tests for validation utilities
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch
from PIL import Image

from src.utils.validators import (
    ImageValidationError,
    ModelValidationError,
    DataValidationError,
    validate_image_path,
    validate_image_array,
    validate_model_input,
    validate_model_output,
    validate_dataset_path,
    validate_class_label,
    validate_batch_size,
    validate_image_for_inference,
)


class TestImagePathValidation:
    """Test cases for image path validation"""

    def test_validate_existing_image_path(self, tmp_path):
        """Test validation of existing image file"""
        # Create a test image file
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake image data")

        result = validate_image_path(test_image)
        assert result == Path(test_image)

    def test_validate_nonexistent_path(self):
        """Test validation fails for non-existent path"""
        with pytest.raises(ImageValidationError, match="not found"):
            validate_image_path("nonexistent_image.jpg")

    def test_validate_directory_path(self, tmp_path):
        """Test validation fails for directory"""
        with pytest.raises(ImageValidationError, match="not a file"):
            validate_image_path(tmp_path)

    def test_validate_unsupported_format(self, tmp_path):
        """Test validation fails for unsupported format"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("not an image")

        with pytest.raises(ImageValidationError, match="Unsupported image format"):
            validate_image_path(test_file)


class TestImageArrayValidation:
    """Test cases for image array validation"""

    def test_validate_valid_grayscale_image(self):
        """Test validation of valid grayscale image"""
        img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        result = validate_image_array(img)
        assert result is img

    def test_validate_valid_color_image(self):
        """Test validation of valid color image"""
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = validate_image_array(img)
        assert result is img

    def test_validate_non_array(self):
        """Test validation fails for non-array"""
        with pytest.raises(ImageValidationError, match="must be numpy array"):
            validate_image_array("not an array")

    def test_validate_empty_array(self):
        """Test validation fails for empty array"""
        with pytest.raises(ImageValidationError, match="empty"):
            validate_image_array(np.array([]))

    def test_validate_wrong_dimensions(self):
        """Test validation fails for wrong dimensions"""
        img = np.random.rand(224, 224, 224, 3)
        with pytest.raises(ImageValidationError, match="must be 2D"):
            validate_image_array(img)

    def test_validate_wrong_channels(self):
        """Test validation fails for wrong number of channels"""
        img = np.random.rand(224, 224, 5)
        with pytest.raises(ImageValidationError, match="channels"):
            validate_image_array(img)

    def test_validate_size_constraints(self):
        """Test validation with size constraints"""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Should pass with valid constraints
        validate_image_array(img, min_size=(50, 50), max_size=(200, 200))

        # Should fail with too small
        with pytest.raises(ImageValidationError, match="smaller than minimum"):
            validate_image_array(img, min_size=(150, 150))

        # Should fail with too large
        with pytest.raises(ImageValidationError, match="larger than maximum"):
            validate_image_array(img, max_size=(50, 50))

    def test_validate_nan_values(self):
        """Test validation fails for NaN values"""
        img = np.random.rand(224, 224, 3).astype(np.float32)
        img[0, 0, 0] = np.nan

        with pytest.raises(ImageValidationError, match="NaN"):
            validate_image_array(img)

    def test_validate_inf_values(self):
        """Test validation fails for Inf values"""
        img = np.random.rand(224, 224, 3).astype(np.float32)
        img[0, 0, 0] = np.inf

        with pytest.raises(ImageValidationError, match="Inf"):
            validate_image_array(img)

    def test_validate_float_range(self):
        img = np.full((32, 32, 3), -0.1, dtype=np.float32)

        with pytest.raises(ImageValidationError, match="range"):
            validate_image_array(img)


class TestModelInputValidation:
    """Test cases for model input validation"""

    def test_validate_valid_input(self):
        """Test validation of valid model input"""
        tensor = torch.randn(2, 3, 224, 224)
        result = validate_model_input(tensor)
        assert result is tensor

    def test_validate_non_tensor(self):
        """Test validation fails for non-tensor"""
        with pytest.raises(ModelValidationError, match="must be torch.Tensor"):
            validate_model_input(np.random.rand(2, 3, 224, 224))

    def test_validate_empty_tensor(self):
        """Test validation fails for empty tensor"""
        with pytest.raises(ModelValidationError, match="empty"):
            validate_model_input(torch.tensor([]))

    def test_validate_wrong_dimensions(self):
        """Test validation fails for wrong dimensions"""
        tensor = torch.randn(2, 3, 224)
        with pytest.raises(ModelValidationError, match="must be 4D"):
            validate_model_input(tensor)

    def test_validate_wrong_batch_size(self):
        """Test validation fails for wrong batch size"""
        tensor = torch.randn(2, 3, 224, 224)
        with pytest.raises(ModelValidationError, match="batch size"):
            validate_model_input(tensor, batch_size=4)

    def test_validate_wrong_channels(self):
        """Test validation fails for wrong channels"""
        tensor = torch.randn(2, 4, 224, 224)
        with pytest.raises(ModelValidationError, match="channels"):
            validate_model_input(tensor, num_channels=3)

    def test_validate_wrong_image_size(self):
        """Test validation fails for wrong image size"""
        tensor = torch.randn(2, 3, 256, 256)
        with pytest.raises(ModelValidationError, match="image size"):
            validate_model_input(tensor, img_size=224)

    def test_validate_nan_values(self):
        """Test validation fails for NaN values"""
        tensor = torch.randn(2, 3, 224, 224)
        tensor[0, 0, 0, 0] = float("nan")

        with pytest.raises(ModelValidationError, match="NaN"):
            validate_model_input(tensor)


class TestModelOutputValidation:
    """Test cases for model output validation"""

    def test_validate_valid_output(self):
        """Test validation of valid model output"""
        output = torch.randn(2, 5)
        result = validate_model_output(output, num_classes=5)
        assert result is output

    def test_validate_wrong_dimensions(self):
        """Test validation fails for wrong dimensions"""
        output = torch.randn(2, 5, 3)
        with pytest.raises(ModelValidationError, match="must be 2D"):
            validate_model_output(output, num_classes=5)

    def test_validate_wrong_num_classes(self):
        """Test validation fails for wrong number of classes"""
        output = torch.randn(2, 3)
        with pytest.raises(ModelValidationError, match="classes"):
            validate_model_output(output, num_classes=5)


class TestDatasetPathValidation:
    """Test cases for dataset path validation"""

    def test_validate_existing_directory(self, tmp_path):
        """Test validation of existing directory"""
        result = validate_dataset_path(tmp_path)
        assert result == Path(tmp_path)

    def test_validate_nonexistent_directory(self):
        """Test validation fails for non-existent directory"""
        with pytest.raises(DataValidationError, match="not found"):
            validate_dataset_path("nonexistent_dir")

    def test_validate_file_path(self, tmp_path):
        """Test validation fails for file path"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(DataValidationError, match="not a directory"):
            validate_dataset_path(test_file)


class TestClassLabelValidation:
    """Test cases for class label validation"""

    def test_validate_valid_int_label(self):
        """Test validation of valid integer label"""
        result = validate_class_label(2, num_classes=5)
        assert result == 2

    def test_validate_valid_string_label(self):
        """Test validation of valid string label"""
        class_names = ["ok", "scratch", "crack", "dent", "corrosion"]
        result = validate_class_label("scratch", num_classes=5, class_names=class_names)
        assert result == 1

    def test_validate_out_of_range_label(self):
        """Test validation fails for out-of-range label"""
        with pytest.raises(DataValidationError, match="out of range"):
            validate_class_label(10, num_classes=5)

    def test_validate_negative_label(self):
        """Test validation fails for negative label"""
        with pytest.raises(DataValidationError, match="out of range"):
            validate_class_label(-1, num_classes=5)

    def test_validate_invalid_string_label(self):
        """Test validation fails for invalid string label"""
        class_names = ["ok", "scratch", "crack"]
        with pytest.raises(DataValidationError, match="Invalid class name"):
            validate_class_label("invalid", num_classes=3, class_names=class_names)


class TestBatchSizeValidation:
    """Test cases for batch size validation"""

    def test_validate_valid_batch_size(self):
        """Test validation of valid batch size"""
        result = validate_batch_size(32)
        assert result == 32

    def test_validate_too_small_batch_size(self):
        """Test validation fails for too small batch size"""
        with pytest.raises(DataValidationError, match="less than minimum"):
            validate_batch_size(0)

    def test_validate_too_large_batch_size(self):
        """Test validation fails for too large batch size"""
        with pytest.raises(DataValidationError, match="greater than maximum"):
            validate_batch_size(10000)


class TestImageForInferenceValidation:
    """Test cases for comprehensive image validation for inference"""

    def test_validate_numpy_array(self):
        """Test validation of numpy array"""
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result, is_file = validate_image_for_inference(img)
        assert isinstance(result, np.ndarray)
        assert not is_file

    def test_validate_pil_image(self):
        """Test validation of PIL Image"""
        img = Image.new("RGB", (224, 224), color="red")
        result, is_file = validate_image_for_inference(img)
        assert isinstance(result, np.ndarray)
        assert not is_file

    def test_validate_file_path(self, tmp_path):
        """Test validation of file path"""
        # Create a valid image file
        test_image = tmp_path / "test.png"
        img = Image.new("RGB", (224, 224), color="blue")
        img.save(test_image)

        result, is_file = validate_image_for_inference(test_image)
        assert isinstance(result, np.ndarray)
        assert is_file

    def test_validate_invalid_type(self):
        """Test validation fails for invalid type"""
        with pytest.raises(ImageValidationError, match="must be"):
            validate_image_for_inference(12345)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
