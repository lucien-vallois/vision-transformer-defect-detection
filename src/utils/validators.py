"""
Input validation utilities for Vision Transformer defect detection.
Provides robust validation for images, model inputs, and data formats.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image


class ValidationError(Exception):
    """Base exception for validation errors"""

    pass


class ImageValidationError(ValidationError):
    """Exception raised for image validation errors"""

    pass


class ModelValidationError(ValidationError):
    """Exception raised for model validation errors"""

    pass


class DataValidationError(ValidationError):
    """Exception raised for data validation errors"""

    pass


def validate_image_path(image_path: Union[str, Path]) -> Path:
    """
    Validate that an image path exists and is readable.

    Args:
        image_path: Path to image file

    Returns:
        Path object if valid

    Raises:
        ImageValidationError: If path is invalid or file doesn't exist
    """
    path = Path(image_path)

    if not path.exists():
        raise ImageValidationError(f"Image file not found: {image_path}")

    if not path.is_file():
        raise ImageValidationError(f"Path is not a file: {image_path}")

    # Check if file is readable
    if not os.access(path, os.R_OK):
        raise ImageValidationError(f"Image file is not readable: {image_path}")

    # Check file extension
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    if path.suffix.lower() not in valid_extensions:
        raise ImageValidationError(
            f"Unsupported image format: {path.suffix}. " f"Supported formats: {valid_extensions}"
        )

    return path


def validate_image_array(
    image: np.ndarray,
    expected_shape: Optional[Tuple[int, ...]] = None,
    min_size: Optional[Tuple[int, int]] = None,
    max_size: Optional[Tuple[int, int]] = None,
    dtype: Optional[type] = None,
) -> np.ndarray:
    """
    Validate numpy array representing an image.

    Args:
        image: Image as numpy array
        expected_shape: Expected shape (H, W) or (H, W, C)
        min_size: Minimum (height, width)
        max_size: Maximum (height, width)
        dtype: Expected dtype

    Returns:
        Validated image array

    Raises:
        ImageValidationError: If image is invalid
    """
    if not isinstance(image, np.ndarray):
        raise ImageValidationError(f"Image must be numpy array, got {type(image)}")

    if image.size == 0:
        raise ImageValidationError("Image array is empty")

    # Check dimensions
    if image.ndim not in [2, 3]:
        raise ImageValidationError(f"Image must be 2D (grayscale) or 3D (color), got {image.ndim}D")

    if image.ndim == 3:
        h, w, c = image.shape
        if c not in [1, 3, 4]:
            raise ImageValidationError(f"Color image must have 1, 3, or 4 channels, got {c}")
    else:
        h, w = image.shape

    # Check size constraints
    if min_size is not None:
        min_h, min_w = min_size
        if h < min_h or w < min_w:
            raise ImageValidationError(
                f"Image size ({h}x{w}) is smaller than minimum ({min_h}x{min_w})"
            )

    if max_size is not None:
        max_h, max_w = max_size
        if h > max_h or w > max_w:
            raise ImageValidationError(
                f"Image size ({h}x{w}) is larger than maximum ({max_h}x{max_w})"
            )

    # Check dtype
    if dtype is not None and image.dtype != dtype:
        raise ImageValidationError(f"Image dtype must be {dtype}, got {image.dtype}")

    # Check value range for uint8
    if image.dtype == np.uint8:
        if image.min() < 0 or image.max() > 255:
            raise ImageValidationError("uint8 image values must be in range [0, 255]")
    elif image.dtype in [np.float32, np.float64]:
        if np.isfinite(image).all() and (image.min() < 0 or image.max() > 255):
            raise ImageValidationError("Floating-point image values must be in range [0, 255]")

    # Check for NaN or Inf
    if np.any(np.isnan(image)):
        raise ImageValidationError("Image contains NaN values")

    if np.any(np.isinf(image)):
        raise ImageValidationError("Image contains Inf values")

    return image


def validate_model_input(
    tensor: torch.Tensor,
    expected_shape: Optional[Tuple[int, ...]] = None,
    batch_size: Optional[int] = None,
    num_channels: int = 3,
    img_size: int = 224,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Validate model input tensor.

    Args:
        tensor: Input tensor
        expected_shape: Expected shape (B, C, H, W)
        batch_size: Expected batch size
        num_channels: Expected number of channels
        img_size: Expected image size (H=W)
        dtype: Expected dtype

    Returns:
        Validated tensor

    Raises:
        ModelValidationError: If tensor is invalid
    """
    if not isinstance(tensor, torch.Tensor):
        raise ModelValidationError(f"Input must be torch.Tensor, got {type(tensor)}")

    if tensor.numel() == 0:
        raise ModelValidationError("Input tensor is empty")

    # Check dimensions
    if tensor.ndim != 4:
        raise ModelValidationError(f"Input tensor must be 4D (B, C, H, W), got {tensor.ndim}D")

    b, c, h, w = tensor.shape

    # Check batch size
    if batch_size is not None and b != batch_size:
        raise ModelValidationError(f"Expected batch size {batch_size}, got {b}")

    # Check channels
    if c != num_channels:
        raise ModelValidationError(f"Expected {num_channels} channels, got {c}")

    # Check image size
    if h != img_size or w != img_size:
        raise ModelValidationError(f"Expected image size {img_size}x{img_size}, got {h}x{w}")

    # Check dtype
    if dtype is not None and tensor.dtype != dtype:
        raise ModelValidationError(f"Expected dtype {dtype}, got {tensor.dtype}")

    # Check for NaN or Inf
    if torch.any(torch.isnan(tensor)):
        raise ModelValidationError("Input tensor contains NaN values")

    if torch.any(torch.isinf(tensor)):
        raise ModelValidationError("Input tensor contains Inf values")

    return tensor


def validate_model_output(
    output: torch.Tensor, num_classes: int, batch_size: Optional[int] = None
) -> torch.Tensor:
    """
    Validate model output tensor.

    Args:
        output: Model output tensor
        num_classes: Expected number of classes
        batch_size: Expected batch size

    Returns:
        Validated output tensor

    Raises:
        ModelValidationError: If output is invalid
    """
    if not isinstance(output, torch.Tensor):
        raise ModelValidationError(f"Output must be torch.Tensor, got {type(output)}")

    if output.numel() == 0:
        raise ModelValidationError("Output tensor is empty")

    # Check dimensions
    if output.ndim != 2:
        raise ModelValidationError(f"Output tensor must be 2D (B, num_classes), got {output.ndim}D")

    b, c = output.shape

    if batch_size is not None and b != batch_size:
        raise ModelValidationError(f"Expected batch size {batch_size}, got {b}")

    if c != num_classes:
        raise ModelValidationError(f"Expected {num_classes} classes, got {c}")

    # Check for NaN or Inf
    if torch.any(torch.isnan(output)):
        raise ModelValidationError("Output tensor contains NaN values")

    if torch.any(torch.isinf(output)):
        raise ModelValidationError("Output tensor contains Inf values")

    return output


def validate_dataset_path(dataset_path: Union[str, Path]) -> Path:
    """
    Validate dataset directory path.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Path object if valid

    Raises:
        DataValidationError: If path is invalid
    """
    path = Path(dataset_path)

    if not path.exists():
        raise DataValidationError(f"Dataset directory not found: {dataset_path}")

    if not path.is_dir():
        raise DataValidationError(f"Path is not a directory: {dataset_path}")

    if not os.access(path, os.R_OK):
        raise DataValidationError(f"Dataset directory is not readable: {dataset_path}")

    return path


def validate_class_label(
    label: Union[int, str], num_classes: int, class_names: Optional[List[str]] = None
) -> int:
    """
    Validate class label.

    Args:
        label: Class label (int or string)
        num_classes: Number of classes
        class_names: Optional list of class names

    Returns:
        Validated integer label

    Raises:
        DataValidationError: If label is invalid
    """
    if isinstance(label, str):
        if class_names is None:
            raise DataValidationError("Cannot validate string label without class_names")
        if label not in class_names:
            raise DataValidationError(f"Invalid class name '{label}'. Valid classes: {class_names}")
        label = class_names.index(label)

    if not isinstance(label, int):
        raise DataValidationError(f"Label must be int or str, got {type(label)}")

    if label < 0 or label >= num_classes:
        raise DataValidationError(f"Label {label} is out of range [0, {num_classes-1}]")

    return label


def validate_batch_size(batch_size: int, min_size: int = 1, max_size: int = 1024) -> int:
    """
    Validate batch size.

    Args:
        batch_size: Batch size to validate
        min_size: Minimum batch size
        max_size: Maximum batch size

    Returns:
        Validated batch size

    Raises:
        DataValidationError: If batch size is invalid
    """
    if not isinstance(batch_size, int):
        raise DataValidationError(f"Batch size must be int, got {type(batch_size)}")

    if batch_size < min_size:
        raise DataValidationError(f"Batch size {batch_size} is less than minimum {min_size}")

    if batch_size > max_size:
        raise DataValidationError(f"Batch size {batch_size} is greater than maximum {max_size}")

    return batch_size


def validate_image_for_inference(
    image: Union[str, Path, np.ndarray, Image.Image], img_size: int = 224
) -> Tuple[np.ndarray, bool]:
    """
    Comprehensive validation for inference input image.

    Args:
        image: Input image (path, numpy array, or PIL Image)
        img_size: Expected image size after preprocessing

    Returns:
        Tuple of (validated_image_array, is_file_path)

    Raises:
        ImageValidationError: If image is invalid
    """
    is_file_path = False

    if isinstance(image, (str, Path)):
        # Validate file path
        path = validate_image_path(image)
        is_file_path = True

        # Try to load image
        try:
            pil_image = Image.open(path)
            image = np.array(pil_image)
        except Exception as e:
            raise ImageValidationError(f"Failed to load image from {path}: {e}")

    elif isinstance(image, Image.Image):
        image = np.array(image)

    elif not isinstance(image, np.ndarray):
        raise ImageValidationError(
            f"Image must be str, Path, PIL.Image, or np.ndarray, got {type(image)}"
        )

    # Validate array
    image = validate_image_array(
        image,
        min_size=(32, 32),  # Minimum reasonable size
        max_size=(4096, 4096),  # Maximum reasonable size
    )

    return image, is_file_path
