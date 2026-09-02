"""Tests for the persistent synthetic dataset generator."""

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from scripts.generate_synthetic_data import (
    generate_base_component,
    generate_synthetic_dataset,
)


def test_small_base_component_supports_rivet_pattern():
    """The generator must honor the image size used by the smoke config."""
    np.random.seed(42)
    with patch("scripts.generate_synthetic_data.np.random.random", return_value=0.8):
        image = generate_base_component(32)

    assert image.shape == (32, 32, 3)
    assert image.dtype == np.uint8


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("num_samples", 0),
        ("img_size", 7),
        ("num_base_components", 0),
        ("num_workers", 0),
    ],
)
def test_generator_rejects_invalid_sizes(argument, value, tmp_path):
    options = {
        "num_samples": 7,
        "output_dir": str(tmp_path),
        "img_size": 32,
        "num_base_components": 1,
        "num_workers": 1,
    }
    options[argument] = value

    with pytest.raises(ValueError):
        generate_synthetic_dataset(**options)


def test_generator_writes_isolated_splits_and_refuses_overwrite(tmp_path):
    output_dir = tmp_path / "dataset"
    options = {
        "num_samples": 10,
        "output_dir": str(output_dir),
        "img_size": 32,
        "num_base_components": 1,
        "num_workers": 1,
        "seed": 7,
    }

    stats = generate_synthetic_dataset(**options)

    assert [stats[split]["num_samples"] for split in ("train", "val", "test")] == [8, 1, 1]
    assert len(list((output_dir / "images").glob("*.png"))) == 10
    assert "\\" not in (output_dir / "train_annotations.json").read_text(encoding="utf-8")
    train_image = cv2.imread(str(next((output_dir / "images").glob("train_000000_ok.png"))))
    val_image = cv2.imread(str(next((output_dir / "images").glob("val_000000_ok.png"))))
    assert not np.array_equal(train_image, val_image)

    with pytest.raises(FileExistsError, match="Choose an empty output directory"):
        generate_synthetic_dataset(**options)
