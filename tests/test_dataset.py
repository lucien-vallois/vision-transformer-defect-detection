"""
Unit tests for dataset classes
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.dataset import SyntheticDefectDataset, DefectDetectionDataset


class TestSyntheticDefectDataset:
    """Test cases for SyntheticDefectDataset"""

    def test_dataset_creation(self):
        """Test dataset can be created"""
        dataset = SyntheticDefectDataset(
            root_dir="./data", split="train", generate_on_fly=True, num_samples=10, img_size=224
        )
        assert len(dataset) == 10

    def test_dataset_getitem(self):
        """Test dataset returns correct item format"""
        dataset = SyntheticDefectDataset(
            root_dir="./data", split="train", generate_on_fly=True, num_samples=10, img_size=224
        )

        img, label, metadata = dataset[0]

        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)
        assert isinstance(label, int)
        assert 0 <= label < 5
        assert isinstance(metadata, dict)
        assert "class" in metadata

    def test_dataset_classes(self):
        """Test dataset has correct classes"""
        dataset = SyntheticDefectDataset(
            root_dir="./data", split="train", generate_on_fly=True, num_samples=10
        )

        expected_classes = ["ok", "scratch", "crack", "dent", "corrosion"]
        assert dataset.classes == expected_classes

    def test_dataset_splits(self):
        """Test dataset works for different splits"""
        for split in ["train", "val", "test"]:
            dataset = SyntheticDefectDataset(
                root_dir="./data", split=split, generate_on_fly=True, num_samples=10
            )
            assert len(dataset) == 10

    def test_validation_sample_is_stable(self):
        dataset = SyntheticDefectDataset(
            root_dir="./data",
            split="val",
            generate_on_fly=True,
            num_samples=10,
            img_size=32,
            seed=7,
        )

        first_image, first_label, first_metadata = dataset[0]
        second_image, second_label, second_metadata = dataset[0]

        assert torch.equal(first_image, second_image)
        assert first_label == second_label
        assert first_metadata == second_metadata

    @pytest.mark.parametrize("argument", ["img_size", "num_samples"])
    def test_dataset_rejects_non_positive_sizes(self, argument):
        options = {
            "root_dir": "./data",
            "generate_on_fly": True,
            "num_samples": 10,
            "img_size": 32,
        }
        options[argument] = 0

        with pytest.raises(ValueError, match="positive integer"):
            SyntheticDefectDataset(**options)

    def test_persistent_annotations_cannot_escape_dataset_root(self, tmp_path):
        (tmp_path / "train_annotations.json").write_text(
            json.dumps([{"id": 0, "class": "ok", "label": 0, "path": "../outside.png"}]),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="outside the dataset root"):
            SyntheticDefectDataset(tmp_path, split="train", generate_on_fly=False)

    def test_persistent_annotations_validate_class_and_label(self, tmp_path):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(images_dir / "sample.png")
        (tmp_path / "train_annotations.json").write_text(
            json.dumps([{"id": 0, "class": "ok", "label": 4, "path": "images/sample.png"}]),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="does not match class"):
            SyntheticDefectDataset(tmp_path, split="train", generate_on_fly=False)


class TestDefectDetectionDataset:
    """Test cases for DefectDetectionDataset wrapper"""

    def test_get_datasets(self):
        """Test get_datasets method"""
        config = {
            "data": {
                "root_dir": "./data",
                "generate_on_fly": True,
                "train_samples": 100,
                "val_samples": 20,
                "test_samples": 20,
                "img_size": 224,
            }
        }

        dataset_handler = DefectDetectionDataset(config)
        train_dataset, val_dataset, test_dataset = dataset_handler.get_datasets()

        assert len(train_dataset) == 100
        assert len(val_dataset) == 20
        assert len(test_dataset) == 20

    def test_get_dataloaders(self):
        """Test get_dataloaders method"""
        config = {
            "data": {
                "root_dir": "./data",
                "generate_on_fly": True,
                "train_samples": 100,
                "val_samples": 20,
                "test_samples": 20,
                "img_size": 224,
            }
        }

        dataset_handler = DefectDetectionDataset(config)
        train_loader, val_loader, test_loader = dataset_handler.get_dataloaders(
            batch_size=32, num_workers=0  # Use 0 for testing to avoid multiprocessing issues
        )

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

        # Test one batch
        batch = next(iter(train_loader))
        images, labels, metadata = batch
        assert images.shape[0] <= 32
        assert images.shape[1:] == (3, 224, 224)

    def test_get_dataloaders_uses_config_defaults(self):
        config = {
            "data": {
                "root_dir": "./data",
                "generate_on_fly": True,
                "train_samples": 8,
                "val_samples": 4,
                "test_samples": 4,
                "img_size": 32,
                "batch_size": 4,
                "num_workers": 0,
            }
        }

        train_loader, val_loader, test_loader = DefectDetectionDataset(config).get_dataloaders()

        assert train_loader.batch_size == 4
        assert val_loader.batch_size == 4
        assert test_loader.batch_size == 4
        assert train_loader.num_workers == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
