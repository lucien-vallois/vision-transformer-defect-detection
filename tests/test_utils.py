"""
Unit tests for utility modules
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.utils.metrics import compute_metrics, compute_multiclass_metrics, EarlyStopping
from src.utils.config_validator import validate_config, validate_config_file, ConfigValidationError


class TestMetrics:
    """Test cases for metrics utilities"""

    def test_compute_metrics(self):
        """Test basic metrics computation"""
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 1])

        precision, recall, f1 = compute_metrics(y_true, y_pred)

        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_compute_multiclass_metrics(self):
        """Test multiclass metrics computation"""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        y_prob = np.random.rand(6, 3)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

        class_names = ["ok", "scratch", "crack"]

        metrics = compute_multiclass_metrics(y_true, y_pred, y_prob, class_names)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert metrics["accuracy"] == 1.0

    def test_compute_multiclass_metrics_keeps_missing_configured_classes(self):
        metrics = compute_multiclass_metrics(
            np.array([0, 0]),
            np.array([0, 0]),
            np.array([[1, 0, 0], [1, 0, 0]]),
            ["ok", "scratch", "crack"],
        )

        assert np.asarray(metrics["confusion_matrix"]).shape == (3, 3)
        assert set(["ok", "scratch", "crack"]) <= set(metrics["per_class_report"])

    def test_early_stopping(self):
        """Test early stopping functionality"""
        early_stop = EarlyStopping(patience=3, min_delta=0.01)

        # Simulate improving losses
        losses = [1.0, 0.9, 0.8, 0.7, 0.75, 0.72, 0.71]

        should_stop = False
        for loss in losses:
            should_stop = early_stop(loss)
            if should_stop:
                break

        # Should stop after patience is exceeded
        assert should_stop is True


class TestConfigValidator:
    """Test cases for configuration validator"""

    def test_validate_config_valid(self):
        """Test validation of valid configuration"""
        config = {
            "model": {
                "img_size": 224,
                "patch_size": 16,
                "in_channels": 3,
                "num_classes": 5,
                "embed_dim": 64,
                "depth": 2,
                "num_heads": 4,
            },
            "data": {
                "root_dir": "./data",
                "generate_on_fly": True,
                "train_samples": 1000,
                "val_samples": 200,
                "test_samples": 200,
            },
            "training": {
                "epochs": 10,
                "lr": 0.001,
                "weight_decay": 0.01,
                "min_lr": 0.00001,
                "output_dir": "./experiments",
                "early_stopping_patience": 3,
                "early_stopping_delta": 0.001,
            },
        }

        is_valid, errors = validate_config(config, strict=False)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_config_missing_key(self):
        """Test validation fails on missing required key"""
        config = {
            "model": {"img_size": 224}
            # Missing 'data' and 'training'
        }

        is_valid, errors = validate_config(config, strict=False)
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_config_invalid_value(self):
        """Test validation fails on invalid value"""
        config = {
            "model": {
                "img_size": -224,  # Invalid: negative
                "patch_size": 16,
                "in_channels": 3,
                "num_classes": 5,
            },
            "data": {"root_dir": "./data", "train_samples": 1000},
            "training": {"epochs": 10, "lr": 0.001, "output_dir": "./experiments"},
        }

        is_valid, errors = validate_config(config, strict=False)
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_config_rejects_incompatible_class_count(self):
        config = {
            "model": {
                "img_size": 32,
                "patch_size": 8,
                "in_channels": 3,
                "num_classes": 4,
                "embed_dim": 32,
                "depth": 1,
                "num_heads": 4,
            },
            "data": {
                "root_dir": "./data",
                "generate_on_fly": True,
                "train_samples": 8,
                "val_samples": 4,
                "test_samples": 4,
            },
            "training": {
                "epochs": 1,
                "lr": 0.001,
                "weight_decay": 0.01,
                "min_lr": 0.00001,
                "output_dir": "./experiments",
                "early_stopping_patience": 3,
                "early_stopping_delta": 0.001,
            },
        }

        is_valid, errors = validate_config(config, strict=False)

        assert is_valid is False
        assert any("num_classes must be 5" in error for error in errors)

    def test_validate_config_rejects_distributed_training(self):
        config = {
            "model": {
                "img_size": 32,
                "patch_size": 8,
                "in_channels": 3,
                "num_classes": 5,
                "embed_dim": 32,
                "depth": 1,
                "num_heads": 4,
            },
            "data": {
                "root_dir": "./data",
                "generate_on_fly": True,
                "train_samples": 8,
                "val_samples": 4,
                "test_samples": 4,
            },
            "training": {
                "epochs": 1,
                "lr": 0.001,
                "weight_decay": 0.01,
                "min_lr": 0.00001,
                "output_dir": "./experiments",
                "early_stopping_patience": 3,
                "early_stopping_delta": 0.001,
            },
            "hardware": {"device": "cpu", "distributed": True},
        }

        is_valid, errors = validate_config(config, strict=False)

        assert is_valid is False
        assert any("distributed is not supported" in error for error in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
