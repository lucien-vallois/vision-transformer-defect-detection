"""
Unit tests for inference module
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

from src.inference import DefectDetector, benchmark_inference
from src.utils.validators import ImageValidationError


class TestDefectDetector:
    """Test cases for DefectDetector class"""

    @pytest.fixture
    def dummy_model_path(self, tmp_path):
        """Create a dummy model checkpoint for testing"""
        from src.model import create_vit_model

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
        model_path = tmp_path / "test_model.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": config,
                "classes": ["ok", "scratch", "crack", "dent", "corrosion"],
            },
            model_path,
        )
        return str(model_path)

    def test_detector_initialization(self, dummy_model_path):
        """Test detector can be initialized"""
        detector = DefectDetector(model_path=dummy_model_path, device="cpu")
        assert detector is not None

    def test_predict_numpy_array(self, dummy_model_path):
        """Test prediction on numpy array"""
        detector = DefectDetector(model_path=dummy_model_path, device="cpu")

        # Create dummy image
        dummy_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        result = detector.predict(dummy_image)

        assert "class" in result
        assert "confidence" in result
        assert "latency_ms" in result
        assert result["class"] in ["ok", "scratch", "crack", "dent", "corrosion"]
        assert 0 <= result["confidence"] <= 1
        assert set(result["probabilities"]) == {"ok", "scratch", "crack", "dent", "corrosion"}

    def test_predict_batch(self, dummy_model_path):
        """Test batch prediction"""
        detector = DefectDetector(model_path=dummy_model_path, device="cpu", batch_size=2)

        # Create dummy images
        images = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]

        results = detector.predict_batch(images)

        assert len(results) == 3
        for result in results:
            assert "class" in result
            assert "confidence" in result

    @pytest.mark.parametrize(
        "image",
        [
            np.array([], dtype=np.uint8),
            np.full((64, 64, 3), np.nan, dtype=np.float32),
            np.full((64, 64, 3), 256, dtype=np.int16),
        ],
    )
    def test_predict_rejects_invalid_numpy_arrays(self, dummy_model_path, image):
        detector = DefectDetector(dummy_model_path, device="cpu")

        with pytest.raises(ImageValidationError):
            detector.predict(image)

    def test_predict_accepts_single_channel_and_normalized_float_arrays(self, dummy_model_path):
        detector = DefectDetector(dummy_model_path, device="cpu")

        single_channel_result = detector.predict(np.zeros((64, 64, 1), dtype=np.uint8))
        normalized_result = detector.predict(np.full((64, 64, 3), 0.5, dtype=np.float32))

        assert single_channel_result["class_idx"] in range(5)
        assert normalized_result["class_idx"] in range(5)

    def test_benchmark_rejects_empty_run(self, dummy_model_path):
        detector = DefectDetector(dummy_model_path, device="cpu")

        with pytest.raises(ValueError, match="num_runs"):
            benchmark_inference(detector, num_runs=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
