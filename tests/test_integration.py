"""
Integration tests for Vision Transformer defect detection system.
Tests the interaction between multiple components.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SyntheticDefectDataset, DefectDetectionDataset
from src.inference import DefectDetector
from src.model import VisionTransformer, create_vit_model
from src.utils.metrics import compute_metrics, compute_multiclass_metrics

TINY_CONFIG = {
    "img_size": 64,
    "patch_size": 8,
    "in_channels": 3,
    "num_classes": 5,
    "embed_dim": 64,
    "depth": 2,
    "num_heads": 4,
    "mlp_ratio": 2.0,
}


def make_model():
    return create_vit_model(TINY_CONFIG)


class TestTrainingPipeline:
    """Integration tests for training pipeline"""

    def test_dataset_to_model_pipeline(self):
        """Test complete pipeline from dataset to model"""
        # Create dataset
        dataset = SyntheticDefectDataset(
            root_dir="./data", split="train", generate_on_fly=True, num_samples=100, img_size=64
        )

        # Create model
        model = make_model()
        model.eval()

        # Test forward pass with dataset samples
        dataloader = DataLoader(dataset, batch_size=4, num_workers=0)

        for images, labels, metadata in dataloader:
            with torch.no_grad():
                outputs = model(images)

            assert outputs.shape[0] == images.shape[0]
            assert outputs.shape[1] == 5
            break  # Test one batch

    def test_training_step_simulation(self):
        """Test a simulated training step"""
        # Setup
        dataset = SyntheticDefectDataset(
            root_dir="./data", split="train", generate_on_fly=True, num_samples=50, img_size=64
        )

        model = make_model()
        model.train()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        dataloader = DataLoader(dataset, batch_size=8, num_workers=0)

        # Simulate one training step
        images, labels, _ = next(iter(dataloader))

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_validation_step_simulation(self):
        """Test a simulated validation step"""
        dataset = SyntheticDefectDataset(
            root_dir="./data", split="val", generate_on_fly=True, num_samples=20, img_size=64
        )

        model = make_model()
        model.eval()

        criterion = torch.nn.CrossEntropyLoss()
        dataloader = DataLoader(dataset, batch_size=4, num_workers=0)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels, _ in dataloader:
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        precision, recall, f1 = compute_metrics(np.array(all_labels), np.array(all_preds))

        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1


class TestInferencePipeline:
    """Integration tests for inference pipeline"""

    @pytest.fixture
    def dummy_model_path(self, tmp_path):
        """Create a dummy model checkpoint for testing"""
        model = make_model()
        model_path = tmp_path / "test_model.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": TINY_CONFIG,
                "classes": ["ok", "scratch", "crack", "dent", "corrosion"],
            },
            model_path,
        )
        return str(model_path)

    def test_inference_with_synthetic_data(self, dummy_model_path):
        """Test inference on synthetic data"""
        detector = DefectDetector(model_path=dummy_model_path, device="cpu")

        # Create synthetic image
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        result = detector.predict(image)

        assert "class" in result
        assert "confidence" in result
        assert "latency_ms" in result
        assert result["class"] in ["ok", "scratch", "crack", "dent", "corrosion"]
        assert 0 <= result["confidence"] <= 1

    def test_batch_inference(self, dummy_model_path):
        """Test batch inference"""
        detector = DefectDetector(model_path=dummy_model_path, device="cpu", batch_size=4)

        images = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(6)]

        results = detector.predict_batch(images)

        assert len(results) == 6
        for result in results:
            assert "class" in result
            assert "confidence" in result

    def test_inference_with_dataset(self, dummy_model_path):
        """Test inference on dataset samples"""
        detector = DefectDetector(model_path=dummy_model_path, device="cpu")

        dataset = SyntheticDefectDataset(
            root_dir="./data", split="test", generate_on_fly=True, num_samples=10, img_size=64
        )

        results = []
        for i in range(min(5, len(dataset))):
            img, label, metadata = dataset[i]
            # Convert tensor to numpy for inference
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            result = detector.predict(img_np)
            results.append(result)

        assert len(results) == 5


class TestConfigIntegration:
    """Integration tests for configuration handling"""

    def test_config_to_model_creation(self):
        """Test creating model from config"""
        config = {
            "img_size": 64,
            "patch_size": 8,
            "in_channels": 3,
            "num_classes": 5,
            "embed_dim": 64,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "dropout": 0.1,
            "attention_dropout": 0.1,
        }

        model = create_vit_model(config)

        assert model.num_classes == 5
        assert model.num_patches == (64 // 8) ** 2

    def test_config_to_dataset_creation(self):
        """Test creating dataset from config"""
        config = {
            "data": {
                "root_dir": "./data",
                "generate_on_fly": True,
                "train_samples": 100,
                "val_samples": 20,
                "test_samples": 20,
                "img_size": 64,
            }
        }

        dataset_handler = DefectDetectionDataset(config)
        train_dataset, val_dataset, test_dataset = dataset_handler.get_datasets()

        assert len(train_dataset) == 100
        assert len(val_dataset) == 20
        assert len(test_dataset) == 20


class TestMetricsIntegration:
    """Integration tests for metrics computation"""

    def test_metrics_with_model_outputs(self):
        """Test metrics computation with model outputs"""
        model = make_model()
        model.eval()

        dataset = SyntheticDefectDataset(
            root_dir="./data", split="test", generate_on_fly=True, num_samples=50, img_size=64
        )

        dataloader = DataLoader(dataset, batch_size=8, num_workers=0)

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels, _ in dataloader:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Compute comprehensive metrics
        metrics = compute_multiclass_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
            class_names=["ok", "scratch", "crack", "dent", "corrosion"],
        )

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "confusion_matrix" in metrics
        assert 0 <= metrics["accuracy"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
