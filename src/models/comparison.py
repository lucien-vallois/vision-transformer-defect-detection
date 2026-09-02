"""
Model comparison utilities for comparing different architectures

Supports comparison between ViT, ResNet, and EfficientNet models.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from ..model import create_vit_model
from ..utils.metrics import compute_multiclass_metrics


class ModelComparator:
    """Compare different model architectures on the same dataset"""

    def __init__(self, device: str = "auto"):
        """
        Initialize model comparator.

        Args:
            device: Device to run inference on ('cpu', 'cuda', 'auto')
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.models = {}
        self.results = {}

    @staticmethod
    def _checkpoint_state(model_path: str) -> Tuple[Dict, Dict]:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            metadata = checkpoint
        else:
            state_dict = checkpoint
            metadata = {}
        if state_dict and all(key.startswith("module.") for key in state_dict):
            state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        return state_dict, metadata

    def load_vit_model(self, model_path: str, config: Optional[Dict] = None):
        """Load Vision Transformer model"""
        state_dict, metadata = self._checkpoint_state(model_path)
        if config is None:
            config = metadata.get("model_config")
        if not config:
            raise ValueError("A model config is required for a weights-only ViT checkpoint")

        model = create_vit_model(config)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self.models["ViT"] = model
        return model

    def load_resnet50(self, model_path: str, num_classes: int = 5):
        """Load ResNet-50 model"""
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for ResNet models. Install with: pip install timm")

        model = timm.create_model("resnet50", pretrained=False, num_classes=num_classes)
        state_dict, _ = self._checkpoint_state(model_path)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self.models["ResNet-50"] = model
        return model

    def load_efficientnet_b0(self, model_path: str, num_classes: int = 5):
        """Load EfficientNet-B0 model"""
        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for EfficientNet models. Install with: pip install timm"
            )

        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
        state_dict, _ = self._checkpoint_state(model_path)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self.models["EfficientNet-B0"] = model
        return model

    def evaluate_model(
        self, model: nn.Module, model_name: str, dataloader: DataLoader, class_names: List[str]
    ) -> Dict:
        """
        Evaluate a single model on a dataset.

        Args:
            model: PyTorch model
            model_name: Name of the model
            dataloader: DataLoader for evaluation
            class_names: List of class names

        Returns:
            Dictionary with evaluation metrics
        """
        all_preds = []
        all_probs = []
        all_labels = []
        latencies = []

        model.eval()
        with torch.no_grad():
            for images, labels, _ in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Measure inference time
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    start_event.record()
                    outputs = model(images)
                    end_event.record()

                    torch.cuda.synchronize()
                    latency = start_event.elapsed_time(end_event) / len(images)  # ms per image
                else:
                    start = time.perf_counter()
                    outputs = model(images)
                    latency = (time.perf_counter() - start) * 1000 / len(images)

                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                latencies.extend([latency] * len(images))

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        latencies = np.array(latencies)

        if not len(all_labels):
            raise ValueError("Cannot compare models with an empty dataloader")

        detailed_metrics = compute_multiclass_metrics(all_labels, all_preds, all_probs, class_names)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "model_name": model_name,
            "accuracy": detailed_metrics["accuracy"],
            "precision": detailed_metrics["precision"]["weighted"],
            "recall": detailed_metrics["recall"]["weighted"],
            "f1": detailed_metrics["f1"]["weighted"],
            "detailed_metrics": detailed_metrics,
            "mean_latency_ms": float(np.mean(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "predictions": all_preds,
            "probabilities": all_probs,
            "true_labels": all_labels,
        }

    def compare_models(
        self,
        dataloader: DataLoader,
        class_names: List[str],
        model_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Compare all loaded models on the same dataset.

        Args:
            dataloader: DataLoader for evaluation
            class_names: List of class names
            model_names: Optional list of model names to compare (default: all loaded models)

        Returns:
            Dictionary with comparison results
        """
        if model_names is None:
            model_names = list(self.models.keys())

        results = {}

        for model_name in model_names:
            if model_name not in self.models:
                print(f"Warning: Model {model_name} not loaded. Skipping.")
                continue

            print(f"Evaluating {model_name}...")
            model = self.models[model_name]
            result = self.evaluate_model(model, model_name, dataloader, class_names)
            results[model_name] = result

        self.results = results
        return results

    def generate_comparison_table(self) -> str:
        """Generate a formatted comparison table"""
        if not self.results:
            return "No results available. Run compare_models() first."

        # Create table
        lines = []
        lines.append("=" * 100)
        lines.append(
            f"{'Model':<20} {'Accuracy':<12} {'F1-Score':<12} {'Params':<15} {'Latency (ms)':<15}"
        )
        lines.append("=" * 100)

        for model_name, result in self.results.items():
            lines.append(
                f"{model_name:<20} "
                f"{result['accuracy']:.4f}      "
                f"{result['f1']:.4f}      "
                f"{result['total_parameters']/1e6:.1f}M          "
                f"{result['mean_latency_ms']:.2f}"
            )

        lines.append("=" * 100)

        return "\n".join(lines)

    def get_best_model(self, metric: str = "accuracy") -> Tuple[str, Dict]:
        """
        Get the best performing model based on a metric.

        Args:
            metric: Metric to use for comparison ('accuracy', 'f1', 'mean_latency_ms')

        Returns:
            Tuple of (model_name, result_dict)
        """
        if not self.results:
            raise ValueError("No results available. Run compare_models() first.")

        # For latency, lower is better
        reverse = metric == "mean_latency_ms"

        best_model = max(
            self.results.items(), key=lambda x: x[1][metric] if not reverse else -x[1][metric]
        )

        return best_model
