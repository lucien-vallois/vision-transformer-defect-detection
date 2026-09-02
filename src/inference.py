"""Checkpoint-backed PyTorch and ONNX inference."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from .data.dataset import SyntheticDefectDataset
from .model import create_vit_model
from .utils.metrics import compute_metrics
from .utils.validators import ImageValidationError


class DefectDetector:
    """
    Defect detection inference class with support for multiple model formats.

    This class provides a unified interface for running inference on defect detection
    models, supporting PyTorch and ONNX formats. It handles image preprocessing,
    model loading, and result post-processing automatically.

    Attributes:
        model_path: Path to model weights/checkpoint
        img_size: Input image size
        batch_size: Batch size for inference
        device: Device used for inference
        model_format: Format of the loaded model
        classes: List of class names
        model: PyTorch model (if using PyTorch format)
        session: ONNX Runtime session (if using ONNX format)
        transform: Image preprocessing transform
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        model_format: str = "auto",
        img_size: Optional[int] = None,
        batch_size: int = 1,
        model_config: Optional[Dict] = None,
        warmup: bool = False,
    ):
        """
        Initialize defect detector.

        Args:
            model_path: Path to model weights/checkpoint file
            device: Device to run inference on ('cpu', 'cuda', 'auto').
                   'auto' will use CUDA if available, otherwise CPU
            model_format: Model format ('pytorch', 'onnx', 'auto').
                         'auto' will detect format from file extension
            img_size: Input image size. Defaults to checkpoint metadata or 224
            batch_size: Batch size for batch inference (default: 1)

        Raises:
            FileNotFoundError: If model_path does not exist
            ValueError: If model_format is unsupported
            ImportError: If the optional ONNX Runtime dependency is missing
        """
        # Validate inputs
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if img_size is not None and img_size <= 0:
            raise ValueError(f"img_size must be positive, got {img_size}")

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.model_path = Path(model_path)
        self.requested_device = device
        configured_img_size = (model_config or {}).get("img_size")
        self.img_size = int(img_size or configured_img_size or 224)
        self.batch_size = batch_size
        self.classes = ["ok", "scratch", "crack", "dent", "corrosion"]
        self.model_config = model_config

        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Determine model format
        if model_format == "auto":
            if self.model_path.suffix == ".onnx":
                self.model_format = "onnx"
            elif self.model_path.suffix in {".pth", ".pt"}:
                self.model_format = "pytorch"
            else:
                raise ValueError(
                    f"Cannot detect model format from extension: {self.model_path.suffix}"
                )
        else:
            self.model_format = model_format

        # Setup model
        self._load_model()

        # Setup transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        if warmup:
            self._warmup()

    def _load_model(self):
        """Load the appropriate model format"""
        if self.model_format == "pytorch":
            self._load_pytorch_model()
        elif self.model_format == "onnx":
            self._load_onnx_model()
        else:
            raise ValueError(f"Unsupported model format: {self.model_format}")

    def _load_pytorch_model(self):
        """Load PyTorch model"""
        default_config = {
            "img_size": self.img_size,
            "patch_size": 16,
            "in_channels": 3,
            "num_classes": len(self.classes),
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 8,
            "mlp_ratio": 4.0,
        }

        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            checkpoint_config = checkpoint.get("model_config", {})
            checkpoint_classes = checkpoint.get("classes")
        else:
            state_dict = checkpoint
            checkpoint_config = {}
            checkpoint_classes = None

        config = {**default_config, **checkpoint_config, **(self.model_config or {})}
        self.img_size = int(config["img_size"])
        if checkpoint_classes:
            self.classes = list(checkpoint_classes)
        if config["num_classes"] != len(self.classes):
            self.classes = [f"class_{index}" for index in range(config["num_classes"])]

        self.model = create_vit_model(config)
        if state_dict and all(key.startswith("module.") for key in state_dict):
            state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    def _load_onnx_model(self):
        """Load ONNX model"""
        if not ONNX_AVAILABLE:
            raise ImportError(
                "ONNX Runtime not available. Install the package with the export extra"
            )

        available_providers = set(ort.get_available_providers())
        if self.device.type == "cuda" and "CUDAExecutionProvider" not in available_providers:
            if self.requested_device == "auto":
                self.device = torch.device("cpu")
            else:
                raise RuntimeError(
                    "CUDAExecutionProvider is unavailable; use --device cpu or install "
                    "a compatible ONNX Runtime GPU package"
                )

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device.type == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)

        # Get input/output names
        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()[0]
        self.input_name = input_info.name
        self.output_name = output_info.name

        metadata = self.session.get_modelmeta().custom_metadata_map
        if "classes" in metadata:
            stored_classes = json.loads(metadata["classes"])
            if isinstance(stored_classes, list) and stored_classes:
                self.classes = [str(class_name) for class_name in stored_classes]

        if len(input_info.shape) == 4 and isinstance(input_info.shape[1], int):
            if input_info.shape[1] != 3:
                raise ValueError(
                    f"ONNX model expects {input_info.shape[1]} channels; RGB input requires 3"
                )
        if len(input_info.shape) == 4 and isinstance(input_info.shape[-1], int):
            self.img_size = int(input_info.shape[-1])
        if len(output_info.shape) == 2 and isinstance(output_info.shape[-1], int):
            num_classes = int(output_info.shape[-1])
            if num_classes != len(self.classes):
                self.classes = [f"class_{index}" for index in range(num_classes)]

    def _warmup(self):
        """Warm up the model with dummy data"""
        dummy_input = torch.randn(1, 3, self.img_size, self.img_size).to(self.device)

        if self.model_format == "pytorch":
            with torch.no_grad():
                _ = self.model(dummy_input)
        elif self.model_format == "onnx":
            dummy_input_np = dummy_input.cpu().numpy()
            _ = self.session.run([self.output_name], {self.input_name: dummy_input_np})

    def _preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess input image for model inference.

        Args:
            image: Input image (path, numpy array, or PIL Image)

        Returns:
            Preprocessed image tensor of shape (1, C, H, W)

        Raises:
            ImageValidationError: If image cannot be loaded or is invalid
        """
        if isinstance(image, (str, os.PathLike)):
            image_path = Path(image)
            try:
                with Image.open(image_path) as opened_image:
                    image = opened_image.convert("RGB")
            except (OSError, ValueError) as exc:
                raise ImageValidationError(
                    f"Failed to load image from {image_path}: {exc}"
                ) from exc
        elif isinstance(image, np.ndarray):
            if image.size == 0:
                raise ImageValidationError("Image array is empty")
            if image.ndim not in {2, 3}:
                raise ImageValidationError(f"Expected a 2D or 3D image array, got {image.shape}")
            if image.ndim == 3 and image.shape[2] not in {1, 3, 4}:
                raise ImageValidationError(f"Unsupported channel count: {image.shape[2]}")
            if not (
                np.issubdtype(image.dtype, np.integer) or np.issubdtype(image.dtype, np.floating)
            ):
                raise ImageValidationError(f"Expected a numeric image array, got {image.dtype}")
            if not np.isfinite(image).all():
                raise ImageValidationError("Image array contains NaN or infinite values")

            image_min = float(image.min())
            image_max = float(image.max())
            if image_min < 0 or image_max > 255:
                raise ImageValidationError("Image array values must be in [0, 255]")
            if np.issubdtype(image.dtype, np.floating) and image_max <= 1:
                image = image * 255
            image = np.rint(image).astype(np.uint8)
            if image.ndim == 3 and image.shape[2] == 1:
                image = image[:, :, 0]
            image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise ImageValidationError(
                f"Unsupported image type: {type(image)}. " f"Expected str, np.ndarray, or PIL.Image"
            )

        return self.transform(image).unsqueeze(0)

    def _inference_pytorch(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, float]:
        """Run inference with PyTorch model"""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start_time = time.perf_counter()

        with torch.no_grad():
            output = self.model(input_tensor.to(self.device))
            probabilities = F.softmax(output, dim=1)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_time = (time.perf_counter() - start_time) * 1000
        return probabilities.cpu().numpy(), inference_time

    def _inference_onnx(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, float]:
        """Run inference with ONNX model"""
        input_np = input_tensor.cpu().numpy()

        start_time = time.perf_counter()
        output = self.session.run([self.output_name], {self.input_name: input_np})[0]
        inference_time = (time.perf_counter() - start_time) * 1000

        # Apply softmax
        probabilities = F.softmax(torch.from_numpy(output), dim=1).numpy()
        return probabilities, inference_time

    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> Dict:
        """
        Run inference on a single image.

        Args:
            image: Input image. Can be:
                  - str/Path: Path to image file
                  - np.ndarray: Image as numpy array (H, W, C) with values in [0, 255]
                  - PIL.Image: PIL Image object

        Returns:
            Dictionary containing:
            - 'class': Predicted class name (str)
            - 'class_idx': Predicted class index (int)
            - 'confidence': Confidence score [0, 1] (float)
            - 'probabilities': Mapping from class names to probabilities (Dict[str, float])
            - 'latency_ms': Inference latency in milliseconds (float)
            - 'all_classes': List of all class names (List[str])

        Raises:
            ImageValidationError: If image is invalid or cannot be loaded
            RuntimeError: If inference fails
        """
        # Preprocess
        input_tensor = self._preprocess_image(image)

        # Run inference
        if self.model_format == "pytorch":
            probabilities, latency = self._inference_pytorch(input_tensor)
        elif self.model_format == "onnx":
            probabilities, latency = self._inference_onnx(input_tensor)
        else:
            raise ValueError(f"Inference not implemented for format: {self.model_format}")

        # Get prediction
        pred_class_idx = np.argmax(probabilities[0])
        confidence = float(probabilities[0][pred_class_idx])

        return {
            "class": self.classes[pred_class_idx],
            "class_idx": int(pred_class_idx),
            "confidence": confidence,
            "probabilities": {
                class_name: float(probability)
                for class_name, probability in zip(self.classes, probabilities[0])
            },
            "latency_ms": latency,
            "all_classes": self.classes,
        }

    def predict_batch(self, images: List[Union[str, np.ndarray, Image.Image]]) -> List[Dict]:
        """
        Run batch inference on multiple images.

        This method processes images in batches for efficiency. If the number of
        images exceeds the batch_size, they will be processed in multiple batches.

        Args:
            images: List of input images. Each image can be:
                   - str/Path: Path to image file
                   - np.ndarray: Image as numpy array (H, W, C)
                   - PIL.Image: PIL Image object

        Returns:
            List of prediction dictionaries, one for each input image.
            Each dictionary has the same structure as returned by predict().

        Raises:
            ValueError: If images list is empty
            ImageValidationError: If any image is invalid
        """
        if not images:
            raise ValueError("Images list cannot be empty")

        if len(images) > self.batch_size:
            # Process in batches
            results = []
            for i in range(0, len(images), self.batch_size):
                batch_images = images[i : i + self.batch_size]
                batch_results = self._predict_batch_internal(batch_images)
                results.extend(batch_results)
            return results
        else:
            return self._predict_batch_internal(images)

    def _predict_batch_internal(
        self, images: List[Union[str, np.ndarray, Image.Image]]
    ) -> List[Dict]:
        """Internal batch prediction"""
        # Preprocess batch
        batch_tensors = []
        for image in images:
            tensor = self._preprocess_image(image)
            batch_tensors.append(tensor)

        batch_input = torch.cat(batch_tensors, dim=0)

        # Run inference
        if self.model_format == "pytorch" and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start_time = time.perf_counter()

        if self.model_format == "pytorch":
            with torch.no_grad():
                outputs = self.model(batch_input.to(self.device))
                probabilities = F.softmax(outputs, dim=1).cpu().numpy()
        elif self.model_format == "onnx":
            input_np = batch_input.cpu().numpy()
            outputs = self.session.run([self.output_name], {self.input_name: input_np})[0]
            probabilities = F.softmax(torch.from_numpy(outputs), dim=1).numpy()
        else:
            raise ValueError(f"Inference not implemented for format: {self.model_format}")

        if self.model_format == "pytorch" and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        batch_latency = (time.perf_counter() - start_time) * 1000
        avg_latency = batch_latency / len(images)

        # Process results
        results = []
        for i, probs in enumerate(probabilities):
            pred_class_idx = np.argmax(probs)
            confidence = float(probs[pred_class_idx])

            results.append(
                {
                    "class": self.classes[pred_class_idx],
                    "class_idx": int(pred_class_idx),
                    "confidence": confidence,
                    "probabilities": {
                        class_name: float(probability)
                        for class_name, probability in zip(self.classes, probs)
                    },
                    "latency_ms": avg_latency,
                    "all_classes": self.classes,
                }
            )

        return results

    def evaluate_dataset(self, dataset_path: str, batch_size: int = 32) -> Dict:
        """
        Evaluate model on a dataset

        Args:
            dataset_path: Path to dataset directory
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        # Create dataset
        dataset = SyntheticDefectDataset(
            root_dir=dataset_path,
            split="test",
            img_size=self.img_size,
            augment=False,
            generate_on_fly=False,
        )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        all_preds = []
        all_targets = []
        latencies = []

        self.model.eval() if self.model_format == "pytorch" else None

        with torch.no_grad():
            for inputs, targets, _ in dataloader:
                if self.model_format == "pytorch" and self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                batch_start = time.perf_counter()

                if self.model_format == "pytorch":
                    outputs = self.model(inputs.to(self.device))
                elif self.model_format == "onnx":
                    input_np = inputs.cpu().numpy()
                    outputs = self.session.run([self.output_name], {self.input_name: input_np})[0]
                    outputs = torch.from_numpy(outputs)

                if self.model_format == "pytorch" and self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                batch_time = (time.perf_counter() - batch_start) * 1000
                latencies.extend([batch_time / inputs.size(0)] * inputs.size(0))

                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Compute metrics
        accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
        precision, recall, f1 = compute_metrics(all_targets, all_preds)

        # Latency statistics
        latencies = np.array(latencies)
        latency_stats = {
            "mean": float(np.mean(latencies)),
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
        }

        return {
            "accuracy": float(accuracy),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "latency_ms": latency_stats,
            "num_samples": len(all_preds),
        }


def benchmark_inference(detector: DefectDetector, num_runs: int = 100) -> Dict:
    """Benchmark inference performance"""
    if num_runs <= 0:
        raise ValueError("num_runs must be a positive integer")
    # Create dummy input
    dummy_image = np.random.randint(
        0, 255, (detector.img_size, detector.img_size, 3), dtype=np.uint8
    )

    latencies = []

    # Warm up
    for _ in range(10):
        _ = detector.predict(dummy_image)

    # Benchmark
    for _ in range(num_runs):
        result = detector.predict(dummy_image)
        latencies.append(result["latency_ms"])

    latencies = np.array(latencies)

    mean_latency = float(np.mean(latencies))
    return {
        "mean_latency_ms": mean_latency,
        "std_latency_ms": float(np.std(latencies)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "throughput_fps": 1000 / mean_latency if mean_latency > 0 else 0.0,
        "num_runs": num_runs,
    }


def main():
    parser = argparse.ArgumentParser(description="Defect Detection Inference")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to model weights/checkpoint"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--image-path", type=str, help="Path to input image for single prediction")
    action.add_argument(
        "--batch-dir", type=str, help="Directory containing images for batch prediction"
    )
    action.add_argument("--evaluate-dataset", type=str, help="Path to dataset for evaluation")
    action.add_argument("--benchmark", action="store_true", help="Run inference benchmark")
    parser.add_argument(
        "--output", type=str, default="results.json", help="Output file for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for inference",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument(
        "--model-format",
        type=str,
        default="auto",
        choices=["auto", "pytorch", "onnx"],
        help="Model format",
    )
    parser.add_argument(
        "--config", type=str, help="Optional YAML config for weights-only checkpoints"
    )

    args = parser.parse_args()

    model_config = None
    if args.config:
        import yaml

        with open(args.config, "r", encoding="utf-8") as config_file:
            loaded_config = yaml.safe_load(config_file) or {}
        model_config = loaded_config.get("model", loaded_config)

    # Create detector with error handling
    try:
        detector = DefectDetector(
            model_path=args.model_path,
            device=args.device,
            model_format=args.model_format,
            batch_size=args.batch_size,
            model_config=model_config,
        )
    except FileNotFoundError as e:
        print(f"Error: Model file not found: {args.model_path}")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to initialize detector: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    results = {}

    if args.image_path:
        # Single image prediction
        try:
            print(f"Running inference on: {args.image_path}")
            result = detector.predict(args.image_path)
            print(f"Prediction: {result['class']} (confidence: {result['confidence']:.2%})")
            print(f"Latency: {result['latency_ms']:.2f}ms")
            results["single_prediction"] = result
        except FileNotFoundError:
            print(f"Error: Image file not found: {args.image_path}")
            sys.exit(1)
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    elif args.batch_dir:
        # Batch prediction
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_paths = sorted(
            path
            for path in Path(args.batch_dir).rglob("*")
            if path.is_file() and path.suffix.lower() in image_extensions
        )

        if not image_paths:
            parser.error(f"No images found in {args.batch_dir}")

        print(f"Running batch inference on {len(image_paths)} images")
        batch_results = detector.predict_batch(image_paths)
        for image_path, result in zip(image_paths, batch_results):
            result["image_path"] = str(image_path)

        # Summary statistics
        classes = [r["class"] for r in batch_results]
        confidences = [r["confidence"] for r in batch_results]
        latencies = [r["latency_ms"] for r in batch_results]

        from collections import Counter

        class_counts = Counter(classes)

        results["batch_prediction"] = {
            "num_images": len(batch_results),
            "results": batch_results,
            "summary": {
                "class_distribution": dict(class_counts),
                "mean_confidence": float(np.mean(confidences)),
                "mean_latency_ms": float(np.mean(latencies)),
                "total_time_ms": sum(latencies),
            },
        }

    elif args.evaluate_dataset:
        # Dataset evaluation
        print(f"Evaluating on dataset: {args.evaluate_dataset}")
        eval_results = detector.evaluate_dataset(args.evaluate_dataset, args.batch_size)
        print(f"Accuracy: {eval_results['accuracy']:.2%}")
        print(f"F1 Score: {eval_results['f1']:.4f}")
        print(f"P95 Latency: {eval_results['latency_ms']['p95']:.2f}ms")
        results["evaluation"] = eval_results

    elif args.benchmark:
        # Benchmark
        print("Running inference benchmark...")
        bench_results = benchmark_inference(detector, num_runs=100)
        print(f"Mean latency: {bench_results['mean_latency_ms']:.2f}ms")
        print(f"P95 latency: {bench_results['p95_latency_ms']:.2f}ms")
        print(f"Throughput: {bench_results['throughput_fps']:.1f} FPS")
        results["benchmark"] = bench_results

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
