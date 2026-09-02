#!/usr/bin/env python3
"""Export a trained checkpoint to ONNX and verify it with ONNX Runtime."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

try:
    import onnx
    import onnxruntime as ort
    import onnxscript  # noqa: F401 - required by the modern PyTorch exporter

    ONNX_AVAILABLE = True
except ImportError:
    onnx = None
    ort = None
    ONNX_AVAILABLE = False

from src.model import create_vit_model


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_model_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load a model section from YAML, if one was supplied."""
    if config_path is None:
        return {}

    with open(config_path, "r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}

    model_config = loaded.get("model", loaded)
    if not isinstance(model_config, dict) or not model_config:
        raise ValueError(f"No model configuration found in {config_path}")
    return model_config


def load_pytorch_model(
    model_path: str, supplied_config: Dict[str, Any]
) -> Tuple[nn.Module, Dict[str, Any], list[str]]:
    """Load a checkpoint and return the model, effective config, and classes."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        checkpoint_config = checkpoint.get("model_config", {})
        checkpoint_classes = checkpoint.get("classes")
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        checkpoint_config = {}
        checkpoint_classes = None
    else:
        raise ValueError("Checkpoint must contain a PyTorch state dictionary")

    conflicts = sorted(
        key
        for key in supplied_config.keys() & checkpoint_config.keys()
        if supplied_config[key] != checkpoint_config[key]
    )
    if conflicts:
        raise ValueError(
            "The supplied config conflicts with checkpoint metadata for: " + ", ".join(conflicts)
        )

    model_config = {**supplied_config, **checkpoint_config}
    if not model_config:
        raise ValueError("This weights-only checkpoint requires --config")

    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

    model = create_vit_model(model_config)
    model.load_state_dict(state_dict)
    model.eval()

    num_classes = int(model_config.get("num_classes", 5))
    if checkpoint_classes and len(checkpoint_classes) == num_classes:
        classes = list(checkpoint_classes)
    elif num_classes == 5:
        classes = ["ok", "scratch", "crack", "dent", "corrosion"]
    else:
        classes = [f"class_{index}" for index in range(num_classes)]

    return model, model_config, classes


def _sample_input(batch_size: int, in_channels: int, img_size: int) -> torch.Tensor:
    """Create deterministic input without changing the application's global RNG."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(batch_size)
    return torch.randn(
        batch_size,
        in_channels,
        img_size,
        img_size,
        generator=generator,
    )


def export_and_verify(
    model: nn.Module,
    model_config: Dict[str, Any],
    classes: list[str],
    output_path: Path,
    opset_version: int = 18,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Export one ONNX file and require PyTorch/ONNX Runtime parity."""
    img_size = int(model_config["img_size"])
    in_channels = int(model_config.get("in_channels", 3))
    if in_channels != 3:
        raise ValueError("The built-in image pipeline supports RGB models with 3 channels")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample = _sample_input(1, in_channels, img_size)
    torch.onnx.export(
        model,
        (sample,),
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset_version,
        dynamo=True,
        dynamic_shapes=({0: torch.export.Dim("batch_size")},),
        external_data=False,
        verbose=verbose,
    )

    onnx_model = onnx.load(str(output_path))
    actual_opset = next(
        item.version for item in onnx_model.opset_import if item.domain in {"", "ai.onnx"}
    )
    if actual_opset != opset_version:
        raise RuntimeError(f"Exporter produced opset {actual_opset}, requested {opset_version}")

    del onnx_model.metadata_props[:]
    for key, value in {
        "model_config": json.dumps(model_config, sort_keys=True),
        "classes": json.dumps(classes),
    }.items():
        metadata = onnx_model.metadata_props.add()
        metadata.key = key
        metadata.value = value

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(output_path))

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    max_abs_difference = 0.0

    for batch_size in (1, 2):
        input_tensor = _sample_input(batch_size, in_channels, img_size)
        with torch.no_grad():
            expected = model(input_tensor).numpy()
        actual = session.run([output_name], {input_name: input_tensor.numpy()})[0]

        if expected.shape != actual.shape or not np.allclose(
            expected, actual, rtol=1e-4, atol=1e-5
        ):
            difference = float(np.max(np.abs(expected - actual)))
            raise RuntimeError(
                f"ONNX verification failed for batch {batch_size}; "
                f"max difference {difference:.3e}"
            )
        max_abs_difference = max(max_abs_difference, float(np.max(np.abs(expected - actual))))

    return {
        "opset_version": actual_opset,
        "verified_batch_sizes": [1, 2],
        "max_abs_difference": max_abs_difference,
        "model_config": model_config,
        "classes": classes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a checkpoint to ONNX and verify it locally"
    )
    parser.add_argument("--model-path", required=True, help="PyTorch checkpoint path")
    parser.add_argument(
        "--output-path",
        default="experiments/exports/model.onnx",
        help="Destination ONNX path",
    )
    parser.add_argument(
        "--config",
        "--config-path",
        dest="config_path",
        help="YAML config; required only for checkpoints without model metadata",
    )
    parser.add_argument("--opset-version", type=int, default=18)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not ONNX_AVAILABLE:
        parser.error('ONNX support is missing; install with: pip install -e ".[export]"')
    if args.opset_version < 18:
        parser.error("opset-version must be 18 or newer")

    try:
        supplied_config = load_model_config(args.config_path)
        model, model_config, classes = load_pytorch_model(args.model_path, supplied_config)
        output_path = Path(args.output_path)
        export_info = export_and_verify(
            model,
            model_config,
            classes,
            output_path,
            opset_version=args.opset_version,
            verbose=args.verbose,
        )
        export_info.update(
            {
                "source_checkpoint": args.model_path,
                "onnx_model": str(output_path),
            }
        )
        info_path = output_path.with_suffix(".json")
        info_path.write_text(json.dumps(export_info, indent=2, sort_keys=True), encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as error:
        logger.error("ONNX export failed: %s", error)
        return 1

    logger.info("ONNX model verified: %s", output_path)
    logger.info("Export metadata: %s", info_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
