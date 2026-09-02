"""
Configuration validator for Vision Transformer training
Validates YAML configuration files before training starts
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors"""

    pass


def validate_config(config: Dict[str, Any], strict: bool = True) -> Tuple[bool, List[str]]:
    """
    Validate configuration dictionary

    Args:
        config: Configuration dictionary
        strict: If True, raise exception on validation failure

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    if not isinstance(config, dict):
        errors = ["Configuration root must be a mapping"]
        if strict:
            raise ConfigValidationError(errors[0])
        return False, errors

    errors = []

    # Check required top-level keys
    required_keys = ["model", "data", "training"]
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required key: '{key}'")
        elif not isinstance(config[key], dict):
            errors.append(f"'{key}' must be a mapping")

    if errors:
        if strict:
            raise ConfigValidationError(f"Configuration validation failed: {', '.join(errors)}")
        return False, errors

    # Validate model section
    model_errors = _validate_model_config(config.get("model", {}))
    errors.extend(model_errors)

    # Validate data section
    data_errors = _validate_data_config(config.get("data", {}))
    errors.extend(data_errors)

    # Validate training section
    training_errors = _validate_training_config(config.get("training", {}))
    errors.extend(training_errors)

    hardware = config.get("hardware", {})
    if not isinstance(hardware, dict):
        errors.append("'hardware' must be a mapping")
        hardware = {}
    if hardware.get("distributed"):
        errors.append("hardware.distributed is not supported; run one training process per device")
    device = hardware.get("device", "auto")
    if not isinstance(device, str) or not re.fullmatch(r"auto|cpu|cuda(?::\d+)?", device):
        errors.append(f"hardware.device must be auto, cpu, cuda, or cuda:N, got {device!r}")

    if errors:
        if strict:
            raise ConfigValidationError(f"Configuration validation failed:\n" + "\n".join(errors))
        return False, errors

    return True, []


def _validate_model_config(model_config: Dict[str, Any]) -> List[str]:
    """Validate model configuration section"""
    errors = []

    required_keys = [
        "img_size",
        "patch_size",
        "in_channels",
        "num_classes",
        "embed_dim",
        "depth",
        "num_heads",
    ]
    for key in required_keys:
        if key not in model_config:
            errors.append(f"model.{key} is required")

    # Validate types and values
    if "img_size" in model_config:
        img_size = model_config["img_size"]
        if not isinstance(img_size, int) or img_size <= 0:
            errors.append(f"model.img_size must be a positive integer, got {img_size}")
        elif isinstance(model_config.get("patch_size"), int) and model_config["patch_size"] > 0:
            if img_size % model_config["patch_size"] != 0:
                errors.append(
                    f"model.img_size ({img_size}) must be divisible by "
                    f"model.patch_size ({model_config['patch_size']})"
                )

    if "patch_size" in model_config:
        patch_size = model_config["patch_size"]
        if not isinstance(patch_size, int) or patch_size <= 0:
            errors.append(f"model.patch_size must be a positive integer, got {patch_size}")

    if not isinstance(model_config.get("in_channels"), int) or model_config.get("in_channels") != 3:
        errors.append("model.in_channels must be 3 for the built-in RGB image pipeline")

    if "num_classes" in model_config:
        num_classes = model_config["num_classes"]
        if not isinstance(num_classes, int) or num_classes != 5:
            errors.append(
                "model.num_classes must be 5 for the built-in defect dataset and class mapping"
            )

    if "embed_dim" in model_config:
        embed_dim = model_config["embed_dim"]
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            errors.append(f"model.embed_dim must be a positive integer, got {embed_dim}")

    if "depth" in model_config:
        depth = model_config["depth"]
        if not isinstance(depth, int) or depth <= 0:
            errors.append(f"model.depth must be a positive integer, got {depth}")

    if "num_heads" in model_config:
        num_heads = model_config["num_heads"]
        if not isinstance(num_heads, int) or num_heads <= 0:
            errors.append(f"model.num_heads must be a positive integer, got {num_heads}")
        if "embed_dim" in model_config and isinstance(num_heads, int) and num_heads > 0:
            embed_dim = model_config["embed_dim"]
            if isinstance(embed_dim, int) and embed_dim % num_heads != 0:
                errors.append(
                    f"model.embed_dim ({embed_dim}) must be divisible by model.num_heads ({num_heads})"
                )

    if "dropout" in model_config:
        dropout = model_config["dropout"]
        if not isinstance(dropout, (int, float)) or not (0 <= dropout < 1):
            errors.append(f"model.dropout must be a float between 0 and 1, got {dropout}")

    if "attention_dropout" in model_config:
        attention_dropout = model_config["attention_dropout"]
        if not isinstance(attention_dropout, (int, float)) or not (0 <= attention_dropout < 1):
            errors.append(
                "model.attention_dropout must be a float between 0 and 1, "
                f"got {attention_dropout}"
            )

    return errors


def _validate_data_config(data_config: Dict[str, Any]) -> List[str]:
    """Validate data configuration section"""
    errors = []

    required_keys = ["root_dir", "generate_on_fly", "train_samples", "val_samples", "test_samples"]
    for key in required_keys:
        if key not in data_config:
            errors.append(f"data.{key} is required")

    if "root_dir" in data_config:
        root_dir = data_config["root_dir"]
        if not isinstance(root_dir, str) or not root_dir.strip():
            errors.append("data.root_dir must be a non-empty string")

    if "generate_on_fly" in data_config and not isinstance(data_config["generate_on_fly"], bool):
        errors.append("data.generate_on_fly must be true or false")

    if "seed" in data_config and (
        not isinstance(data_config["seed"], int) or isinstance(data_config["seed"], bool)
    ):
        errors.append(f"data.seed must be an integer, got {data_config['seed']!r}")

    if "train_samples" in data_config:
        train_samples = data_config["train_samples"]
        if not isinstance(train_samples, int) or train_samples <= 0:
            errors.append(f"data.train_samples must be a positive integer, got {train_samples}")

    if "val_samples" in data_config:
        val_samples = data_config["val_samples"]
        if not isinstance(val_samples, int) or val_samples <= 0:
            errors.append(f"data.val_samples must be a positive integer, got {val_samples}")

    if "test_samples" in data_config:
        test_samples = data_config["test_samples"]
        if not isinstance(test_samples, int) or test_samples <= 0:
            errors.append(f"data.test_samples must be a positive integer, got {test_samples}")

    if "batch_size" in data_config:
        batch_size = data_config["batch_size"]
        if not isinstance(batch_size, int) or batch_size <= 0:
            errors.append(f"data.batch_size must be a positive integer, got {batch_size}")

    if "num_workers" in data_config:
        num_workers = data_config["num_workers"]
        if not isinstance(num_workers, int) or num_workers < 0:
            errors.append(f"data.num_workers must be a non-negative integer, got {num_workers}")

    return errors


def _validate_training_config(training_config: Dict[str, Any]) -> List[str]:
    """Validate training configuration section"""
    errors = []

    required_keys = [
        "epochs",
        "output_dir",
        "lr",
        "weight_decay",
        "min_lr",
        "early_stopping_patience",
        "early_stopping_delta",
    ]
    for key in required_keys:
        if key not in training_config:
            errors.append(f"training.{key} is required")

    if "epochs" in training_config:
        epochs = training_config["epochs"]
        if not isinstance(epochs, int) or epochs <= 0:
            errors.append(f"training.epochs must be a positive integer, got {epochs}")

    if "lr" in training_config:
        lr = training_config["lr"]
        if not isinstance(lr, (int, float)) or lr <= 0:
            errors.append(f"training.lr must be a positive number, got {lr}")

    if "weight_decay" in training_config:
        weight_decay = training_config["weight_decay"]
        if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
            errors.append(
                f"training.weight_decay must be a non-negative number, got {weight_decay}"
            )

    if "output_dir" in training_config:
        output_dir = training_config["output_dir"]
        if not isinstance(output_dir, str) or not output_dir.strip():
            errors.append("training.output_dir must be a non-empty string")

    if "min_lr" in training_config:
        min_lr = training_config["min_lr"]
        if not isinstance(min_lr, (int, float)) or min_lr < 0:
            errors.append(f"training.min_lr must be a non-negative number, got {min_lr}")
        elif isinstance(training_config.get("lr"), (int, float)) and min_lr > training_config["lr"]:
            errors.append("training.min_lr cannot exceed training.lr")

    if "early_stopping_patience" in training_config:
        patience = training_config["early_stopping_patience"]
        if not isinstance(patience, int) or patience < 0:
            errors.append(
                f"training.early_stopping_patience must be a non-negative integer, got {patience}"
            )

    if "early_stopping_delta" in training_config:
        delta = training_config["early_stopping_delta"]
        if not isinstance(delta, (int, float)) or delta < 0:
            errors.append(
                f"training.early_stopping_delta must be a non-negative number, got {delta}"
            )

    if "mixed_precision" in training_config and not isinstance(
        training_config["mixed_precision"], bool
    ):
        errors.append("training.mixed_precision must be true or false")

    return errors


def validate_config_file(config_path: str, strict: bool = True) -> Tuple[bool, List[str]]:
    """
    Validate configuration from YAML file

    Args:
        config_path: Path to YAML configuration file
        strict: If True, raise exception on validation failure

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    config_file = Path(config_path)

    if not config_file.exists():
        error_msg = f"Configuration file not found: {config_path}"
        if strict:
            raise ConfigValidationError(error_msg)
        return False, [error_msg]

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        if config is None:
            error_msg = f"Configuration file is empty: {config_path}"
            if strict:
                raise ConfigValidationError(error_msg)
            return False, [error_msg]

        return validate_config(config, strict=strict)

    except yaml.YAMLError as e:
        error_msg = f"Error parsing YAML file {config_path}: {str(e)}"
        if strict:
            raise ConfigValidationError(error_msg)
        return False, [error_msg]
    except Exception as e:
        error_msg = f"Error reading configuration file {config_path}: {str(e)}"
        if strict:
            raise ConfigValidationError(error_msg)
        return False, [error_msg]


if __name__ == "__main__":
    # Test validation
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        is_valid, errors = validate_config_file(config_path, strict=False)

        if is_valid:
            print(f"OK: Configuration file '{config_path}' is valid")
            sys.exit(0)
        else:
            print(f"ERROR: Configuration file '{config_path}' has errors:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
    else:
        print("Usage: python config_validator.py <config_file.yaml>")
