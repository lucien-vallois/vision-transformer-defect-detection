"""
Training script for Vision Transformer defect detection
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Try to import optional dependencies
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

import numpy as np

from .data.dataset import DefectDetectionDataset
from .model import create_vit_model
from .utils.config_validator import validate_config
from .utils.metrics import EarlyStopping, compute_metrics


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    log_file = Path(log_dir) / "training.log"
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    loss: float,
    accuracy: float,
    checkpoint_dir: str,
    model_config: Dict,
    best_accuracy: float,
    is_best: bool = False,
):
    """Save model checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "classes": ["ok", "scratch", "crack", "dent", "corrosion"],
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict(),
        "loss": loss,
        "accuracy": accuracy,
        "best_accuracy": best_accuracy,
    }

    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_path = Path(checkpoint_dir) / "best_model.pth"
        torch.save(checkpoint, best_path)
        # Also save just the model weights for inference
        torch.save(model.state_dict(), Path(checkpoint_dir) / "best_model_weights.pth")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    state_dict = checkpoint["model_state_dict"]
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    best_accuracy = checkpoint.get("best_accuracy", checkpoint.get("accuracy", 0))
    return checkpoint.get("epoch", 0), checkpoint.get("loss", 0), best_accuracy


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: Any,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    writer=None,
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_idx, (inputs, targets, _) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Log batch progress
        if batch_idx % 10 == 0:
            batch_loss = running_loss / (batch_idx + 1)
            batch_acc = 100.0 * correct / total
            logger.info(
                f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                f"Loss: {batch_loss:.4f} | Acc: {batch_acc:.2f}%"
            )

    epoch_time = time.time() - start_time
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total

    metrics = {"loss": epoch_loss, "accuracy": epoch_acc, "epoch_time": epoch_time}

    # Log to TensorBoard
    if writer:
        writer.add_scalar("Train/Loss", epoch_loss, epoch)
        writer.add_scalar("Train/Accuracy", epoch_acc, epoch)
        writer.add_scalar("Train/Time", epoch_time, epoch)

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    writer=None,
) -> Dict[str, float]:
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    start_time = time.time()

    for inputs, targets, _ in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    val_time = time.time() - start_time
    val_loss = running_loss / len(val_loader)
    val_acc = 100.0 * correct / total

    # Compute additional metrics
    precision, recall, f1 = compute_metrics(all_targets, all_preds)

    metrics = {
        "loss": val_loss,
        "accuracy": val_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "val_time": val_time,
    }

    logger.info(
        f"Validation | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | "
        f"F1: {f1:.4f} | Time: {val_time:.2f}s"
    )

    # Log to TensorBoard
    if writer:
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Accuracy", val_acc, epoch)
        writer.add_scalar("Val/F1", f1, epoch)
        writer.add_scalar("Val/Precision", precision, epoch)
        writer.add_scalar("Val/Recall", recall, epoch)

    return metrics


def train(config: Dict):
    """Train one model on one configured device."""
    validate_config(config, strict=True)
    seed = config.get("data", {}).get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    requested_device = config.get("hardware", {}).get("device", "auto")
    if requested_device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    exp_dir = Path(config["training"]["output_dir"])
    checkpoint_dir = exp_dir / "checkpoints"
    log_dir = exp_dir / "logs"
    resume_value = config["training"].get("resume_from")
    checkpoint_path = Path(resume_value) if resume_value else None

    if checkpoint_path is None:
        managed_outputs = [exp_dir / "config.yaml", exp_dir / "training_history.json"]
        existing_checkpoints = checkpoint_dir.exists() and any(checkpoint_dir.glob("*.pth"))
        if existing_checkpoints or any(path.exists() for path in managed_outputs):
            raise FileExistsError(
                f"Training output already contains an experiment: {exp_dir}. "
                "Choose a new output directory or configure resume_from."
            )
    else:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        if checkpoint_path.resolve().parent.parent != exp_dir.resolve():
            raise ValueError(
                "Resume checkpoint must belong to training.output_dir; "
                f"expected a checkpoint under {exp_dir}"
            )

    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    logger = setup_logging(str(log_dir))

    model = create_vit_model(config["model"]).to(device)
    train_loader, val_loader, _ = DefectDetectionDataset(config).get_dataloaders()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    use_amp = bool(config["training"].get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    early_stopping = EarlyStopping(
        patience=config["training"]["early_stopping_patience"],
        min_delta=config["training"]["early_stopping_delta"],
    )

    start_epoch = 0
    best_acc = float("-inf")
    if checkpoint_path is not None:
        loaded_epoch, _, best_acc = load_checkpoint(
            str(checkpoint_path), model, optimizer, scaler=scaler
        )
        start_epoch = loaded_epoch + 1
        if start_epoch >= config["training"]["epochs"]:
            raise ValueError(
                f"Checkpoint already completed {start_epoch} epoch(s); "
                "increase training.epochs to continue"
            )
        logger.info(f"Resumed from epoch {start_epoch}, best accuracy: {best_acc:.2f}%")

        # The current config/CLI learning rate is authoritative for the rebuilt schedule.
        for group in optimizer.param_groups:
            group["initial_lr"] = config["training"]["lr"]

    scheduler = CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"], eta_min=config["training"]["min_lr"]
    )
    if start_epoch:
        resumed_lrs = []
        for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
            resumed_lr = (
                config["training"]["min_lr"]
                + (base_lr - config["training"]["min_lr"])
                * (1 + math.cos(math.pi * start_epoch / scheduler.T_max))
                / 2
            )
            group["lr"] = resumed_lr
            resumed_lrs.append(resumed_lr)
        scheduler.last_epoch = start_epoch
        scheduler._last_lr = resumed_lrs

    history_path = exp_dir / "training_history.json"
    training_history = []
    if checkpoint_path is not None and history_path.is_file():
        with open(history_path, "r", encoding="utf-8") as history_file:
            stored_history = json.load(history_file)
        if not isinstance(stored_history, list):
            raise ValueError(f"Training history must be a JSON list: {history_path}")
        training_history = [
            item
            for item in stored_history
            if isinstance(item, dict) and item.get("epoch", 0) <= start_epoch
        ]

    with open(exp_dir / "config.yaml", "w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, default_flow_style=False)

    writer = SummaryWriter(str(log_dir)) if TENSORBOARD_AVAILABLE else None

    for epoch in range(start_epoch, config["training"]["epochs"]):
        epoch_start_time = time.time()
        epoch_lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, logger, writer
        )
        val_metrics = validate(model, val_loader, criterion, device, epoch, logger, writer)
        scheduler.step()

        is_best = val_metrics["accuracy"] > best_acc
        if is_best:
            best_acc = val_metrics["accuracy"]

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            val_metrics["loss"],
            val_metrics["accuracy"],
            str(checkpoint_dir),
            config["model"],
            best_acc,
            is_best,
        )

        epoch_time = time.time() - epoch_start_time
        logger.info(
            f"Epoch {epoch+1}/{config['training']['epochs']} completed in {epoch_time:.2f}s"
        )
        logger.info(f"Best accuracy so far: {best_acc:.2f}%")

        training_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "lr": epoch_lr,
            }
        )
        with open(history_path, "w", encoding="utf-8") as history_file:
            json.dump(training_history, history_file, indent=2)

        if early_stopping(val_metrics["loss"]):
            logger.info("Early stopping triggered")
            break

    if writer:
        writer.close()
    logger.info(f"Training completed. Best accuracy: {best_acc:.2f}%")
    best_model_path = checkpoint_dir / "best_model.pth"
    if best_model_path.is_file():
        logger.info(f"Best model saved to {best_model_path}")
    else:
        logger.warning("No best-model checkpoint was produced")


def main():
    parser = argparse.ArgumentParser(description="Train Vision Transformer for Defect Detection")
    parser.add_argument("--config", type=str, required=True, help="Path to a training YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument(
        "--resume-from", type=str, default=None, help="Resume training from checkpoint"
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable CUDA mixed precision",
    )

    args = parser.parse_args()

    # Load config with error handling
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"Configuration file is empty: {args.config}")

        validate_config(config, strict=True)

    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Override config with command line arguments
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.resume_from:
        config["training"]["resume_from"] = args.resume_from
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir

    if args.mixed_precision is not None:
        config["training"]["mixed_precision"] = args.mixed_precision

    try:
        validate_config(config, strict=True)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    print(f"Starting training on {config.get('hardware', {}).get('device', 'auto')}")
    train(config)


if __name__ == "__main__":
    main()
