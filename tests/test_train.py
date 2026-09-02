"""Tests for training state and resume behavior."""

import json
import sys

import pytest
import torch
import yaml

from src.train import main, train


def make_config(output_dir, epochs=1):
    return {
        "model": {
            "img_size": 32,
            "patch_size": 8,
            "in_channels": 3,
            "num_classes": 5,
            "embed_dim": 16,
            "depth": 1,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "attention_dropout": 0.0,
        },
        "data": {
            "root_dir": "./data",
            "generate_on_fly": True,
            "train_samples": 8,
            "val_samples": 5,
            "test_samples": 5,
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "seed": 7,
        },
        "training": {
            "epochs": epochs,
            "output_dir": str(output_dir),
            "lr": 0.001,
            "weight_decay": 0.01,
            "min_lr": 0.00001,
            "mixed_precision": False,
            "early_stopping_patience": 3,
            "early_stopping_delta": 0.001,
            "resume_from": None,
        },
        "hardware": {"device": "cpu"},
    }


def test_resume_keeps_history_best_accuracy_and_rebuilds_schedule(tmp_path):
    output_dir = tmp_path / "experiment"
    first_config = make_config(output_dir, epochs=1)
    train(first_config)

    checkpoint_path = output_dir / "checkpoints" / "checkpoint_epoch_0.pth"
    resumed_config = make_config(output_dir, epochs=2)
    resumed_config["training"]["resume_from"] = str(checkpoint_path)
    train(resumed_config)

    history = json.loads((output_dir / "training_history.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(
        output_dir / "checkpoints" / "checkpoint_epoch_1.pth",
        map_location="cpu",
        weights_only=True,
    )

    assert [entry["epoch"] for entry in history] == [1, 2]
    assert history[0]["lr"] == pytest.approx(0.001)
    assert history[1]["lr"] == pytest.approx(0.000505)
    assert checkpoint["scheduler_state_dict"]["T_max"] == 2
    assert checkpoint["best_accuracy"] >= checkpoint["accuracy"]


def test_resume_requires_existing_checkpoint(tmp_path):
    config = make_config(tmp_path / "experiment", epochs=2)
    config["training"]["resume_from"] = str(tmp_path / "missing.pth")

    with pytest.raises(FileNotFoundError, match="Resume checkpoint not found"):
        train(config)


def test_new_training_refuses_existing_experiment(tmp_path):
    output_dir = tmp_path / "experiment"
    output_dir.mkdir()
    (output_dir / "training_history.json").write_text("[]", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Choose a new output directory"):
        train(make_config(output_dir))


def test_cli_revalidates_epoch_override(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(make_config(tmp_path / "experiment")),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["vit-train", "--config", str(config_path), "--epochs", "0"],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
