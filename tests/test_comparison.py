"""Tests for checkpoint-backed model comparison."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import create_vit_model
from src.models.comparison import ModelComparator


TINY_CONFIG = {
    "img_size": 32,
    "patch_size": 8,
    "in_channels": 3,
    "num_classes": 5,
    "embed_dim": 32,
    "depth": 1,
    "num_heads": 4,
    "mlp_ratio": 2.0,
}


def test_comparator_evaluates_checkpoint_backed_model(tmp_path):
    model = create_vit_model(TINY_CONFIG)
    checkpoint_path = tmp_path / "model.pth"
    torch.save(
        {"model_state_dict": model.state_dict(), "model_config": TINY_CONFIG},
        checkpoint_path,
    )
    comparator = ModelComparator(device="cpu")
    loaded_model = comparator.load_vit_model(str(checkpoint_path))
    dataloader = DataLoader(
        TensorDataset(
            torch.randn(4, 3, 32, 32),
            torch.tensor([0, 1, 2, 3]),
            torch.arange(4),
        ),
        batch_size=2,
    )

    result = comparator.evaluate_model(
        loaded_model,
        "ViT",
        dataloader,
        ["ok", "scratch", "crack", "dent", "corrosion"],
    )

    assert 0 <= result["accuracy"] <= 1
    assert result["predictions"].shape == (4,)
    assert result["probabilities"].shape == (4, 5)
    comparator.results = {"ViT": result}
    assert "ViT" in comparator.generate_comparison_table()
