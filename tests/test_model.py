"""
Unit tests for Vision Transformer model
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest

from src.model import MultiHeadAttention, VisionTransformer, create_vit_model

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


def make_model(**overrides):
    return create_vit_model({**TINY_CONFIG, **overrides})


class TestVisionTransformer:
    """Test cases for VisionTransformer model"""

    def test_model_creation(self):
        """Test model can be created with default parameters"""
        model = make_model()
        assert model is not None
        assert model.num_classes == 5

    def test_model_forward(self):
        """Test model forward pass"""
        model = make_model()
        model.eval()

        batch_size = 2
        x = torch.randn(batch_size, 3, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 5)

    def test_model_parameters(self):
        """Test model has trainable parameters"""
        model = make_model()
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0

    def test_create_vit_model_from_config(self):
        """Test model creation from config dictionary"""
        model = create_vit_model(TINY_CONFIG)
        assert model is not None
        assert model.num_classes == 5

    def test_attention_maps_extraction(self):
        """Test attention maps can be extracted"""
        model = make_model()
        model.eval()

        x = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            attn_weights, patches = model.get_attention_maps(x)

        assert attn_weights is not None
        assert patches is not None
        assert attn_weights.shape[0] == 1  # batch size
        assert attn_weights.shape[1] == 4  # num_heads
        assert patches.shape[1] == 64  # num_patches

    def test_model_different_input_sizes(self):
        """Test model with different input sizes"""
        for img_size in [32, 64, 96]:
            model = make_model(img_size=img_size)
            model.eval()

            x = torch.randn(1, 3, img_size, img_size)
            with torch.no_grad():
                output = model(x)

            assert output.shape == (1, 5)

    def test_model_different_patch_sizes(self):
        """Test model with different patch sizes"""
        for patch_size in [4, 8, 16]:
            img_size = 64
            if img_size % patch_size == 0:
                model = make_model(img_size=img_size, patch_size=patch_size)
                model.eval()

                x = torch.randn(1, 3, img_size, img_size)
                with torch.no_grad():
                    output = model(x)

                assert output.shape == (1, 5)

    def test_model_gradient_flow(self):
        """Test that gradients flow through the model"""
        model = make_model()
        model.train()

        x = torch.randn(2, 3, 64, 64, requires_grad=True)
        output = model(x)

        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break

        assert has_grad, "No gradients found in model parameters"

    def test_model_parameter_count(self):
        """Test the small model stays suitable for fast checks."""
        model = make_model()
        num_params = sum(p.numel() for p in model.parameters())

        assert 50_000 < num_params < 500_000

    def test_model_attention_heads_validation(self):
        """Test that invalid num_heads raises error"""
        with pytest.raises(ValueError, match="divisible"):
            make_model(embed_dim=62, num_heads=4)

    def test_model_patch_size_validation(self):
        """Test that invalid patch_size raises error"""
        with pytest.raises(ValueError, match="divisible"):
            make_model(img_size=64, patch_size=10)

    def test_model_rejects_wrong_channel_count(self):
        model = make_model()

        with pytest.raises(ValueError, match="Expected 3 input channels"):
            model(torch.randn(1, 1, 64, 64))


class TestMultiHeadAttention:
    def test_batch_attention_mask_broadcasts_over_heads(self):
        attention = MultiHeadAttention(embed_dim=8, num_heads=2, dropout=0)
        inputs = torch.randn(3, 4, 8)
        mask = torch.ones(3, 4, 4)
        mask[:, :, -1] = 0

        output = attention(inputs, mask)

        assert output.shape == inputs.shape
        assert torch.isfinite(output).all()

    def test_attention_mask_rejects_fully_masked_rows(self):
        attention = MultiHeadAttention(embed_dim=8, num_heads=2, dropout=0)
        inputs = torch.randn(1, 4, 8)
        mask = torch.ones(4, 4)
        mask[0] = 0

        with pytest.raises(ValueError, match="must allow at least one"):
            attention(inputs, mask)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
