"""
Vision Transformer implementation for defect detection
Based on "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    Convert image patches to embeddings using convolutional projection.

    This module splits an image into non-overlapping patches and projects each patch
    into an embedding space using a convolutional layer. This is the first step in
    the Vision Transformer architecture.

    Attributes:
        img_size: Size of input image (assumed square)
        patch_size: Size of each patch (assumed square)
        patches_per_side: Number of patches per side of the image
        num_patches: Total number of patches (patches_per_side ** 2)
        in_channels: Number of input channels (typically 3 for RGB)
        embed_dim: Dimension of patch embeddings
        projection: Convolutional layer that projects patches to embeddings
    """

    def __init__(
        self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768
    ):
        """
        Initialize patch embedding module.

        Args:
            img_size: Size of input image (default: 224)
            patch_size: Size of each patch (default: 16)
            in_channels: Number of input channels (default: 3)
            embed_dim: Dimension of patch embeddings (default: 768)

        Raises:
            ValueError: If img_size is not divisible by patch_size
        """
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
            )

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_per_side = img_size // patch_size
        self.num_patches = self.patches_per_side**2

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: convert image to patch embeddings.

        Args:
            x: Input tensor of shape (batch_size, in_channels, img_size, img_size)

        Returns:
            Patch embeddings of shape (batch_size, num_patches, embed_dim)

        Raises:
            RuntimeError: If input tensor shape is invalid
        """
        if x.ndim != 4:
            raise ValueError(f"Expected input with shape (N, C, H, W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}")
        if x.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"Expected spatial size {self.img_size}x{self.img_size}, "
                f"got {x.shape[-2]}x{x.shape[-1]}"
            )

        x = self.projection(x)  # (batch_size, embed_dim, patches_per_side, patches_per_side)
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x


class PositionalEncoding(nn.Module):
    """
    Add positional information to patch embeddings using learnable position embeddings.

    This module adds a learnable CLS (classification) token at the beginning of the
    patch sequence and adds learnable positional embeddings to all tokens (CLS + patches).
    The positional embeddings allow the model to understand spatial relationships
    between patches.

    Attributes:
        position_embeddings: Learnable positional embeddings for all tokens
        cls_token: Learnable CLS token for classification
    """

    def __init__(self, num_patches: int, embed_dim: int):
        """
        Initialize positional encoding module.

        Args:
            num_patches: Number of image patches
            embed_dim: Dimension of embeddings
        """
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Initialize with normal distribution
        nn.init.normal_(self.position_embeddings, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: add CLS token and positional embeddings.

        Args:
            x: Patch embeddings of shape (batch_size, num_patches, embed_dim)

        Returns:
            Embeddings with positional encoding and CLS token:
            (batch_size, num_patches + 1, embed_dim)
        """
        batch_size = x.shape[0]

        # Add CLS token at the beginning
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embeddings
        x = x + self.position_embeddings
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Implements scaled dot-product attention with multiple heads, allowing the model
    to attend to different aspects of the input simultaneously. Each head learns
    different attention patterns.

    Attributes:
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads
        head_dim: Dimension per attention head (embed_dim // num_heads)
        qkv_proj: Linear projection for query, key, and value
        out_proj: Output projection layer
        dropout: Dropout layer for attention weights
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention module.

        Args:
            embed_dim: Total embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability for attention weights (default: 0.1)

        Raises:
            ValueError: If embed_dim is not divisible by num_heads
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass: compute multi-head self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
                  where 0 indicates positions to mask

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Generate Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention computation
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        if mask is not None:
            if mask.ndim == 2:
                if tuple(mask.shape) != (seq_len, seq_len):
                    raise ValueError(
                        f"Expected mask shape ({seq_len}, {seq_len}), got {tuple(mask.shape)}"
                    )
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.ndim == 3:
                if tuple(mask.shape) != (batch_size, seq_len, seq_len):
                    raise ValueError(
                        f"Expected mask shape ({batch_size}, {seq_len}, {seq_len}), "
                        f"got {tuple(mask.shape)}"
                    )
                mask = mask.unsqueeze(1)
            elif mask.ndim == 4:
                if (
                    mask.shape[0] not in {1, batch_size}
                    or mask.shape[1] not in {1, self.num_heads}
                    or tuple(mask.shape[-2:]) != (seq_len, seq_len)
                ):
                    raise ValueError(
                        "Expected a broadcastable 4D mask with shape "
                        f"(1|{batch_size}, 1|{self.num_heads}, {seq_len}, {seq_len}), "
                        f"got {tuple(mask.shape)}"
                    )
            else:
                raise ValueError(f"Expected a 2D, 3D, or 4D attention mask, got {mask.ndim}D")

            mask = mask.to(device=attn_weights.device)
            if not torch.all(torch.any(mask != 0, dim=-1)):
                raise ValueError("Each attention-mask row must allow at least one position")
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, embed_dim)

        # Output projection
        output = self.out_proj(attn_output)
        return output


class MLP(nn.Module):
    """Multi-layer perceptron for transformer block"""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.attn = MultiHeadAttention(embed_dim, num_heads, attention_dropout)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Multi-head attention with residual connection
        x = x + self.attn(self.norm1(x), mask)

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model for defect detection.

    This implementation follows the architecture described in "An Image is Worth 16x16 Words:
    Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020), adapted for
    multi-class defect classification.

    The model processes images by:
    1. Splitting images into patches
    2. Embedding patches into a learned space
    3. Adding positional encodings and a CLS token
    4. Processing through transformer encoder blocks
    5. Using the CLS token for final classification

    Attributes:
        num_classes: Number of output classes
        num_patches: Number of image patches
        patch_embed: Patch embedding module
        pos_embed: Positional encoding module
        blocks: List of transformer encoder blocks
        norm: Layer normalization before classification head
        head: Classification head (linear layer)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 5,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        """
        Initialize Vision Transformer model.

        Args:
            img_size: Size of input images (assumed square, default: 224)
            patch_size: Size of each patch (assumed square, default: 16)
            in_channels: Number of input channels (default: 3 for RGB)
            num_classes: Number of output classes (default: 5)
            embed_dim: Embedding dimension (default: 768)
            depth: Number of transformer blocks (default: 12)
            num_heads: Number of attention heads (default: 8)
            mlp_ratio: Ratio of MLP hidden dimension to embed_dim (default: 4.0)
            dropout: Dropout probability (default: 0.1)
            attention_dropout: Dropout probability for attention weights (default: 0.1)

        Raises:
            ValueError: If img_size is not divisible by patch_size
            ValueError: If embed_dim is not divisible by num_heads
        """
        super().__init__()

        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")

        self.num_classes = num_classes
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # Positional encoding
        self.pos_embed = PositionalEncoding(self.num_patches, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Vision Transformer.

        Args:
            x: Input images of shape (batch_size, in_channels, img_size, img_size)
               Expected to be normalized with ImageNet statistics

        Returns:
            Classification logits of shape (batch_size, num_classes)

        Raises:
            RuntimeError: If input tensor shape is invalid
        """
        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)

        # Add positional encoding and CLS token
        x = self.pos_embed(x)  # (batch_size, num_patches + 1, embed_dim)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification: use CLS token
        x = self.norm(x[:, 0])  # Take CLS token
        x = self.head(x)
        return x

    def get_attention_maps(
        self, x: torch.Tensor, layer_idx: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract attention maps from a specific transformer layer for visualization.

        This method is useful for understanding what parts of the image the model
        focuses on when making predictions. The attention weights show the importance
        of each patch relative to others.

        Args:
            x: Input images of shape (batch_size, in_channels, img_size, img_size)
            layer_idx: Index of transformer layer to extract attention from.
                      Use -1 for the last layer (default: -1)

        Returns:
            Tuple containing:
            - attention_weights: Tensor of shape (batch_size, num_heads, seq_len, seq_len)
              containing attention weights
            - patch_embeddings: Tensor of shape (batch_size, num_patches, embed_dim)
              containing patch embeddings before positional encoding

        Raises:
            IndexError: If layer_idx is out of range
        """
        if not -len(self.blocks) <= layer_idx < len(self.blocks):
            raise IndexError(f"layer_idx {layer_idx} is out of range for {len(self.blocks)} layers")
        layer_idx %= len(self.blocks)

        # Patch embedding
        patches = self.patch_embed(x)

        # Add positional encoding and CLS token
        x = self.pos_embed(patches)

        # Apply transformer blocks up to the specified layer
        for i, block in enumerate(self.blocks):
            if i == layer_idx:
                # Get attention weights from this layer
                with torch.no_grad():
                    batch_size, seq_len, embed_dim = x.shape
                    qkv = block.attn.qkv_proj(block.norm1(x))
                    qkv = qkv.reshape(
                        batch_size, seq_len, 3, block.attn.num_heads, block.attn.head_dim
                    )
                    qkv = qkv.permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]

                    scale = math.sqrt(block.attn.head_dim)
                    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
                    attn_weights = F.softmax(attn_weights, dim=-1)

                break
            x = block(x)

        return attn_weights, patches


def create_vit_model(config: dict) -> VisionTransformer:
    """
    Create a Vision Transformer model from a configuration dictionary.

    This is a convenience function that creates a ViT model with parameters
    specified in a configuration dictionary. Useful for loading models from
    configuration files.

    Args:
        config: Configuration dictionary containing model parameters.
                Expected keys:
                - img_size (int): Image size
                - patch_size (int): Patch size
                - in_channels (int): Number of input channels
                - num_classes (int): Number of output classes
                - embed_dim (int): Embedding dimension
                - depth (int): Number of transformer blocks
                - num_heads (int): Number of attention heads
                - mlp_ratio (float): MLP ratio
                - dropout (float): Dropout probability
                - attention_dropout (float): Attention dropout probability

    Returns:
        Initialized VisionTransformer model

    Raises:
        KeyError: If required configuration keys are missing
        ValueError: If configuration values are invalid
    """
    return VisionTransformer(
        img_size=config.get("img_size", 224),
        patch_size=config.get("patch_size", 16),
        in_channels=config.get("in_channels", 3),
        num_classes=config.get("num_classes", 5),
        embed_dim=config.get("embed_dim", 768),
        depth=config.get("depth", 12),
        num_heads=config.get("num_heads", 8),
        mlp_ratio=config.get("mlp_ratio", 4.0),
        dropout=config.get("dropout", 0.1),
        attention_dropout=config.get("attention_dropout", 0.1),
    )
