"""
Visualization utilities for Vision Transformer defect detection
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from ..model import VisionTransformer
except ImportError:
    from src.model import VisionTransformer


def _normalize_array(values: np.ndarray) -> np.ndarray:
    """Scale an array to [0, 1] without dividing by zero."""
    value_range = values.max() - values.min()
    if value_range == 0:
        return np.zeros_like(values, dtype=np.float32)
    return (values - values.min()) / value_range


def plot_attention_maps(
    model: VisionTransformer,
    image: torch.Tensor,
    layer_idx: int = -1,
    head_idx: Optional[int] = 0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize attention maps from Vision Transformer

    Args:
        model: Vision Transformer model
        image: Input image tensor (C, H, W) or (1, C, H, W)
        layer_idx: Which transformer layer to visualize (-1 for last layer)
        head_idx: Which attention head to visualize (None for average across heads)
        save_path: Path to save the plot

    Returns:
        Matplotlib figure object
    """
    model.eval()

    # Ensure image has batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)

    # Get attention maps
    with torch.no_grad():
        attn_weights, patches = model.get_attention_maps(image, layer_idx)

    # Average across heads if not specified
    if head_idx is not None:
        if not 0 <= head_idx < attn_weights.shape[1]:
            raise IndexError(f"head_idx {head_idx} is out of range")
        attn_map = attn_weights[0, head_idx]  # (seq_len, seq_len)
    else:
        attn_map = attn_weights[0].mean(dim=0)  # Average across heads

    # Remove CLS token and get patch-to-patch attention
    attn_map = attn_map[1:, 1:]  # (num_patches, num_patches)

    # Reshape to spatial attention map
    patches_per_side = int(np.sqrt(attn_map.shape[0]))

    # Average attention for each patch (mean across all queries)
    patch_attention = attn_map.mean(dim=0)  # (num_patches,)

    # Reshape to 2D grid
    attn_grid = patch_attention.view(patches_per_side, patches_per_side).cpu().numpy()

    # Upsample attention map to original image size
    img_size = (image.shape[3], image.shape[2])
    attn_upsampled = cv2.resize(attn_grid, img_size, interpolation=cv2.INTER_CUBIC)

    # Normalize attention map
    attn_upsampled = _normalize_array(attn_upsampled)

    # Convert image to numpy for plotting
    img_np = image[0].permute(1, 2, 0).cpu().numpy()
    img_np = _normalize_array(img_np)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Attention map overlay
    axes[1].imshow(img_np)
    im = axes[1].imshow(attn_upsampled, cmap="jet", alpha=0.5)
    axes[1].set_title(
        f'Attention Map (Layer {layer_idx}, Head {head_idx if head_idx is not None else "Avg"})'
    )
    axes[1].axis("off")

    # Pure attention map
    axes[2].imshow(attn_grid, cmap="viridis")
    axes[2].set_title("Attention Grid")
    axes[2].axis("off")

    # Add colorbar
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_multi_head_attention(
    model: VisionTransformer,
    image: torch.Tensor,
    layer_idx: int = -1,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize attention patterns from all heads in a layer

    Args:
        model: Vision Transformer model
        image: Input image tensor
        layer_idx: Which transformer layer to visualize
        save_path: Path to save the plot

    Returns:
        Matplotlib figure object
    """
    model.eval()

    if image.dim() == 3:
        image = image.unsqueeze(0)

    with torch.no_grad():
        attn_weights, _ = model.get_attention_maps(image, layer_idx)

    num_heads = attn_weights.shape[1]

    # Calculate grid dimensions
    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.asarray(axes, dtype=object).reshape(rows, cols)

    # Get image for background
    img_np = image[0].permute(1, 2, 0).cpu().numpy()
    img_np = _normalize_array(img_np)

    for head_idx in range(num_heads):
        row, col = head_idx // cols, head_idx % cols
        ax = axes[row, col]

        # Get attention for this head
        attn_map = attn_weights[0, head_idx, 1:, 1:]  # Remove CLS token
        patch_attention = attn_map.mean(dim=0)  # Average across queries

        # Reshape to grid
        patches_per_side = int(np.sqrt(patch_attention.shape[0]))
        attn_grid = patch_attention.view(patches_per_side, patches_per_side).cpu().numpy()

        # Upsample
        img_size = (image.shape[3], image.shape[2])
        attn_upsampled = cv2.resize(attn_grid, img_size, interpolation=cv2.INTER_CUBIC)
        attn_upsampled = _normalize_array(attn_upsampled)

        # Plot
        ax.imshow(img_np)
        im = ax.imshow(attn_upsampled, cmap="jet", alpha=0.5)
        ax.set_title(f"Head {head_idx}")
        ax.axis("off")

    # Remove empty subplots
    for head_idx in range(num_heads, rows * cols):
        row, col = head_idx // cols, head_idx % cols
        axes[row, col].remove()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_patch_embeddings(
    model: VisionTransformer, image: torch.Tensor, save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize patch embeddings and their spatial arrangement

    Args:
        model: Vision Transformer model
        image: Input image tensor
        save_path: Path to save the plot

    Returns:
        Matplotlib figure object
    """
    model.eval()

    if image.dim() == 3:
        image = image.unsqueeze(0)

    with torch.no_grad():
        patches = model.patch_embed(image)  # (1, num_patches, embed_dim)

    patches = patches[0]  # Remove batch dim
    num_patches = patches.shape[0]
    embed_dim = patches.shape[1]

    # Reshape to spatial grid
    patches_per_side = int(np.sqrt(num_patches))
    patches_grid = patches.view(patches_per_side, patches_per_side, embed_dim)

    # Visualize different embedding dimensions
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Show first 8 embedding dimensions as spatial maps
    for i in range(8):
        row, col = i // 4, i % 4
        ax = axes[row, col]

        embed_map = patches_grid[:, :, i].cpu().numpy()
        im = ax.imshow(embed_map, cmap="viridis")
        ax.set_title(f"Embedding Dim {i}")
        ax.axis("off")

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return fig


def plot_model_predictions(
    model: VisionTransformer,
    images: List[torch.Tensor],
    true_labels: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize model predictions on multiple images

    Args:
        model: Vision Transformer model
        images: List of input image tensors
        true_labels: Optional true labels for comparison
        class_names: Class names for display
        save_path: Path to save the plot

    Returns:
        Matplotlib figure object
    """
    if class_names is None:
        class_names = ["ok", "scratch", "crack", "dent", "corrosion"]

    model.eval()

    predictions = []
    probabilities = []

    with torch.no_grad():
        for img in images:
            if img.dim() == 3:
                img = img.unsqueeze(0)
            output = model(img)
            probs = F.softmax(output, dim=1)[0]
            pred_class = torch.argmax(probs).item()
            predictions.append(pred_class)
            probabilities.append(probs.cpu().numpy())

    n_images = len(images)
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i, (img, pred, probs) in enumerate(zip(images, predictions, probabilities)):
        row, col = i // cols, i % cols
        ax = axes[row, col]

        # Convert tensor to image
        img_np = (
            img.permute(1, 2, 0).cpu().numpy()
            if img.dim() == 3
            else img[0].permute(1, 2, 0).cpu().numpy()
        )
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        ax.imshow(img_np)
        ax.axis("off")

        # Title with prediction
        title = f"Pred: {class_names[pred]} ({probs[pred]:.2%})"
        if true_labels and i < len(true_labels):
            title += f"\nTrue: {true_labels[i]}"
            if true_labels[i] != class_names[pred]:
                ax.set_title(title, color="red")
            else:
                ax.set_title(title, color="green")
        else:
            ax.set_title(title)

    # Remove empty subplots
    for i in range(n_images, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].remove()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return fig


def plot_gradcam(
    model: VisionTransformer,
    image: torch.Tensor,
    target_class: int,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate Grad-CAM visualization for Vision Transformer

    Args:
        model: Vision Transformer model
        image: Input image tensor
        target_class: Target class for Grad-CAM
        save_path: Path to save the plot

    Returns:
        Matplotlib figure object
    """
    model.eval()

    if image.dim() == 3:
        image = image.unsqueeze(0)
    image.requires_grad_(True)

    # Forward pass
    output = model(image)
    target_output = output[0, target_class]

    # Backward pass
    model.zero_grad()
    target_output.backward()

    # Get gradients from patch embeddings
    gradients = model.patch_embed.projection.weight.grad
    activations = model.patch_embed(image).detach()

    # Global average pooling of gradients
    weights = torch.mean(gradients, dim=[1, 2, 3])  # Average across spatial dims

    # Weighted combination of activations
    cam = torch.zeros(activations.shape[2], activations.shape[3], device=activations.device)
    for i, w in enumerate(weights):
        cam += w * activations[0, :, i, :].view(14, 14)

    # Apply ReLU
    cam = F.relu(cam)

    # Normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    # Convert to numpy
    cam_np = cam.cpu().numpy()

    # Upsample to original image size
    img_size = image.shape[2:]
    cam_upsampled = cv2.resize(cam_np, img_size, interpolation=cv2.INTER_CUBIC)

    # Convert image to numpy
    img_np = image[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_upsampled), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap / 255.0

    # Overlay heatmap on image
    overlay = 0.6 * img_np + 0.4 * heatmap

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Grad-CAM heatmap
    axes[1].imshow(cam_upsampled, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return fig


def create_attention_gif(
    model: VisionTransformer, image: torch.Tensor, save_path: str, duration: int = 500
) -> None:
    """
    Create animated GIF showing attention across layers

    Args:
        model: Vision Transformer model
        image: Input image tensor
        save_path: Path to save GIF
        duration: Duration between frames in ms
    """
    try:
        from PIL import Image as PILImage
    except ImportError:
        warnings.warn("PIL not available, cannot create GIF")
        return

    frames = []

    for layer_idx in range(len(model.blocks)):
        fig = plot_attention_maps(model, image, layer_idx=layer_idx)
        fig.canvas.draw()

        # Convert to PIL Image
        buf = fig.canvas.buffer_rgba()
        img = PILImage.frombytes("RGBA", fig.canvas.get_width_height(), buf)
        frames.append(img)

        plt.close(fig)

    # Save as GIF
    frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)


def visualize_synthetic_defects(num_samples: int = 5, save_path: Optional[str] = None):
    """
    Visualize examples of synthetic defects

    Args:
        num_samples: Number of samples to generate per defect type
        save_path: Path to save the visualization
    """
    try:
        from ..data.augmentation import apply_random_defect
    except ImportError:
        from src.data.augmentation import apply_random_defect

    defect_types = ["ok", "scratch", "crack", "dent", "corrosion"]

    # Create base image
    base_image = np.full((224, 224, 3), 180, dtype=np.uint8)
    base_image += np.random.normal(0, 20, base_image.shape).astype(np.uint8)

    fig, axes = plt.subplots(
        len(defect_types), num_samples, figsize=(3 * num_samples, 3 * len(defect_types))
    )

    for i, defect_type in enumerate(defect_types):
        for j in range(num_samples):
            # Generate defect
            augmented, _ = apply_random_defect(base_image.copy(), defect_type)

            # Convert to RGB for display
            display_img = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)

            axes[i, j].imshow(display_img)
            axes[i, j].set_title(f"{defect_type.capitalize()} #{j+1}")
            axes[i, j].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
