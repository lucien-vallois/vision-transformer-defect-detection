# Architecture

The model implements an encoder-only Vision Transformer for image classification.

```text
RGB image
  -> non-overlapping convolutional patch projection
  -> learned class token and positional embeddings
  -> repeated pre-normalized self-attention and MLP blocks
  -> class-token layer normalization
  -> linear classification head
```

The implementation lives in `src/model.py` and contains five building blocks:

1. `PatchEmbedding` converts an image into a patch sequence.
2. `PositionalEncoding` prepends a learned class token and adds learned positions.
3. `MultiHeadAttention` computes scaled dot-product self-attention.
4. `TransformerBlock` applies pre-normalization, residual attention, and a residual MLP.
5. `VisionTransformer` stacks blocks and classifies the final class token.

## Configurations

`configs/smoke.yaml` is deliberately small and exists to verify the pipeline on CPU.
`configs/vit_base.yaml` describes a substantially larger ViT-Base-style model. Neither config
implies a pretrained model or a measured quality level.

Image size must be divisible by patch size, and embedding dimension must be divisible by the number
of heads. The config validator enforces these constraints before training.

## Attention maps

`get_attention_maps` exposes attention from a selected encoder block for inspection. Attention is
not an explanation guarantee and should not be treated as defect localization ground truth.

## Boundaries

- Classification labels are fixed to five surface conditions.
- Input images are resized to a square, which can distort aspect ratio.
- Synthetic augmentation is useful for pipeline testing, not a substitute for representative,
  labeled inspection data.
- PyTorch and optional ONNX inference are implemented; TensorRT inference is not.
