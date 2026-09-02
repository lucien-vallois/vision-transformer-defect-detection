# Python API

The stable public surface is intentionally small.

## Model

```python
from src.model import VisionTransformer, create_vit_model
```

`create_vit_model(config)` accepts the keys in `configs/smoke.yaml` and returns a
`VisionTransformer`. Calling the model with a tensor shaped `(batch, channels, height, width)`
returns class logits shaped `(batch, num_classes)`.

`model.get_attention_maps(images, layer_idx=-1)` returns attention weights and patch embeddings.
Negative layer indexes follow normal Python indexing.

## Dataset

```python
from src.data.dataset import DefectDetectionDataset, SyntheticDefectDataset
```

`SyntheticDefectDataset` can generate samples on demand or load split annotations from disk. Each
item is `(image_tensor, integer_label, metadata)`. With on-demand data, validation and test images
are stable for a configured `data.seed`; training augmentation remains stochastic across epochs.

`DefectDetectionDataset(config).get_dataloaders()` creates train, validation, and test loaders from
the `data` section of a config.

## Inference

```python
from src.inference import DefectDetector

detector = DefectDetector("checkpoint.pth", device="auto")
result = detector.predict("image.png")
```

`predict` accepts a path, NumPy array, or PIL image and returns:

```python
{
    "class": "scratch",
    "class_idx": 1,
    "confidence": 0.73,
    "probabilities": {
        "ok": 0.04,
        "scratch": 0.73,
        "crack": 0.08,
        "dent": 0.09,
        "corrosion": 0.06,
    },
    "latency_ms": 12.4,
    "all_classes": ["ok", "scratch", "crack", "dent", "corrosion"],
}
```

The numeric values above illustrate the response shape; they are not project benchmark results.

`predict_batch(images)` returns one result dictionary per input image. The configured batch size
controls chunking.

## Checkpoint contract

Training checkpoints contain:

- `model_state_dict`;
- `model_config`;
- ordered `classes`;
- optimizer, scheduler, scaler, epoch, loss, current accuracy, and best accuracy state for resume.

Inference uses model metadata when present. For a plain state dictionary with a non-default
architecture, pass a model config to `DefectDetector` or use `--config` on the CLI.

Resume with the same `training.output_dir` and increase `training.epochs` beyond the checkpoint's
completed epoch count. Existing history is retained and the cosine schedule is recalculated for
the new total. Resume restores optimization state, but it is not a bit-for-bit replay of random
training augmentation.
