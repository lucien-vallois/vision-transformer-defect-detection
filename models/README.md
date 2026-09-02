# Model files

No pretrained model is included or published by this repository.

Training writes checkpoints under the configured `training.output_dir`. The smoke configuration
writes to `experiments/smoke/checkpoints/`. Checkpoints and exported models are ignored by Git
because they are generated artifacts and can be large.

Create a checkpoint with:

```bash
python -m src.train --config configs/smoke.yaml
```

Then load it with:

```bash
python -m src.inference --model-path experiments/smoke/checkpoints/best_model.pth --image-path path/to/image.png
```

Project-generated checkpoints contain model configuration and class names. A weights-only file
without that metadata is interpreted as the default ViT-Base architecture unless `--config` is
provided.

Only load model files whose origin and integrity you trust.
