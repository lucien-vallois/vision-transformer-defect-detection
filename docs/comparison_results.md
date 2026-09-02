# Model comparison protocol

This repository does not publish model-comparison results. A trustworthy comparison must be
generated from the same immutable dataset split and record enough evidence to reproduce it.

The optional comparison command requires the `comparison` dependency group:

```bash
python -m pip install -e ".[comparison]"
python -m scripts.compare_models --data-dir data/synthetic_defects --vit-model experiments/smoke/checkpoints/best_model.pth --output-dir experiments/comparison
```

Every selected architecture requires a trained checkpoint. The command refuses to rank randomly
initialized models. Use `--models` plus the matching `--resnet-model` or `--efficientnet-model`
argument when comparing optional `timm` architectures.

Before interpreting output, record:

- source revision and dependency versions;
- dataset manifest or digest and exact split;
- model configuration and checkpoint digest;
- whether weights were pretrained;
- hardware, device, precision, warm-up, and timing method;
- per-class support and all requested metrics.

Synthetic-only results describe that generated distribution. They do not establish performance on
real components or suitability for operational inspection.
