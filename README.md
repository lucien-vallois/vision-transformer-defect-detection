# Vision Transformer Defect Detection

A reusable PyTorch reference implementation for classifying five surface conditions with a
Vision Transformer: `ok`, `scratch`, `crack`, `dent`, and `corrosion`.

The repository is train-first. It includes synthetic data generation, training, checkpoint-based
inference, evaluation helpers, tests, and optional local interfaces. It does not include a trained
checkpoint or claim benchmark results. Models trained only on the included synthetic data are not
validated for industrial or safety-critical inspection.

## Scope

| Capability | Status |
| --- | --- |
| Synthetic data generated on demand | Included |
| Training on CPU or CUDA | Included |
| PyTorch checkpoint inference | Included |
| ONNX export and inference | Optional |
| Gradio and Streamlit local interfaces | Optional |
| Pretrained weights | Not included |
| Production deployment | Out of scope |
| Real-world accuracy or latency benchmark | Not published |

## Requirements

- Python 3.10, 3.11, or 3.12
- 8 GB RAM for the smoke configuration
- More memory, storage, and preferably a CUDA-capable GPU for `configs/vit_base.yaml`

## Install

Create an isolated environment from the repository root.

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the package and its core runtime dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

See [INSTALL.md](INSTALL.md) for optional dependency groups and troubleshooting.

## Verify the installation

The quick start uses a small model and generates samples in memory. It does not download data or
weights.

```bash
python examples/quick_start.py
python -m src.utils.config_validator configs/smoke.yaml
```

For the full test suite:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Train a local smoke model

This exercises data generation, a training epoch, validation, and checkpoint writing on CPU:

```bash
python -m src.train --config configs/smoke.yaml
```

Outputs are written to `experiments/smoke/`. Checkpoints include the model configuration and class
names needed by inference. A new run refuses to mix with an existing experiment directory; choose
another `training.output_dir` or resume a checkpoint. Generated artifacts are intentionally ignored
by Git.

For a larger model, use:

```bash
python -m src.train --config configs/vit_base.yaml
```

The base configuration is computationally expensive. Adjust sample counts, batch size, workers,
and model dimensions for the available hardware.

## Run inference

Inference requires a checkpoint created by training:

```bash
python -m src.inference --model-path experiments/smoke/checkpoints/best_model.pth --image-path path/to/image.png --output results.json
```

Python API:

```python
from src.inference import DefectDetector

detector = DefectDetector("experiments/smoke/checkpoints/best_model.pth")
result = detector.predict("path/to/image.png")

print(result["class"], result["confidence"])
print(result["probabilities"])
```

Do not load checkpoints from untrusted sources. The loader uses PyTorch's weights-only mode, but
model provenance and integrity still belong to the caller.

## Generate a persistent synthetic dataset

On-demand generation is the default for training. To write an inspectable dataset to disk:

```bash
python -m scripts.generate_synthetic_data --num-samples 1000 --output-dir data/synthetic_defects
```

The command writes images, split annotation files, and `dataset_stats.json`. The destination must
not already contain generated dataset files. To train from those files, copy a config and set:

```yaml
data:
  root_dir: ./data/synthetic_defects
  generate_on_fly: false
```

## Optional local interfaces

Install interface dependencies:

```bash
python -m pip install -e ".[demo]"
```

Then start one interface and load a local checkpoint in the UI:

```bash
python demo/gradio_app.py
streamlit run demo/streamlit_app.py
```

Gradio binds to `127.0.0.1` by default. These interfaces are local inspection tools, not hosted
services.

## Optional ONNX support

```bash
python -m pip install -e ".[export]"
vit-export-onnx --model-path experiments/smoke/checkpoints/best_model.pth
vit-inference --model-path experiments/exports/model.onnx --image-path path/to/image.png --output onnx-results.json
```

The exporter reads configuration and class names from current checkpoints, embeds them in the ONNX
file, and verifies PyTorch/ONNX Runtime parity for batch sizes 1 and 2. Pass `--config` only for an
older weights-only checkpoint. TensorRT execution is not implemented.

## Dataset contract

For a persistent dataset, the root directory must contain:

```text
dataset-root/
├── images/
├── train_annotations.json
├── val_annotations.json
└── test_annotations.json
```

Each annotation is a JSON object with `id`, `class`, and an image `path` relative to the dataset
root. Supported class names are fixed to the five labels listed at the top of this document.

## Documentation

- [Installation](INSTALL.md)
- [Local usage](docs/LOCAL_USAGE.md)
- [Python API](docs/API.md)
- [Architecture](docs/architecture.md)
- [Model comparison protocol](docs/comparison_results.md)
- [Contributing](CONTRIBUTING.md)

## License

MIT. See [LICENSE](LICENSE).
