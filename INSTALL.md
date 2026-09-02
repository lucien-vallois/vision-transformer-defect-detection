# Installation

## Core environment

Use Python 3.10, 3.11, or 3.12 in an isolated virtual environment.

```bash
python -m venv .venv
```

Activate `.venv` before installing on a new shell. On Windows PowerShell use
`.\.venv\Scripts\Activate.ps1`; on Linux or macOS use `source .venv/bin/activate`.

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

The core install contains the packages needed for model construction, synthetic data, training,
metrics, and PyTorch inference. PyTorch installation varies by operating system and accelerator;
if a specific CUDA build is required, install the matching PyTorch build first and then run the
project installation command.

## Optional groups

Install only what is needed:

```bash
# Tests and package checks
python -m pip install -e ".[dev]"

# Gradio and Streamlit local interfaces
python -m pip install -e ".[demo]"

# ONNX export and runtime
python -m pip install -e ".[export]"

# Model comparison helpers
python -m pip install -e ".[comparison]"

# TensorBoard logging
python -m pip install -e ".[tensorboard]"
```

## Verification

```bash
python -c "import src; print(src.__file__)"
python examples/quick_start.py
python -m src.utils.config_validator configs/smoke.yaml
```

Run the tests after installing the development group:

```bash
python -m pytest -q
```

## Common problems

### Commands cannot import `src`

Run commands from an activated environment after `python -m pip install -e .`. For module commands,
use `python -m src.train` and `python -m src.inference`, not direct execution of files under `src/`.

### CUDA is unavailable

The smoke config uses CPU. Check the installed PyTorch build with:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Training uses too much memory

Start with `configs/smoke.yaml`. For larger runs, reduce `data.batch_size`, `data.train_samples`,
`model.embed_dim`, or `model.depth` in a copied config.

### There is no model file after cloning

No pretrained checkpoint is distributed. Run the smoke training command or train with a custom
configuration before using inference or a local interface.
