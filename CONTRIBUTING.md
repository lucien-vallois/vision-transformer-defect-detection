# Contributing

Contributions should keep the repository runnable from a clean clone and should not add benchmark
claims without the dataset manifest, checkpoint digest, configuration, hardware, and raw result
artifact needed to reproduce them.

## Development setup

```bash
python -m venv .venv
```

Use `.\.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on Linux and
macOS.

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Before opening a pull request

```bash
python -m compileall -q src scripts examples demo
python -m pytest -q
python examples/quick_start.py
```

If packaging changed, also verify a non-editable install from outside the checkout.

## Pull requests

Include:

- the problem and the chosen scope;
- tests or a reproducible check;
- user-facing documentation changes;
- compatibility or checkpoint-format impact;
- evidence for any accuracy, latency, or resource claim.

Keep generated datasets, checkpoints, experiment logs, and local environments out of Git. Never
commit credentials or checkpoints from an untrusted source.
