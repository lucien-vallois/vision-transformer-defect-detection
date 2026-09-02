# Local usage

This project supports local training, evaluation, inference, and optional browser interfaces. It
does not provide or validate a production deployment.

## Minimal end-to-end check

```bash
python -m src.train --config configs/smoke.yaml
python -m scripts.generate_synthetic_data --num-samples 20 --img-size 32 --num-base-components 2 --num-workers 1 --output-dir data/smoke
python -m src.inference --model-path experiments/smoke/checkpoints/best_model.pth --image-path data/smoke/images/train_000000_ok.png --output experiments/smoke/prediction.json
```

This proves that the local code path runs; it does not establish useful accuracy. The smoke model
is trained briefly on generated images and its prediction is not meaningful for real inspection.

## Train from generated files

Generate the dataset, copy `configs/smoke.yaml`, set `data.root_dir` to the generated directory, and
set `data.generate_on_fly` to `false`. The loader reads the three split annotation files from the
dataset root. Generate into a new or empty destination; the command refuses to overwrite an
existing generated dataset.

## Train from another dataset

Convert the images and annotations to the documented JSON contract. The built-in training path is
fixed to the five supported classes and their documented order.

## Local interfaces

After installing `.[demo]`, run either interface from the repository root. Load the checkpoint in
the interface before submitting images. Gradio listens only on the loopback interface by default.
