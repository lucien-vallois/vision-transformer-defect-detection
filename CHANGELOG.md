# Changelog

This project uses the structure from [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Changed

- Reframed the repository as a reusable, train-first implementation with explicit limitations.
- Added package boundaries and a build-system declaration.
- Added a small CPU smoke configuration.
- Aligned generated annotations, checkpoint metadata, inference output, and local interfaces.
- Made synthetic generation and validation/test sampling deterministic for a fixed seed.
- Added guarded checkpoint resume, configuration validation, and regression coverage for local UIs.
- Made model comparison require real checkpoints instead of ranking random initializations.
- Reduced ONNX export to a checked PyTorch-to-runtime parity path.
- Reduced core dependencies and separated optional feature groups.
- Consolidated continuous integration into one workflow with wheel and entry-point verification.

### Removed

- Unverifiable benchmark and production-readiness claims.
- Unsupported TensorRT claims and unvalidated deployment recipes.
- The model download path, because no verified pretrained asset is distributed.

## 0.1.0 - 2024-11-22

- Initial source import with a ViT implementation, synthetic data helpers, training, inference,
  examples, and tests.
