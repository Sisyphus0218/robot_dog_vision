# Repository Guidelines

## Project Structure & Module Organization

Package code lives under `src/robot_dog_vision/`. The top-level pipeline is in `robot_dog_vision.py`; focused subpackages handle frame sources (`source/`), person and target tracking (`tracking/`), re-identification (`reid/`), pose and position estimation (`pose/`, `position/`), follow control (`follow_controller/`), and rendering (`visualization/`). Put reusable code in the appropriate subpackage and expose public APIs through its `__init__.py`.

`scripts/download_checkpoints.py` retrieves model weights. Local media, checkpoints, and generated output belong in `inputs/`, `checkpoints/`, and `results/`; these paths are ignored by Git. Integration scripts currently live in `tests/`.

## Build, Test, and Development Commands

- `uv sync --extra cpu` installs Python 3.12 dependencies with CPU PyTorch.
- `uv sync --extra cu130` installs the CUDA 13.0 PyTorch variant.
- `uv run python scripts/download_checkpoints.py` downloads YOLO, OSNet, and MediaPipe models and records their paths in `.env`.
- `uv build` creates wheel and source distributions.
- `uv run python tests/test_video.py` runs the offline video pipeline. Other `tests/test_*.py` scripts exercise registration or specific camera sources and may require local input files or hardware.

## Coding Style & Naming Conventions

Use four-space indentation, PEP 8 spacing, and type annotations for public interfaces. Follow existing names: `snake_case` for modules, functions, and variables; `PascalCase` for classes and enums; `UPPER_SNAKE_CASE` for constants. Keep hardware-specific acquisition behind the `source` interfaces and prefer `pathlib.Path` over manual path strings. No formatter or linter is configured; keep imports grouped as standard library, third-party, then local package imports.

## Testing Guidelines

Name new tests `test_<behavior>.py` and test functions `test_<behavior>`. Separate fast, deterministic unit tests from model-, media-, and hardware-dependent integration tests. Use fixtures or configurable paths instead of committing recordings or generated results. Always close frame sources, estimators, and visualizers in `finally` blocks. Document required devices and checkpoint setup in the test docstring.

## Commit & Pull Request Guidelines

This repository has no existing commit history, so use concise imperative subjects, optionally with a scope, such as `tracking: handle lost targets`. Keep each commit focused. Pull requests should explain the behavior change, list commands run, identify required hardware/models, and link relevant issues. Include representative screenshots or short output samples for visualization changes; never commit `.env`, checkpoints, input recordings, or generated results.
