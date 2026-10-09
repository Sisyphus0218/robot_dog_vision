# Robot Dog Vision

A vision pipeline for target-person registration and tracking, pose and 3D position estimation, and robot-dog following control.

## Installation

Clone the repository.

```bash
git clone https://github.com/Sisyphus0218/robot_dog_vision.git
cd robot_dog_vision
```

Choose a command to create the virtual environment and install the dependencies.

```bash
# CPU
uv sync --extra cpu

# NVIDIA GPU with CUDA 13.0
uv sync --extra cu130
```

## Download Checkpoints

Download the model checkpoints.

```bash
uv run python scripts/download_checkpoints.py
```

| Model | Purpose | Source |
| --- | --- | --- |
| YOLO26-N | Person detection and tracking | Ultralytics |
| OSNet x0.25, trained on MSMT17 | Appearance feature extraction | OSNet |
| MediaPipe Pose Landmarker Full | Pose estimation | MediaPipe |

The checkpoints are saved under `checkpoints/`. The script saves the checkpoint paths to `.env`, which is used by the package to locate the models.
