# Robot Dog Vision

A vision pipeline for target-person registration and tracking, pose and 3D position estimation, and robot-dog following control.

## Installation

Clone the repository.

```bash
git clone https://github.com/Sisyphus0218/robot_dog_vision.git
cd robot_dog_vision
```

Create the virtual environment and install the CPU dependencies.

```bash
uv sync --extra cpu
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

## Testing

### Video File

Run the vision pipeline on a color video:

```powershell
uv run python tests/test_video.py `
  --video-path "PATH_TO_VIDEO" `
  --target-dir "PATH_TO_TARGET_IMAGES" `
  --results-dir "PATH_TO_RESULTS" `
  --device cpu `
  --reid
```

To run without ReID:

```powershell
uv run python tests/test_video.py `
  --video-path "PATH_TO_VIDEO" `
  --target-dir "PATH_TO_TARGET_IMAGES" `
  --results-dir "PATH_TO_RESULTS" `
  --device cpu `
  --no-reid
```

The annotated video is saved as `visualization.mp4` in the selected results directory. A color video does not provide depth, so detection, tracking, ReID, and pose estimation can be evaluated, but the generated follow command is not valid for 3D control.

### RealSense DB3

Run the vision pipeline on a recorded RealSense ROS 2 DB3 bag:

```powershell
uv run python tests/test_db3.py `
  --bag-path "PATH_TO_DB3" `
  --target-dir "PATH_TO_TARGET_IMAGES" `
  --results-dir "PATH_TO_RESULTS" `
  --device cpu `
  --reid
```

To run without ReID:

```powershell
uv run python tests/test_db3.py `
  --bag-path "PATH_TO_DB3" `
  --target-dir "PATH_TO_TARGET_IMAGES" `
  --results-dir "PATH_TO_RESULTS" `
  --device cpu `
  --no-reid
```

The annotated recording is saved as `visualization.mp4` in the selected results directory.
