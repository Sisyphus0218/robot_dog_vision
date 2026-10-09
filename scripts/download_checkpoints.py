import argparse
from pathlib import Path
from urllib.request import urlretrieve

import openvino as ov
import torch
from dotenv import set_key
from huggingface_hub import hf_hub_download
from torchreid.reid import models
from torchreid.reid.utils import load_pretrained_weights
from tqdm import tqdm
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_DIR / ".env"
DEFAULT_CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download checkpoints for YOLO, OSNet, and MediaPipe."
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default=DEFAULT_CHECKPOINTS_DIR,
        help="Directory to save the downloaded checkpoints.",
    )
    return parser.parse_args()


def download_checkpoints(checkpoints_dir: str | Path) -> None:
    checkpoints_dir = Path(checkpoints_dir).expanduser().resolve()
    print(f"Checkpoints directory: {checkpoints_dir}")
    print()

    # YOLO
    print("Preparing YOLO26-N...")
    yolo_path = checkpoints_dir / "yolo" / "yolo26n.pt"
    yolo_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(yolo_path)
    yolo_model_path = Path(
        model.export(format="openvino", imgsz=512, device="cpu")
    ).resolve()
    print(f"YOLO26-N is ready: {yolo_model_path}")

    # OSNet
    print("\nPreparing OSNet...")
    osnet_path = Path(
        hf_hub_download(
            repo_id="kaiyangzhou/osnet",
            filename=(
                "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_"
                "lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
            ),
            local_dir=checkpoints_dir / "osnet",
        )
    ).resolve()

    osnet_model = models.build_model(
        name="osnet_x0_25",
        num_classes=1000,
        loss="softmax",
        pretrained=False,
        use_gpu=False,
    )
    load_pretrained_weights(osnet_model, str(osnet_path))
    osnet_model.eval()

    dummy_input = torch.randn(1, 3, 256, 128)
    openvino_osnet = ov.convert_model(osnet_model, example_input=dummy_input)
    openvino_osnet.reshape(
        {openvino_osnet.input(0): ov.PartialShape([-1, 3, 256, 128])}
    )

    osnet_model_path = (
        checkpoints_dir / "osnet" / "osnet_x0_25_openvino" / "osnet_x0_25.xml"
    )
    osnet_model_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(openvino_osnet, osnet_model_path, compress_to_fp16=False)
    print(f"OSNet is ready: {osnet_model_path}")

    # MediaPipe
    print("\nDownloading MediaPipe...")
    mediapipe_path = checkpoints_dir / "mediapipe" / "pose_landmarker.task"
    mediapipe_path.parent.mkdir(parents=True, exist_ok=True)
    if not mediapipe_path.is_file():
        partial = mediapipe_path.with_suffix(".part")
        with tqdm(
            desc="MediaPipe", unit="B", unit_scale=True, unit_divisor=1024
        ) as progress:

            def report_progress(
                block_count: int, block_size: int, total_size: int
            ) -> None:
                if total_size > 0:
                    progress.total = total_size
                downloaded = block_count * block_size
                if total_size > 0:
                    downloaded = min(downloaded, total_size)
                progress.update(downloaded - progress.n)

            urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_full/float16/1/pose_landmarker_full.task",
                partial,
                reporthook=report_progress,
            )
        partial.replace(mediapipe_path)

    if not mediapipe_path.is_file():
        raise FileNotFoundError(
            f"MediaPipe checkpoint was not created: {mediapipe_path}"
        )
    print(f"MediaPipe is ready: {mediapipe_path}")

    ENV_PATH.touch(exist_ok=True)
    model_paths = {
        "CHECKPOINTS_DIR": checkpoints_dir,
        "YOLO_MODEL_PATH": yolo_model_path,
        "OSNET_MODEL_PATH": osnet_model_path,
        "POSE_MODEL_PATH": mediapipe_path,
    }
    for key, path in model_paths.items():
        set_key(
            dotenv_path=ENV_PATH,
            key_to_set=key,
            value_to_set=str(path),
            quote_mode="auto",
        )

    print("\nAll checkpoints are ready.")
    print(f"Model paths saved to {ENV_PATH}")


if __name__ == "__main__":
    args = parse_args()
    download_checkpoints(args.checkpoints_dir)
