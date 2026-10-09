from pathlib import Path

import cv2
from dotenv import load_dotenv
import numpy as np
from ultralytics import YOLO

from robot_dog_vision.model_path import load_model_path
from robot_dog_vision.reid import FeatureExtractor, FeatureMemory


def register(
    image_dir: str | Path,
    results_dir: str | Path,
    device: str = "cpu",
) -> FeatureMemory:
    """Register a target and return its feature memory."""
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {image_dir}")

    load_dotenv()

    # ---------- Detect Target ----------
    target_detector_model_path = load_model_path("YOLO_MODEL_PATH")
    target_detector = YOLO(target_detector_model_path)
    predict_kwargs = {
        "source": str(image_dir),
        "classes": [0],  # Class 0 in COCO is person.
        "conf": 0.5,
        "device": device,
        "verbose": False,
    }
    results = target_detector.predict(**predict_kwargs)

    # ---------- Crop Target ----------
    crops_dir = Path(results_dir) / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    crops: list[np.ndarray] = []
    for result in results:  # results for all images, result for a single image
        if result.boxes is None or len(result.boxes) == 0:
            print(f"Skipped {Path(result.path).name}: No person detected.")
            continue

        # Find the person with the highest confidence score.
        target_index = int(result.boxes.conf.argmax())
        box = result.boxes.xyxy[target_index].cpu().numpy()

        # Crop the detected person and save it to the target_crop_dir
        image = result.orig_img  # bgr
        height, width = image.shape[:2]

        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1 = int(np.clip(x1, 0, width - 1))
        y1 = int(np.clip(y1, 0, height - 1))
        x2 = int(np.clip(x2, 0, width))
        y2 = int(np.clip(y2, 0, height))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2].copy()  # bgr

        crop_path = crops_dir / Path(result.path).name
        if not cv2.imwrite(crop_path, crop):
            raise RuntimeError(f"Failed to save cropped image: {crop_path}")

        crops.append(crop)

    if not crops:
        raise RuntimeError(f"Registration failed. No persons detected in {image_dir}.")

    # ---------- Extract Features ----------
    feature_extractor_model_path = load_model_path("OSNET_MODEL_PATH")
    feature_extractor = FeatureExtractor(
        model_name="osnet_x0_25",
        model_path=feature_extractor_model_path,
        feature_dim=512,
        device=device,
    )
    features = feature_extractor.extract_features(crops)

    # ---------- Save Features ----------
    target_feature_memory = FeatureMemory(
        feature_dim=512,
        short_term_capacity=100,
        long_term_capacity=500,
        device=device,
    )
    target_feature_memory.add_long_term_memory(features)

    print("Registration successful.")

    return target_feature_memory
