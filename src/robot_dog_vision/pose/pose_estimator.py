from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from robot_dog_vision.utils import BBox, Point2D

MEDIAPIPE_POSE_KEYPOINT_NAMES = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)


@dataclass(frozen=True)
class Keypoint2D:
    name: str
    position_2d: Point2D
    confidence: float


class PoseEstimator:
    def __init__(
        self,
        model_path: str | Path,
    ) -> None:
        self.model_path = Path(model_path)

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def estimate(
        self,
        frame_bgr: np.ndarray,
        bbox: BBox | None,
    ) -> dict[str, Keypoint2D] | None:
        if bbox is None:
            return None

        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must have shape (height, width, 3)")

        frame_height, frame_width = frame_bgr.shape[:2]
        x1 = int(np.clip(bbox.x1, 0, frame_width))
        x2 = int(np.clip(bbox.x2, 0, frame_width))
        y1 = int(np.clip(bbox.y1, 0, frame_height))
        y2 = int(np.clip(bbox.y2, 0, frame_height))

        if x2 <= x1 or y2 <= y1:
            return None

        crop_bgr = frame_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(crop_rgb),
        )
        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        crop_height, crop_width = crop_bgr.shape[:2]
        landmarks = result.pose_landmarks[0]
        keypoints: dict[str, Keypoint2D] = {}

        for name, landmark in zip(MEDIAPIPE_POSE_KEYPOINT_NAMES, landmarks):
            confidence = min(landmark.visibility, landmark.presence)

            u = x1 + float(landmark.x) * crop_width
            v = y1 + float(landmark.y) * crop_height
            u = float(np.clip(u, 0, frame_width - 1))
            v = float(np.clip(v, 0, frame_height - 1))

            keypoints[name] = Keypoint2D(
                name=name,
                position_2d=Point2D(u, v),
                confidence=confidence,
            )

        if not keypoints:
            return None

        return keypoints

    def close(self) -> None:
        self._landmarker.close()
