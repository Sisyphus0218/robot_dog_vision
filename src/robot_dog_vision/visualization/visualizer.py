from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from robot_dog_vision.pose import Keypoint2D
from robot_dog_vision.utils import BBox

POSE_CONNECTIONS = (
    ("left_ear", "left_eye_outer"),
    ("left_eye_outer", "left_eye"),
    ("left_eye", "left_eye_inner"),
    ("left_eye_inner", "nose"),
    ("nose", "right_eye_inner"),
    ("right_eye_inner", "right_eye"),
    ("right_eye", "right_eye_outer"),
    ("right_eye_outer", "right_ear"),
    ("mouth_left", "mouth_right"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_pinky"),
    ("left_wrist", "left_index"),
    ("left_wrist", "left_thumb"),
    ("left_pinky", "left_index"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_pinky"),
    ("right_wrist", "right_index"),
    ("right_wrist", "right_thumb"),
    ("right_pinky", "right_index"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("left_ankle", "left_foot_index"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
    ("right_ankle", "right_foot_index"),
)


class VisionVisualizer:
    def __init__(
        self,
        output_path: str | Path | None = None,
        fps: float = 30.0,
        window_name: str = "Robot Dog Vision",
        confidence_threshold: float = 0.5,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

        self.output_path = Path(output_path) if output_path is not None else None
        self.fps = float(fps)
        self.window_name = window_name
        self.confidence_threshold = confidence_threshold
        self.current_frame: np.ndarray | None = None
        self._video_writer: cv2.VideoWriter | None = None

    def draw(
        self,
        frame_bgr: np.ndarray,
        target_bbox: BBox | None,
        pose: dict[str, Keypoint2D] | None,
        tracking_state: str | None = None,
    ) -> np.ndarray:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must have shape (height, width, 3)")

        frame = frame_bgr.copy()
        if target_bbox is not None:
            cv2.rectangle(
                frame,
                (target_bbox.x1, target_bbox.y1),
                (target_bbox.x2, target_bbox.y2),
                (0, 255, 0),
                3,
            )
            cv2.putText(
                frame,
                "TARGET",
                (target_bbox.x1, max(25, target_bbox.y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if pose is not None:
            self._draw_pose(frame, pose)

        if tracking_state is not None:
            cv2.putText(
                frame,
                f"State: {tracking_state}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        self.current_frame = frame
        return frame

    def _draw_pose(
        self,
        frame: np.ndarray,
        pose: dict[str, Keypoint2D],
    ) -> None:
        for first_name, second_name in POSE_CONNECTIONS:
            first = pose.get(first_name)
            second = pose.get(second_name)
            if not self._is_visible(first) or not self._is_visible(second):
                continue
            cv2.line(
                frame,
                self._point(first),
                self._point(second),
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        for keypoint in pose.values():
            if self._is_visible(keypoint):
                cv2.circle(
                    frame,
                    self._point(keypoint),
                    4,
                    (0, 0, 255),
                    -1,
                    cv2.LINE_AA,
                )

    def _is_visible(self, keypoint: Keypoint2D | None) -> bool:
        return keypoint is not None and keypoint.confidence >= self.confidence_threshold

    @staticmethod
    def _point(keypoint: Keypoint2D) -> tuple[int, int]:
        return (
            int(round(keypoint.position_2d.u)),
            int(round(keypoint.position_2d.v)),
        )

    def show(self, delay_ms: int = 1) -> bool:
        frame = self._require_current_frame()
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(delay_ms) & 0xFF
        return key in (ord("q"), 27)

    def save(self) -> None:
        if self.output_path is None:
            return

        frame = self._require_current_frame()
        if self._video_writer is None:
            self._video_writer = self._create_video_writer(frame)
        self._video_writer.write(frame)

    def _create_video_writer(self, frame: np.ndarray) -> cv2.VideoWriter:
        assert self.output_path is not None
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Unable to create video: {self.output_path}")
        return writer

    def _require_current_frame(self) -> np.ndarray:
        if self.current_frame is None:
            raise RuntimeError("draw() must be called before show() or save()")
        return self.current_frame

    def close(self) -> None:
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        cv2.destroyAllWindows()
