from enum import Enum

import numpy as np
from dotenv import load_dotenv

from robot_dog_vision.model_path import load_model_path
from robot_dog_vision.source import RGBDFrame
from robot_dog_vision.reid import FeatureExtractor, FeatureMemory, TargetClassifier
from robot_dog_vision.tracking import PersonTracker, TargetTracker
from robot_dog_vision.pose import PoseEstimator, Keypoint2D
from robot_dog_vision.position import PositionEstimator
from robot_dog_vision.follow_controller import (
    FollowController,
    RobotHardwareConfig,
    VelocityCommand,
    VelocityFeedback,
)


class ControlMode(Enum):
    HUMAN_FOLLOW = "human_follow"
    POSE_RECOGNITION = "pose_recognition"


class RobotDogVision:
    def __init__(
        self,
        target_feature_memory: FeatureMemory,
        hardware: RobotHardwareConfig,
        intrinsics: np.ndarray | None = None,
        mode: ControlMode = ControlMode.HUMAN_FOLLOW,
        device: str = "cpu",
        enable_reid: bool = True,
    ) -> None:
        load_dotenv()
        yolo_model_path = load_model_path("YOLO_MODEL_PATH")
        osnet_model_path = load_model_path("OSNET_MODEL_PATH")
        pose_model_path = load_model_path("POSE_MODEL_PATH")

        feature_dim = target_feature_memory.feature_dim
        feature_extractor = FeatureExtractor(
            model_path=osnet_model_path,
            feature_dim=feature_dim,
            device=device,
        )
        negative_feature_memory = FeatureMemory(
            feature_dim=feature_dim,
            short_term_capacity=100,
            long_term_capacity=500,
            device=device,
        )
        target_classifier = TargetClassifier(
            feature_dim=feature_dim,
            device=device,
        )

        self.person_tracker = PersonTracker(
            model_path=yolo_model_path,
            device=device,
        )
        self.target_tracker = TargetTracker(
            feature_extractor=feature_extractor,
            positive_feature_memory=target_feature_memory,
            negative_feature_memory=negative_feature_memory,
            target_classifier=target_classifier,
            config=None,
            enable_reid=enable_reid,
        )
        self.pose_estimator = PoseEstimator(model_path=pose_model_path)

        self.position_estimator = PositionEstimator(intrinsics=intrinsics)
        self.follow_controller = FollowController(hardware=hardware)

        self.mode = mode

    def process_frame(
        self,
        rgbd_frame: RGBDFrame,
        velocity_feedback: VelocityFeedback | None = None,
    ):
        persons = self.person_tracker.track(rgbd_frame.color_bgr)
        target_bbox = self.target_tracker.track(
            rgbd_frame.color_bgr,
            persons,
            rgbd_frame.frame_index,
        )
        pose = self.pose_estimator.estimate(rgbd_frame.color_bgr, target_bbox)

        if self.mode is ControlMode.HUMAN_FOLLOW:
            velocity_command = self.follow_human(
                rgbd_frame=rgbd_frame,
                pose=pose,
                velocity_feedback=velocity_feedback,
            )
        elif self.mode is ControlMode.POSE_RECOGNITION:
            velocity_command = self.recognize_pose(
                rgbd_frame=rgbd_frame,
                pose=pose,
                velocity_feedback=velocity_feedback,
            )
        else:
            raise ValueError(f"Unknown control mode: {self.mode}")

        return target_bbox, pose, velocity_command

    def follow_human(
        self,
        rgbd_frame: RGBDFrame,
        pose: dict[str, Keypoint2D] | None,
        velocity_feedback: VelocityFeedback | None,
    ) -> VelocityCommand:
        position = self.position_estimator.estimate(rgbd_frame.depth_mm, pose)
        velocity_command = self.follow_controller.update(
            position,
            velocity_feedback,
            rgbd_frame.timestamp_ms,
        )
        return velocity_command

    def recognize_pose(
        self,
        rgbd_frame: RGBDFrame,
        pose: dict[str, Keypoint2D] | None,
        velocity_feedback: VelocityFeedback | None,
    ) -> VelocityCommand:
        # TODO: Implement pose recognition logic here
        raise NotImplementedError("POSE_RECOGNITION mode is not implemented yet")
