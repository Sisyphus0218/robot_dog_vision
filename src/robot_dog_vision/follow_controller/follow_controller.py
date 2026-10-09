from dataclasses import dataclass
import math

import numpy as np

from robot_dog_vision.utils import Point3D

from .robot_velocity import VelocityCommand, VelocityFeedback


@dataclass(frozen=True)
class RobotHardwareConfig:
    camera_to_robot_rotation: np.ndarray
    camera_to_robot_translation_mm: np.ndarray

    max_positive_x_speed_mps: float
    max_negative_x_speed_mps: float
    max_y_speed_mps: float
    max_yaw_rate_radps: float

    max_x_accel_mps2: float
    max_y_accel_mps2: float
    max_yaw_accel_radps2: float


@dataclass
class FollowSettings:
    desired_distance_m: float = 1.0

    gain_x: float = 0.8
    gain_y: float = 0.8
    gain_yaw: float = 1.2

    x_deadband_m: float = 0.05
    y_deadband_m: float = 0.05
    yaw_deadband_rad: float = 0.03

    position_filter_alpha: float = 0.7
    velocity_filter_alpha: float = 0.7

    max_target_speed_mps: float = 1.5

    timeout_threshold: float = 1.0


class FollowController:
    """camera-frame human position (mm) --> velocity command."""

    def __init__(
        self,
        hardware: RobotHardwareConfig,
        settings: FollowSettings | None = None,
    ) -> None:
        self.hardware = hardware
        self.settings = settings or FollowSettings()

        # state
        self.last_valid_timestamp_ms: float | None = None
        self.last_command = np.zeros(3, dtype=np.float64)
        self.last_human_position: np.ndarray | None = None
        self.last_human_velocity = np.zeros(2, dtype=np.float64)

    def reset(self) -> None:
        self.last_valid_timestamp_ms = None
        self.last_human_position = None
        self.last_command.fill(0.0)
        self.last_human_velocity.fill(0.0)

    def update(
        self,
        camera_space_human_position: Point3D | None,
        velocity_feedback: VelocityFeedback | None,
        timestamp_ms: float,
    ) -> VelocityCommand:
        hardware = self.hardware
        settings = self.settings

        if velocity_feedback is None:
            return VelocityCommand(0.0, 0.0, 0.0, valid=False)

        if self.last_valid_timestamp_ms is not None:
            elapsed = (timestamp_ms - self.last_valid_timestamp_ms) / 1000.0
            if elapsed >= settings.timeout_threshold:
                self.reset()

        if camera_space_human_position is None:
            if self.last_valid_timestamp_ms is None:
                return VelocityCommand(0.0, 0.0, 0.0, valid=False)

            return VelocityCommand(
                vx=float(self.last_command[0]),
                vy=float(self.last_command[1]),
                wz=float(self.last_command[2]),
                valid=True,
            )

        robot_space_human_position = camera_space_human_position.transform(
            hardware.camera_to_robot_rotation,
            hardware.camera_to_robot_translation_mm,
        )
        # Use robot-frame horizontal coordinates in meters.
        human_position = (
            np.asarray(
                [robot_space_human_position.x, robot_space_human_position.y],
                dtype=np.float64,
            )
            / 1000.0
        )

        if self.last_valid_timestamp_ms is None:
            self.last_valid_timestamp_ms = timestamp_ms
            self.last_human_position = human_position.copy()
            return VelocityCommand(0.0, 0.0, 0.0, valid=True)

        dt = (timestamp_ms - self.last_valid_timestamp_ms) / 1000.0
        if dt <= 0.0:
            return VelocityCommand(0.0, 0.0, 0.0, valid=False)

        # Filter position.
        human_position = self.filter(
            human_position,
            self.last_human_position,
            settings.position_filter_alpha,
        )

        # Estimate velocity.
        human_velocity = self.estimate_human_velocity(
            human_position=human_position,
            velocity_feedback=velocity_feedback,
            dt=dt,
        )
        human_velocity = self.filter(
            human_velocity,
            self.last_human_velocity,
            settings.velocity_filter_alpha,
        )

        x, y = human_position[:2]

        heading_error = math.atan2(y, x)
        forward_error = x - settings.desired_distance_m

        last_vx, last_vy, last_wz = self.last_command
        max_vx_change = hardware.max_x_accel_mps2 * dt
        max_vy_change = hardware.max_y_accel_mps2 * dt
        max_wz_change = hardware.max_yaw_accel_radps2 * dt

        # wz = K_yaw * atan2(y, x)
        raw_wz = (
            0.0
            if abs(heading_error) <= settings.yaw_deadband_rad
            else settings.gain_yaw * heading_error
        )
        clipped_wz = float(
            np.clip(
                raw_wz,
                -hardware.max_yaw_rate_radps,
                hardware.max_yaw_rate_radps,
            )
        )  # limit velocity
        wz = float(
            np.clip(
                clipped_wz,
                last_wz - max_wz_change,
                last_wz + max_wz_change,
            )
        )  # limit acceleration

        # v_cmd = v_h - wz_cmd * J * p + K * (p - p_target)

        # vx = v_hx + wz_cmd * y + K_x * (x - x_target)
        correction_x = (
            0.0
            if abs(forward_error) <= settings.x_deadband_m
            else settings.gain_x * forward_error
        )
        raw_vx = float(human_velocity[0]) + wz * y + correction_x
        clipped_vx = float(
            np.clip(
                raw_vx,
                -hardware.max_negative_x_speed_mps,
                hardware.max_positive_x_speed_mps,
            )
        )
        vx = float(
            np.clip(
                clipped_vx,
                last_vx - max_vx_change,
                last_vx + max_vx_change,
            )
        )

        # vy_raw = v_hy - wz_cmd * x + K_y * (y - y_target)
        correction_y = 0.0 if abs(y) <= settings.y_deadband_m else settings.gain_y * y
        raw_vy = float(human_velocity[1]) - wz * x + correction_y
        clipped_vy = float(
            np.clip(
                raw_vy,
                -hardware.max_y_speed_mps,
                hardware.max_y_speed_mps,
            )
        )
        vy = float(
            np.clip(
                clipped_vy,
                last_vy - max_vy_change,
                last_vy + max_vy_change,
            )
        )

        command = np.asarray([vx, vy, wz], dtype=np.float64)

        self.last_valid_timestamp_ms = timestamp_ms
        self.last_command = command
        self.last_human_velocity = human_velocity.copy()
        self.last_human_position = human_position.copy()

        return VelocityCommand(
            vx=vx,
            vy=vy,
            wz=wz,
            valid=True,
        )

    def estimate_human_velocity(
        self,
        human_position: np.ndarray,
        velocity_feedback: VelocityFeedback,
        dt: float,
    ) -> np.ndarray:
        """
        Estimate target world velocity, expressed in the current body frame.
        """
        # Estimate velocity
        relative_velocity = (human_position[:2] - self.last_human_position[:2]) / dt
        rotation_velocity = velocity_feedback.wz * np.asarray(
            [-human_position[1], human_position[0]]
        )
        human_velocity = (
            relative_velocity
            + np.asarray([velocity_feedback.vx, velocity_feedback.vy])
            + rotation_velocity
        )

        # Limit velocity
        speed = float(np.linalg.norm(human_velocity))
        max_speed = self.settings.max_target_speed_mps
        if speed > max_speed:
            human_velocity *= max_speed / speed

        return human_velocity

    def filter(
        self,
        current_value: np.ndarray,
        last_value: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        return alpha * current_value + (1.0 - alpha) * last_value
