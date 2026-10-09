from collections import deque
import argparse
from pathlib import Path
import time

import numpy as np

from robot_dog_vision import (
    RobotDogVision,
    RobotHardwareConfig,
    VelocityFeedback,
    register,
)
from robot_dog_vision.source import VideoFileSource
from robot_dog_vision.visualization import VisionVisualizer

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test RobotDogVision with a color video."
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        required=True,
        help="Path to the input video.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Directory containing target registration images.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory used to save registration and visualization results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device (current OpenVINO checkpoints use cpu).",
    )
    parser.add_argument(
        "--reid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable person ReID.",
    )
    return parser.parse_args()


def test_video(
    video_path: str | Path,
    target_dir: str | Path,
    results_dir: str | Path,
    device: str = "cpu",
    enable_reid: bool = True,
) -> None:
    # 1. 注册目标
    target_feature_memory = register(
        image_dir=target_dir,
        results_dir=results_dir,
        device=device,
    )

    # 2. 占位硬件参数；视频没有深度，因此不会产生有效跟随速度
    hardware = RobotHardwareConfig(
        camera_to_robot_rotation=np.eye(3, dtype=np.float64),
        camera_to_robot_translation_mm=np.zeros(3, dtype=np.float64),
        max_positive_x_speed_mps=1.0,
        max_negative_x_speed_mps=0.5,
        max_y_speed_mps=0.5,
        max_yaw_rate_radps=1.0,
        max_x_accel_mps2=1.0,
        max_y_accel_mps2=1.0,
        max_yaw_accel_radps2=1.0,
    )

    # 3. 创建视频帧源
    source = VideoFileSource(
        name=Path(video_path).stem,
        video_path=video_path,
    )

    # 4. 创建视觉系统，所有模型均使用 CPU
    vision = RobotDogVision(
        target_feature_memory=target_feature_memory,
        hardware=hardware,
        intrinsics=source.intrinsics,
        device=device,
        enable_reid=enable_reid,
    )

    visualizer = VisionVisualizer(
        output_path=Path(results_dir) / "visualization.mp4",
        fps=source.fps,
    )

    processed_frames = 0
    processing_times = deque(maxlen=30)

    try:
        source.open()

        while True:
            frame = source.read()
            if frame is None:
                break

            # 离线视频测试假设机器狗保持静止
            velocity_feedback = VelocityFeedback(
                vx=0.0,
                vy=0.0,
                wz=0.0,
                timestamp_ms=frame.timestamp_ms,
                valid=True,
            )

            processing_start = time.perf_counter()
            target_bbox, pose, velocity_command = vision.process_frame(
                rgbd_frame=frame,
                velocity_feedback=velocity_feedback,
            )
            processing_time = time.perf_counter() - processing_start
            processing_times.append(processing_time)
            average_processing_time = sum(processing_times) / len(processing_times)
            processing_fps = 1.0 / average_processing_time

            visualizer.draw(
                frame_bgr=frame.color_bgr,
                target_bbox=target_bbox,
                pose=pose,
                tracking_state=vision.target_tracker.state,
            )
            visualizer.save()

            print(
                f"frame={frame.frame_index}, "
                f"fps={processing_fps:.2f}, "
                f"latency={processing_time * 1000.0:.1f} ms, "
                f"state={vision.target_tracker.state}, "
                f"command=({velocity_command.vx:.3f}, "
                f"{velocity_command.vy:.3f}, "
                f"{velocity_command.wz:.3f}), "
                f"valid={velocity_command.valid}"
            )

            processed_frames += 1

            # if visualizer.show():
            #     break
    finally:
        source.close()
        vision.pose_estimator.close()
        visualizer.close()

    assert processed_frames > 0


if __name__ == "__main__":
    args = parse_args()
    test_video(
        video_path=args.video_path,
        target_dir=args.target_dir,
        results_dir=args.results_dir,
        device=args.device,
        enable_reid=args.reid,
    )
    print("Video RobotDogVision test passed.")
