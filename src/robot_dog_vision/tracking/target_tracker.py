from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from robot_dog_vision.reid import (
    calculate_similarity_scores,
    FeatureExtractor,
    FeatureMemory,
    TargetClassifier,
)
from robot_dog_vision.utils import BBox

from .person_tracker import TrackedPerson


@dataclass(frozen=True)
class TargetTrackerConfig:
    reid_search_threshold: float = 0.65
    reid_keep_threshold: float = 0.35
    reid_reject_threshold: float = 0.25

    reid_validation_interval: int = 5
    reid_validation_tolerance: int = 2
    missing_frame_tolerance: int = 3

    classifier_collect_interval: int = 1
    classifier_max_overlap_iou: float = 0.2

    min_negatives: int = 30
    min_positives: int = 20


class TargetTracker:
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        positive_feature_memory: FeatureMemory,
        negative_feature_memory: FeatureMemory,
        target_classifier: TargetClassifier,
        config: TargetTrackerConfig | None = None,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.positive_feature_memory = positive_feature_memory
        self.negative_feature_memory = negative_feature_memory
        self.target_classifier = target_classifier

        if config is None:
            config = TargetTrackerConfig()

        # ReID state-machine parameters.
        self.state = "SEARCHING"
        self.target_track_id: int | None = None

        self.reid_search_threshold = config.reid_search_threshold
        self.reid_keep_threshold = config.reid_keep_threshold
        self.reid_reject_threshold = config.reid_reject_threshold

        self.reid_validation_interval = config.reid_validation_interval
        self.reid_validation_tolerance = config.reid_validation_tolerance
        self.reid_validation_failures = 0

        self.missing_frame_tolerance = config.missing_frame_tolerance
        self.missing_frames = 0

        # target classifier
        self.classifier_collect_interval = config.classifier_collect_interval
        self.classifier_max_overlap_iou = config.classifier_max_overlap_iou
        self.min_negatives = config.min_negatives
        self.min_positives = config.min_positives

    def reset(self) -> None:
        """Reset state for a new tracking source."""
        self.state = "SEARCHING"
        self.target_track_id = None
        self.missing_frames = 0
        self.reid_validation_failures = 0
        self.positive_feature_memory.clear_short_term_memory()
        self.negative_feature_memory.clear_short_term_memory()
        self.target_classifier.reset()

    @staticmethod
    def crop_person(frame: np.ndarray, person: TrackedPerson) -> np.ndarray:
        """Copy one tracked person's BGR crop from the current frame."""
        bbox = person.bbox
        return frame[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2].copy()

    def update_feature_memory(
        self,
        frame: np.ndarray,
        other_persons: list[TrackedPerson],
        target: TrackedPerson,
        target_features: torch.Tensor,
    ) -> None:
        is_target_clean = True
        safe_negative_crops = []

        for person in other_persons:
            iou = target.bbox.iou(person.bbox)
            if iou <= self.classifier_max_overlap_iou:
                safe_negative_crops.append(self.crop_person(frame, person))
            else:
                is_target_clean = False

        if safe_negative_crops:
            negative_features = self.feature_extractor.extract_features(
                safe_negative_crops
            )
            self.negative_feature_memory.add_short_term_memory(negative_features)

        if is_target_clean:
            self.positive_feature_memory.add_short_term_memory(target_features)

    def update_target_classifier(
        self,
        frame_index: int,
    ) -> None:
        if frame_index % self.classifier_collect_interval != 0:
            return

        positive_count = self.positive_feature_memory.count
        negative_count = self.negative_feature_memory.count
        if positive_count < self.min_positives or negative_count < self.min_negatives:
            tqdm.write(
                f"Frame {frame_index}: not enough samples to update target classifier, "
                f"positives={positive_count}, negatives={negative_count}"
            )
            return

        self.target_classifier.fit(
            positive_features=self.positive_feature_memory.all_features(),
            negative_features=self.negative_feature_memory.all_features(),
        )

        tqdm.write(f"Frame {frame_index}: target classifier updated")

    def calculate_reid_scores(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, str]:
        """Score candidates using the trained classifier or feature similarity."""
        if self.target_classifier.is_trained:
            return self.target_classifier.predict(features), "target classifier"

        return (
            calculate_similarity_scores(
                query_features=features,
                target_feature_memory=self.positive_feature_memory,
                top_k=3,
            ),
            "feature similarity",
        )

    def find_best_candidate(
        self,
        features: torch.Tensor,
    ) -> tuple[int | None, float, str]:
        """Return the best candidate if its ReID score passes the threshold."""
        scores, method = self.calculate_reid_scores(features)
        best_index = int(torch.argmax(scores).item())
        best_score = float(scores[best_index].item())

        if best_score < self.reid_search_threshold:
            return None, best_score, method

        return best_index, best_score, method

    def search_target(
        self,
        frame: np.ndarray,
        tracked_persons: list[TrackedPerson],
        frame_index: int,
    ) -> TrackedPerson | None:
        if not tracked_persons:
            return None

        crops = [self.crop_person(frame, person) for person in tracked_persons]
        features = self.feature_extractor.extract_features(crops)

        best_index, best_score, _ = self.find_best_candidate(features.detach())

        if best_index is None:
            tqdm.write(
                f"Frame {frame_index}: no candidate passed the ReID threshold, "
                f"best score: {best_score:.4f}"
            )
            return None

        target = tracked_persons[best_index]

        self.state = "TRACKING"
        self.target_track_id = target.track_id
        self.reid_validation_failures = 0

        tqdm.write(
            f"Frame {frame_index}: target confirmed, "
            f"track id={self.target_track_id}, "
            f"reid score={best_score:.4f}"
        )

        return target

    def track_target(
        self,
        frame: np.ndarray,
        tracked_persons: list[TrackedPerson],
        frame_index: int,
    ) -> TrackedPerson | None:
        """Follow the locked ID and periodically validate its appearance."""
        # search the locked ID in the current frame
        target = None
        for index, person in enumerate(tracked_persons):
            if person.track_id == self.target_track_id:
                target = person
                target_index = index
                other_persons = (
                    tracked_persons[:target_index] + tracked_persons[target_index + 1 :]
                )
                break

        # the target disappears
        if target is None:
            tqdm.write(
                f"Frame {frame_index}: target ID missing, " f"switch to RECOVERING"
            )
            self.state = "RECOVERING"
            self.reid_validation_failures = 0
            return self.recover_target(
                frame=frame,
                tracked_persons=tracked_persons,
                frame_index=frame_index,
            )

        # validate the target's appearance every N frames
        if frame_index % self.reid_validation_interval != 0:
            return target
        else:  # validate
            # calculate reid score
            target_crop = self.crop_person(frame, target)
            target_features = self.feature_extractor.extract_features(target_crop)
            scores, _ = self.calculate_reid_scores(target_features.detach())
            score = float(scores[0].item())

            # validate target
            if score >= self.reid_keep_threshold:  # pass
                self.reid_validation_failures = 0
                self.update_feature_memory(
                    frame=frame,
                    other_persons=other_persons,
                    target=target,
                    target_features=target_features.detach(),
                )
                self.update_target_classifier(
                    frame_index=frame_index,
                )
            elif score < self.reid_reject_threshold:  # fail
                self.reid_validation_failures = self.reid_validation_tolerance
            else:  # supcious
                self.reid_validation_failures += 1

            if self.reid_validation_failures < self.reid_validation_tolerance:
                return target
            else:  # too many failures, switch to SEARCHING
                tqdm.write(
                    f"Frame {frame_index}: locked ID failed ReID validation, "
                    f"track id={target.track_id}, "
                    f"ReID score={score:.4f}"
                )
                self.state = "SEARCHING"
                self.target_track_id = None
                self.reid_validation_failures = 0
                return self.search_target(
                    frame=frame,
                    tracked_persons=tracked_persons,
                    frame_index=frame_index,
                )

    def recover_target(
        self,
        frame: np.ndarray,
        tracked_persons: list[TrackedPerson],
        frame_index: int,
    ) -> TrackedPerson | None:
        if self.missing_frames >= self.missing_frame_tolerance:
            self.state = "SEARCHING"
            self.target_track_id = None
            self.missing_frames = 0
            return self.search_target(
                frame=frame,
                tracked_persons=tracked_persons,
                frame_index=frame_index,
            )

        if not tracked_persons:
            self.missing_frames += 1
            return None

        for person in tracked_persons:
            if person.track_id == self.target_track_id:
                self.state = "TRACKING"
                self.missing_frames = 0
                tqdm.write(
                    f"Frame {frame_index}: recovered original track ID, "
                    f"track id={person.track_id}"
                )
                return person

        crops = [self.crop_person(frame, person) for person in tracked_persons]
        features = self.feature_extractor.extract_features(crops)
        best_index, best_score, recovery_method = self.find_best_candidate(
            features.detach()
        )
        if best_index is not None:
            target = tracked_persons[best_index]
            self.missing_frames = 0
            self.target_track_id = target.track_id
            self.state = "TRACKING"
            tqdm.write(
                f"Frame {frame_index}: recovered target with {recovery_method}, "
                f"track id={target.track_id}, ReID score={best_score:.4f}"
            )
            return target

        self.missing_frames += 1

        return None

    def update_target_state(
        self,
        frame: np.ndarray,
        tracked_persons: list[TrackedPerson],
        frame_index: int,
    ) -> TrackedPerson | None:
        """
        SEARCHING:
            Look at all detected people and select the one that best matches
            the registered target.

        TRACKING:
            Follow the selected person's track ID and periodically check that
            their appearance still matches the registered target.

        RECOVERING:
            When the selected track ID disappears, try to find the target again.
        """
        if self.state == "SEARCHING":
            return self.search_target(frame, tracked_persons, frame_index)

        if self.state == "TRACKING":
            return self.track_target(frame, tracked_persons, frame_index)

        if self.state == "RECOVERING":
            return self.recover_target(frame, tracked_persons, frame_index)

        raise RuntimeError(f"Unknown tracking state: {self.state}")

    def track(
        self,
        frame_bgr: np.ndarray,
        tracked_persons: list[TrackedPerson],
        frame_index: int,
    ) -> BBox | None:
        """Update tracking state and return the target box, or None if absent."""
        target = self.update_target_state(
            frame=frame_bgr,
            tracked_persons=tracked_persons,
            frame_index=frame_index,
        )

        return target.bbox if target is not None else None
