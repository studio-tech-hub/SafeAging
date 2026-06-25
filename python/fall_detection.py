"""
Fall Detection Module
Detects falls using velocity, angle change, aspect ratio, and position tracking.
Integrates with person tracking system.
"""

import math
import logging
from typing import Dict, Tuple, List, Optional
from collections import deque

logger = logging.getLogger(__name__)


class FallDetectionTracker:
    """
    Tracks person motion and detects fall events.
    Uses smoothed position/angle history to avoid false positives.
    """

    def __init__(
        self,
        person_id: int,
        velocity_threshold: float = 20.0,
        angle_change_threshold: float = 45.0,
        aspect_ratio_threshold: float = 1.5,
        position_history_size: int = 5,
        confidence_threshold: float = 0.8,
        confirm_frames: int = 3,
        velocity_ref_height: float = 120.0,
        pose_torso_angle_threshold: float = 55.0,
    ):
        self.person_id = person_id
        self.velocity_threshold = velocity_threshold
        self.angle_change_threshold = angle_change_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.position_history_size = position_history_size
        self.confidence_threshold = confidence_threshold
        self.confirm_frames = max(1, int(confirm_frames))
        self.velocity_ref_height = max(20.0, float(velocity_ref_height))
        self.pose_torso_angle_threshold = max(10.0, float(pose_torso_angle_threshold))

        self.position_history = deque(maxlen=position_history_size)
        self.angle_history = deque(maxlen=position_history_size)
        self.aspect_ratio_history = deque(maxlen=position_history_size)
        self.fall_detected = False
        self.fall_frame_count = 0
        self.consecutive_fall_signals = 0
        self.last_detection_frame = 0

    def _raw_fall_signal(
        self,
        velocity: float,
        angle_change: float,
        avg_aspect_ratio: float,
        aspect_ratio: float,
        bbox_height: float,
    ) -> Tuple[bool, List[str]]:
        eff_velocity = self.velocity_threshold * max(0.5, bbox_height / self.velocity_ref_height)
        reasons: List[str] = []
        is_falling = False

        if velocity > eff_velocity:
            is_falling = True
            reasons.append(f"high_velocity({velocity:.1f}>{eff_velocity:.1f})")

        if angle_change > self.angle_change_threshold:
            is_falling = True
            reasons.append(f"angle_change({angle_change:.1f}°)")

        if avg_aspect_ratio > self.aspect_ratio_threshold:
            is_falling = True
            reasons.append(f"aspect_ratio({avg_aspect_ratio:.2f})")

        if len(self.aspect_ratio_history) >= 2:
            ratio_change = avg_aspect_ratio - self.aspect_ratio_history[0]
            if ratio_change > 0.5 and aspect_ratio > 1.2:
                is_falling = True
                reasons.append(f"ratio_increase({ratio_change:.2f})")

        return is_falling, reasons

    def update(
        self,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        frame_idx: int,
        pose_hint: Optional[Dict] = None,
    ) -> bool:
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1

        if height < 10 or width < 10:
            return False

        center_y = (y1 + y2) // 2
        aspect_ratio = width / height if height > 0 else 0
        angle = math.degrees(math.atan2(height, width))

        self.position_history.append(center_y)
        self.angle_history.append(angle)
        self.aspect_ratio_history.append(aspect_ratio)
        self.last_detection_frame = frame_idx

        if len(self.position_history) < 2:
            return False

        if confidence < self.confidence_threshold:
            self.consecutive_fall_signals = 0
            return False

        velocity = abs(self.position_history[-1] - self.position_history[-2])
        angle_change = abs(self.angle_history[-1] - self.angle_history[-2])
        avg_aspect_ratio = sum(self.aspect_ratio_history) / len(self.aspect_ratio_history)

        raw_fall, fall_reasons = self._raw_fall_signal(
            velocity, angle_change, avg_aspect_ratio, aspect_ratio, float(height)
        )

        if pose_hint is not None:
            torso_angle = pose_hint.get("torso_angle_from_vertical")
            if (
                torso_angle is not None
                and float(torso_angle) >= self.pose_torso_angle_threshold
            ):
                raw_fall = True
                fall_reasons.append(f"pose_torso({float(torso_angle):.1f}°)")

        if raw_fall:
            self.consecutive_fall_signals += 1
        else:
            self.consecutive_fall_signals = 0
            return False

        if self.consecutive_fall_signals < self.confirm_frames:
            return False

        self.fall_detected = True
        self.fall_frame_count += 1
        logger.info(
            "Fall confirmed for person %s (%d/%d frames): %s",
            self.person_id,
            self.consecutive_fall_signals,
            self.confirm_frames,
            ", ".join(fall_reasons),
        )
        return True

    def apply_pose_hint(self, pose_hint: Dict) -> bool:
        """Refine a pending bbox fall signal using pose on a person crop."""
        if self.fall_detected:
            return True
        torso_angle = pose_hint.get("torso_angle_from_vertical")
        if torso_angle is None or float(torso_angle) < self.pose_torso_angle_threshold:
            return False
        if self.consecutive_fall_signals < 1:
            return False
        self.consecutive_fall_signals += 1
        if self.consecutive_fall_signals < self.confirm_frames:
            return False
        self.fall_detected = True
        self.fall_frame_count += 1
        logger.info(
            "Fall confirmed via pose crop for person %s (torso=%.1f°)",
            self.person_id,
            float(torso_angle),
        )
        return True

    def is_stale(self, current_frame: int, ttl: int = 30) -> bool:
        return (current_frame - self.last_detection_frame) > ttl

    def reset_fall(self):
        self.fall_detected = False
        self.fall_frame_count = 0
        self.consecutive_fall_signals = 0


class FallDetectionManager:
    """Manages fall detection for multiple people."""

    def __init__(
        self,
        velocity_threshold: float = 20.0,
        angle_change_threshold: float = 45.0,
        aspect_ratio_threshold: float = 1.5,
        confidence_threshold: float = 0.8,
        tracker_ttl: int = 30,
        confirm_frames: int = 3,
        velocity_ref_height: float = 120.0,
        pose_torso_angle_threshold: float = 55.0,
    ):
        self.velocity_threshold = velocity_threshold
        self.angle_change_threshold = angle_change_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.confidence_threshold = confidence_threshold
        self.tracker_ttl = tracker_ttl
        self.confirm_frames = max(1, int(confirm_frames))
        self.velocity_ref_height = max(20.0, float(velocity_ref_height))
        self.pose_torso_angle_threshold = max(10.0, float(pose_torso_angle_threshold))

        self.trackers: Dict[int, FallDetectionTracker] = {}
        self.current_frame = 0

    def update(
        self,
        detections: List[Dict],
        frame_idx: int,
        pose_hints: Optional[Dict[int, Dict]] = None,
    ) -> Dict[int, bool]:
        self.current_frame = frame_idx
        fall_results = {}
        hints = pose_hints or {}

        for det in detections:
            track_id = det.get("track_id", -1)
            if track_id < 0:
                continue

            if track_id not in self.trackers:
                self.trackers[track_id] = FallDetectionTracker(
                    person_id=track_id,
                    velocity_threshold=self.velocity_threshold,
                    angle_change_threshold=self.angle_change_threshold,
                    aspect_ratio_threshold=self.aspect_ratio_threshold,
                    confidence_threshold=self.confidence_threshold,
                    confirm_frames=self.confirm_frames,
                    velocity_ref_height=self.velocity_ref_height,
                    pose_torso_angle_threshold=self.pose_torso_angle_threshold,
                )

            bbox = det.get("bbox", (0, 0, 0, 0))
            conf = det.get("confidence", 0.0)
            is_falling = self.trackers[track_id].update(
                bbox, conf, frame_idx, pose_hint=hints.get(track_id)
            )
            fall_results[track_id] = is_falling

        stale_ids = [
            pid for pid, tracker in self.trackers.items()
            if tracker.is_stale(frame_idx, self.tracker_ttl)
        ]
        for pid in stale_ids:
            del self.trackers[pid]

        return fall_results

    def get_pose_candidate_track_ids(self) -> List[int]:
        """Tracks with bbox fall signal pending confirmation (tier-2 pose check)."""
        return [
            pid
            for pid, tracker in self.trackers.items()
            if tracker.consecutive_fall_signals >= 1 and not tracker.fall_detected
        ]

    def apply_pose_hints(self, pose_hints: Dict[int, Dict]) -> Dict[int, bool]:
        results: Dict[int, bool] = {}
        for track_id, hint in pose_hints.items():
            tracker = self.trackers.get(track_id)
            if tracker is None:
                continue
            results[track_id] = tracker.apply_pose_hint(hint)
        return results

    def get_fall_detections(self) -> List[Tuple[int, Dict]]:
        results = []
        for person_id, tracker in self.trackers.items():
            if tracker.fall_detected:
                results.append((person_id, {
                    "person_id": person_id,
                    "frame_count": tracker.fall_frame_count,
                    "position_history": list(tracker.position_history),
                    "angle_history": list(tracker.angle_history),
                }))
        return results

    def reset_fall(self, person_id: Optional[int] = None):
        if person_id is None:
            for tracker in self.trackers.values():
                tracker.reset_fall()
        elif person_id in self.trackers:
            self.trackers[person_id].reset_fall()

    def get_stats(self) -> Dict:
        return {
            "total_tracked": len(self.trackers),
            "total_fallen": sum(1 for t in self.trackers.values() if t.fall_detected),
            "current_frame": self.current_frame,
        }
