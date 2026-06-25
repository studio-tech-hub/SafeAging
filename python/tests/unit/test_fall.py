"""
Unit tests for FallDetectionTracker and FallDetectionManager (python/fall_detection.py).
No service, model, or database required.
"""
from __future__ import annotations

import sys
import os
import pytest

# fall_detection.py lives in python/ (not in the package)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from fall_detection import FallDetectionTracker, FallDetectionManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _upright_bbox(cx: int = 100, cy: int = 200, w: int = 60, h: int = 120):
    """Return (x1, y1, x2, y2) for an upright person."""
    return (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)


def _horizontal_bbox(cx: int = 100, cy: int = 200, w: int = 120, h: int = 40):
    """Return bbox for a person lying down (wide, low height)."""
    return (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)


# ---------------------------------------------------------------------------
# FallDetectionTracker
# ---------------------------------------------------------------------------

class TestFallDetectionTracker:
    def _tracker(self, **kw) -> FallDetectionTracker:
        defaults = dict(
            person_id=1,
            velocity_threshold=15.0,
            angle_change_threshold=30.0,
            aspect_ratio_threshold=1.2,
            confidence_threshold=0.5,
            confirm_frames=1,
        )
        defaults.update(kw)
        return FallDetectionTracker(**defaults)

    # --- first frame never triggers fall ---------------------------------

    def test_first_frame_no_fall(self):
        t = self._tracker()
        result = t.update(_upright_bbox(), confidence=0.9, frame_idx=1)
        assert result is False

    # --- low confidence skipped ------------------------------------------

    def test_low_confidence_no_fall(self):
        t = self._tracker(confidence_threshold=0.9)
        # Seed with an upright frame
        t.update(_upright_bbox(cy=100), confidence=0.95, frame_idx=1)
        # Send a horizontal bbox but with low confidence
        result = t.update(_horizontal_bbox(), confidence=0.3, frame_idx=2)
        assert result is False

    # --- velocity trigger -------------------------------------------------

    def test_high_velocity_triggers_fall(self):
        t = self._tracker(velocity_threshold=10.0)
        t.update(_upright_bbox(cy=100), confidence=0.9, frame_idx=1)
        # person "drops" 60 pixels in one frame → velocity = 60 > 10
        result = t.update(_upright_bbox(cy=160), confidence=0.9, frame_idx=2)
        assert result is True

    def test_low_velocity_no_fall(self):
        t = self._tracker(velocity_threshold=50.0)
        t.update(_upright_bbox(cy=100), confidence=0.9, frame_idx=1)
        # tiny movement of 3 pixels
        result = t.update(_upright_bbox(cy=103), confidence=0.9, frame_idx=2)
        assert result is False

    # --- aspect ratio trigger --------------------------------------------

    def test_horizontal_aspect_ratio_triggers_fall(self):
        t = self._tracker(aspect_ratio_threshold=1.2)
        t.update(_upright_bbox(), confidence=0.9, frame_idx=1)
        # width=180, height=20 → ratio=9.0 >> 1.2
        result = t.update((10, 190, 190, 210), confidence=0.9, frame_idx=2)
        assert result is True

    def test_normal_aspect_ratio_no_fall(self):
        t = self._tracker(aspect_ratio_threshold=2.0, velocity_threshold=100.0, angle_change_threshold=90.0)
        t.update(_upright_bbox(), confidence=0.9, frame_idx=1)
        result = t.update(_upright_bbox(), confidence=0.9, frame_idx=2)
        assert result is False

    # --- tiny bbox ignored -----------------------------------------------

    def test_tiny_bbox_ignored(self):
        t = self._tracker()
        result = t.update((100, 100, 105, 108), confidence=0.9, frame_idx=1)
        assert result is False

    # --- stale detection -------------------------------------------------

    def test_stale_tracker(self):
        t = self._tracker()
        t.update(_upright_bbox(), confidence=0.9, frame_idx=1)
        assert t.is_stale(current_frame=50, ttl=30)
        assert not t.is_stale(current_frame=10, ttl=30)

    # --- reset -----------------------------------------------------------

    def test_reset_clears_fall_state(self):
        t = self._tracker(velocity_threshold=10.0, confirm_frames=1)
        t.update(_upright_bbox(cy=100), confidence=0.9, frame_idx=1)
        t.update(_upright_bbox(cy=160), confidence=0.9, frame_idx=2)
        assert t.fall_detected is True
        t.reset_fall()
        assert t.fall_detected is False
        assert t.fall_frame_count == 0

    def test_pose_torso_hint_triggers_fall(self):
        t = self._tracker(velocity_threshold=50.0, confirm_frames=1)
        t.update(_upright_bbox(cy=100), confidence=0.9, frame_idx=1)
        result = t.update(
            _upright_bbox(cy=103),
            confidence=0.9,
            frame_idx=2,
            pose_hint={"torso_angle_from_vertical": 70.0},
        )
        assert result is True

        t = self._tracker(velocity_threshold=10.0, confirm_frames=3)
        t.update(_upright_bbox(cy=100), confidence=0.9, frame_idx=1)
        assert t.update(_upright_bbox(cy=160), confidence=0.9, frame_idx=2) is False
        assert t.update(_upright_bbox(cy=220), confidence=0.9, frame_idx=3) is False
        assert t.update(_upright_bbox(cy=280), confidence=0.9, frame_idx=4) is True


# ---------------------------------------------------------------------------
# FallDetectionManager
# ---------------------------------------------------------------------------

class TestFallDetectionManager:
    def _manager(self, **kw) -> FallDetectionManager:
        defaults = dict(
            velocity_threshold=15.0,
            angle_change_threshold=30.0,
            aspect_ratio_threshold=1.2,
            confidence_threshold=0.5,
            tracker_ttl=10,
            confirm_frames=1,
        )
        defaults.update(kw)
        return FallDetectionManager(**defaults)

    def _det(self, track_id: int, bbox, confidence: float = 0.9):
        return {"track_id": track_id, "bbox": bbox, "confidence": confidence}

    # --- basic result types -----------------------------------------------

    def test_returns_dict_of_bool(self):
        mgr = self._manager()
        result = mgr.update([self._det(1, _upright_bbox())], frame_idx=1)
        assert isinstance(result, dict)
        assert 1 in result
        assert isinstance(result[1], bool)

    def test_empty_detections(self):
        mgr = self._manager()
        result = mgr.update([], frame_idx=1)
        assert result == {}

    # --- multiple persons ------------------------------------------------

    def test_multiple_persons_tracked_independently(self):
        mgr = self._manager(velocity_threshold=10.0)
        dets_f1 = [
            self._det(1, _upright_bbox(cx=100, cy=200)),
            self._det(2, _upright_bbox(cx=300, cy=200)),
        ]
        mgr.update(dets_f1, frame_idx=1)

        # person 2 falls (velocity 80 > 10), person 1 stays
        dets_f2 = [
            self._det(1, _upright_bbox(cx=100, cy=202)),   # barely moved
            self._det(2, _upright_bbox(cx=300, cy=280)),   # dropped 80px
        ]
        result = mgr.update(dets_f2, frame_idx=2)
        assert result[1] is False
        assert result[2] is True

    # --- stale tracker cleanup -------------------------------------------

    def test_stale_trackers_removed(self):
        mgr = self._manager(tracker_ttl=3)
        mgr.update([self._det(1, _upright_bbox())], frame_idx=1)
        assert 1 in mgr.trackers
        # Advance far beyond TTL without seeing person 1
        mgr.update([], frame_idx=50)
        assert 1 not in mgr.trackers

    # --- get_stats -------------------------------------------------------

    def test_get_stats_keys(self):
        mgr = self._manager()
        mgr.update([self._det(1, _upright_bbox())], frame_idx=1)
        stats = mgr.get_stats()
        assert "total_tracked" in stats
        assert "total_fallen" in stats

    def test_get_stats_counts_fall(self):
        mgr = self._manager(velocity_threshold=10.0)
        mgr.update([self._det(1, _upright_bbox(cy=100))], frame_idx=1)
        mgr.update([self._det(1, _upright_bbox(cy=200))], frame_idx=2)
        stats = mgr.get_stats()
        assert stats["total_fallen"] >= 1
