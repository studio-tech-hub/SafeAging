"""
Unit tests for people_analytics_service.tracking pure functions.
No service, model, or database required.
"""
from __future__ import annotations

import numpy as np
import pytest

from people_analytics_service.tracking import (
    center_distance_ratio,
    combined_track_score,
    dedupe_output_tracks,
    has_duplicate_track_overlap,
    hungarian_minimize,
    iou,
    post_nms_dedupe,
    predict_track_bbox,
    smooth_bbox,
    update_track_motion,
)


# ---------------------------------------------------------------------------
# iou
# ---------------------------------------------------------------------------

class TestIou:
    def test_perfect_overlap(self):
        box = (0.0, 0.0, 10.0, 10.0)
        assert iou(box, box) == pytest.approx(1.0, abs=1e-5)

    def test_no_overlap(self):
        a = (0.0, 0.0, 5.0, 5.0)
        b = (10.0, 10.0, 15.0, 15.0)
        assert iou(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_half_overlap(self):
        # a covers [0,10]×[0,10] (area=100)
        # b covers [5,15]×[0,10] (area=100)
        # intersection [5,10]×[0,10] = 50
        # union = 100+100-50 = 150
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 0.0, 15.0, 10.0)
        assert iou(a, b) == pytest.approx(50 / 150, abs=1e-4)

    def test_zero_area_box(self):
        assert iou((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 5.0, 5.0)) == pytest.approx(0.0)

    def test_contained_box(self):
        outer = (0.0, 0.0, 10.0, 10.0)
        inner = (2.0, 2.0, 8.0, 8.0)
        # inner area = 36, outer = 100, intersection = 36, union = 100
        assert iou(outer, inner) == pytest.approx(36 / 100, abs=1e-4)

    def test_symmetry(self):
        a = (0.0, 0.0, 8.0, 6.0)
        b = (3.0, 2.0, 11.0, 9.0)
        assert iou(a, b) == pytest.approx(iou(b, a), abs=1e-8)


# ---------------------------------------------------------------------------
# center_distance_ratio
# ---------------------------------------------------------------------------

class TestCenterDistanceRatio:
    def test_same_box_is_zero(self):
        box = (0.0, 0.0, 10.0, 10.0)
        assert center_distance_ratio(box, box) == pytest.approx(0.0, abs=1e-6)

    def test_far_boxes_greater_than_one(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (1000.0, 1000.0, 1010.0, 1010.0)
        assert center_distance_ratio(a, b) > 1.0

    def test_adjacent_small_boxes(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (10.0, 0.0, 20.0, 10.0)
        # distance between centres = 10, diagonal ≈ 14.14, norm ≈ 14.14
        ratio = center_distance_ratio(a, b)
        assert 0.0 < ratio < 2.0


# ---------------------------------------------------------------------------
# smooth_bbox
# ---------------------------------------------------------------------------

class TestSmoothBbox:
    def test_alpha_zero_returns_old(self):
        old = (0.0, 0.0, 10.0, 10.0)
        new = (5.0, 5.0, 15.0, 15.0)
        result = smooth_bbox(old, new, alpha=0.0)
        assert result == old

    def test_alpha_one_returns_new(self):
        old = (0.0, 0.0, 10.0, 10.0)
        new = (5.0, 5.0, 15.0, 15.0)
        result = smooth_bbox(old, new, alpha=1.0)
        assert result == new

    def test_intermediate_alpha_between_old_and_new(self):
        old = (0.0, 0.0, 10.0, 10.0)
        new = (10.0, 10.0, 20.0, 20.0)
        result = smooth_bbox(old, new, alpha=0.5)
        # adaptive alpha may be higher for large moves; just check bounds
        rx1, ry1, rx2, ry2 = result
        assert old[0] <= rx1 <= new[0]
        assert old[1] <= ry1 <= new[1]


# ---------------------------------------------------------------------------
# post_nms_dedupe
# ---------------------------------------------------------------------------

class TestPostNmsDedupe:
    def _det(self, score, x1, y1, x2, y2):
        return {"score": score, "bbox": (float(x1), float(y1), float(x2), float(y2))}

    def test_single_det_unchanged(self):
        dets = [self._det(0.9, 0, 0, 10, 10)]
        assert post_nms_dedupe(dets, 0.5) == dets

    def test_non_overlapping_kept(self):
        dets = [self._det(0.9, 0, 0, 10, 10), self._det(0.8, 50, 50, 60, 60)]
        result = post_nms_dedupe(dets, 0.5)
        assert len(result) == 2

    def test_duplicate_suppressed(self):
        dets = [
            self._det(0.9, 0, 0, 10, 10),
            self._det(0.7, 1, 1, 11, 11),  # ~90% overlap with first
        ]
        result = post_nms_dedupe(dets, 0.5)
        assert len(result) == 1
        assert result[0]["score"] == 0.9  # highest score kept

    def test_empty_input(self):
        assert post_nms_dedupe([], 0.5) == []

    def test_order_independent_of_input(self):
        low = self._det(0.3, 0, 0, 10, 10)
        high = self._det(0.9, 0, 0, 10, 10)
        assert post_nms_dedupe([low, high], 0.5)[0]["score"] == 0.9
        assert post_nms_dedupe([high, low], 0.5)[0]["score"] == 0.9


# ---------------------------------------------------------------------------
# has_duplicate_track_overlap
# ---------------------------------------------------------------------------

class TestHasDuplicateTrackOverlap:
    def _track(self, x1, y1, x2, y2, last_seen=0.0):
        return {"bbox": (float(x1), float(y1), float(x2), float(y2)), "last_seen": last_seen}

    def test_overlapping_track_detected(self):
        track_by_id = {1: self._track(0, 0, 10, 10, last_seen=1.0)}
        det_box = (1.0, 1.0, 11.0, 11.0)
        assert has_duplicate_track_overlap(det_box, track_by_id, now_ts=1.5, overlap_iou=0.5, recent_only_sec=5.0)

    def test_non_overlapping_no_duplicate(self):
        track_by_id = {1: self._track(0, 0, 10, 10, last_seen=1.0)}
        det_box = (50.0, 50.0, 60.0, 60.0)
        assert not has_duplicate_track_overlap(det_box, track_by_id, now_ts=1.5, overlap_iou=0.5, recent_only_sec=5.0)

    def test_stale_track_ignored(self):
        track_by_id = {1: self._track(0, 0, 10, 10, last_seen=0.0)}
        det_box = (1.0, 1.0, 11.0, 11.0)
        # recent_only_sec=1, now_ts=10 → stale
        assert not has_duplicate_track_overlap(det_box, track_by_id, now_ts=10.0, overlap_iou=0.5, recent_only_sec=1.0)


# ---------------------------------------------------------------------------
# predict_track_bbox
# ---------------------------------------------------------------------------

class TestPredictTrackBbox:
    def _track(self, bbox, vx=0.0, vy=0.0, vw=0.0, vh=0.0, last_seen=0.0):
        return {
            "bbox": bbox,
            "measurement_bbox": bbox,
            "vx": vx, "vy": vy, "vw": vw, "vh": vh,
            "last_seen": last_seen,
        }

    def test_no_velocity_no_movement(self):
        t = self._track((10.0, 20.0, 50.0, 80.0), last_seen=0.0)
        result = predict_track_bbox(t, now_ts=1.0)
        # With zero velocity prediction should be equal to original
        assert result == pytest.approx((10.0, 20.0, 50.0, 80.0), abs=1e-3)

    def test_velocity_applied_over_dt(self):
        # vx=10 pixels/sec, dt=1s → x shifts by 10
        t = self._track((0.0, 0.0, 20.0, 40.0), vx=10.0, vy=5.0, last_seen=0.0)
        rx1, ry1, rx2, ry2 = predict_track_bbox(t, now_ts=1.0)
        assert rx1 == pytest.approx(10.0, abs=0.5)
        assert ry1 == pytest.approx(5.0, abs=0.5)


# ---------------------------------------------------------------------------
# update_track_motion
# ---------------------------------------------------------------------------

class TestUpdateTrackMotion:
    def test_velocity_computed_from_displacement(self):
        track = {"last_seen": 0.0, "vx": 0.0, "vy": 0.0, "vw": 0.0, "vh": 0.0}
        prev = (0.0, 0.0, 20.0, 40.0)
        new = (10.0, 5.0, 30.0, 45.0)  # moved +10x, +5y
        update_track_motion(track, prev, new, now_ts=1.0)
        assert track["vx"] == pytest.approx(10.0, abs=0.1)
        assert track["vy"] == pytest.approx(5.0, abs=0.1)

    def test_zero_movement_zero_velocity(self):
        track = {"last_seen": 0.0, "vx": 99.0, "vy": 99.0, "vw": 0.0, "vh": 0.0}
        box = (10.0, 10.0, 30.0, 50.0)
        update_track_motion(track, box, box, now_ts=1.0)
        assert track["vx"] == pytest.approx(0.0, abs=0.01)
        assert track["vy"] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# combined_track_score
# ---------------------------------------------------------------------------

class TestCombinedTrackScore:
    def test_perfect_match(self):
        score = combined_track_score(iou_score=1.0, appearance_dist=0.0)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_zero_iou(self):
        score = combined_track_score(iou_score=0.0, appearance_dist=1.0)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_higher_iou_yields_higher_score(self):
        s1 = combined_track_score(iou_score=0.8, appearance_dist=0.3)
        s2 = combined_track_score(iou_score=0.4, appearance_dist=0.3)
        assert s1 > s2


# ---------------------------------------------------------------------------
# hungarian_minimize
# ---------------------------------------------------------------------------

class TestHungarianMinimize:
    def test_empty_matrix(self):
        assert hungarian_minimize(np.empty((0, 0))) == []

    def test_1x1_assignment(self):
        cost = np.array([[0.5]])
        result = hungarian_minimize(cost)
        assert result == [(0, 0)]

    def test_2x2_optimal(self):
        cost = np.array([[1.0, 4.0],
                         [4.0, 1.0]])
        result = hungarian_minimize(cost)
        assignments = set(result)
        assert (0, 0) in assignments
        assert (1, 1) in assignments
        total = sum(cost[r, c] for r, c in result)
        assert total == pytest.approx(2.0, abs=1e-6)

    def test_2x3_non_square(self):
        cost = np.array([[3.0, 1.0, 2.0],
                         [2.0, 3.0, 1.0]])
        result = hungarian_minimize(cost)
        # 2 rows → 2 assignments
        assert len(result) == 2
        rows, cols = zip(*result)
        assert len(set(rows)) == 2  # no duplicate row
        assert len(set(cols)) == 2  # no duplicate col

    def test_3x2_non_square(self):
        cost = np.array([[1.0, 5.0],
                         [5.0, 1.0],
                         [3.0, 3.0]])
        result = hungarian_minimize(cost)
        assert len(result) == 2
