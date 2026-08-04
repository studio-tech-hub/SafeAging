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
    greedy_nms_indices,
    has_duplicate_track_overlap,
    hungarian_minimize,
    iou,
    overlaps_confirmed_track,
    post_nms_dedupe,
    predict_track_bbox,
    select_output_bbox,
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
# select_output_bbox (P1-1: the rendered/output box must be the smoothed one,
# not the raw measurement — smooth_bbox() was being computed and discarded).
# ---------------------------------------------------------------------------

class TestSelectOutputBbox:
    def test_prefers_smoothed_bbox_over_raw_measurement(self):
        # Mirrors api.py's matched-track update: tr["bbox"] holds the smoothed
        # value, tr["measurement_bbox"] holds the latest raw YOLO measurement.
        old = (0.0, 0.0, 10.0, 10.0)
        new = (50.0, 50.0, 60.0, 60.0)
        smoothed = smooth_bbox(old, new, alpha=0.85)
        track = {"bbox": smoothed, "measurement_bbox": new}
        assert select_output_bbox(track) == smoothed

    def test_end_to_end_smoothing_differs_from_raw_measurement(self):
        """Proves smoothing is actually applied end-to-end (the P1-1 bug: it was
        computed via smooth_bbox() but never reached the output)."""
        old = (0.0, 0.0, 10.0, 10.0)
        new = (20.0, 20.0, 30.0, 30.0)
        smoothed = smooth_bbox(old, new, alpha=0.85)
        track = {"bbox": smoothed, "measurement_bbox": new}
        selected = select_output_bbox(track)
        assert selected != new
        assert selected == smoothed

    def test_regression_guard_large_jump_still_uses_smoothed_box(self):
        """Pins the P1-1 bug so it can't silently regress: even on a large, fast
        jump (which pushes smooth_bbox()'s adaptive alpha up via
        center_distance_ratio so it "snaps" quickly), the selected output must
        still be the smoothed value, not the raw measurement."""
        old = (0.0, 0.0, 10.0, 10.0)
        new = (200.0, 200.0, 210.0, 210.0)  # large jump
        smoothed = smooth_bbox(old, new, alpha=0.85)
        track = {"bbox": smoothed, "measurement_bbox": new}
        assert select_output_bbox(track) != new
        assert select_output_bbox(track) == smoothed

    def test_smoothing_disabled_is_backward_compatible_with_raw_output(self):
        """BBOX_SMOOTHING=0 semantics: api.py's matched-track update sets
        tr["bbox"] = det_box directly in that case (smooth_bbox() is never
        called), so "bbox" already equals the raw measurement — operators who
        want the old razor-sharp/no-lag boxes see identical behavior."""
        raw = (5.0, 5.0, 15.0, 15.0)
        track = {"bbox": raw, "measurement_bbox": raw}
        assert select_output_bbox(track) == raw

    def test_missing_bbox_falls_back_to_measurement_bbox(self):
        track = {"measurement_bbox": (1.0, 2.0, 3.0, 4.0)}
        assert select_output_bbox(track) == (1.0, 2.0, 3.0, 4.0)


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
# greedy_nms_indices (P1-3: xyxy-native NMS via iou(), replaces the
# cv2.dnn.NMSBoxes-based yolo_backend._nms_indices, which silently expected
# [x, y, w, h] input and mis-scaled xyxy boxes into much larger ones anchored
# at the same top-left corner -- making suppression depend on a box's
# absolute position in the frame instead of only its overlap with others.)
# ---------------------------------------------------------------------------

class TestGreedyNmsIndices:
    def test_empty_input(self):
        assert greedy_nms_indices([], [], 0.5) == []

    def test_single_box_kept(self):
        assert greedy_nms_indices([(0.0, 0.0, 10.0, 10.0)], [0.9], 0.5) == [0]

    def test_non_overlapping_both_kept(self):
        boxes = [(0.0, 0.0, 10.0, 10.0), (50.0, 50.0, 60.0, 60.0)]
        scores = [0.9, 0.8]
        assert greedy_nms_indices(boxes, scores, 0.5) == [0, 1]

    def test_duplicate_suppressed_highest_score_first(self):
        boxes = [(0.0, 0.0, 10.0, 10.0), (1.0, 1.0, 11.0, 11.0)]  # ~90% overlap
        scores = [0.7, 0.9]
        assert greedy_nms_indices(boxes, scores, 0.5) == [1]

    def test_translation_invariant_near_and_far_from_origin(self):
        """P1-3 regression: same relative geometry must NMS identically
        regardless of absolute frame position.

        True IoU of this box pair is ~0.39, below the 0.45 threshold, so
        both boxes must survive whether the pair sits near the frame origin
        or far away from it. The pre-fix cv2.dnn.NMSBoxes-based
        implementation passed xyxy straight in as if it were [x, y, w, h]:
        near the origin (x1=y1=0) that happens to coincide with the true
        box, but far from the origin it inflates each box into one anchored
        at (x1, y1) with far corner at (x1 + x2, y1 + y2) -- e.g. a box at
        (300, 300, 340, 340) (true 40x40) became a 340x340 box, which
        pushed the *apparent* IoU with its neighbor up to ~0.84 and wrongly
        suppressed it (empirically measured against the pre-fix
        cv2.dnn.NMSBoxes-based implementation before this test was added).
        """
        near = [(0.0, 0.0, 40.0, 40.0), (10.0, 10.0, 50.0, 50.0)]
        far = [(300.0, 300.0, 340.0, 340.0), (310.0, 310.0, 350.0, 350.0)]
        scores = [0.9, 0.8]
        true_iou_near = iou(near[0], near[1])
        true_iou_far = iou(far[0], far[1])
        assert true_iou_near == pytest.approx(true_iou_far)
        assert true_iou_near < 0.45  # below threshold -> neither should suppress the other

        assert greedy_nms_indices(near, scores, 0.45) == [0, 1]
        assert greedy_nms_indices(far, scores, 0.45) == [0, 1]

    def test_matches_post_nms_dedupe_on_same_input(self):
        dets = [
            {"score": 0.9, "bbox": (0.0, 0.0, 10.0, 10.0)},
            {"score": 0.7, "bbox": (1.0, 1.0, 11.0, 11.0)},
            {"score": 0.8, "bbox": (50.0, 50.0, 60.0, 60.0)},
        ]
        boxes = [d["bbox"] for d in dets]
        scores = [d["score"] for d in dets]
        picked = greedy_nms_indices(boxes, scores, 0.5)
        expected = post_nms_dedupe(dets, 0.5)
        assert [dets[i] for i in picked] == expected


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
# overlaps_confirmed_track (P1-2: makes PERSON_MIN_HW_RATIO track-state-aware —
# a wide/low box overlapping an already-confirmed track should bypass the
# furniture-rejecting ratio filter, since it's far more likely a fall than a
# new piece of furniture; brand-new candidates still get the strict filter.)
# ---------------------------------------------------------------------------

class TestOverlapsConfirmedTrack:
    def _track(self, x1, y1, x2, y2, status="confirmed"):
        return {
            "bbox": (float(x1), float(y1), float(x2), float(y2)),
            "status": status,
        }

    def test_wide_box_over_confirmed_track_bypasses_filter(self):
        """The core P1-2 fix: a fallen-posture-shaped box overlapping a track
        that was already confirmed must be allowed through (bypass=True)."""
        track_by_id = {1: self._track(0, 0, 100, 100, status="confirmed")}
        wide_fallen_box = (5.0, 40.0, 95.0, 90.0)  # low h/w ratio, overlaps track 1
        assert overlaps_confirmed_track(wide_fallen_box, track_by_id, iou_threshold=0.12)

    def test_wide_box_with_no_nearby_track_stays_filtered(self):
        """A brand-new wide candidate with no overlapping track must NOT bypass
        the filter — this is exactly the furniture false-positive case the
        filter exists to catch."""
        track_by_id = {1: self._track(0, 0, 100, 100, status="confirmed")}
        far_away_box = (500.0, 500.0, 600.0, 550.0)
        assert not overlaps_confirmed_track(far_away_box, track_by_id, iou_threshold=0.12)

    def test_tentative_track_does_not_bypass_filter(self):
        """Only CONFIRMED tracks may bypass the ratio filter — a tentative
        (not-yet-established) track candidate must still be held to the strict
        furniture filter, per the P1-2 spec ('only applies to new/tentative
        track candidates')."""
        track_by_id = {1: self._track(0, 0, 100, 100, status="tentative")}
        wide_box = (5.0, 40.0, 95.0, 90.0)
        assert not overlaps_confirmed_track(wide_box, track_by_id, iou_threshold=0.12)

    def test_low_overlap_below_threshold_does_not_bypass(self):
        """Overlap must meet the same IoU bar the tracker itself uses to
        consider a detection a candidate match — a box that barely grazes a
        confirmed track's last-known position should not bypass the filter."""
        track_by_id = {1: self._track(0, 0, 10, 10, status="confirmed")}
        barely_touching_box = (9.5, 9.5, 30.0, 30.0)
        assert not overlaps_confirmed_track(barely_touching_box, track_by_id, iou_threshold=0.12)

    def test_no_tracks_returns_false(self):
        assert not overlaps_confirmed_track((0.0, 0.0, 10.0, 10.0), {}, iou_threshold=0.12)

    def test_falls_back_to_measurement_bbox_when_bbox_missing(self):
        track_by_id = {1: {"status": "confirmed", "measurement_bbox": (0.0, 0.0, 100.0, 100.0)}}
        wide_box = (5.0, 40.0, 95.0, 90.0)
        assert overlaps_confirmed_track(wide_box, track_by_id, iou_threshold=0.12)


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
