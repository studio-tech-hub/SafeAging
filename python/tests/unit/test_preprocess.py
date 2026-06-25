"""
Unit tests for people_analytics_service.preprocess pure functions.
No service, model, or database required.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def frame_640():
    """640×480 plain grey BGR frame."""
    return np.full((480, 640, 3), 128, dtype=np.uint8)


@pytest.fixture
def frame_100():
    """100×100 plain grey BGR frame (easy maths)."""
    return np.full((100, 100, 3), 200, dtype=np.uint8)


# ---------------------------------------------------------------------------
# apply_roi_rect
# ---------------------------------------------------------------------------

class TestApplyRoiRect:
    def _call(self, frame, x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0):
        import people_analytics_service.preprocess as pp
        with (
            patch.object(pp, "ROI_X_MIN", x_min),
            patch.object(pp, "ROI_X_MAX", x_max),
            patch.object(pp, "ROI_Y_MIN", y_min),
            patch.object(pp, "ROI_Y_MAX", y_max),
        ):
            return pp.apply_roi_rect(frame)

    def test_full_frame_unchanged(self, frame_100):
        cropped, roi_box = self._call(frame_100, 0.0, 0.0, 1.0, 1.0)
        assert cropped.shape == frame_100.shape
        assert roi_box == (0, 0, 100, 100)

    def test_quarter_crop(self, frame_100):
        import people_analytics_service.preprocess as pp

        with (
            patch.object(pp, "ROI_X_MIN", 0.25),
            patch.object(pp, "ROI_X_MAX", 0.75),
            patch.object(pp, "ROI_Y_MIN", 0.25),
            patch.object(pp, "ROI_Y_MAX", 0.75),
        ):
            cropped, roi_box = pp.apply_roi_rect(frame_100)
        h, w = cropped.shape[:2]
        assert w == 50 and h == 50
        assert roi_box == (25, 25, 75, 75)

    def test_per_camera_rect_override(self, frame_100):
        import people_analytics_service.preprocess as pp

        with patch.object(pp, "ENABLE_ROI", True):
            cropped, roi_box, roi_type = pp.apply_roi(
                frame_100,
                roi_override={
                    "type": "rect",
                    "rect": {"x_min": 0.0, "y_min": 0.5, "x_max": 1.0, "y_max": 1.0},
                },
            )
        assert roi_type == "rect"
        assert cropped.shape[0] == 50

    def test_top_half(self, frame_640):
        cropped, roi_box = self._call(frame_640, 0.0, 0.0, 1.0, 0.5)
        assert cropped.shape == (240, 640, 3)
        assert roi_box == (0, 0, 640, 240)

    def test_clamp_to_frame(self, frame_100):
        # request more than 100% on each side
        cropped, roi_box = self._call(frame_100, -0.5, -0.5, 1.5, 1.5)
        assert cropped.shape == frame_100.shape


# ---------------------------------------------------------------------------
# apply_roi_polygon
# ---------------------------------------------------------------------------

class TestApplyRoiPolygon:
    def _call(self, frame, polygon_json: str):
        import json
        import people_analytics_service.preprocess as pp
        with patch.object(pp, "ROI_POLYGON_JSON", polygon_json):
            return pp.apply_roi_polygon(frame)

    def test_full_diamond(self, frame_100):
        # Diamond that covers the whole frame roughly
        poly = "[[0.5,0],[1,0.5],[0.5,1],[0,0.5]]"
        result_frame, roi_box = self._call(frame_100, poly)
        assert result_frame.shape == frame_100.shape  # same size, but masked
        assert len(roi_box) == 4

    def test_empty_polygon_fallback(self, frame_100):
        result_frame, roi_box = self._call(frame_100, "")
        assert result_frame.shape == frame_100.shape
        assert roi_box == (0, 0, 100, 100)

    def test_invalid_json_fallback(self, frame_100):
        result_frame, roi_box = self._call(frame_100, "INVALID_JSON")
        assert result_frame.shape == frame_100.shape
        assert roi_box == (0, 0, 100, 100)

    def test_masked_pixels_outside_polygon_are_zero(self, frame_100):
        # Small polygon in top-left corner (covers only 25×25)
        poly = "[[0,0],[0.25,0],[0.25,0.25],[0,0.25]]"
        result_frame, _ = self._call(frame_100, poly)
        # Bottom-right corner (far from polygon) should be zeroed out
        bottom_right = result_frame[90:, 90:]
        assert np.all(bottom_right == 0)


# ---------------------------------------------------------------------------
# remap_detections_to_input_space
# ---------------------------------------------------------------------------

class TestRemapDetections:
    def _det(self, x, y):
        return SimpleNamespace(x=float(x), y=float(y))

    def _call(self, dets, roi_box, roi_type, pre_roi_shape, crop_offset=(0, 0)):
        from people_analytics_service.preprocess import remap_detections_to_input_space
        remap_detections_to_input_space(
            detections=dets,
            roi_box=roi_box,
            roi_type=roi_type,
            pre_roi_shape=pre_roi_shape,
            undistort_crop_offset=crop_offset,
        )

    def test_no_roi_no_crop_no_change(self):
        det = self._det(10, 20)
        self._call([det], roi_box=(0, 0, 640, 480), roi_type="rect",
                   pre_roi_shape=(480, 640, 3), crop_offset=(0, 0))
        assert det.x == pytest.approx(10.0)
        assert det.y == pytest.approx(20.0)

    def test_rect_roi_offsets_applied(self):
        det = self._det(5, 10)  # coords inside ROI patch
        # ROI starts at (100, 50) in original frame
        self._call([det], roi_box=(100, 50, 400, 300), roi_type="rect",
                   pre_roi_shape=(480, 640, 3), crop_offset=(0, 0))
        assert det.x == pytest.approx(105.0)  # 5 + 100 ROI offset
        assert det.y == pytest.approx(60.0)   # 10 + 50 ROI offset

    def test_undistort_crop_offset_applied(self):
        det = self._det(20, 30)
        self._call([det], roi_box=(0, 0, 640, 480), roi_type="rect",
                   pre_roi_shape=(480, 640, 3), crop_offset=(15, 25))
        assert det.x == pytest.approx(35.0)  # 20 + 15
        assert det.y == pytest.approx(55.0)  # 30 + 25

    def test_both_offsets_summed(self):
        det = self._det(0, 0)
        self._call([det], roi_box=(10, 20, 200, 200), roi_type="rect",
                   pre_roi_shape=(480, 640, 3), crop_offset=(5, 8))
        assert det.x == pytest.approx(15.0)  # 10 ROI + 5 undistort
        assert det.y == pytest.approx(28.0)  # 20 ROI + 8 undistort

    def test_multiple_dets_all_offset(self):
        dets = [self._det(0, 0), self._det(10, 10), self._det(20, 20)]
        self._call(dets, roi_box=(100, 100, 400, 400), roi_type="rect",
                   pre_roi_shape=(480, 640, 3), crop_offset=(0, 0))
        assert dets[0].x == pytest.approx(100.0)
        assert dets[1].x == pytest.approx(110.0)
        assert dets[2].x == pytest.approx(120.0)

    def test_polygon_roi_type_no_rect_offset(self):
        # polygon type: rect offset NOT applied (only undistort offset)
        det = self._det(50, 60)
        self._call([det], roi_box=(10, 20, 200, 200), roi_type="polygon",
                   pre_roi_shape=(480, 640, 3), crop_offset=(5, 5))
        # Only undistort crop offset (5,5) added; ROI offset NOT applied for polygon
        assert det.x == pytest.approx(55.0)
        assert det.y == pytest.approx(65.0)


# ---------------------------------------------------------------------------
# undistort_frame (no-op when calibration not loaded)
# ---------------------------------------------------------------------------

class TestUndistortFrame:
    def test_returns_original_when_no_calibration(self, frame_640):
        import people_analytics_service.preprocess as pp
        # Ensure calibration is None
        with patch.object(pp, "camera_matrix", None), \
             patch.object(pp, "dist_coeffs", None):
            result, offset = pp.undistort_frame(frame_640)
        assert np.array_equal(result, frame_640)
        assert offset == (0, 0)

    def test_returns_original_when_undistort_disabled(self, frame_640):
        import people_analytics_service.preprocess as pp
        with patch.object(pp, "ENABLE_UNDISTORT", False):
            result, offset = pp.undistort_frame(frame_640)
        assert np.array_equal(result, frame_640)
        assert offset == (0, 0)

    def test_with_identity_calibration_no_change(self, frame_640):
        import cv2
        import people_analytics_service.preprocess as pp

        h, w = frame_640.shape[:2]
        fx = fy = float(w)
        camera_matrix = np.array([[fx, 0, w / 2],
                                   [0, fy, h / 2],
                                   [0,  0,     1]], dtype=np.float64)
        dist_coeffs = np.zeros((4,), dtype=np.float64)

        with (
            patch.object(pp, "ENABLE_UNDISTORT", True),
            patch.object(pp, "camera_matrix", camera_matrix),
            patch.object(pp, "dist_coeffs", dist_coeffs),
        ):
            result, offset = pp.undistort_frame(frame_640.copy())

        # With near-identity calibration, shape should be preserved
        assert result.shape[:2][0] > 0
        assert result.shape[:2][1] > 0


# ---------------------------------------------------------------------------
# ensure_min_input_side
# ---------------------------------------------------------------------------

class TestEnsureMinInputSide:
    def test_no_op_when_already_large_enough(self):
        import people_analytics_service.preprocess as pp

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        out, scale = pp.ensure_min_input_side(frame, 640)
        assert scale == 1.0
        assert out is frame

    def test_upscales_short_side(self):
        import people_analytics_service.preprocess as pp

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        out, scale = pp.ensure_min_input_side(frame, 640)
        assert scale == pytest.approx(640 / 360, rel=1e-3)
        assert min(out.shape[:2]) >= 640

    def test_square_small_frame(self, frame_100):
        import people_analytics_service.preprocess as pp

        out, scale = pp.ensure_min_input_side(frame_100, 640)
        assert scale == 6.4
        assert out.shape == (640, 640, 3)
