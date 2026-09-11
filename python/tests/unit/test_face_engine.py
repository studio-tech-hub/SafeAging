"""Unit tests for face_engine.py's P1-6 cost-control helpers.

These tests target `select_det_size()` (pure function, no insightface import
needed) and `_detect_faces()`'s fallback-safety behavior. The adaptive-size
detection path itself (`_detect_faces_with_size`) reaches into
insightface-internal objects (app.det_model, app.models) that only exist once
the real insightface package + model pack are loaded; that path is validated
against real hardware as part of the P1-6 deployment checklist (see
CPU_PRODUCTION_PROFILE.md), not here.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# select_det_size — pure function, exercised across a range of crop sizes.
# ---------------------------------------------------------------------------

class TestSelectDetSize:
    def test_small_square_crop_uses_smallest_bucket(self):
        from people_analytics_service.face_engine import select_det_size

        assert select_det_size(160, 160, ceiling=640) == 320

    def test_crop_at_smallest_bucket_boundary_uses_that_bucket(self):
        from people_analytics_service.face_engine import select_det_size

        assert select_det_size(320, 200, ceiling=640) == 320

    def test_crop_just_above_smallest_bucket_uses_next_bucket(self):
        from people_analytics_service.face_engine import select_det_size

        assert select_det_size(321, 200, ceiling=640) == 480

    def test_crop_at_middle_bucket_boundary_uses_that_bucket(self):
        from people_analytics_service.face_engine import select_det_size

        assert select_det_size(480, 480, ceiling=640) == 480

    def test_large_crop_uses_ceiling(self):
        from people_analytics_service.face_engine import select_det_size

        assert select_det_size(640, 640, ceiling=640) == 640

    def test_very_large_crop_uses_ceiling_not_upscaled_bucket(self):
        from people_analytics_service.face_engine import select_det_size

        assert select_det_size(2000, 3000, ceiling=640) == 640

    def test_decision_driven_by_long_side_not_short_side(self):
        """A tall, narrow crop (e.g. a distant person's head-region crop) must
        be sized off its longer dimension, matching SCRFD's own square-canvas
        letterbox-fit behavior in detect()."""
        from people_analytics_service.face_engine import select_det_size

        assert select_det_size(50, 470, ceiling=640) == 480
        assert select_det_size(470, 50, ceiling=640) == 480

    def test_ceiling_below_smallest_bucket_is_never_exceeded(self):
        """If an operator configures FACE_DET_SIZE below 320 (e.g. 288), the
        adaptive buckets must never recommend a *larger* size than that
        explicit ceiling -- the ceiling always wins."""
        from people_analytics_service.face_engine import select_det_size

        assert select_det_size(160, 160, ceiling=288) == 288
        assert select_det_size(2000, 2000, ceiling=288) == 288

    def test_ceiling_between_buckets(self):
        """A ceiling strictly between the two small buckets (e.g. 400) should
        only ever offer 320 or the ceiling itself -- never the unreachable 480
        bucket above the ceiling."""
        from people_analytics_service.face_engine import select_det_size

        assert select_det_size(160, 160, ceiling=400) == 320
        assert select_det_size(321, 160, ceiling=400) == 400
        assert select_det_size(2000, 2000, ceiling=400) == 400

    def test_defaults_ceiling_from_config_face_det_size(self, monkeypatch):
        monkeypatch.setenv("FACE_DET_SIZE", "640")
        from people_analytics_service import config

        config.CONFIG.__init__()
        from people_analytics_service.face_engine import select_det_size

        assert select_det_size(100, 100) == 320
        assert select_det_size(5000, 5000) == config.FACE_DET_SIZE

    def test_zero_or_negative_dimensions_do_not_crash(self):
        from people_analytics_service.face_engine import select_det_size

        assert select_det_size(0, 0, ceiling=640) == 320
        assert select_det_size(-5, -5, ceiling=640) == 320


# ---------------------------------------------------------------------------
# _detect_faces — falls back to app.get() if the adaptive-size path errors.
# ---------------------------------------------------------------------------

class TestDetectFacesFallback:
    def test_falls_back_to_app_get_on_internal_api_mismatch(self, monkeypatch):
        from people_analytics_service import face_engine

        def _boom(app, bgr_crop, det_size):
            raise AttributeError("simulated insightface internal API change")

        monkeypatch.setattr(face_engine, "_detect_faces_with_size", _boom)

        sentinel = ["fallback-result"]

        class FakeApp:
            def get(self, img):
                return sentinel

        result = face_engine._detect_faces(FakeApp(), object(), 480)
        assert result is sentinel

    def test_uses_adaptive_path_when_it_succeeds(self, monkeypatch):
        from people_analytics_service import face_engine

        calls = []

        def _fake_adaptive(app, bgr_crop, det_size):
            calls.append(det_size)
            return ["adaptive-result"]

        monkeypatch.setattr(face_engine, "_detect_faces_with_size", _fake_adaptive)

        class FakeApp:
            def get(self, img):
                raise AssertionError("should not fall back when adaptive path succeeds")

        result = face_engine._detect_faces(FakeApp(), object(), 320)
        assert result == ["adaptive-result"]
        assert calls == [320]
