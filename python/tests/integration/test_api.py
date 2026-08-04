"""
Integration tests for the SafeAging analytics service.

Requires a running service (docker compose up -d or direct run).
Run with:
    pytest -m integration python/tests/integration/

SERVICE_URL defaults to http://localhost:18000.
"""
from __future__ import annotations

import base64
import json
import time
import uuid

import pytest
import requests


SERVICE_URL = __import__("os").getenv("SERVICE_URL", "http://localhost:18000")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _svc(path: str, **kw) -> str:
    return SERVICE_URL + path


def _camera() -> str:
    return f"itst_{uuid.uuid4().hex[:8]}"


def _infer(frame_b64: str, camera_id: str, timeout: int = 20) -> requests.Response:
    return requests.post(
        _svc("/infer"),
        json={"image": frame_b64, "camera_id": camera_id},
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestHealth:
    def test_health_returns_200(self):
        r = requests.get(_svc("/health"), timeout=5)
        assert r.status_code == 200

    def test_health_has_status_field(self):
        r = requests.get(_svc("/health"), timeout=5)
        body = r.json()
        assert "status" in body
        assert body["status"] in {"healthy", "degraded", "not_ready"}

    def test_health_has_dependencies(self):
        r = requests.get(_svc("/health"), timeout=5)
        body = r.json()
        assert "dependencies" in body
        deps = body["dependencies"]
        assert isinstance(deps, dict)

    def test_health_has_pipeline_field(self):
        r = requests.get(_svc("/health"), timeout=5)
        body = r.json()
        assert "pipeline" in body


# ---------------------------------------------------------------------------
# Authentication (P0-2 — fail-closed API key enforcement)
#
# These tests adapt to however the target service is configured rather than
# assuming a fixed mode, so the same suite works against a secured deployment
# (API_KEY_REQUIRED=true) and an intentionally open lab/dev instance
# (API_KEY_REQUIRED=false or ALLOW_INSECURE_NO_AUTH=true).
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAuthEnforcement:
    @staticmethod
    def _auth_enabled() -> bool:
        r = requests.get(_svc("/health"), timeout=5)
        return bool(r.json().get("auth_enabled"))

    def test_infer_without_key_matches_service_auth_mode(self, frame_b64):
        cam = _camera()
        r = _infer(frame_b64, cam)
        if self._auth_enabled():
            assert r.status_code == 401
        else:
            assert r.status_code == 200

    def test_infer_with_wrong_key_is_rejected_when_auth_enabled(self, frame_b64):
        if not self._auth_enabled():
            pytest.skip("target service has auth disabled (API_KEY_REQUIRED=false)")
        cam = _camera()
        r = requests.post(
            _svc("/infer"),
            json={"image": frame_b64, "camera_id": cam},
            headers={"X-API-Key": "definitely-the-wrong-key"},
            timeout=20,
        )
        assert r.status_code == 401

    def test_infer_with_correct_key_succeeds_when_auth_enabled(self, frame_b64):
        if not self._auth_enabled():
            pytest.skip("target service has auth disabled (API_KEY_REQUIRED=false)")
        api_key = __import__("os").getenv("API_KEY")
        if not api_key:
            pytest.skip("set API_KEY env var for the test client to exercise the authenticated path")
        cam = _camera()
        r = requests.post(
            _svc("/infer"),
            json={"image": frame_b64, "camera_id": cam},
            headers={"X-API-Key": api_key},
            timeout=20,
        )
        assert r.status_code == 200

    def test_admin_endpoint_without_key_matches_service_auth_mode(self):
        r = requests.get(_svc("/admin/zones"), timeout=5)
        if self._auth_enabled():
            assert r.status_code == 401
        else:
            assert r.status_code == 200

    def test_health_reports_auth_disabled_reason_code_when_insecure(self):
        r = requests.get(_svc("/health"), timeout=5)
        body = r.json()
        if not body.get("auth_enabled"):
            assert "auth_disabled" in body.get("reason_codes", [])


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestMetrics:
    def test_metrics_returns_200(self):
        r = requests.get(_svc("/metrics"), timeout=5)
        assert r.status_code == 200

    def test_metrics_content_type_prometheus(self):
        r = requests.get(_svc("/metrics"), timeout=5)
        assert "text/plain" in r.headers.get("content-type", "")

    def test_metrics_contains_safeaging_prefix(self):
        r = requests.get(_svc("/metrics"), timeout=5)
        assert "safeaging_" in r.text

    def test_outbox_metrics_present_after_infer(self, frame_b64):
        cam = _camera()
        # Seed some requests so infer metrics exist
        for _ in range(3):
            _infer(frame_b64, cam)
        time.sleep(0.5)
        r = requests.get(_svc("/metrics"), timeout=5)
        assert "safeaging_infer_requests_total" in r.text


# ---------------------------------------------------------------------------
# /infer
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestInfer:
    def test_infer_returns_200(self, frame_b64):
        cam = _camera()
        r = _infer(frame_b64, cam)
        assert r.status_code == 200

    def test_infer_returns_list(self, frame_b64):
        cam = _camera()
        r = _infer(frame_b64, cam)
        body = r.json()
        assert isinstance(body, list)

    def test_infer_detection_fields(self, frame_b64):
        cam = _camera()
        # Send several frames to confirm tracks
        for _ in range(5):
            r = _infer(frame_b64, cam)
        dets = r.json()
        if dets:
            d = dets[0]
            assert "cls" in d
            assert "score" in d
            assert "track_id" in d
            assert "x" in d and "y" in d
            assert "w" in d and "h" in d
            assert isinstance(d["fall_detected"], bool)

    def test_infer_same_camera_stable_track_ids(self, frame_b64):
        cam = _camera()
        ids_per_frame: list[set] = []
        for _ in range(6):
            r = _infer(frame_b64, cam)
            ids_per_frame.append({d["track_id"] for d in r.json()})
        # At least some frames should share track IDs (stability)
        non_empty = [s for s in ids_per_frame if s]
        if len(non_empty) >= 2:
            assert non_empty[-1] & non_empty[-2], "Track IDs should persist across adjacent frames"

    def test_infer_different_cameras_independent(self, frame_b64):
        cam_a = _camera()
        cam_b = _camera()
        # Prime both cameras
        for _ in range(4):
            _infer(frame_b64, cam_a)
            _infer(frame_b64, cam_b)
        r_a = _infer(frame_b64, cam_a)
        r_b = _infer(frame_b64, cam_b)
        # Both should return detections independently
        assert r_a.status_code == 200
        assert r_b.status_code == 200

    def test_infer_bad_base64_returns_error(self):
        cam = _camera()
        r = requests.post(_svc("/infer"), json={"image": "NOTBASE64!!", "camera_id": cam}, timeout=10)
        assert r.status_code in {400, 422, 200}  # service returns [] on decode error

    def test_infer_missing_image_422(self):
        r = requests.post(_svc("/infer"), json={"camera_id": "x"}, timeout=5)
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# /infer/binary — additive multipart/binary transport (P1-4)
#
# Same detection/tracking pipeline as /infer, reached via multipart/form-data
# (raw JPEG bytes) instead of JSON+base64. /infer itself must stay untouched
# and remain the default — covered by TestInfer above, which is unmodified.
# ---------------------------------------------------------------------------

def _infer_binary(jpeg_bytes: bytes, camera_id: str, timeout: int = 20, **kw) -> requests.Response:
    return requests.post(
        _svc("/infer/binary"),
        data={"camera_id": camera_id},
        files={"image": ("frame.jpg", jpeg_bytes, "image/jpeg")},
        timeout=timeout,
        **kw,
    )


@pytest.mark.integration
class TestInferBinary:
    def test_infer_binary_returns_200(self, frame_b64):
        cam = _camera()
        jpeg_bytes = base64.b64decode(frame_b64)
        r = _infer_binary(jpeg_bytes, cam)
        assert r.status_code == 200

    def test_infer_binary_returns_list(self, frame_b64):
        cam = _camera()
        jpeg_bytes = base64.b64decode(frame_b64)
        r = _infer_binary(jpeg_bytes, cam)
        assert isinstance(r.json(), list)

    def test_infer_binary_detection_fields_match_json_transport(self, frame_b64):
        cam = _camera()
        jpeg_bytes = base64.b64decode(frame_b64)
        for _ in range(5):
            r = _infer_binary(jpeg_bytes, cam)
        dets = r.json()
        if dets:
            d = dets[0]
            assert "cls" in d
            assert "score" in d
            assert "track_id" in d
            assert "x" in d and "y" in d
            assert "w" in d and "h" in d
            assert isinstance(d["fall_detected"], bool)

    def test_infer_binary_same_camera_stable_track_ids(self, frame_b64):
        cam = _camera()
        jpeg_bytes = base64.b64decode(frame_b64)
        ids_per_frame: list[set] = []
        for _ in range(6):
            r = _infer_binary(jpeg_bytes, cam)
            ids_per_frame.append({d["track_id"] for d in r.json()})
        non_empty = [s for s in ids_per_frame if s]
        if len(non_empty) >= 2:
            assert non_empty[-1] & non_empty[-2], "Track IDs should persist across adjacent frames"

    def test_infer_binary_first_frame_matches_json_transport(self, frame_b64):
        """Both transports must feed the identical pipeline (P1-4): a brand-new
        camera's very first frame should produce the same detection count and
        boxes regardless of which transport delivered the JPEG bytes."""
        jpeg_bytes = base64.b64decode(frame_b64)

        cam_json = _camera()
        r_json = _infer(frame_b64, cam_json)
        assert r_json.status_code == 200

        cam_binary = _camera()
        r_binary = _infer_binary(jpeg_bytes, cam_binary)
        assert r_binary.status_code == 200

        dets_json = r_json.json()
        dets_binary = r_binary.json()
        assert len(dets_json) == len(dets_binary)
        boxes_json = sorted((d["x"], d["y"], d["w"], d["h"]) for d in dets_json)
        boxes_binary = sorted((d["x"], d["y"], d["w"], d["h"]) for d in dets_binary)
        assert boxes_json == boxes_binary

    def test_infer_binary_bad_jpeg_returns_empty_list(self):
        cam = _camera()
        r = _infer_binary(b"not a real jpeg", cam)
        assert r.status_code in {400, 422, 200}  # service returns [] on decode error

    def test_infer_binary_missing_image_422(self):
        r = requests.post(_svc("/infer/binary"), data={"camera_id": "x"}, timeout=5)
        assert r.status_code == 422

    def test_infer_binary_defaults_camera_id_when_omitted(self, frame_b64):
        jpeg_bytes = base64.b64decode(frame_b64)
        r = requests.post(
            _svc("/infer/binary"),
            files={"image": ("frame.jpg", jpeg_bytes, "image/jpeg")},
            timeout=20,
        )
        assert r.status_code == 200


@pytest.mark.integration
class TestInferBinaryAuthEnforcement:
    @staticmethod
    def _auth_enabled() -> bool:
        r = requests.get(_svc("/health"), timeout=5)
        return bool(r.json().get("auth_enabled"))

    def test_infer_binary_without_key_matches_service_auth_mode(self, frame_b64):
        cam = _camera()
        jpeg_bytes = base64.b64decode(frame_b64)
        r = _infer_binary(jpeg_bytes, cam)
        if self._auth_enabled():
            assert r.status_code == 401
        else:
            assert r.status_code == 200

    def test_infer_binary_with_correct_key_succeeds_when_auth_enabled(self, frame_b64):
        if not self._auth_enabled():
            pytest.skip("target service has auth disabled (API_KEY_REQUIRED=false)")
        api_key = __import__("os").getenv("API_KEY")
        if not api_key:
            pytest.skip("set API_KEY env var for the test client to exercise the authenticated path")
        cam = _camera()
        jpeg_bytes = base64.b64decode(frame_b64)
        r = _infer_binary(jpeg_bytes, cam, headers={"X-API-Key": api_key})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# PERSON_MIN_HW_RATIO track-state-aware filter (P1-2)
#
# The ratio filter must only gate *new* track candidates. Once a track is
# confirmed, it must survive a transition to a wide/fallen posture instead of
# being silently dropped — that used to directly undermine fall detection.
# Simulated by warping a real photo's aspect ratio wide (stretch width, squash
# height) after establishing a confirmed track on the unmodified photo, since
# the actual filter decision only happens inside the real detection pipeline.
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFallPostureAspectRatioFilter:
    @staticmethod
    def _widen_b64(frame_b64: str, x_scale: float = 2.2, y_scale: float = 0.65) -> str:
        import cv2
        import numpy as np
        raw = base64.b64decode(frame_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        warped = cv2.resize(img, (max(1, int(w * x_scale)), max(1, int(h * y_scale))))
        ok, buf = cv2.imencode(".jpg", warped, [cv2.IMWRITE_JPEG_QUALITY, 90])
        assert ok, "cv2.imencode failed while widening test frame"
        return base64.b64encode(buf).decode()

    def test_confirmed_track_survives_wide_posture_transition(self, real_frame_b64):
        cam = _camera()

        # Establish a confirmed track on the unmodified (upright) real photo.
        last_dets: list = []
        for _ in range(6):
            r = _infer(real_frame_b64, cam)
            assert r.status_code == 200
            last_dets = r.json()
        confirmed_ids_before = {d["track_id"] for d in last_dets if d.get("stable")}
        if not confirmed_ids_before:
            pytest.skip(
                "No stable/confirmed track established on the unmodified real photo; "
                "cannot exercise the mid-track wide-posture transition"
            )

        # Now force a low h/w ratio (stretch wide, squash short) on the same
        # camera_id so any surviving detection is treated as the same track,
        # never a brand-new candidate.
        wide_b64 = self._widen_b64(real_frame_b64)
        wide_dets: list = []
        for _ in range(3):
            r = _infer(wide_b64, cam)
            assert r.status_code == 200
            wide_dets = r.json()

        if not wide_dets:
            pytest.skip(
                "YOLO produced no detections at all on the warped frame (model-dependent); "
                "cannot isolate the aspect-ratio filter's effect without a baseline detection"
            )

        wide_ids = {d["track_id"] for d in wide_dets}
        assert confirmed_ids_before & wide_ids, (
            f"Confirmed track(s) {confirmed_ids_before} disappeared after transitioning to a "
            f"wide/fallen posture (got {wide_ids}) — PERSON_MIN_HW_RATIO must not blanket-drop "
            "an already-confirmed track, only gate brand-new track candidates."
        )


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestStatus:
    def test_status_200(self):
        r = requests.get(_svc("/status"), timeout=5)
        assert r.status_code == 200

    def test_status_has_service_key(self):
        r = requests.get(_svc("/status"), timeout=5)
        body = r.json()
        assert "service" in body


# ---------------------------------------------------------------------------
# /admin/persons (CRUD)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAdminPersons:
    BASE = "/admin/persons"

    def test_list_persons_returns_list(self):
        r = requests.get(_svc(self.BASE), timeout=5)
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_create_read_update_delete_person(self):
        # CREATE
        payload = {"name": f"Test Person {uuid.uuid4().hex[:6]}", "notes": "integration test"}
        r = requests.post(_svc(self.BASE), json=payload, timeout=5)
        assert r.status_code == 201, r.text
        created = r.json()
        pid = created["id"]
        assert created["name"] == payload["name"]

        # READ
        r = requests.get(_svc(f"{self.BASE}/{pid}"), timeout=5)
        assert r.status_code == 200
        assert r.json()["id"] == pid

        # UPDATE
        r = requests.put(_svc(f"{self.BASE}/{pid}"), json={"notes": "updated"}, timeout=5)
        assert r.status_code == 200
        assert r.json()["notes"] == "updated"

        # DELETE (hard) — 204
        r = requests.delete(_svc(f"{self.BASE}/{pid}"), timeout=5)
        assert r.status_code in {200, 204}

        r2 = requests.get(_svc(f"{self.BASE}/{pid}"), timeout=5)
        assert r2.status_code == 404

    def test_get_nonexistent_person_404(self):
        fake_id = str(uuid.uuid4())
        r = requests.get(_svc(f"{self.BASE}/{fake_id}"), timeout=5)
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# /admin/zones (CRUD)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAdminZones:
    BASE = "/admin/zones"

    def test_list_zones(self):
        r = requests.get(_svc(self.BASE), timeout=5)
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_create_and_deactivate_zone(self):
        payload = {
            "camera_id": _camera(),
            "name": f"Zone {uuid.uuid4().hex[:6]}",
            "zone_type": "roi",
            "geometry": {"points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]},
        }
        r = requests.post(_svc(self.BASE), json=payload, timeout=5)
        assert r.status_code == 201, r.text
        zid = r.json()["id"]

        # Deactivate — 200 or 204 both acceptable
        r = requests.delete(_svc(f"{self.BASE}/{zid}"), timeout=5)
        assert r.status_code in {200, 204}


# ---------------------------------------------------------------------------
# /admin/events (read-only)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAdminEvents:
    def test_list_events_returns_list(self):
        r = requests.get(_svc("/admin/events"), timeout=5)
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_events_persisted_after_infer(self, real_frame_b64):
        cam = _camera()
        # Warm up with real image (bus.jpg) so YOLO finds people, tracks confirm
        for _ in range(6):
            _infer(real_frame_b64, cam)
        time.sleep(4.0)  # wait for async outbox worker to drain

        r = requests.get(_svc(f"/admin/events?camera_id={cam}&limit=20"), timeout=5)
        assert r.status_code == 200
        events = r.json()
        # At least one detection event should be persisted for this camera
        assert any(e["camera_id"] == cam for e in events), \
            f"No events for camera {cam}. All events: {[e['camera_id'] for e in events]}"


# ---------------------------------------------------------------------------
# /reset
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestReset:
    def test_reset_all(self):
        r = requests.post(_svc("/reset_all"), timeout=5)
        assert r.status_code == 200
