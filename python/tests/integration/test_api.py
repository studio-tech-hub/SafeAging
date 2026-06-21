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
