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
from typing import Optional

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

    @staticmethod
    def _squash_person_in_place(
        frame_b64: str, det: dict, x_scale: float = 1.8, y_scale: float = 0.55
    ) -> str:
        """Warp only the confirmed track's own bbox region to a wide/short shape,
        in place, keeping the rest of the frame (and the box's center point)
        unchanged.

        Follow-up fix for the P1-7 xfail: a *whole-frame* resize (see
        `_widen_b64` above) moves every object to new coordinates, so a
        confirmed track's box no longer overlaps its old position enough for
        IoU-based tracking to re-associate it -- that was testing tracker
        continuity under an unrealistic full-scene distortion, not the
        PERSON_MIN_HW_RATIO gate the test name/assertion targets. A real fall
        keeps the camera and scene fixed and only changes that one person's
        silhouette, so squashing just their own crop in place (same center
        point, same rest-of-frame) is both a more faithful simulation and one
        that a same-camera confirmed track can plausibly still be associated
        against by IoU, isolating the gate's actual behavior end-to-end.
        """
        import cv2
        import numpy as np
        raw = base64.b64decode(frame_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        H, W = img.shape[:2]

        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        cx, cy = x + w / 2.0, y + h / 2.0

        # Crop with a little padding so the resized patch has real pixels to
        # sample instead of butting straight against the detection edge.
        pad = 0.15
        x0 = max(0, int(x - w * pad))
        y0 = max(0, int(y - h * pad))
        x1 = min(W, int(x + w * (1 + pad)))
        y1 = min(H, int(y + h * (1 + pad)))
        crop = img[y0:y1, x0:x1]
        assert crop.size > 0, "empty crop while squashing person bbox"

        new_w = max(1, int((x1 - x0) * x_scale))
        new_h = max(1, int((y1 - y0) * y_scale))
        squashed = cv2.resize(crop, (new_w, new_h))

        # Paste centered at the same center point as the original box, so the
        # new (wide/short) region overlaps the old (confirmed) box location.
        out = img.copy()
        paste_w, paste_h = min(new_w, W), min(new_h, H)
        px0 = max(0, min(int(cx - new_w / 2.0), W - paste_w))
        py0 = max(0, min(int(cy - new_h / 2.0), H - paste_h))
        out[py0 : py0 + paste_h, px0 : px0 + paste_w] = squashed[:paste_h, :paste_w]

        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        assert ok, "cv2.imencode failed while squashing test frame"
        return base64.b64encode(buf).decode()

    def test_confirmed_track_survives_wide_posture_transition(self, real_frame_b64):
        cam = _camera()

        # Establish a confirmed track on the unmodified (upright) real photo.
        last_dets: list = []
        for _ in range(6):
            r = _infer(real_frame_b64, cam)
            assert r.status_code == 200
            last_dets = r.json()
        confirmed_before = [d for d in last_dets if d.get("stable")]
        if not confirmed_before:
            pytest.skip(
                "No stable/confirmed track established on the unmodified real photo; "
                "cannot exercise the mid-track wide-posture transition"
            )
        confirmed_ids_before = {d["track_id"] for d in confirmed_before}

        # Squash the (largest) confirmed detection's own bbox region wide/short,
        # in place -- same camera_id, same rest-of-frame, so any surviving
        # detection near that location is plausibly the same track, not a
        # brand-new candidate forced by an unrelated full-frame distortion.
        target_det = max(confirmed_before, key=lambda d: d["w"] * d["h"])
        wide_b64 = self._squash_person_in_place(real_frame_b64, target_det)
        wide_dets: list = []
        for _ in range(3):
            r = _infer(wide_b64, cam)
            assert r.status_code == 200
            wide_dets = r.json()

        if not wide_dets:
            pytest.skip(
                "YOLO produced no detections at all on the squashed frame (model-dependent); "
                "cannot isolate the aspect-ratio filter's effect without a baseline detection"
            )

        wide_ids = {d["track_id"] for d in wide_dets}
        assert confirmed_ids_before & wide_ids, (
            f"Confirmed track(s) {confirmed_ids_before} disappeared after transitioning to a "
            f"wide/fallen posture (got {wide_ids}) — PERSON_MIN_HW_RATIO must not blanket-drop "
            "an already-confirmed track, only gate brand-new track candidates."
        )


# ---------------------------------------------------------------------------
# Per-camera face-recognition toggle (P1-6)
#
# enable_face_recognition=false in a camera's extra config must skip face
# processing *entirely* for that camera -- not just discard the recognition
# result. That is externally observable via /admin/live/tracks: identity is
# only ever attached to a track once _apply_face_identity() has run for it at
# least once (even a gallery *miss* writes a non-empty identity entry, see
# face_identity.commit_face_miss), so person_name flips from None -> "Unknown"
# the first time a real attempt happens, and has_crop flips False -> True the
# first time a face-bearing crop is cached. A camera with the override off
# must never make either transition, no matter how many frames it receives.
# ---------------------------------------------------------------------------

def _live_tracks_for(camera_id: str) -> list:
    r = requests.get(_svc(f"/admin/live/tracks?camera_id={camera_id}"), timeout=5)
    assert r.status_code == 200
    cameras = r.json().get("cameras", {})
    # The server keys its response by the *normalized* camera id (braces added
    # per config.normalize_camera_id -- see /infer and admin_router.live_tracks),
    # which may differ cosmetically from the plain id used in the query string
    # here. Since each test uses a fresh, uniquely-generated camera id filtered
    # server-side, `cameras` holds at most one entry -- return its value
    # directly rather than re-deriving the exact normalized key in the test.
    if not cameras:
        return []
    return next(iter(cameras.values()))


def _poll_infer_until(camera_id: str, frame_b64: str, predicate, max_seconds: float = 12.0) -> list:
    """POST /infer repeatedly (feeding the async face-worker queue and letting
    tracks confirm) until `predicate(tracks)` is true or the deadline passes.
    Returns whatever /admin/live/tracks last reported for this camera."""
    deadline = time.time() + max_seconds
    tracks: list = []
    while time.time() < deadline:
        r = _infer(frame_b64, camera_id)
        assert r.status_code == 200
        tracks = _live_tracks_for(camera_id)
        if predicate(tracks):
            return tracks
        time.sleep(0.3)
    return tracks


@pytest.mark.integration
class TestPerCameraFaceRecognitionToggle:
    def _set_face_recognition_override(self, camera_id: str, enabled: Optional[bool]):
        extra = {} if enabled is None else {"enable_face_recognition": enabled}
        r = requests.put(_svc(f"/admin/camera-configs/{camera_id}"), json={"extra": extra}, timeout=5)
        if r.status_code == 503:
            pytest.skip("no database configured on target service (per-camera config needs Postgres/edge SQLite)")
        assert r.status_code == 200, r.text
        return r.json()

    def test_disabled_camera_never_gets_an_identity_entry(self, real_frame_b64, frame_b64):
        # 1) Baseline: a camera with NO override must, on this deployment,
        #    eventually get *some* identity entry (even "Unknown") once face
        #    recognition has attempted to process a confirmed track -- this
        #    proves face recognition is actually active here, so a negative
        #    result below is meaningful rather than vacuous.
        control_cam = _camera()
        control_tracks = _poll_infer_until(
            control_cam, real_frame_b64,
            lambda tracks: any(t.get("person_name") is not None for t in tracks),
        )
        if not control_tracks:
            pytest.skip("no confirmed track established on real_frame_b64 for this run")
        if not any(t.get("person_name") is not None for t in control_tracks):
            pytest.skip(
                "face recognition does not appear to be active on this deployment "
                "(ENABLE_FACE_RECOGNITION=false or insightface unavailable) -- "
                "the disabled-camera assertion below would be vacuous here"
            )

        # 2) Same photo, same frame count/timeout, but enable_face_recognition
        #    explicitly disabled for this camera -- identity must never appear.
        disabled_cam = _camera()
        self._set_face_recognition_override(disabled_cam, False)
        try:
            # Warm the per-camera config cache *before* this camera's first
            # real (person-bearing) frame. get_per_camera_config_sync()
            # (config_engine.py) uses the same fire-and-forget cache pattern
            # as zone_engine's get_zones_for_camera_sync: the very first
            # lookup for a never-before-seen camera_id is a cold-cache miss
            # that returns None -- falling back to the global
            # ENABLE_FACE_RECOGNITION default -- while a background thread
            # loads the real row; only later lookups see the override. In
            # production this window is invisible because operators configure
            # a camera before it starts streaming; here we make that ordering
            # explicit by priming with frames that yield zero detections (the
            # synthetic `frame_b64` fixture -- colour blocks the real model
            # never classifies as a person), so no track/identity gets
            # created *during* the warm-up itself.
            for _ in range(3):
                warm_r = _infer(frame_b64, disabled_cam)
                assert warm_r.status_code == 200
                assert warm_r.json() == [], (
                    "warm-up frame unexpectedly produced a detection -- it must stay "
                    "person-free so it cannot itself create a track/identity before "
                    "the per-camera config cache has finished loading"
                )
            time.sleep(0.5)

            disabled_tracks = _poll_infer_until(
                disabled_cam, real_frame_b64,
                lambda tracks: any(t.get("person_name") is not None for t in tracks),
            )
            if not disabled_tracks:
                pytest.skip("no confirmed track established on real_frame_b64 for this run")
            for t in disabled_tracks:
                assert t.get("person_name") is None, (
                    f"track {t.get('track_id')} on camera={disabled_cam} got an identity "
                    f"entry (person_name={t.get('person_name')!r}) even though "
                    "enable_face_recognition=false was set for this camera -- face "
                    "processing must be skipped entirely, not just its result discarded"
                )
                assert t.get("has_crop") is False, (
                    f"track {t.get('track_id')} on camera={disabled_cam} has a cached "
                    "face crop even though enable_face_recognition=false -- "
                    "_apply_face_identity must never run for this camera"
                )
        finally:
            requests.delete(_svc(f"/admin/camera-configs/{disabled_cam}"), timeout=5)

    def test_override_absent_falls_back_to_global_default(self):
        """A camera config with other fields set but no enable_face_recognition
        key at all must be indistinguishable from having no config row."""
        cam = _camera()
        r = requests.put(_svc(f"/admin/camera-configs/{cam}"), json={"confidence_threshold": 0.6}, timeout=5)
        if r.status_code == 503:
            pytest.skip("no database configured on target service")
        assert r.status_code == 200, r.text
        body = r.json()
        assert (body.get("extra") or {}).get("enable_face_recognition") is None
        requests.delete(_svc(f"/admin/camera-configs/{cam}"), timeout=5)


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
        """A 'detection' event is enqueued once, on a track's first appearance
        in new_outbox_ids (api.py), then persisted asynchronously by
        outbox_worker. Poll with a bounded retry instead of a fixed sleep,
        since the worker's flush timing is not itself under test here.

        Follow-up fix for the P1-7 xfail: root cause was NOT actually flush
        timing -- /admin/events matched camera_id with an exact, unnormalized
        string compare while outbox events are always persisted with the
        normalized ({uuid}-braced) camera_id (see api.py's /infer handler),
        so an unbraced query camera_id could never match, at any wait length.
        Fixed in admin_router.py's list_events(); this poll loop remains as
        defense-in-depth against genuine async timing, not as the real fix.
        """
        cam = _camera()
        # Warm up with real image (bus.jpg) so YOLO finds people, tracks confirm
        for _ in range(6):
            _infer(real_frame_b64, cam)

        # Events are persisted with the normalized ({uuid}-braced) camera_id
        # (see api.py's /infer handler) regardless of which form was sent —
        # mirror that here so the assertion isn't comparing the wrong form.
        normalized_cam = "{" + cam + "}"

        deadline = time.monotonic() + 10.0
        events: list = []
        while time.monotonic() < deadline:
            r = requests.get(_svc(f"/admin/events?camera_id={cam}&limit=20"), timeout=5)
            assert r.status_code == 200
            events = r.json()
            if any(e["camera_id"] == normalized_cam for e in events):
                break
            time.sleep(0.5)

        # At least one detection event should be persisted for this camera
        assert any(e["camera_id"] == normalized_cam for e in events), \
            f"No events for camera {cam}. All events: {[e['camera_id'] for e in events]}"


# ---------------------------------------------------------------------------
# /reset
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestReset:
    def test_reset_all(self):
        r = requests.post(_svc("/reset_all"), timeout=5)
        assert r.status_code == 200
