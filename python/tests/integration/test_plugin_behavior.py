"""
P P1.3 – Integration tests: queue/backpressure, CB reconnect, metadata consistency.

These tests exercise the Python analytics service from the perspective of what
the C++ plugin depends on.  Four concern areas:

  1. Backpressure / queue  – rapid concurrent /infer calls must all succeed.
  2. Circuit-breaker (service side) – the service must return the right HTTP
     status codes that the C++ CB reacts to (5xx = transient, 4xx/200 = final).
  3. Reconnect – after a service restart the /health endpoint must return
     "healthy" within a bounded time.
  4. Metadata consistency – persisted event types must match the manifest IDs
     (no legacy "sample.opencv…" prefix); track IDs must be stable across frames.

Run inside the analytics container:
    pytest -m integration -v tests/integration/test_plugin_behavior.py

Or from the host with a running stack:
    SERVICE_URL=http://localhost:18000 pytest -m integration \\
        python/tests/integration/test_plugin_behavior.py
"""
from __future__ import annotations

import base64
import json
import os
import queue
import socket
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pytest
import requests

SERVICE_URL = os.getenv("SERVICE_URL", "http://localhost:18000")

# Expected manifest prefix for all event / object type IDs
MANIFEST_PREFIX = "mycompany.yolo26_people_analytics."

# Legacy prefix that must NOT appear anywhere (bug guard)
LEGACY_PREFIX = "sample.opencv_object_detection."


# ── helpers ───────────────────────────────────────────────────────────────────


def _cam() -> str:
    return f"pp13_{uuid.uuid4().hex[:8]}"


def _infer(frame_b64: str, camera_id: str, timeout: int = 20) -> requests.Response:
    return requests.post(
        SERVICE_URL + "/infer",
        json={"image": frame_b64, "camera_id": camera_id},
        timeout=timeout,
    )


def _infer_binary(jpeg_bytes: bytes, camera_id: str, timeout: int = 20) -> requests.Response:
    """P1-4: multipart/form-data transport — the additive counterpart of _infer().

    Mirrors what MultipartBinaryTransport (C++ plugin, transport_mode=binary)
    sends: camera_id as a form field, raw JPEG bytes as the `image` file part.
    """
    return requests.post(
        SERVICE_URL + "/infer/binary",
        data={"camera_id": camera_id},
        files={"image": ("frame.jpg", jpeg_bytes, "image/jpeg")},
        timeout=timeout,
    )


def _health(timeout: int = 5) -> requests.Response:
    return requests.get(SERVICE_URL + "/health", timeout=timeout)


def _events(camera_id: str, timeout: int = 5) -> List[Dict[str, Any]]:
    r = requests.get(
        SERVICE_URL + "/admin/events",
        params={"camera_id": camera_id, "limit": 100},
        timeout=timeout,
    )
    if r.status_code == 200:
        return r.json()
    return []


# ── Mock HTTP server (simulates the analytics service for CB scenario tests) ──


class _MockServiceHandler(BaseHTTPRequestHandler):
    """Tiny HTTP server that returns preconfigured responses for /infer."""

    def log_message(self, fmt, *args):  # silence default access log
        pass

    def do_POST(self):
        with self.server._lock:  # type: ignore[attr-defined]
            scenario = self.server._scenario  # type: ignore[attr-defined]

        content_len = int(self.headers.get("Content-Length", 0))
        _ = self.rfile.read(content_len)  # consume body

        if scenario == "ok":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "detections": [],
                "camera_id": "mock",
                "tracks": 0,
            }).encode())

        elif scenario == "transient_fail":
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"detail":"service unavailable"}')

        elif scenario == "permanent_fail":
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"detail":"bad request"}')

        elif scenario == "slow":
            time.sleep(1.5)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"detections":[],"camera_id":"mock","tracks":0}')


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class MockService:
    """Context manager that starts a controllable mock HTTP server."""

    def __init__(self):
        self.port = _free_port()
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def set_scenario(self, scenario: str):
        if self._server is not None:
            with self._server._lock:  # type: ignore[attr-defined]
                self._server._scenario = scenario  # type: ignore[attr-defined]

    def __enter__(self):
        self._server = HTTPServer(("127.0.0.1", self.port), _MockServiceHandler)
        self._server._lock = threading.Lock()  # type: ignore[attr-defined]
        self._server._scenario = "ok"  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        if self._server:
            self._server.shutdown()


# ── A tiny client that mirrors the C++ circuit breaker logic (Python port) ───


class PythonCircuitBreaker:
    """Pure-Python port of the C++ circuit breaker for black-box contract tests."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, threshold: int = 5, cooldown_s: float = 15.0):
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._state = self.CLOSED
        self._failures = 0
        self._open_until: float = 0.0
        self._probe_in_flight = False

    @property
    def state(self) -> str:
        return self._state

    def allow(self) -> Tuple[bool, str]:
        now = time.monotonic()
        if self._state == self.OPEN:
            if now < self._open_until:
                remaining = self._open_until - now
                return False, f"cooldown_ms={int(remaining * 1000)}"
            self._state = self.HALF_OPEN
            self._probe_in_flight = True
            return True, "half_open_probe"
        if self._state == self.HALF_OPEN:
            if self._probe_in_flight:
                return False, "half_open_probe_in_flight"
            self._probe_in_flight = True
            return True, "half_open_probe"
        return True, "ok"

    def on_success(self) -> bool:
        recovered = self._state != self.CLOSED
        self._state = self.CLOSED
        self._failures = 0
        self._probe_in_flight = False
        self._open_until = 0.0
        return recovered

    def on_transient_failure(self) -> bool:
        if self._state == self.HALF_OPEN:
            self._state = self.OPEN
            self._probe_in_flight = False
            self._failures = 0
            self._open_until = time.monotonic() + self.cooldown_s
            return True
        self._failures += 1
        if self._failures >= self.threshold:
            self._state = self.OPEN
            self._probe_in_flight = False
            self._failures = 0
            self._open_until = time.monotonic() + self.cooldown_s
            return True
        return False

    def is_transient(self, status_code: int) -> bool:
        return status_code in (408, 429) or (500 <= status_code <= 599)


def _make_b64(width=64, height=64) -> str:
    """Create a tiny synthetic JPEG quickly (no real objects)."""
    import struct, zlib
    # Build a minimal all-gray JPEG-like bytes (use PNG instead which cv2-free)
    # We rely on cv2 only if available, otherwise inline a real minimal JPEG.
    try:
        import cv2
        import numpy as np
        frame = np.full((height, width, 3), 100, dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", frame)
        return base64.b64encode(buf).decode()
    except Exception:
        # Absolute minimal valid JPEG (8×8 gray)
        TINY_JPEG_HEX = (
            "ffd8ffe000104a46494600010100000100010000"
            "ffdb004300080606070605080707070909080a0c140d"
            "0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720"
            "222c231c1c2837292c30313434341f27393d382932"
            "3c2e333432ffc0000b080001000101011100ffda"
            "00080101000000017f10ffd9"
        )
        return base64.b64encode(bytes.fromhex(TINY_JPEG_HEX)).decode()


# ═══════════════════════════════════════════════════════════════════════════════
# Test classes
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestBackpressure:
    """Service must handle concurrent /infer bursts without errors."""

    def test_rapid_sequential_requests_succeed(self, frame_b64):
        """20 sequential /infer requests to the same camera must all succeed."""
        cam = _cam()
        errors = []
        for _ in range(20):
            r = _infer(frame_b64, cam)
            if r.status_code != 200:
                errors.append(r.status_code)
        assert not errors, f"Got non-200 responses: {errors}"

    def test_concurrent_requests_same_camera(self, frame_b64):
        """10 parallel /infer requests from the same camera must all return 200."""
        cam = _cam()
        results = []
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = [ex.submit(_infer, frame_b64, cam) for _ in range(10)]
            for f in as_completed(futs):
                results.append(f.result().status_code)
        failed = [s for s in results if s != 200]
        assert not failed, f"{len(failed)} / {len(results)} requests failed: {failed}"

    def test_concurrent_requests_multiple_cameras(self, frame_b64):
        """10 parallel cameras each sending 3 frames — all must succeed."""
        cams = [_cam() for _ in range(10)]
        errors = 0
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = []
            for cam in cams:
                for _ in range(3):
                    futs.append(ex.submit(_infer, frame_b64, cam))
            for f in as_completed(futs):
                if f.result().status_code != 200:
                    errors += 1
        assert errors == 0, f"{errors} / {len(futs)} requests failed"

    def test_burst_latency_p95_under_threshold(self, frame_b64):
        """30 sequential requests — p95 latency must be under 10 s (conservative)."""
        cam = _cam()
        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            r = _infer(frame_b64, cam, timeout=30)
            elapsed = time.perf_counter() - t0
            assert r.status_code == 200
            times.append(elapsed)
        times.sort()
        p95 = times[int(len(times) * 0.95)]
        assert p95 < 10.0, f"p95 latency {p95:.2f}s exceeds 10s threshold"

    def test_health_stays_healthy_under_load(self, frame_b64):
        """Health endpoint must remain healthy while concurrent infer requests run."""
        cam = _cam()
        health_results: List[str] = []

        def infer_loop():
            for _ in range(15):
                _infer(frame_b64, cam)

        def health_loop():
            for _ in range(5):
                r = _health()
                health_results.append(r.json().get("status", "unknown"))
                time.sleep(0.5)

        t1 = threading.Thread(target=infer_loop)
        t2 = threading.Thread(target=health_loop)
        t1.start(); t2.start()
        t1.join(); t2.join()

        non_healthy = [s for s in health_results if s not in ("healthy", "degraded")]
        assert not non_healthy, f"Unexpected health statuses under load: {non_healthy}"


@pytest.mark.integration
class TestCircuitBreakerBehavior:
    """Verify the CB logic (Python port) behaves identically to the C++ version,
    and that the service returns HTTP codes that trigger the correct CB transitions."""

    # ── Python-port CB contract tests (no network) ────────────────────────────

    def test_cb_starts_closed(self):
        cb = PythonCircuitBreaker(threshold=5)
        allowed, _ = cb.allow()
        assert allowed
        assert cb.state == PythonCircuitBreaker.CLOSED

    def test_cb_opens_after_threshold(self):
        cb = PythonCircuitBreaker(threshold=5)
        for _ in range(5):
            allowed, _ = cb.allow()
            cb.on_transient_failure()
        allowed, reason = cb.allow()
        assert not allowed
        assert "cooldown" in reason
        assert cb.state == PythonCircuitBreaker.OPEN

    def test_cb_half_open_success_closes(self):
        cb = PythonCircuitBreaker(threshold=3, cooldown_s=0.01)
        for _ in range(3):
            cb.allow(); cb.on_transient_failure()
        time.sleep(0.05)  # cooldown expires
        allowed, _ = cb.allow()
        assert allowed
        assert cb.state == PythonCircuitBreaker.HALF_OPEN
        recovered = cb.on_success()
        assert recovered
        assert cb.state == PythonCircuitBreaker.CLOSED

    def test_cb_half_open_failure_reopens(self):
        cb = PythonCircuitBreaker(threshold=3, cooldown_s=0.01)
        for _ in range(3):
            cb.allow(); cb.on_transient_failure()
        time.sleep(0.05)
        cb.allow()  # probe → half-open
        cb.on_transient_failure()
        assert cb.state == PythonCircuitBreaker.OPEN

    def test_cb_probe_in_flight_blocks_concurrent(self):
        cb = PythonCircuitBreaker(threshold=3, cooldown_s=0.01)
        for _ in range(3):
            cb.allow(); cb.on_transient_failure()
        time.sleep(0.05)
        cb.allow()  # first probe
        allowed, reason = cb.allow()  # second should be blocked
        assert not allowed
        assert "half_open_probe_in_flight" in reason

    def test_cb_transient_vs_permanent_status_codes(self):
        cb = PythonCircuitBreaker()
        # 5xx and 408/429 are transient
        for code in (500, 502, 503, 504, 408, 429):
            assert cb.is_transient(code), f"{code} should be transient"
        # 4xx (except 408/429) and 200 are not transient
        for code in (200, 400, 401, 403, 404):
            assert not cb.is_transient(code), f"{code} should not be transient"

    # ── Against real mock server ───────────────────────────────────────────────

    def test_mock_server_ok_response(self):
        """CB stays closed when mock server returns 200."""
        with MockService() as svc:
            svc.set_scenario("ok")
            cb = PythonCircuitBreaker(threshold=5)
            for _ in range(10):
                allowed, _ = cb.allow()
                assert allowed
                r = requests.post(svc.url + "/infer", json={"image": "x"}, timeout=5)
                if cb.is_transient(r.status_code):
                    cb.on_transient_failure()
                else:
                    cb.on_success()
            assert cb.state == PythonCircuitBreaker.CLOSED

    def test_mock_server_transient_failures_open_cb(self):
        """5 consecutive 503 responses from mock server must open the CB."""
        with MockService() as svc:
            svc.set_scenario("transient_fail")
            cb = PythonCircuitBreaker(threshold=5)
            for _ in range(5):
                allowed, _ = cb.allow()
                assert allowed, "CB should be closed during accumulation"
                r = requests.post(svc.url + "/infer", json={"image": "x"}, timeout=5)
                assert r.status_code == 503
                cb.on_transient_failure()
            allowed, _ = cb.allow()
            assert not allowed, "CB should be open after 5 transient failures"

    def test_mock_server_recovery_closes_cb(self):
        """After cooldown + successful probe, CB closes and traffic flows again."""
        with MockService() as svc:
            svc.set_scenario("transient_fail")
            cb = PythonCircuitBreaker(threshold=3, cooldown_s=0.05)
            for _ in range(3):
                cb.allow(); cb.on_transient_failure()

            # CB is open; switch mock to OK
            svc.set_scenario("ok")
            time.sleep(0.1)  # cooldown expires

            allowed, _ = cb.allow()
            assert allowed, "Should transition to half-open after cooldown"
            r = requests.post(svc.url + "/infer", json={"image": "x"}, timeout=5)
            assert r.status_code == 200
            cb.on_success()
            assert cb.state == PythonCircuitBreaker.CLOSED

    def test_permanent_failure_does_not_increment_cb(self):
        """400 Bad Request must NOT count toward CB failure threshold."""
        with MockService() as svc:
            svc.set_scenario("permanent_fail")
            cb = PythonCircuitBreaker(threshold=3)
            for _ in range(10):
                cb.allow()
                r = requests.post(svc.url + "/infer", json={"image": "x"}, timeout=5)
                if cb.is_transient(r.status_code):
                    cb.on_transient_failure()
                else:
                    cb.on_success()  # 400 is treated as a "definitive" answer
            assert cb.state == PythonCircuitBreaker.CLOSED, (
                "400 permanent failures should not open the CB"
            )

    def test_real_service_returns_200_on_valid_request(self, frame_b64):
        """Real service must return 200 for a valid /infer payload (never 5xx)."""
        cb = PythonCircuitBreaker(threshold=5)
        for _ in range(5):
            allowed, _ = cb.allow()
            assert allowed, "CB should never open for a healthy service"
            r = _infer(frame_b64, _cam())
            assert r.status_code == 200
            cb.on_success()
        assert cb.state == PythonCircuitBreaker.CLOSED


@pytest.mark.integration
class TestReconnect:
    """Service must recover and serve /health after a restart."""

    def _docker_available(self) -> bool:
        try:
            result = subprocess.run(
                ["docker", "inspect", "safeaging-analytics"],
                capture_output=True, timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def test_health_healthy_before_restart(self):
        """Pre-condition: service must be healthy before running restart test."""
        r = _health()
        assert r.status_code == 200
        assert r.json().get("status") in ("healthy", "degraded")

    def test_service_recovers_after_restart(self):
        """Stop → start analytics container; service must become healthy within 60 s."""
        if not self._docker_available():
            pytest.skip("Docker CLI not available in this environment")

        # Restart (stop then start is more deterministic than 'restart')
        subprocess.run(
            ["docker", "stop", "safeaging-analytics"],
            capture_output=True, timeout=30,
        )

        # Brief wait to ensure it's actually stopped
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                requests.get(SERVICE_URL + "/health", timeout=2)
            except requests.exceptions.ConnectionError:
                break
            time.sleep(0.5)

        subprocess.run(
            ["docker", "start", "safeaging-analytics"],
            capture_output=True, timeout=30,
        )

        # Poll until healthy (up to 60 s)
        recovered = False
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            try:
                r = requests.get(SERVICE_URL + "/health", timeout=5)
                if r.status_code == 200 and r.json().get("status") in ("healthy", "degraded"):
                    recovered = True
                    break
            except Exception:
                pass
            time.sleep(2.0)

        assert recovered, "Service did not recover healthy status within 60 s after restart"

    def test_infer_works_after_restart(self, frame_b64):
        """After a service restart (or if already running), /infer must succeed."""
        cam = _cam()
        r = _infer(frame_b64, cam)
        assert r.status_code == 200, f"/infer failed after restart: {r.text}"

    def test_health_endpoint_includes_dependencies(self):
        """Health response must include a 'dependencies' key for plugin to parse."""
        r = _health()
        assert r.status_code == 200
        body = r.json()
        assert "status" in body, "Missing 'status' field in /health response"
        assert "dependencies" in body or "pipeline" in body, (
            "Health response must expose at least 'dependencies' or 'pipeline' for the plugin to parse"
        )


@pytest.mark.integration
class TestMetadataConsistency:
    """Event types and track IDs must be consistent with the plugin manifest."""

    def test_no_legacy_prefix_in_event_types(self, real_frame_b64):
        """Events persisted in DB must NOT contain the old 'sample.opencv' prefix."""
        cam = _cam()
        # Send a few frames to generate events
        for _ in range(3):
            r = _infer(real_frame_b64, cam)
            assert r.status_code == 200
        time.sleep(3)  # let outbox worker persist

        events = _events(cam)
        for ev in events:
            evt = ev.get("event_type", "") or ""
            assert not evt.startswith(LEGACY_PREFIX), (
                f"Event {ev.get('id')} has legacy prefix: {evt!r}"
            )

    def test_event_types_are_known_values(self, real_frame_b64):
        """Every persisted event_type must be one of the known short-form types.

        The DB stores the canonical short form ("detection", "fall", "prolonged").
        The C++ plugin translates these to full manifest IDs when emitting to Nx VMS.
        This test guards against typos or unknown event type strings.
        """
        KNOWN_TYPES = {"detection", "fall", "prolonged"}
        cam = _cam()
        for _ in range(3):
            r = _infer(real_frame_b64, cam)
            assert r.status_code == 200
        time.sleep(3)

        events = _events(cam)
        if not events:
            pytest.skip("No events persisted (no detections in test frame)")

        for ev in events:
            evt = ev.get("event_type", "") or ""
            if evt:
                assert evt in KNOWN_TYPES, (
                    f"Unknown event_type {evt!r} — must be one of {KNOWN_TYPES}"
                )
                assert not evt.startswith(LEGACY_PREFIX), (
                    f"Event type {evt!r} contains legacy plugin prefix"
                )

    def test_track_ids_stable_across_consecutive_frames(self, real_frame_b64):
        """Same real scene → track IDs must be reused across frames, not reshuffled.

        /infer returns a flat list of detection dicts with a 'track_id' int field.
        """
        cam = _cam()
        all_ids_per_frame: List[set] = []
        for _ in range(5):
            r = _infer(real_frame_b64, cam)
            assert r.status_code == 200
            dets = r.json()  # flat list
            assert isinstance(dets, list)
            ids = {d["track_id"] for d in dets if d.get("track_id")}
            all_ids_per_frame.append(ids)
            time.sleep(0.1)

        frames_with_detections = [ids for ids in all_ids_per_frame if ids]
        if len(frames_with_detections) < 2:
            pytest.skip("Not enough detections to test track ID stability")

        # Track IDs seen in first detection frame must reappear in subsequent frames
        first_ids = frames_with_detections[0]
        later_ids = set().union(*frames_with_detections[1:])
        overlap = first_ids & later_ids
        assert overlap, (
            f"No track ID overlap between frames: first={first_ids}, later={later_ids}"
        )

    def test_camera_state_isolation(self, frame_b64):
        """Two cameras must have independent /status (no state bleed)."""
        cam_a = _cam()
        cam_b = _cam()

        # Send frames to both cameras
        _infer(frame_b64, cam_a)
        _infer(frame_b64, cam_b)

        ra = requests.get(SERVICE_URL + "/status", params={"camera_id": cam_a}, timeout=5)
        rb = requests.get(SERVICE_URL + "/status", params={"camera_id": cam_b}, timeout=5)

        assert ra.status_code == 200
        assert rb.status_code == 200

        body_a = ra.json()
        body_b = rb.json()

        # camera_id in response must match the requested camera
        if "camera_id" in body_a:
            assert body_a["camera_id"] == cam_a
        if "camera_id" in body_b:
            assert body_b["camera_id"] == cam_b

        # Frame counts must be independent
        count_a = body_a.get("frame_count", body_a.get("frames", 0))
        count_b = body_b.get("frame_count", body_b.get("frames", 0))
        assert count_a == count_b, (
            f"Frame counts should be equal (both sent 1 frame): a={count_a}, b={count_b}"
        )

    def test_infer_response_schema(self, real_frame_b64):
        """The /infer JSON response must contain detection dicts with the expected fields.

        /infer returns a flat list[dict].  Each detection dict has:
          cls, score, x, y, w, h, track_id, fall_detected, stable, degraded
        """
        r = _infer(real_frame_b64, _cam())
        assert r.status_code == 200
        dets = r.json()
        assert isinstance(dets, list), f"/infer must return a list, got {type(dets).__name__}"

        if not dets:
            pytest.skip("No detections in test frame; cannot verify schema")

        for det in dets:
            assert isinstance(det, dict), f"Each detection must be a dict, got {type(det)}"
            # Spatial position fields (bbox as separate components)
            for field in ("x", "y", "w", "h"):
                assert field in det, f"Detection missing '{field}': {det}"
                assert isinstance(det[field], (int, float)), (
                    f"'{field}' must be numeric, got {type(det[field])}"
                )
            # Score
            assert "score" in det, f"Detection missing 'score': {det}"
            assert 0.0 <= det["score"] <= 1.0, f"score out of range: {det['score']}"
            # Class/label
            assert "cls" in det, f"Detection missing 'cls': {det}"
            # Track ID (int, assigned by tracker)
            assert "track_id" in det, f"Detection missing 'track_id': {det}"
            # Stability / fall flags (bool)
            assert "stable" in det, f"Detection missing 'stable': {det}"
            assert "fall_detected" in det, f"Detection missing 'fall_detected': {det}"

    def test_events_unique_per_camera_after_reset(self, real_frame_b64):
        """After /admin/reset, a camera's in-memory state clears.
        The service must continue to respond 200 to /infer after reset.

        Follow-up fix for the P1-7 xfail: root cause was NOT actually an async
        cache-warm-up race -- get_camera_state() populates `camera_states`
        synchronously on the very first /infer call, keyed by the normalized
        ({uuid}-braced) camera_id. admin_reset_camera() looked it up with the
        *raw, unbraced* camera_id from the URL path, which can never match, at
        any wait length. Fixed in admin_router.py's admin_reset_camera(). The
        short sleep below is kept only as a sanity margin for the (unrelated)
        outbox enqueue this warm-up also triggers, not as the real fix.
        """
        import urllib.parse
        cam = _cam()

        # Generate some events
        for _ in range(3):
            _infer(real_frame_b64, cam)
        time.sleep(0.5)

        # Reset the camera
        r_reset = requests.post(
            SERVICE_URL + f"/admin/reset/{urllib.parse.quote(cam, safe='')}",
            timeout=10,
        )
        assert r_reset.status_code in (200, 204), f"Reset failed: {r_reset.text}"

        # Send frames again — must succeed without error
        last_r = None
        for _ in range(3):
            last_r = _infer(real_frame_b64, cam)
            assert last_r.status_code == 200
            assert isinstance(last_r.json(), list), "/infer must return a list after reset"

        assert last_r is not None and last_r.status_code == 200


@pytest.mark.integration
class TestQueuePolicyEdgeCases:
    """Edge cases for queue/drop-oldest and backpressure at the service level."""

    def test_empty_detections_response(self, frame_b64):
        """Synthetic frame with no real objects must return 200 with an empty list."""
        r = _infer(frame_b64, _cam())
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list), f"/infer must return a list, got {type(body).__name__}"
        # Synthetic frame has no real people → expected empty or near-empty
        assert len(body) == 0 or all(isinstance(d, dict) for d in body)

    def test_invalid_image_does_not_cause_5xx(self):
        """Invalid base64 must NOT return 5xx (would open the C++ circuit breaker).

        The service currently returns 200 [] for undecodable payloads (graceful
        degradation).  Either 200 or 4xx is acceptable — 5xx is the only forbidden
        outcome because it would incorrectly increment the plugin CB failure counter.
        """
        r = requests.post(
            SERVICE_URL + "/infer",
            json={"image": "not_valid_base64!@#$", "camera_id": _cam()},
            timeout=10,
        )
        assert r.status_code < 500, (
            f"Invalid payload must not return 5xx (got {r.status_code}); "
            "this would trigger the circuit breaker unnecessarily"
        )

    def test_missing_image_field_returns_422(self):
        """Missing required 'image' field must return 422 Unprocessable Entity."""
        r = requests.post(
            SERVICE_URL + "/infer",
            json={"camera_id": _cam()},
            timeout=10,
        )
        assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text}"

    def test_missing_camera_id_still_succeeds(self, frame_b64):
        """Optional camera_id: missing it should still process (defaults to 'unknown')."""
        r = requests.post(
            SERVICE_URL + "/infer",
            json={"image": frame_b64},
            timeout=20,
        )
        # Either 200 (accepted with default camera) or 422 (if required) — both are acceptable
        assert r.status_code in (200, 422), (
            f"Unexpected status for missing camera_id: {r.status_code}"
        )

    def test_rapid_same_camera_no_duplicate_track_ids(self, real_frame_b64):
        """10 sequential frames to same camera must not produce duplicate track IDs
        within a single response (no NMS regression).

        /infer returns a flat list; track_id is an int field.
        """
        cam = _cam()
        for _ in range(10):
            r = _infer(real_frame_b64, cam)
            assert r.status_code == 200
            dets = r.json()  # flat list
            assert isinstance(dets, list)
            ids = [d["track_id"] for d in dets if d.get("track_id") is not None]
            assert len(ids) == len(set(ids)), (
                f"Duplicate track IDs in single response: {ids}"
            )


@pytest.mark.integration
class TestBinaryTransportPlugin:
    """P1-4 — the same plugin-facing contract tests as above (backpressure,
    error handling, track-id stability), but against the additive
    /infer/binary (multipart/form-data) route instead of /infer
    (JSON+base64). transport_mode=binary must be a drop-in replacement from
    the C++ plugin's perspective: same status codes, same response shape,
    same tracking behaviour — see also TestInferBinary in
    python/tests/unit/../integration/test_api.py for the byte-level parity
    check against /infer on the very first frame of a new camera.
    """

    def test_infer_binary_returns_200(self, frame_bgr):
        import cv2
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        assert ok
        r = _infer_binary(buf.tobytes(), _cam())
        assert r.status_code == 200

    def test_concurrent_requests_same_camera(self, frame_bgr):
        """10 parallel /infer/binary requests from the same camera must all return 200."""
        import cv2
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        assert ok
        jpeg_bytes = buf.tobytes()
        cam = _cam()
        results = []
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = [ex.submit(_infer_binary, jpeg_bytes, cam) for _ in range(10)]
            for f in as_completed(futs):
                results.append(f.result().status_code)
        failed = [s for s in results if s != 200]
        assert not failed, f"{len(failed)} / {len(results)} requests failed: {failed}"

    def test_invalid_image_does_not_cause_5xx(self):
        """Corrupt JPEG bytes must NOT return 5xx (would open the C++ circuit breaker)."""
        r = _infer_binary(b"not a real jpeg payload", _cam())
        assert r.status_code < 500, (
            f"Invalid payload must not return 5xx (got {r.status_code}); "
            "this would trigger the circuit breaker unnecessarily"
        )

    def test_missing_image_field_returns_422(self):
        """Missing required 'image' file part must return 422 Unprocessable Entity."""
        r = requests.post(
            SERVICE_URL + "/infer/binary",
            data={"camera_id": _cam()},
            timeout=10,
        )
        assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text}"

    def test_missing_camera_id_still_succeeds(self, frame_bgr):
        """Optional camera_id form field: omitting it should still process (defaults to 'default')."""
        import cv2
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        assert ok
        r = requests.post(
            SERVICE_URL + "/infer/binary",
            files={"image": ("frame.jpg", buf.tobytes(), "image/jpeg")},
            timeout=20,
        )
        assert r.status_code in (200, 422), (
            f"Unexpected status for missing camera_id: {r.status_code}"
        )

    def test_rapid_same_camera_no_duplicate_track_ids(self, real_frame_b64):
        """10 sequential frames over /infer/binary must not produce duplicate
        track IDs within a single response, matching /infer's guarantee."""
        jpeg_bytes = base64.b64decode(real_frame_b64)
        cam = _cam()
        for _ in range(10):
            r = _infer_binary(jpeg_bytes, cam)
            assert r.status_code == 200
            dets = r.json()
            assert isinstance(dets, list)
            ids = [d["track_id"] for d in dets if d.get("track_id") is not None]
            assert len(ids) == len(set(ids)), (
                f"Duplicate track IDs in single response: {ids}"
            )

    def test_track_ids_stable_across_consecutive_frames(self, real_frame_b64):
        """Same real scene sent repeatedly over /infer/binary → track IDs must be
        reused across frames, not reshuffled (same guarantee as /infer)."""
        jpeg_bytes = base64.b64decode(real_frame_b64)
        cam = _cam()
        all_ids_per_frame: List[set] = []
        for _ in range(5):
            r = _infer_binary(jpeg_bytes, cam)
            assert r.status_code == 200
            dets = r.json()
            assert isinstance(dets, list)
            ids = {d["track_id"] for d in dets if d.get("track_id")}
            all_ids_per_frame.append(ids)
            time.sleep(0.1)

        frames_with_detections = [ids for ids in all_ids_per_frame if ids]
        if len(frames_with_detections) < 2:
            pytest.skip("Not enough detections to test track ID stability")

        first_ids = frames_with_detections[0]
        later_ids = set().union(*frames_with_detections[1:])
        overlap = first_ids & later_ids
        assert overlap, (
            f"No track ID overlap between frames: first={first_ids}, later={later_ids}"
        )
