"""
Load tests — simulate multiple cameras sending requests concurrently.

Run explicitly with:
    pytest -m load python/tests/load/ -v

Env vars:
    SERVICE_URL      — default http://localhost:18000
    LOAD_CAMERAS     — number of concurrent cameras (default 4)
    LOAD_FRAMES      — frames per camera (default 10)
    LOAD_P95_MAX_MS  — p95 latency budget in ms (default 2000)
"""
from __future__ import annotations

import concurrent.futures
import os
import statistics
import time
import uuid
from dataclasses import dataclass, field
from typing import List

import pytest
import requests

SERVICE_URL = os.getenv("SERVICE_URL", "http://localhost:18000")
CAMERAS = int(os.getenv("LOAD_CAMERAS", "4"))
FRAMES_PER_CAM = int(os.getenv("LOAD_FRAMES", "10"))
P95_BUDGET_MS = float(os.getenv("LOAD_P95_MAX_MS", "2000"))


@dataclass
class CameraResult:
    camera_id: str
    latencies_ms: List[float] = field(default_factory=list)
    errors: int = 0
    total: int = 0


def _run_camera(camera_id: str, frame_b64: str, n_frames: int) -> CameraResult:
    result = CameraResult(camera_id=camera_id)
    for _ in range(n_frames):
        t0 = time.perf_counter()
        try:
            r = requests.post(
                SERVICE_URL + "/infer",
                json={"image": frame_b64, "camera_id": camera_id},
                timeout=30,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            result.total += 1
            if r.status_code == 200:
                result.latencies_ms.append(elapsed_ms)
            else:
                result.errors += 1
        except Exception:
            result.errors += 1
            result.total += 1
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.load
class TestConcurrentCameras:
    def test_multiple_cameras_parallel(self, frame_b64):
        """Send CAMERAS cameras × FRAMES_PER_CAM frames in parallel."""
        camera_ids = [f"load_{uuid.uuid4().hex[:6]}" for _ in range(CAMERAS)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=CAMERAS) as pool:
            futures = {
                pool.submit(_run_camera, cid, frame_b64, FRAMES_PER_CAM): cid
                for cid in camera_ids
            }
            results: List[CameraResult] = []
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())

        all_latencies = [ms for r in results for ms in r.latencies_ms]
        total_requests = sum(r.total for r in results)
        total_errors = sum(r.errors for r in results)
        error_rate = total_errors / max(1, total_requests)

        p50 = statistics.median(all_latencies) if all_latencies else 9999
        p95 = _percentile(all_latencies, 95) if all_latencies else 9999

        print(f"\n{'='*60}")
        print(f"Load test: {CAMERAS} cameras × {FRAMES_PER_CAM} frames")
        print(f"Total requests : {total_requests}")
        print(f"Errors         : {total_errors}  (rate={error_rate:.1%})")
        print(f"p50 latency    : {p50:.0f} ms")
        print(f"p95 latency    : {p95:.0f} ms  (budget={P95_BUDGET_MS:.0f} ms)")
        for r in results:
            cam_p95 = _percentile(r.latencies_ms, 95) if r.latencies_ms else 9999
            print(f"  {r.camera_id}: {r.total} req, {r.errors} err, p95={cam_p95:.0f}ms")
        print(f"{'='*60}")

        # Hard assertions
        assert error_rate < 0.05, f"Error rate {error_rate:.1%} exceeded 5%"
        assert p95 <= P95_BUDGET_MS, f"p95={p95:.0f}ms exceeds budget={P95_BUDGET_MS:.0f}ms"

    def test_single_camera_sustained(self, frame_b64):
        """Single camera: 20 frames, check p95 and zero errors."""
        cam = f"load_single_{uuid.uuid4().hex[:6]}"
        result = _run_camera(cam, frame_b64, 20)

        assert result.errors == 0, f"{result.errors} errors on single camera"
        p95 = _percentile(result.latencies_ms, 95) if result.latencies_ms else 9999
        assert p95 <= P95_BUDGET_MS, f"p95={p95:.0f}ms exceeded budget"

    def test_health_not_degraded_under_load(self, frame_b64):
        """Health endpoint should remain reachable and report healthy/degraded (not 500)."""
        camera_ids = [f"load_health_{uuid.uuid4().hex[:6]}" for _ in range(CAMERAS)]

        # Start load in background
        with concurrent.futures.ThreadPoolExecutor(max_workers=CAMERAS + 1) as pool:
            load_futs = [
                pool.submit(_run_camera, cid, frame_b64, FRAMES_PER_CAM)
                for cid in camera_ids
            ]
            # Health check during load
            h = requests.get(SERVICE_URL + "/health", timeout=5)
            assert h.status_code == 200, "Health check failed during load"
            assert h.json().get("status") in {"healthy", "degraded"}

            # Wait for load to finish
            for f in concurrent.futures.as_completed(load_futs):
                f.result()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * pct / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]
