"""Shared pytest fixtures for all SafeAging test suites."""
from __future__ import annotations

import base64
import os
import uuid

import cv2
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Service URL (override with SERVICE_URL env var)
# ---------------------------------------------------------------------------

SERVICE_URL = os.getenv("SERVICE_URL", "http://localhost:18000")


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires live service")
    config.addinivalue_line("markers", "load: performance / load tests")


def pytest_collection_modifyitems(config, items):
    """Auto-skip integration / load tests unless the matching -m flag is used."""
    skip_integration = pytest.mark.skip(reason="integration tests need -m integration or SERVICE_URL=<url>")
    skip_load = pytest.mark.skip(reason="load tests need -m load")

    for item in items:
        if "integration" in item.keywords and "integration" not in (config.getoption("-m", default="") or ""):
            item.add_marker(skip_integration)
        if "load" in item.keywords and "load" not in (config.getoption("-m", default="") or ""):
            item.add_marker(skip_load)


# ---------------------------------------------------------------------------
# Synthetic test frames (no network downloads needed)
# ---------------------------------------------------------------------------

def _make_bgr_frame(
    width: int = 640,
    height: int = 480,
    *,
    people: int = 2,
) -> np.ndarray:
    """Generate a realistic-looking synthetic BGR frame with coloured rectangles as 'people'."""
    frame = np.full((height, width, 3), fill_value=128, dtype=np.uint8)
    # Draw a floor gradient
    for y in range(height):
        frame[y] = (int(80 + y * 0.2), int(80 + y * 0.15), int(80 + y * 0.1))

    colours = [(0, 100, 200), (200, 50, 50), (50, 200, 50), (200, 200, 0)]
    step = width // (people + 1)
    for i in range(people):
        cx = step * (i + 1)
        # Upright person-like rectangle (taller than wide)
        x1, x2 = cx - 30, cx + 30
        y1, y2 = height // 4, height * 3 // 4
        c = colours[i % len(colours)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, -1)

    return frame


@pytest.fixture(scope="session")
def frame_bgr() -> np.ndarray:
    """640×480 BGR synthetic frame with 2 person-like rectangles."""
    return _make_bgr_frame(640, 480, people=2)


@pytest.fixture(scope="session")
def frame_b64(frame_bgr) -> str:
    """Base-64 JPEG of the synthetic frame."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok, "cv2.imencode failed in fixture"
    return base64.b64encode(buf).decode()


@pytest.fixture
def unique_camera_id() -> str:
    """A unique camera_id per test to avoid state bleed."""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def real_frame_b64() -> str:
    """Base-64 JPEG of a real photo containing people (bus.jpg from Ultralytics).
    Downloaded once per test session; cached to /tmp to survive container restarts.
    """
    import urllib.request

    cache_path = "/tmp/_safeaging_test_bus.jpg"
    if not os.path.exists(cache_path):
        url = "https://ultralytics.com/images/bus.jpg"
        try:
            urllib.request.urlretrieve(url, cache_path)
        except Exception:
            # Fallback: check if the tools smoke-test already cached it
            alt = os.path.join(os.path.dirname(__file__), "..", "..", "tools", "_test_frame.jpg")
            if os.path.exists(alt):
                import shutil
                shutil.copy(alt, cache_path)
            else:
                # Last resort: return the synthetic frame
                pytest.skip("real_frame_b64 fixture: could not download bus.jpg and no cached frame found")
    with open(cache_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ---------------------------------------------------------------------------
# Integration / HTTP client
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def svc_url() -> str:
    return SERVICE_URL


@pytest.fixture(scope="session")
def http(svc_url):
    """requests.Session pointed at the live service."""
    import requests
    s = requests.Session()
    s.base_url = svc_url  # type: ignore[attr-defined]

    def get(path, **kw):
        return s.get(svc_url + path, **kw)

    def post(path, **kw):
        return s.post(svc_url + path, **kw)

    s.get_ = get   # type: ignore[attr-defined]
    s.post_ = post  # type: ignore[attr-defined]
    return s
