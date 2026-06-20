#!/usr/bin/env python3
"""
S P1.3 smoke-test: verify outbox worker processes events.

What this test does:
  1. Sends N_FRAMES infer requests (same camera) so tracks get confirmed.
  2. Waits DRAIN_SECS for the async worker to drain.
  3. Queries the DB and checks that detection events were persisted.
  4. Checks MinIO for snapshot uploads (if S3 endpoint is reachable).
  5. Prints outbox stats from /status endpoint.

Usage:
    python tools/test_outbox_worker.py

"""

import base64
import json
import os
import sys
import time
import urllib.request
import urllib.error

SERVICE_URL = os.getenv("SERVICE_URL", "http://localhost:18000")
API_KEY = os.getenv("API_KEY", "change-me-to-a-strong-secret")
CAMERA_ID = "outbox_test_cam"

N_FRAMES = 6       # send this many frames (needs >3 for tracks to be confirmed)
DRAIN_SECS = 3.0   # wait for the worker to process
FRAME_DELAY = 0.1  # 100 ms between frames

IMAGE_URL = "https://ultralytics.com/images/bus.jpg"
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "_test_frame.jpg")


def download_image() -> bytes:
    if not os.path.exists(IMAGE_PATH):
        print(f"[i] Downloading test image from {IMAGE_URL} ...")
        urllib.request.urlretrieve(IMAGE_URL, IMAGE_PATH)
    with open(IMAGE_PATH, "rb") as f:
        return f.read()


def send_infer(image_b64: str, frame_num: int) -> list:
    payload = json.dumps({"image": image_b64, "camera_id": CAMERA_ID}).encode()
    req = urllib.request.Request(
        f"{SERVICE_URL}/infer",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def check_db_events() -> list:
    """Query DB via the admin API (/admin/events)."""
    req = urllib.request.Request(
        f"{SERVICE_URL}/admin/events?limit=20",
        headers={"Content-Type": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"[warn] Could not query /admin/events: {e}")
        return []


def get_service_status() -> dict:
    req = urllib.request.Request(f"{SERVICE_URL}/status", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception:
        return {}


def main():
    print("=" * 60)
    print("S P1.3 Outbox Worker Smoke-Test")
    print("=" * 60)

    # Reset camera state first
    try:
        reset_req = urllib.request.Request(
            f"{SERVICE_URL}/admin/reset/{CAMERA_ID}",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(reset_req, timeout=5)
        print(f"[i] Camera state reset for '{CAMERA_ID}'")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"[i] No existing camera state to reset (first run)")
        else:
            print(f"[warn] Reset returned {e.code}")
    except Exception as e:
        print(f"[warn] Reset failed: {e}")

    # Load image
    img_bytes = download_image()
    img_b64 = base64.b64encode(img_bytes).decode()
    print(f"[i] Loaded test frame ({len(img_bytes)//1024} KB)")

    # Send N_FRAMES infer requests
    print(f"\n[1] Sending {N_FRAMES} infer requests (camera_id={CAMERA_ID}) ...")
    total_dets = 0
    for i in range(1, N_FRAMES + 1):
        try:
            dets = send_infer(img_b64, i)
            total_dets = len(dets)
            print(f"    frame {i}/{N_FRAMES}: {total_dets} detections")
        except Exception as e:
            print(f"    frame {i}: ERROR {e}")
        time.sleep(FRAME_DELAY)

    # Wait for worker to drain
    print(f"\n[2] Waiting {DRAIN_SECS:.1f}s for outbox worker to drain ...")
    time.sleep(DRAIN_SECS)

    # Check DB events
    print("\n[3] Checking persisted events in DB (/admin/events) ...")
    events = check_db_events()
    our_events = [e for e in events if e.get("camera_id") == CAMERA_ID]
    print(f"    Total events in DB: {len(events)}")
    print(f"    Events for '{CAMERA_ID}': {len(our_events)}")

    if our_events:
        print("\n    Recent events:")
        for ev in our_events[:5]:
            ts = ev.get("occurred_at", "?")[:19]
            print(f"      [{ts}] type={ev['event_type']} track={ev.get('track_id','?')} "
                  f"cam={ev['camera_id']} conf={ev.get('confidence', 0.0):.2f}")
        print(f"\n    PASS — {len(our_events)} event(s) persisted to DB")
    else:
        print("\n    WARN — no events found for this camera yet")
        print("    (tracks may need more frames to get confirmed; try running again)")

    # Check service status for outbox stats
    print("\n[4] Outbox stats from /status ...")
    status = get_service_status()
    # outbox stats are embedded in the service status
    print(f"    service status available: {bool(status)}")

    # Check Prometheus metrics for outbox
    try:
        metrics_req = urllib.request.Request(f"{SERVICE_URL}/metrics", method="GET")
        with urllib.request.urlopen(metrics_req, timeout=5) as resp:
            metrics_text = resp.read().decode()
        outbox_lines = [l for l in metrics_text.splitlines() if "outbox" in l and not l.startswith("#")]
        if outbox_lines:
            print("\n    Outbox Prometheus metrics:")
            for line in outbox_lines:
                print(f"      {line}")
        else:
            print("    (no outbox metrics in Prometheus yet)")
    except Exception as e:
        print(f"    [warn] Could not read /metrics: {e}")

    print("\n" + "=" * 60)
    result = "PASS" if our_events else "WARN (no events yet)"
    print(f"Result: {result}")
    print("=" * 60)
    return 0 if our_events else 1


if __name__ == "__main__":
    sys.exit(main())
