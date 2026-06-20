#!/usr/bin/env python3
"""
Quick smoke-test for the /infer endpoint.
Downloads a public-domain street image with multiple people,
sends it to the local analytics service, and prints what was detected.

Usage:
    python tools/test_infer_quick.py
    API_KEY=change-me-to-a-strong-secret python tools/test_infer_quick.py
"""
import base64
import json
import os
import sys
import urllib.request

SERVICE_URL = os.getenv("SERVICE_URL", "http://localhost:18000")
API_KEY = os.getenv("API_KEY", "change-me-to-a-strong-secret")
CAMERA_ID = "test_cam_01"

# Ultralytics demo image: a bus stop with ~4 people visible
IMAGE_URL = "https://ultralytics.com/images/bus.jpg"
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "_test_frame.jpg")


def download_image():
    if os.path.exists(IMAGE_PATH):
        print(f"[i] Using cached image: {IMAGE_PATH}")
        return
    print(f"[i] Downloading test image from {IMAGE_URL} ...")
    urllib.request.urlretrieve(IMAGE_URL, IMAGE_PATH)
    print(f"[i] Saved {os.path.getsize(IMAGE_PATH):,} bytes")


def call_infer(image_path: str) -> list:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    payload = json.dumps({"image": b64, "camera_id": CAMERA_ID}).encode()
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
    }
    req = urllib.request.Request(
        f"{SERVICE_URL}/infer", data=payload, headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def main():
    print(f"\n{'='*55}")
    print("  SafeAging — /infer quick smoke-test")
    print(f"{'='*55}")

    # 1. Health
    try:
        with urllib.request.urlopen(f"{SERVICE_URL}/health", timeout=5) as r:
            h = json.loads(r.read())
        print(f"\n[health]  status={h['status']}  ready={h['ready']}  device={h['device']}")
        if not h["ready"]:
            print("[!] Service not ready — aborting infer test")
            sys.exit(1)
    except Exception as e:
        print(f"[!] Cannot reach {SERVICE_URL}/health: {e}")
        sys.exit(1)

    # 2. Download test frame
    try:
        download_image()
    except Exception as e:
        print(f"[!] Could not download test image: {e}")
        print("    Place any JPEG at tools/_test_frame.jpg and re-run.")
        sys.exit(1)

    # 3. Infer
    print(f"\n[infer]   sending frame (camera_id={CAMERA_ID}) ...")
    try:
        detections = call_infer(IMAGE_PATH)
    except Exception as e:
        print(f"[!] /infer failed: {e}")
        sys.exit(1)

    # 4. Print results
    print(f"\n[result]  {len(detections)} detection(s) returned\n")
    if not detections:
        print("  (no objects detected — try a frame with people in it)")
    else:
        print(f"  {'cls':<10} {'score':>6}  {'track_id':>9}  bbox(x,y,w,h)")
        print(f"  {'-'*55}")
        for d in detections:
            print(
                f"  {d['cls']:<10} {d['score']:>6.3f}  {d['track_id']:>9}  "
                f"({d['x']:.1f}, {d['y']:.1f}, {d['w']:.1f}, {d['h']:.1f})"
            )

    # 5. Status
    try:
        req = urllib.request.Request(
            f"{SERVICE_URL}/status",
            headers={"X-API-Key": API_KEY},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            st = json.loads(r.read())
        cams = st.get("cameras", {})
        if CAMERA_ID in cams:
            c = cams[CAMERA_ID]
            print(f"\n[stats]   requests={c.get('request_count')}  "
                  f"errors={c.get('error_count')}  "
                  f"avg_infer={c.get('avg_inference_ms', 0):.0f}ms  "
                  f"tracks={c.get('last_track_count')}")
    except Exception:
        pass

    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    main()
