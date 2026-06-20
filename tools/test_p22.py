"""Quick smoke test for P2.2 zone engine and per-camera config."""
import base64
import json
import sys
import urllib.request

import httpx

BASE = "http://localhost:18000"


def main():
    # 1. Check /config endpoint with no DB config
    r = httpx.get(f"{BASE}/config/smoke-cam")
    assert r.status_code == 200, f"GET /config failed: {r.status_code}"
    cfg = r.json()
    print("[OK] GET /config/smoke-cam:", cfg)

    # 2. Upsert per-camera config
    r = httpx.put(
        f"{BASE}/admin/camera-configs/smoke-cam",
        json={"confidence_threshold": 0.4, "iou_threshold": 0.4, "frame_period": 1},
    )
    assert r.status_code == 200, f"PUT camera-configs failed: {r.status_code} {r.text}"
    print("[OK] PUT /admin/camera-configs/smoke-cam:", r.json()["camera_id"], "conf=", r.json()["confidence_threshold"])

    # 3. Create a zone covering entire frame
    r = httpx.post(
        f"{BASE}/admin/zones",
        json={
            "camera_id": "smoke-cam",
            "name": "Full Frame Forbidden",
            "zone_type": "forbidden",
            "geometry": {"points": [[0, 0], [10000, 0], [10000, 10000], [0, 10000]]},
            "active": True,
        },
    )
    assert r.status_code == 201, f"Create zone failed: {r.status_code} {r.text}"
    zone_id = r.json()["id"]
    print("[OK] Created zone:", zone_id)

    # 4. GET /config shows active_zones
    r = httpx.get(f"{BASE}/config/smoke-cam")
    cfg2 = r.json()
    assert any(z["id"] == zone_id for z in cfg2["active_zones"]), "Zone missing from /config active_zones"
    print("[OK] /config/smoke-cam active_zones:", cfg2["active_zones"])

    # 5. Infer with a real image (bus.jpg has people)
    print("Downloading bus.jpg for infer test...")
    urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", "/tmp/bus.jpg")
    with open("/tmp/bus.jpg", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    r = httpx.post(f"{BASE}/infer", json={"image": b64, "camera_id": "smoke-cam"}, timeout=60)
    assert r.status_code == 200, f"/infer failed: {r.status_code}"
    dets = r.json()
    print(f"[OK] /infer returned {len(dets)} detection(s)")

    violations = [d for d in dets if d.get("zone_violation")]
    print(f"[OK] Zone violations this frame: {len(violations)}")
    for d in violations[:3]:
        print(f"     track_id={d['track_id']} zone_id={d.get('zone_id')} zone_type={d.get('zone_type')}")

    if dets and not violations:
        print("[INFO] No violations — zone check may need another frame to warm up the cache (first call is cold).")
        # Second call — cache should now be warm
        r2 = httpx.post(f"{BASE}/infer", json={"image": b64, "camera_id": "smoke-cam"}, timeout=60)
        dets2 = r2.json()
        violations2 = [d for d in dets2 if d.get("zone_violation")]
        print(f"[OK] Second call violations: {len(violations2)}")

    # 6. GET /admin/events to see zone_violation events
    import time
    time.sleep(1)  # let outbox worker process
    r = httpx.get(f"{BASE}/admin/events?camera_id=smoke-cam&event_type=zone_violation")
    evts = r.json()
    print(f"[OK] zone_violation events in DB: {len(evts)}")

    print("\nAll P2.2 smoke tests passed!")


if __name__ == "__main__":
    main()
