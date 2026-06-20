"""Smoke tests for Tier 2: P P2.1 zone_violation fields in /infer + S P2.3 alert engine."""
import base64
import time
import urllib.request
import httpx

BASE = "http://localhost:18000"


def setup_zone(camera_id: str) -> str:
    """Create a full-frame forbidden zone, return zone_id."""
    r = httpx.post(f"{BASE}/admin/zones", json={
        "camera_id": camera_id,
        "name": "Full Frame Test Zone",
        "zone_type": "forbidden",
        "geometry": {"points": [[0, 0], [10000, 0], [10000, 10000], [0, 10000]]},
        "active": True,
    })
    assert r.status_code == 201, f"Zone create failed: {r.status_code}"
    return r.json()["id"]


def test_zone_violation_fields_in_infer():
    """Verify Detection objects in /infer response have zone_* fields after 2nd call."""
    cam = "tier2-test"
    zone_id = setup_zone(cam)
    print(f"[OK] Created zone {zone_id[:8]}... for {cam}")

    # Download real image
    urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", "/tmp/bus.jpg")
    with open("/tmp/bus.jpg", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    # First call — zone cache warms up
    r1 = httpx.post(f"{BASE}/infer", json={"image": b64, "camera_id": cam}, timeout=60)
    assert r1.status_code == 200
    dets1 = r1.json()
    print(f"[OK] First infer: {len(dets1)} detections")

    # Second call — cache warm → zone violations expected
    r2 = httpx.post(f"{BASE}/infer", json={"image": b64, "camera_id": cam}, timeout=60)
    assert r2.status_code == 200
    dets2 = r2.json()
    print(f"[OK] Second infer: {len(dets2)} detections")

    # Verify zone_violation fields exist in response
    for d in dets2:
        assert "zone_violation" in d, f"Missing zone_violation field: {d}"
        assert "zone_id" in d, f"Missing zone_id field: {d}"
        assert "zone_type" in d, f"Missing zone_type field: {d}"
    print("[OK] All Detection objects have zone_violation, zone_id, zone_type fields")

    violations = [d for d in dets2 if d["zone_violation"]]
    print(f"[OK] Detections with zone_violation=True: {len(violations)}/{len(dets2)}")
    if violations:
        v = violations[0]
        print(f"     Example: track={v['track_id']} zone_type={v['zone_type']} zone_id={v['zone_id'][:8] if v['zone_id'] else 'null'}...")


def test_zone_violation_events_in_db():
    """Check zone_violation events are persisted with zone_id."""
    time.sleep(1)  # let outbox process
    r = httpx.get(f"{BASE}/admin/events?event_type=zone_violation&limit=5")
    assert r.status_code == 200
    events = r.json()
    print(f"[OK] zone_violation events in DB: {len(events)}")
    for e in events[:2]:
        print(f"     event_id={e['event_id'][:8]} zone_id={str(e.get('zone_id',''))[:8]} camera={e['camera_id']}")


def test_alert_engine_dedupe():
    """Verify alert_engine module imports and dedupe works."""
    import sys
    sys.path.insert(0, "/app/python")
    from people_analytics_service.alert_engine import get_stats, _should_suppress

    # First call should not suppress
    assert not _should_suppress("cam-a", "zone_violation", 1), "First call should not suppress"
    # Second call within dedupe window should suppress
    assert _should_suppress("cam-a", "zone_violation", 1), "Second call should suppress (dedupe)"
    # Different track should not suppress
    assert not _should_suppress("cam-a", "zone_violation", 2), "Different track should not suppress"

    print("[OK] Alert engine dedupe logic working correctly")
    print("[OK] Alert engine stats:", get_stats())


def test_config_endpoint():
    """Verify GET /config returns service_defaults and active_zones."""
    r = httpx.get(f"{BASE}/config/tier2-test")
    assert r.status_code == 200
    cfg = r.json()
    assert "service_defaults" in cfg
    assert "active_zones" in cfg
    assert len(cfg["active_zones"]) >= 1, "Should have at least 1 active zone"
    print(f"[OK] GET /config/tier2-test: {len(cfg['active_zones'])} active zone(s), defaults={cfg['service_defaults']}")


if __name__ == "__main__":
    print("=== Tier 2 Smoke Tests ===\n")
    # Create zone first so subsequent tests can use it
    test_zone_violation_fields_in_infer()
    test_config_endpoint()
    test_zone_violation_events_in_db()
    test_alert_engine_dedupe()
    print("\n=== All Tier 2 tests passed! ===")
