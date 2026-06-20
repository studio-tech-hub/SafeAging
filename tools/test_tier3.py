"""Smoke tests for P2 Tier 3 (P P2.3 + S P2.4)."""
import json
import sys
import httpx

BASE = "http://localhost:18000"


def ok(label, cond, detail=""):
    if cond:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label}  {detail}")
        sys.exit(1)


def test_health():
    r = httpx.get(f"{BASE}/health", timeout=5)
    ok("health 200", r.status_code == 200)
    ok("health status not not_ready", r.json()["status"] in ("healthy", "degraded"))


def test_retention_status():
    r = httpx.get(f"{BASE}/admin/retention/status", timeout=5)
    ok("retention_status 200", r.status_code == 200, r.text)
    d = r.json()
    ok("retention_status.enabled key present", "enabled" in d)
    ok("retention_status.retention_days present", "retention_days" in d)
    ok("retention_status.worker_running present", "worker_running" in d)
    print(f"     retention config: {d}")


def test_events_new_filters():
    # track_id filter
    r = httpx.get(f"{BASE}/admin/events?track_id=9999&limit=5", timeout=5)
    ok("events?track_id 200", r.status_code == 200, r.text)
    ok("events?track_id returns list", isinstance(r.json(), list))

    # person_id filter (random UUID — expect empty list)
    r = httpx.get(
        f"{BASE}/admin/events?person_id=00000000-0000-0000-0000-000000000001&limit=5",
        timeout=5,
    )
    ok("events?person_id 200", r.status_code == 200, r.text)

    # start_time / end_time filter
    r = httpx.get(
        f"{BASE}/admin/events?start_time=2026-01-01T00:00:00Z&end_time=2026-12-31T23:59:59Z&limit=5",
        timeout=5,
    )
    ok("events?start_time+end_time 200", r.status_code == 200, r.text)

    # offset pagination
    r = httpx.get(f"{BASE}/admin/events?offset=0&limit=10", timeout=5)
    ok("events?offset 200", r.status_code == 200, r.text)


def test_person_history_endpoint():
    # Create a person first
    r = httpx.post(
        f"{BASE}/admin/persons",
        json={"name": "Test Resident", "status": "active"},
        timeout=5,
    )
    ok("create person 201", r.status_code == 201, r.text)
    person_id = r.json()["id"]

    # History should be empty (no events linked yet)
    r = httpx.get(f"{BASE}/admin/persons/{person_id}/history", timeout=5)
    ok("person history 200", r.status_code == 200, r.text)
    ok("person history is list", isinstance(r.json(), list))
    ok("person history initially empty", len(r.json()) == 0)

    # Cleanup
    httpx.delete(f"{BASE}/admin/persons/{person_id}", timeout=5)


def test_link_event_to_person():
    # Need an event to link — grab the most recent one if any
    r = httpx.get(f"{BASE}/admin/events?limit=1", timeout=5)
    ok("list events 200", r.status_code == 200, r.text)
    events = r.json()

    if not events:
        print("  [SKIP] link-person test: no events in DB yet")
        return

    event_uuid = events[0]["id"]

    # Create a person to link
    r = httpx.post(
        f"{BASE}/admin/persons",
        json={"name": "Link Test Person", "status": "active"},
        timeout=5,
    )
    ok("create person for link 201", r.status_code == 201, r.text)
    person_id = r.json()["id"]

    # Link event to person
    r = httpx.post(
        f"{BASE}/admin/events/{event_uuid}/link-person",
        json={"person_id": person_id},
        timeout=5,
    )
    ok("link-person 200", r.status_code == 200, r.text)
    ok("link-person returns event with person_id", r.json()["person_id"] == person_id)

    # Verify history shows the event
    r = httpx.get(f"{BASE}/admin/persons/{person_id}/history", timeout=5)
    ok("person history after link 200", r.status_code == 200)
    ok("person history has 1 event after link", len(r.json()) == 1)

    # Filter history by event_type
    linked_event_type = events[0]["event_type"]
    r = httpx.get(
        f"{BASE}/admin/persons/{person_id}/history?event_type={linked_event_type}",
        timeout=5,
    )
    ok("person history event_type filter 200", r.status_code == 200)

    # Cleanup
    httpx.delete(f"{BASE}/admin/persons/{person_id}", timeout=5)


if __name__ == "__main__":
    print("=== Tier 3 Smoke Tests ===")
    print("\n[health]")
    test_health()
    print("\n[retention status]")
    test_retention_status()
    print("\n[events new filters]")
    test_events_new_filters()
    print("\n[person history endpoint]")
    test_person_history_endpoint()
    print("\n[link event to person]")
    test_link_event_to_person()
    print("\n=== All Tier 3 tests passed ===")
