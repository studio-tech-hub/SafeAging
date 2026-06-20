"""Quick smoke test for S P1.2 admin endpoints."""
import json
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:18000/admin"


def req(method, path, body=None):
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if data else {}
    try:
        r = urllib.request.urlopen(
            urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
        )
        return json.loads(r.read()) if r.status != 204 else {"status": 204}
    except urllib.error.HTTPError as e:
        return {"error": e.code, "detail": e.read().decode()}


print("=" * 60)
print("S P1.2 Admin API smoke test")
print("=" * 60)

# ── Person CRUD ──────────────────────────────────────────────────
p = req("POST", "/persons", {"name": "Nguyen Van A", "age": 75, "room": "101"})
print(f"\n[POST /admin/persons]")
print(f"  id={p.get('id')}  name={p.get('name')}  status={p.get('status')}")

pid = p.get("id")
if pid:
    p2 = req("GET", f"/persons/{pid}")
    print(f"\n[GET  /admin/persons/{pid[:8]}...]")
    print(f"  age={p2.get('age')}  room={p2.get('room')}")

    p3 = req("PUT", f"/persons/{pid}", {"notes": "Resident since 2024", "room": "102"})
    print(f"\n[PUT  /admin/persons/{pid[:8]}...]")
    print(f"  notes={p3.get('notes')}  room={p3.get('room')}")

all_persons = req("GET", "/persons?limit=5")
print(f"\n[GET  /admin/persons]  count={len(all_persons) if isinstance(all_persons, list) else 'ERR'}")

# ── Zone CRUD ────────────────────────────────────────────────────
z = req("POST", "/zones", {
    "camera_id": "IP Camera",
    "name": "Main Corridor",
    "zone_type": "roi",
    "geometry": {"points": [[0, 0], [1, 0], [1, 1], [0, 1]]},
})
print(f"\n[POST /admin/zones]")
print(f"  id={z.get('id')}  name={z.get('name')}  active={z.get('active')}")

zid = z.get("id")
if zid:
    z2 = req("PUT", f"/zones/{zid}", {"name": "Corridor West", "active": True})
    print(f"\n[PUT  /admin/zones/{zid[:8]}...]")
    print(f"  name={z2.get('name')}  active={z2.get('active')}")

all_zones = req("GET", "/zones?camera_id=IP+Camera")
print(f"\n[GET  /admin/zones?camera_id=IP Camera]  count={len(all_zones) if isinstance(all_zones, list) else 'ERR'}")

# ── Events (read-only) ───────────────────────────────────────────
events = req("GET", "/events?limit=3")
print(f"\n[GET  /admin/events?limit=3]  count={len(events) if isinstance(events, list) else 'ERR'}")
if isinstance(events, list) and events:
    e = events[0]
    print(f"  first: event_type={e.get('event_type')}  camera={e.get('camera_id')}")

# ── Admin reset ──────────────────────────────────────────────────
rs = req("POST", "/reset/IP%20Camera")
print(f"\n[POST /admin/reset/IP Camera]")
print(f"  {rs}")

print("\n" + "=" * 60)
print("Done.")
