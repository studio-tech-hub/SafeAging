"""Tier 5 smoke tests — S P2.5 Dashboard/analytics/reporting."""
import httpx
import sys

BASE = "http://localhost:18000"
PASS = []
FAIL = []


def check(name, ok, detail=""):
    if ok:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


# ── Analytics REST API ────────────────────────────────────────────────────────
print("\n[1] Analytics REST API")
for path in [
    "/admin/analytics/summary",
    "/admin/analytics/cameras",
    "/admin/analytics/zones",
    "/admin/analytics/alerts",
    "/admin/analytics/events/timeseries",
    "/admin/analytics/events/timeseries?granularity=hour&days_back=1",
    "/admin/analytics/events/timeseries?event_type=fall",
]:
    r = httpx.get(f"{BASE}{path}", timeout=8)
    check(f"GET {path.split('?')[0]}", r.status_code == 200, f"status={r.status_code}")

# ── Summary content ───────────────────────────────────────────────────────────
print("\n[2] Summary content")
s = httpx.get(f"{BASE}/admin/analytics/summary").json()
check("summary.total_events >= 0", s.get("total_events", -1) >= 0)
check("summary.events_by_type is dict", isinstance(s.get("events_by_type"), dict))
check("summary.active_cameras >= 0", s.get("active_cameras", -1) >= 0)
check("summary.active_zones >= 0", s.get("active_zones", -1) >= 0)
check("summary.alerts has 'total'", "total" in s.get("alerts", {}))
check("summary.generated_at present", bool(s.get("generated_at")))

# ── Cameras content ───────────────────────────────────────────────────────────
print("\n[3] Camera analytics")
cams = httpx.get(f"{BASE}/admin/analytics/cameras").json()
check("cameras is list", isinstance(cams, list))
if cams:
    cam = cams[0]
    for field in ["camera_id", "total_events", "falls", "zone_violations", "unique_tracks"]:
        check(f"camera.{field} present", field in cam)

# ── Zones content ─────────────────────────────────────────────────────────────
print("\n[4] Zone analytics")
zones = httpx.get(f"{BASE}/admin/analytics/zones").json()
check("zones is list", isinstance(zones, list))
if zones:
    z = zones[0]
    for field in ["zone_id", "name", "zone_type", "camera_id", "active", "violation_count"]:
        check(f"zone.{field} present", field in z)

# ── Timeseries content ────────────────────────────────────────────────────────
print("\n[5] Timeseries")
ts = httpx.get(f"{BASE}/admin/analytics/events/timeseries?granularity=day&days_back=7").json()
check("timeseries is list", isinstance(ts, list))
if ts:
    check("timeseries item has bucket", "bucket" in ts[0])
    check("timeseries item has count", "count" in ts[0])

# ── Prometheus metrics ────────────────────────────────────────────────────────
print("\n[6] New Prometheus metrics")
metrics_text = httpx.get(f"{BASE}/metrics", timeout=5).text
for metric in ["safeaging_events_persisted_total", "safeaging_reid_auto_links_total", "safeaging_alerts_sent_total"]:
    check(f"metric declared: {metric}", metric in metrics_text)

# ── Result ────────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"PASSED: {len(PASS)}   FAILED: {len(FAIL)}")
if FAIL:
    print("FAILED tests:", FAIL)
    sys.exit(1)
else:
    print("All Tier 5 checks passed.")
