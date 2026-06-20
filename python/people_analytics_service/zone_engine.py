"""
S P2.2 — Zone Engine
====================
Polygon-based zone violation detection.

Zone geometry is stored as {"points": [[x,y], ...]} in the same pixel
coordinate space as the /infer input frame (after remap_detections_to_input_space).

Zone types
----------
forbidden   — person enters zone = violation event (stateful, fires once per entry)
entry       — person crosses INTO zone = violation event
exit        — person crosses OUT OF zone = violation event
roi         — informational only; no violation event generated

Design notes
------------
- `infer` runs in a thread-pool worker (sync FastAPI route).
  Async DB calls are executed in a dedicated background thread with its own
  event loop; the sync caller receives the cached result immediately.
- On first call the cache is cold → returns []. A background thread loads
  zones from DB. Next call (a few seconds later) returns the real zones.
- Invalidate the cache after zone CRUD via admin API.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid as _uuid
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── tunables ─────────────────────────────────────────────────────────────────
_ZONE_CACHE_TTL_SEC: float = 30.0

# ── in-memory stores ─────────────────────────────────────────────────────────
# camera_id → (loaded_at: float, zones: list)
_zone_cache: dict[str, tuple[float, list]] = {}
_zone_cache_lock = threading.Lock()

# camera_id → track_id → frozenset[zone_id_str] (current membership)
_zone_track_state: dict[str, dict[int, frozenset]] = {}
_zone_state_lock = threading.Lock()

# guard against spawning duplicate refresh threads per camera
_refresh_in_progress: set[str] = set()
_refresh_guard_lock = threading.Lock()


# ── public types ──────────────────────────────────────────────────────────────

@dataclass
class ZoneViolation:
    zone_id: str
    zone_name: str
    zone_type: str
    track_id: int


# ── zone cache ────────────────────────────────────────────────────────────────

def get_zones_for_camera_sync(camera_id: str) -> list:
    """Thread-safe zone list access from the sync /infer thread.

    Returns the cached list immediately. Schedules a background refresh
    if the cache is cold or older than _ZONE_CACHE_TTL_SEC.
    """
    now = time.monotonic()
    with _zone_cache_lock:
        entry = _zone_cache.get(camera_id)

    if entry is None or (now - entry[0]) >= _ZONE_CACHE_TTL_SEC:
        _schedule_refresh(camera_id)

    return entry[1] if entry else []


def invalidate_zone_cache(camera_id: Optional[str] = None) -> None:
    """Invalidate zone list cache and track state after zone CRUD."""
    with _zone_cache_lock:
        if camera_id:
            _zone_cache.pop(camera_id, None)
        else:
            _zone_cache.clear()
    with _zone_state_lock:
        if camera_id:
            _zone_track_state.pop(camera_id, None)
        else:
            _zone_track_state.clear()


def _schedule_refresh(camera_id: str) -> None:
    with _refresh_guard_lock:
        if camera_id in _refresh_in_progress:
            return
        _refresh_in_progress.add(camera_id)
    t = threading.Thread(target=_bg_load_zones, args=(camera_id,), daemon=True, name=f"zone-refresh-{camera_id[:8]}")
    t.start()


def _bg_load_zones(camera_id: str) -> None:
    """Run in a daemon thread; DB work is scheduled on the FastAPI event loop."""
    try:
        from .async_bridge import run_async

        zones = run_async(_async_load_zones(camera_id))
        with _zone_cache_lock:
            _zone_cache[camera_id] = (time.monotonic(), zones)
        logger.debug("[zone_engine][%s] Loaded %d zone(s) from DB", camera_id, len(zones))
    except Exception as exc:
        logger.warning("[zone_engine][%s] DB load failed: %s", camera_id, exc)
    finally:
        with _refresh_guard_lock:
            _refresh_in_progress.discard(camera_id)


async def _async_load_zones(camera_id: str) -> list:
    from .db import get_session
    from .db.dal import list_zones as dal_list_zones
    async with get_session() as session:
        return await dal_list_zones(session, camera_id=camera_id, active_only=True)


# ── geometry helpers ─────────────────────────────────────────────────────────

def _point_in_polygon(px: float, py: float, polygon: list) -> bool:
    """True if (px, py) is inside or on the boundary of the polygon."""
    if len(polygon) < 3:
        return False
    pts = np.array(polygon, dtype=np.float32).reshape((-1, 1, 2))
    # >= 0: inside or on boundary; < 0: outside
    return cv2.pointPolygonTest(pts, (float(px), float(py)), False) >= 0


# ── zone check ────────────────────────────────────────────────────────────────

def check_zones(detections: list, zones: list, camera_id: str) -> list[ZoneViolation]:
    """Check each stable detection's foot point against all active zones.

    Updates per-camera membership state for entry/exit zone types.
    Returns a list of ZoneViolation fired this frame.

    Side effect: annotates each Detection object with zone_id, zone_type,
    and zone_violation for the highest-priority violation (if any).
    """
    if not zones or not detections:
        return []

    with _zone_state_lock:
        cam_state: dict[int, frozenset] = dict(_zone_track_state.get(camera_id, {}))

    violations: list[ZoneViolation] = []
    new_state: dict[int, frozenset] = {}

    for det in detections:
        if not det.stable or det.degraded:
            continue

        # Foot point: bottom-centre of bbox — best proxy for person's floor position.
        foot_x = det.x + det.w / 2.0
        foot_y = det.y + det.h
        track_id = det.track_id
        prev_inside: frozenset = cam_state.get(track_id, frozenset())
        currently_inside: set[str] = set()
        det_violations: list[ZoneViolation] = []

        for zone in zones:
            geometry = zone.geometry or {}
            polygon = geometry.get("points", [])
            if len(polygon) < 3:
                continue

            zone_id_str = str(zone.id)
            inside = _point_in_polygon(foot_x, foot_y, polygon)

            if inside:
                currently_inside.add(zone_id_str)

            if zone.zone_type == "forbidden":
                # Fire once on first entry; re-fire if they leave and return.
                if inside and zone_id_str not in prev_inside:
                    det_violations.append(ZoneViolation(zone_id=zone_id_str, zone_name=zone.name, zone_type=zone.zone_type, track_id=track_id))
            elif zone.zone_type == "entry":
                if inside and zone_id_str not in prev_inside:
                    det_violations.append(ZoneViolation(zone_id=zone_id_str, zone_name=zone.name, zone_type=zone.zone_type, track_id=track_id))
            elif zone.zone_type == "exit":
                if not inside and zone_id_str in prev_inside:
                    det_violations.append(ZoneViolation(zone_id=zone_id_str, zone_name=zone.name, zone_type=zone.zone_type, track_id=track_id))
            # roi: no violation

        new_state[track_id] = frozenset(currently_inside)

        if det_violations:
            primary = det_violations[0]
            det.zone_id = primary.zone_id
            det.zone_type = primary.zone_type
            det.zone_violation = True
            violations.extend(det_violations)
            logger.info(
                "[zone_engine][%s] Track %d → zone '%s' (%s)",
                camera_id, track_id, primary.zone_name, primary.zone_type,
            )

    # Merge new state; keep tracks that left frame (for exit detection continuity).
    cam_state.update(new_state)
    active_ids = {d.track_id for d in detections}
    for stale_id in [tid for tid in list(cam_state) if tid not in active_ids]:
        cam_state.pop(stale_id, None)

    with _zone_state_lock:
        _zone_track_state[camera_id] = cam_state

    return violations
