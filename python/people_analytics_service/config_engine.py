"""
P2.2 — Per-camera Config Engine
================================
Loads CameraConfig from PostgreSQL per camera_id and caches it with a TTL.

Design notes
------------
- Same sync-safe pattern as zone_engine: cache is read immediately;
  a background daemon thread refreshes it when stale.
- Falls back to service-wide defaults (config.py) when no DB record exists.
- Expose `get_per_camera_config_sync(camera_id)` for the sync /infer thread.
- Expose `invalidate_config_cache(camera_id)` for the admin config endpoints.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CONFIG_CACHE_TTL_SEC: float = 60.0

# camera_id → (loaded_at: float, config_dict | None)
_config_cache: dict[str, tuple[float, Optional[dict]]] = {}
_config_cache_lock = threading.Lock()

_refresh_in_progress: set[str] = set()
_refresh_guard_lock = threading.Lock()


def get_per_camera_config_sync(camera_id: str) -> Optional[dict]:
    """Return per-camera config dict, or None if not set.

    Keys (all Optional):
        confidence_threshold, iou_threshold, frame_period, zone_ids, extra
    """
    now = time.monotonic()
    with _config_cache_lock:
        entry = _config_cache.get(camera_id)

    if entry is None or (now - entry[0]) >= _CONFIG_CACHE_TTL_SEC:
        _schedule_refresh(camera_id)

    return entry[1] if entry else None


def invalidate_config_cache(camera_id: Optional[str] = None) -> None:
    with _config_cache_lock:
        if camera_id:
            _config_cache.pop(camera_id, None)
        else:
            _config_cache.clear()


def _schedule_refresh(camera_id: str) -> None:
    with _refresh_guard_lock:
        if camera_id in _refresh_in_progress:
            return
        _refresh_in_progress.add(camera_id)
    t = threading.Thread(target=_bg_load_config, args=(camera_id,), daemon=True, name=f"cfg-refresh-{camera_id[:8]}")
    t.start()


def _bg_load_config(camera_id: str) -> None:
    try:
        from .async_bridge import run_async

        cfg_dict = run_async(_async_load_config(camera_id))
        with _config_cache_lock:
            _config_cache[camera_id] = (time.monotonic(), cfg_dict)
        if cfg_dict:
            logger.debug("[config_engine][%s] Loaded per-camera config: %s", camera_id, cfg_dict)
    except Exception as exc:
        logger.debug("[config_engine][%s] Config load failed: %s", camera_id, exc)
    finally:
        with _refresh_guard_lock:
            _refresh_in_progress.discard(camera_id)


async def _async_load_config(camera_id: str) -> Optional[dict]:
    from .db import get_session
    from .db.dal import get_camera_config
    async with get_session() as session:
        cfg = await get_camera_config(session, camera_id)
    if cfg is None:
        return None
    return {
        "confidence_threshold": cfg.confidence_threshold,
        "iou_threshold": cfg.iou_threshold,
        "frame_period": cfg.frame_period,
        "zone_ids": cfg.zone_ids or [],
        "extra": cfg.extra or {},
    }
