"""Dependency health probes with TTL caching."""
import time
from typing import Dict, List, Optional, Tuple

import httpx

from .config import DATABASE_URL, S3_ENDPOINT, logger, HEALTH_AVG_DEGRADED_MS, HEALTH_P95_DEGRADED_MS

_PROBE_TTL: float = 30.0  # seconds between live probes

# Module-level cached state — set last_probed=0 to force an immediate first probe
_state: Dict[str, object] = {
    "postgres": "unknown",
    "object_storage": "unknown",
    "last_probed": 0.0,
}


# ── Individual probes ─────────────────────────────────────────────────────────

async def _probe_postgres() -> str:
    """Try Postgres or edge SQLite depending on configuration.

    Returns: "up" | "down" | "not_configured"
    """
    if not DATABASE_URL:
        from .db.edge_store import is_initialized

        return "up" if is_initialized() else "not_configured"
    try:
        import asyncpg  # type: ignore[import]
        conn = await asyncpg.connect(DATABASE_URL, timeout=5)
        await conn.close()
        return "up"
    except Exception as exc:
        logger.debug(f"[health] postgres probe failed: {exc}")
        return "down"


async def _probe_object_storage() -> str:
    """Hit MinIO's liveness endpoint.

    Returns: "up" | "down" | "not_configured"
    """
    if not S3_ENDPOINT:
        return "not_configured"
    try:
        url = S3_ENDPOINT.rstrip("/") + "/minio/health/live"
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url)
        return "up" if r.status_code == 200 else "down"
    except Exception as exc:
        logger.debug(f"[health] object_storage probe failed: {exc}")
        return "down"


# ── Cached refresh ────────────────────────────────────────────────────────────

async def refresh_probes() -> None:
    """Re-probe all dependencies if the TTL has expired.

    Updates module-level _state and Prometheus gauges.
    Concurrent calls within the TTL window are no-ops.
    """
    now = time.monotonic()
    if now - float(_state["last_probed"]) < _PROBE_TTL:
        return

    # Mark probed immediately to prevent concurrent callers from re-probing
    _state["last_probed"] = now

    postgres_result = await _probe_postgres()
    object_storage_result = await _probe_object_storage()

    _state["postgres"] = postgres_result
    _state["object_storage"] = object_storage_result

    # Update Prometheus gauges (imported lazily to avoid circular imports at
    # module load time — metrics.py imports nothing from health.py)
    from .metrics import DB_UP, OBJECT_STORAGE_UP

    DB_UP.set(1.0 if postgres_result == "up" else 0.0)
    OBJECT_STORAGE_UP.set(1.0 if object_storage_result == "up" else 0.0)

    logger.debug(
        f"[health] probed — postgres={postgres_result} "
        f"object_storage={object_storage_result}"
    )


def dependency_snapshot() -> Dict[str, str]:
    """Return the last-known dependency statuses (non-blocking)."""
    return {
        "postgres": str(_state["postgres"]),
        "object_storage": str(_state["object_storage"]),
    }


# ── Status computation ────────────────────────────────────────────────────────

def compute_status(
    model_ok: bool,
    deps: Dict[str, str],
    pipeline: Optional[Dict[str, object]] = None,
) -> Tuple[str, List[str]]:
    """Derive top-level health status and reason codes.

    Args:
        model_ok: True when model is loaded and warmup succeeded.
        deps:     Result of dependency_snapshot().

    Returns:
        (status, reason_codes)
          status:       "healthy" | "degraded" | "not_ready"
          reason_codes: list of machine-readable reason strings
    """
    if not model_ok:
        return "not_ready", []

    reason_codes: List[str] = []

    if deps.get("postgres") == "down":
        reason_codes.append("db_unreachable")

    if deps.get("object_storage") == "down":
        reason_codes.append("object_storage_unreachable")

    if pipeline:
        cameras = pipeline.get("cameras") or {}
        if isinstance(cameras, dict):
            for _cam_id, stats in cameras.items():
                if not isinstance(stats, dict):
                    continue
                reqs = int(stats.get("requests", 0) or 0)
                if reqs < 5:
                    continue
                avg_ms = float(stats.get("avg_infer_ms", 0) or 0)
                p95_ms = float(stats.get("p95_infer_ms", 0) or 0)
                if avg_ms >= HEALTH_AVG_DEGRADED_MS or p95_ms >= HEALTH_P95_DEGRADED_MS:
                    reason_codes.append("pipeline_latency_high")
                    break

    try:
        import psutil

        load1 = psutil.getloadavg()[0]
        cpu_count = psutil.cpu_count(logical=True) or 1
        if load1 / cpu_count >= 0.85:
            reason_codes.append("host_cpu_overloaded")
    except Exception:
        pass

    status = "degraded" if reason_codes else "healthy"
    return status, reason_codes
