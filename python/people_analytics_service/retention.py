"""S P2.4 — Retention policy: periodic cleanup of old events and alerts.

A background asyncio task runs every RETENTION_CHECK_HOURS hours and hard-deletes
Event rows (+ cascaded Alert rows) whose `occurred_at` is older than RETENTION_DAYS
days.  MinIO snapshot objects referenced by the deleted events are also removed on a
best-effort basis.

Usage
-----
Call `start_retention_worker()` once from the FastAPI `startup_event` handler.
The worker loops forever; it is never explicitly stopped (process exit ends it).
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# Sentinel so we can skip the worker entirely when RETENTION_DAYS <= 0.
_retention_task: asyncio.Task | None = None


async def _delete_old_events(retention_days: int) -> int:
    """Delete events older than `retention_days` days. Returns count deleted."""
    from .db import get_session
    from .db.models import Event
    from sqlalchemy import delete

    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

    async with get_session() as session:
        stmt = delete(Event).where(Event.occurred_at < cutoff)
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount


async def _delete_old_minio_snapshots(retention_days: int) -> int:
    """Remove MinIO snapshot objects older than retention_days days. Best-effort."""
    from .config import S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET_SNAPSHOTS

    if not S3_ENDPOINT or not S3_ACCESS_KEY:
        return 0

    try:
        import aioboto3  # type: ignore[import]
    except ImportError:
        # Fall back to synchronous boto3 in a thread
        return await asyncio.to_thread(_sync_delete_old_minio_snapshots, retention_days)

    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    deleted = 0
    try:
        session = aioboto3.Session()
        async with session.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
        ) as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=S3_BUCKET_SNAPSHOTS):
                for obj in page.get("Contents", []):
                    if obj["LastModified"] < cutoff:
                        await s3.delete_object(
                            Bucket=S3_BUCKET_SNAPSHOTS, Key=obj["Key"]
                        )
                        deleted += 1
    except Exception as exc:
        logger.warning(f"[retention] MinIO cleanup error: {exc}")
    return deleted


def _sync_delete_old_minio_snapshots(retention_days: int) -> int:
    """Synchronous fallback for MinIO cleanup (used when aioboto3 is unavailable)."""
    from .config import S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET_SNAPSHOTS

    try:
        import boto3  # type: ignore[import]

        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        s3 = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
        )
        deleted = 0
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET_SNAPSHOTS):
            for obj in page.get("Contents", []):
                if obj["LastModified"].replace(tzinfo=timezone.utc) < cutoff:
                    s3.delete_object(Bucket=S3_BUCKET_SNAPSHOTS, Key=obj["Key"])
                    deleted += 1
        return deleted
    except Exception as exc:
        logger.warning(f"[retention] MinIO sync cleanup error: {exc}")
        return 0


async def _retention_loop(retention_days: int, check_hours: float) -> None:
    """Main loop: sleep `check_hours` between runs."""
    interval_sec = max(check_hours * 3600, 60)  # minimum 1 min in tests
    logger.info(
        f"[retention] Worker started: retention_days={retention_days} "
        f"check_every={check_hours}h"
    )
    while True:
        await asyncio.sleep(interval_sec)
        try:
            events_deleted = await _delete_old_events(retention_days)
            snapshots_deleted = await _delete_old_minio_snapshots(retention_days)
            if events_deleted or snapshots_deleted:
                logger.info(
                    f"[retention] Cleanup done: events_deleted={events_deleted} "
                    f"snapshots_deleted={snapshots_deleted}"
                )
            else:
                logger.debug("[retention] Cleanup run: nothing to delete")
        except Exception as exc:
            logger.warning(f"[retention] Cleanup error (will retry next cycle): {exc}")


def start_retention_worker(retention_days: int, check_hours: float) -> None:
    """Schedule the background retention loop.  Call once from startup_event."""
    global _retention_task
    if retention_days <= 0:
        logger.info("[retention] Disabled (RETENTION_DAYS<=0)")
        return
    loop = asyncio.get_event_loop()
    _retention_task = loop.create_task(_retention_loop(retention_days, check_hours))
    logger.info(
        f"[retention] Worker scheduled (retention_days={retention_days} "
        f"check_hours={check_hours})"
    )


def get_status() -> dict:
    """Return current retention worker status for the admin API."""
    from .config import RETENTION_DAYS, RETENTION_CHECK_HOURS

    return {
        "enabled": RETENTION_DAYS > 0,
        "retention_days": RETENTION_DAYS,
        "check_hours": RETENTION_CHECK_HOURS,
        "worker_running": _retention_task is not None and not _retention_task.done(),
    }
