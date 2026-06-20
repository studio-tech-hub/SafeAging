"""
S P2.3 — Alert Engine
=====================
Centralised alert sender with:
  - Deduplication  — same (camera_id, event_type, track_id) suppressed within DEDUPE_SEC
  - Rate limiting  — max N alerts per camera per sliding window
  - Retry          — exponential backoff up to MAX_RETRIES attempts
  - Multi-type     — fall, zone_violation (future: SMS/push via _dispatch_alert routing)

Public API
----------
configure(...)     — call once at startup to inject config values
send_alert(event, db_event_pk)  — coroutine; returns True if sent, False if suppressed/failed
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
import threading
import time
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import uuid as _uuid_mod
    from .outbox_worker import OutboxEvent

logger = logging.getLogger(__name__)

# ── runtime config (populated by configure() at startup) ─────────────────────
_DEDUPE_SEC: float = 60.0
_RATE_LIMIT_MAX: int = 10           # max alerts per camera per window
_RATE_LIMIT_WINDOW_SEC: float = 600.0
_MAX_RETRIES: int = 3
_RETRY_BACKOFF_BASE: float = 2.0    # seconds; attempt-1 uses 1s, attempt-2 uses 2s, etc.


def configure(
    dedupe_sec: float = 60.0,
    rate_limit_max: int = 10,
    rate_limit_window_sec: float = 600.0,
    max_retries: int = 3,
) -> None:
    global _DEDUPE_SEC, _RATE_LIMIT_MAX, _RATE_LIMIT_WINDOW_SEC, _MAX_RETRIES
    _DEDUPE_SEC = dedupe_sec
    _RATE_LIMIT_MAX = rate_limit_max
    _RATE_LIMIT_WINDOW_SEC = rate_limit_window_sec
    _MAX_RETRIES = max_retries
    logger.info(
        "[alert_engine] Configured: dedupe=%ss rate=%d/%ss retries=%d",
        dedupe_sec, rate_limit_max, rate_limit_window_sec, max_retries,
    )


# ── deduplication ─────────────────────────────────────────────────────────────
# key: (camera_id, event_type, track_id) → last_sent_at (monotonic)
_dedupe_cache: dict[tuple, float] = {}
_dedupe_lock = threading.Lock()


def _should_suppress(camera_id: str, event_type: str, track_id: Optional[int]) -> bool:
    key = (camera_id, event_type, track_id)
    now = time.monotonic()
    with _dedupe_lock:
        last = _dedupe_cache.get(key)
        if last is not None and (now - last) < _DEDUPE_SEC:
            return True
        _dedupe_cache[key] = now
        # GC: evict entries older than 2× TTL while we have the lock
        stale = [k for k, t in _dedupe_cache.items() if (now - t) > _DEDUPE_SEC * 2]
        for k in stale:
            del _dedupe_cache[k]
    return False


# ── rate limiting ─────────────────────────────────────────────────────────────
_rate_buckets: dict[str, list[float]] = {}
_rate_lock = threading.Lock()


def _check_rate_limit(camera_id: str) -> bool:
    """True = allowed (under limit); False = rate-limited."""
    now = time.monotonic()
    with _rate_lock:
        bucket = _rate_buckets.setdefault(camera_id, [])
        _rate_buckets[camera_id] = [t for t in bucket if (now - t) < _RATE_LIMIT_WINDOW_SEC]
        if len(_rate_buckets[camera_id]) >= _RATE_LIMIT_MAX:
            return False
        _rate_buckets[camera_id].append(now)
    return True


# ── public entry point ────────────────────────────────────────────────────────

async def send_alert(
    event: "OutboxEvent",
    db_event_pk: "Optional[_uuid_mod.UUID]" = None,
) -> bool:
    """Send alert for OutboxEvent with dedupe + rate limit + retry.

    Returns True if the alert was dispatched successfully, False otherwise.
    Does NOT raise — all errors are logged.
    """
    if _should_suppress(event.camera_id, event.event_type, event.track_id):
        logger.debug(
            "[alert_engine][%s] Dedupe-suppressed %s alert track=%s",
            event.camera_id, event.event_type, event.track_id,
        )
        return False

    if not _check_rate_limit(event.camera_id):
        logger.warning(
            "[alert_engine][%s] Rate-limited: skipping %s alert",
            event.camera_id, event.event_type,
        )
        return False

    last_exc: Optional[Exception] = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            await _dispatch_alert(event)
            logger.info(
                "[alert_engine][%s] Alert sent type=%s track=%s attempt=%d",
                event.camera_id, event.event_type, event.track_id, attempt,
            )
            _increment_stat("sent")
            await _update_alert_status(db_event_pk, success=True, attempts=attempt)
            return True
        except Exception as exc:
            last_exc = exc
            _increment_stat("errors")
            if attempt < _MAX_RETRIES:
                backoff = _RETRY_BACKOFF_BASE ** (attempt - 1)
                logger.warning(
                    "[alert_engine][%s] Attempt %d/%d failed: %s — retrying in %.1fs",
                    event.camera_id, attempt, _MAX_RETRIES, exc, backoff,
                )
                await asyncio.sleep(backoff)

    logger.error(
        "[alert_engine][%s] Alert FAILED after %d attempts: %s",
        event.camera_id, _MAX_RETRIES, last_exc,
    )
    await _update_alert_status(db_event_pk, success=False, attempts=_MAX_RETRIES)
    return False


# ── dispatch (channel routing) ────────────────────────────────────────────────

async def _dispatch_alert(event: "OutboxEvent") -> None:
    """Route to the appropriate channel. Extend here for SMS/push."""
    from .config import SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM, ALERT_EMAIL_TO

    if not SMTP_HOST or not ALERT_EMAIL_TO:
        return  # no channel configured — silent no-op

    await asyncio.to_thread(
        _smtp_send,
        SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD,
        SMTP_FROM, ALERT_EMAIL_TO, event,
    )


def _smtp_send(
    host: str,
    port: int,
    user: str,
    password: str,
    from_addr: str,
    to_addr: str,
    event: "OutboxEvent",
) -> None:
    """Blocking SMTP send — runs inside asyncio.to_thread."""
    ts = event.occurred_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    # Use the recognised person's name when available; fall back to a generic label.
    who = getattr(event, "person_name", None) or f"Người (track {event.track_id})"

    if event.event_type == "fall":
        subject = f"[SafeAging] {who} bị té — {event.camera_id}"
        body = (
            f"CẢNH BÁO TÉ NGÃ\n"
            f"====================\n"
            f"Nội dung  : {who} té\n"
            f"Camera    : {event.camera_id}\n"
            f"Track ID  : {event.track_id}\n"
            f"Thời gian : {ts}\n"
            f"Confidence: {event.confidence:.0%}\n"
            f"Event ID  : {event.event_id}\n"
        )
    elif event.event_type == "zone_violation":
        subject = f"[SafeAging] {who} vào vùng nguy hiểm — {event.camera_id}"
        body = (
            f"CẢNH BÁO VÙNG NGUY HIỂM\n"
            f"====================\n"
            f"Nội dung  : {who} đi vào vùng nguy hiểm\n"
            f"Camera    : {event.camera_id}\n"
            f"Zone ID   : {event.zone_id or 'unknown'}\n"
            f"Track ID  : {event.track_id}\n"
            f"Thời gian : {ts}\n"
            f"Confidence: {event.confidence:.0%}\n"
            f"Event ID  : {event.event_id}\n"
        )
    else:
        subject = f"[SafeAging] {event.event_type} — {event.camera_id}"
        body = f"Cảnh báo\nCamera: {event.camera_id}\nLoại: {event.event_type}\nThời gian: {ts}\n"

    if getattr(event, "snapshot_jpeg", None):
        body += f"Snapshot: {_snapshot_key_str(event)}\n"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.attach(MIMEText(body, "plain"))

    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=10) as server:
            if user:
                server.login(user, password)
            server.sendmail(from_addr, [to_addr], msg.as_string())
    else:
        with smtplib.SMTP(host, port, timeout=10) as server:
            server.ehlo()
            try:
                server.starttls()
                server.ehlo()
            except smtplib.SMTPNotSupportedError:
                pass
            if user:
                server.login(user, password)
            server.sendmail(from_addr, [to_addr], msg.as_string())


def _snapshot_key_str(event: "OutboxEvent") -> str:
    date_str = event.occurred_at.strftime("%Y/%m/%d")
    return f"snapshots/{event.camera_id}/{date_str}/{event.event_id}.jpg"


# ── DB status update ──────────────────────────────────────────────────────────

async def _update_alert_status(
    db_event_pk: "Optional[_uuid_mod.UUID]",
    *,
    success: bool,
    attempts: int,
) -> None:
    if db_event_pk is None:
        return
    now = datetime.now(tz=timezone.utc)
    try:
        from sqlalchemy import update as sa_update
        from .db import get_session
        from .db.models import Alert
        async with get_session() as session:
            await session.execute(
                sa_update(Alert)
                .where(Alert.event_id == db_event_pk)
                .values(
                    status="sent" if success else "failed",
                    attempts=attempts,
                    last_attempt_at=now,
                    sent_at=now if success else None,
                )
            )
            await session.commit()
    except Exception as exc:
        logger.debug("[alert_engine] DB status update failed: %s", exc)


# ── stats ─────────────────────────────────────────────────────────────────────
_stats: dict = {"sent": 0, "errors": 0, "suppressed": 0, "rate_limited": 0}


def _increment_stat(key: str) -> None:
    _stats[key] = _stats.get(key, 0) + 1


def get_stats() -> dict:
    return dict(_stats)
