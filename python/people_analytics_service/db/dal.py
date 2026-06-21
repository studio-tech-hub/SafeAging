"""Data Access Layer — async CRUD for all schema tables.

Every public function accepts an AsyncSession as its first argument.
Session lifecycle (commit / rollback) is managed by the caller via
get_session() in session.py.

Design notes
------------
- flush() after add() writes the SQL without committing, so the generated
  PK is available immediately within the same transaction.
- upsert_* functions are idempotent: they check-then-insert rather than
  using ON CONFLICT to keep the code portable and easy to test.
- update_* functions explicitly set updated_at because bulk UPDATE statements
  bypass SQLAlchemy's onupdate column event.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import case, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Alert, CameraConfig, Event, Person, PersonEmbedding, Zone
from .session import is_edge_mode


async def _edge_call(func_name: str, *args, **kwargs):
    from . import edge_dal

    return await getattr(edge_dal, func_name)(*args, **kwargs)


# ── Person ────────────────────────────────────────────────────────────────────

async def get_person(session: AsyncSession, person_id: uuid.UUID) -> Person | None:
    if is_edge_mode():
        return await _edge_call("get_person", person_id)
    return await session.get(Person, person_id)


async def list_persons(
    session: AsyncSession,
    status: str | None = None,
    limit: int = 100,
) -> list[Person]:
    if is_edge_mode():
        return await _edge_call("list_persons", status=status, limit=limit)
    stmt = select(Person)
    if status:
        stmt = stmt.where(Person.status == status)
    stmt = stmt.limit(limit).order_by(Person.created_at.desc())
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def create_person(session: AsyncSession, **kwargs: Any) -> Person:
    if is_edge_mode():
        return await _edge_call("create_person", **kwargs)
    person = Person(**kwargs)
    session.add(person)
    await session.flush()
    return person


async def update_person(
    session: AsyncSession,
    person_id: uuid.UUID,
    **kwargs: Any,
) -> Person | None:
    if is_edge_mode():
        return await _edge_call("update_person", person_id, **kwargs)
    kwargs.setdefault("updated_at", datetime.now(timezone.utc))
    stmt = (
        update(Person)
        .where(Person.id == person_id)
        .values(**kwargs)
        .returning(Person)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def delete_person(
    session: AsyncSession,
    person_id: uuid.UUID,
) -> bool:
    """Permanently delete a person and their embeddings. Returns False if not found."""
    if is_edge_mode():
        return await _edge_call("delete_person", person_id)
    person = await session.get(Person, person_id)
    if person is None:
        return False
    await session.execute(
        update(Event).where(Event.person_id == person_id).values(person_id=None)
    )
    await session.delete(person)
    await session.flush()
    return True


# ── PersonEmbedding ───────────────────────────────────────────────────────────

async def create_person_embedding(
    session: AsyncSession, **kwargs: Any
) -> PersonEmbedding:
    if is_edge_mode():
        return await _edge_call("create_person_embedding", **kwargs)
    emb = PersonEmbedding(**kwargs)
    session.add(emb)
    await session.flush()
    return emb


async def list_person_embeddings(
    session: AsyncSession, person_id: uuid.UUID
) -> list[PersonEmbedding]:
    if is_edge_mode():
        return await _edge_call("list_person_embeddings", person_id)
    stmt = (
        select(PersonEmbedding)
        .where(PersonEmbedding.person_id == person_id)
        .order_by(PersonEmbedding.created_at)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_person_embedding(
    session: AsyncSession, emb_id: uuid.UUID
) -> PersonEmbedding | None:
    if is_edge_mode():
        return await _edge_call("get_person_embedding", emb_id)
    return await session.get(PersonEmbedding, emb_id)


async def delete_person_embedding(
    session: AsyncSession, emb_id: uuid.UUID
) -> bool:
    """Hard-delete a single embedding. Returns True if it existed."""
    if is_edge_mode():
        return await _edge_call("delete_person_embedding", emb_id)
    emb = await session.get(PersonEmbedding, emb_id)
    if emb is None:
        return False
    await session.delete(emb)
    await session.flush()
    return True


async def list_face_gallery(session: AsyncSession) -> list[dict]:
    """Return enrolled face embeddings joined with person profile fields.

    Used by the real-time face recognition gallery cache. Only active persons
    and embeddings of type 'face' are returned.
    Each row: {person_id, name, gender, embedding (bytes)}.
    """
    if is_edge_mode():
        return await _edge_call("list_face_gallery")
    from .models import Person as PersonModel

    stmt = (
        select(
            PersonEmbedding.person_id,
            PersonModel.name,
            PersonModel.gender,
            PersonModel.date_of_birth,
            PersonModel.age,
            PersonEmbedding.embedding,
        )
        .join(PersonModel, PersonEmbedding.person_id == PersonModel.id)
        .where(
            PersonModel.status == "active",
            PersonEmbedding.embedding_type == "face",
        )
        .order_by(PersonEmbedding.created_at)
    )
    result = await session.execute(stmt)
    return [
        {
            "person_id": row.person_id,
            "name": row.name,
            "gender": row.gender,
            "date_of_birth": row.date_of_birth,
            "age": row.age,
            "embedding": row.embedding,
        }
        for row in result.fetchall()
    ]


async def list_all_embeddings_for_matching(
    session: AsyncSession,
) -> list[PersonEmbedding]:
    """Return all embeddings for all persons (used for ReID matching).

    Only includes embeddings for active persons. The caller is responsible
    for not calling this too frequently — cache results if needed.
    """
    if is_edge_mode():
        return await _edge_call("list_all_embeddings_for_matching")
    from .models import Person as PersonModel

    stmt = (
        select(PersonEmbedding)
        .join(PersonModel, PersonEmbedding.person_id == PersonModel.id)
        .where(PersonModel.status == "active")
        .order_by(PersonEmbedding.created_at)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


# ── Zone ──────────────────────────────────────────────────────────────────────

async def get_zone(session: AsyncSession, zone_id: uuid.UUID) -> Zone | None:
    if is_edge_mode():
        return await _edge_call("get_zone", zone_id)
    return await session.get(Zone, zone_id)


async def list_zones(
    session: AsyncSession,
    camera_id: str | None = None,
    active_only: bool = True,
) -> list[Zone]:
    if is_edge_mode():
        return await _edge_call("list_zones", camera_id, active_only)
    stmt = select(Zone)
    if camera_id:
        stmt = stmt.where(Zone.camera_id == camera_id)
    if active_only:
        stmt = stmt.where(Zone.active.is_(True))
    stmt = stmt.order_by(Zone.created_at)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def create_zone(session: AsyncSession, **kwargs: Any) -> Zone:
    if is_edge_mode():
        return await _edge_call("create_zone", **kwargs)
    zone = Zone(**kwargs)
    session.add(zone)
    await session.flush()
    return zone


async def update_zone(
    session: AsyncSession,
    zone_id: uuid.UUID,
    **kwargs: Any,
) -> Zone | None:
    if is_edge_mode():
        return await _edge_call("update_zone", zone_id, **kwargs)
    kwargs.setdefault("updated_at", datetime.now(timezone.utc))
    stmt = (
        update(Zone)
        .where(Zone.id == zone_id)
        .values(**kwargs)
        .returning(Zone)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


# ── Event ─────────────────────────────────────────────────────────────────────

async def get_event(session: AsyncSession, event_id: uuid.UUID) -> Event | None:
    if is_edge_mode():
        return await _edge_call("get_event", event_id)
    return await session.get(Event, event_id)


async def get_event_by_event_id(
    session: AsyncSession, event_id: str
) -> Event | None:
    if is_edge_mode():
        return await _edge_call("get_event_by_event_id", event_id)
    stmt = select(Event).where(Event.event_id == event_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_events(
    session: AsyncSession,
    camera_id: str | None = None,
    event_type: str | None = None,
    track_id: int | None = None,
    person_id: uuid.UUID | None = None,
    zone_id: uuid.UUID | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Event]:
    if is_edge_mode():
        return await _edge_call(
            "list_events",
            camera_id=camera_id,
            event_type=event_type,
            track_id=track_id,
            person_id=person_id,
            zone_id=zone_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset,
        )
    stmt = select(Event)
    if camera_id:
        stmt = stmt.where(Event.camera_id == camera_id)
    if event_type:
        stmt = stmt.where(Event.event_type == event_type)
    if track_id is not None:
        stmt = stmt.where(Event.track_id == track_id)
    if person_id is not None:
        stmt = stmt.where(Event.person_id == person_id)
    if zone_id is not None:
        stmt = stmt.where(Event.zone_id == zone_id)
    if start_time is not None:
        stmt = stmt.where(Event.occurred_at >= start_time)
    if end_time is not None:
        stmt = stmt.where(Event.occurred_at <= end_time)
    stmt = stmt.order_by(Event.occurred_at.desc()).offset(offset).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def link_event_to_person(
    session: AsyncSession,
    event_uuid: uuid.UUID,
    person_id: uuid.UUID,
) -> Event | None:
    """Attach an existing Event record to a Person (manual labelling)."""
    if is_edge_mode():
        return await _edge_call("link_event_to_person", event_uuid, person_id)
    stmt = (
        update(Event)
        .where(Event.id == event_uuid)
        .values(person_id=person_id)
        .returning(Event)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def delete_events_before(
    session: AsyncSession,
    cutoff: datetime,
) -> int:
    """Hard-delete events older than cutoff. Returns count deleted."""
    if is_edge_mode():
        return await _edge_call("delete_events_before", cutoff)
    stmt = delete(Event).where(Event.occurred_at < cutoff)
    result = await session.execute(stmt)
    return result.rowcount


async def upsert_event(
    session: AsyncSession,
    event_id: str,
    **kwargs: Any,
) -> Event:
    """Insert an event or return the existing record (idempotent by event_id).

    The DB-level UNIQUE constraint on event_id is the final safeguard;
    this function avoids the constraint violation in the normal path.
    """
    if is_edge_mode():
        return await _edge_call("upsert_event", event_id, **kwargs)
    existing = await get_event_by_event_id(session, event_id)
    if existing is not None:
        return existing
    event = Event(event_id=event_id, **kwargs)
    session.add(event)
    await session.flush()
    return event


# ── Alert ─────────────────────────────────────────────────────────────────────

async def create_alert(session: AsyncSession, **kwargs: Any) -> Alert:
    if is_edge_mode():
        return await _edge_call("create_alert", **kwargs)
    alert = Alert(**kwargs)
    session.add(alert)
    await session.flush()
    return alert


async def update_alert(
    session: AsyncSession,
    alert_id: uuid.UUID,
    **kwargs: Any,
) -> Alert | None:
    if is_edge_mode():
        return await _edge_call("update_alert", alert_id, **kwargs)
    stmt = (
        update(Alert)
        .where(Alert.id == alert_id)
        .values(**kwargs)
        .returning(Alert)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_pending_alerts(
    session: AsyncSession,
    limit: int = 50,
) -> list[Alert]:
    if is_edge_mode():
        return await _edge_call("list_pending_alerts", limit)
    stmt = (
        select(Alert)
        .where(Alert.status == "pending")
        .order_by(Alert.created_at)
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


# ── CameraConfig ──────────────────────────────────────────────────────────────

async def get_camera_config(
    session: AsyncSession, camera_id: str
) -> CameraConfig | None:
    if is_edge_mode():
        return await _edge_call("get_camera_config", camera_id)
    stmt = select(CameraConfig).where(CameraConfig.camera_id == camera_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def upsert_camera_config(
    session: AsyncSession,
    camera_id: str,
    **kwargs: Any,
) -> CameraConfig:
    """Insert or update a camera config row (upsert by camera_id)."""
    if is_edge_mode():
        return await _edge_call("upsert_camera_config", camera_id, **kwargs)
    now = datetime.now(timezone.utc)
    existing = await get_camera_config(session, camera_id)
    if existing is not None:
        for key, value in kwargs.items():
            setattr(existing, key, value)
        existing.updated_at = now
        await session.flush()
        return existing
    config = CameraConfig(camera_id=camera_id, updated_at=now, **kwargs)
    session.add(config)
    await session.flush()
    return config


async def delete_camera_config(session: AsyncSession, camera_id: str) -> int:
    """Delete a camera config row. Returns number of rows deleted."""
    if is_edge_mode():
        return await _edge_call("delete_camera_config", camera_id)
    stmt = delete(CameraConfig).where(CameraConfig.camera_id == camera_id)
    result = await session.execute(stmt)
    await session.flush()
    return result.rowcount


async def record_email_alert_attempt(
    session: AsyncSession,
    db_event_pk: uuid.UUID,
    *,
    success: bool,
    attempted_at: datetime,
) -> None:
    if is_edge_mode():
        return await _edge_call(
            "record_email_alert_attempt",
            db_event_pk,
            success=success,
            attempted_at=attempted_at,
        )
    result = await session.execute(
        select(Alert).where(
            Alert.event_id == db_event_pk, Alert.alert_type == "email"
        )
    )
    alert = result.scalars().first()
    if alert is None:
        return
    alert.attempts = (alert.attempts or 0) + 1
    alert.last_attempt_at = attempted_at
    if success:
        alert.status = "sent"
        alert.sent_at = attempted_at
    else:
        alert.status = "failed"
    await session.flush()


# ── Analytics (S P2.5) ────────────────────────────────────────────────────────

async def get_analytics_summary(session: AsyncSession) -> dict:
    """Return aggregate summary statistics across all tables."""
    if is_edge_mode():
        return await _edge_call("get_analytics_summary")
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    day_ago = now - timedelta(days=1)
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)

    total_events = await session.scalar(select(func.count()).select_from(Event)) or 0

    type_rows = await session.execute(
        select(Event.event_type, func.count().label("n"))
        .group_by(Event.event_type)
        .order_by(func.count().desc())
    )
    events_by_type = {r.event_type: r.n for r in type_rows}

    events_24h = await session.scalar(
        select(func.count()).select_from(Event).where(Event.occurred_at >= day_ago)
    ) or 0
    events_7d = await session.scalar(
        select(func.count()).select_from(Event).where(Event.occurred_at >= week_ago)
    ) or 0
    events_30d = await session.scalar(
        select(func.count()).select_from(Event).where(Event.occurred_at >= month_ago)
    ) or 0
    falls_24h = await session.scalar(
        select(func.count()).select_from(Event).where(
            Event.event_type == "fall", Event.occurred_at >= day_ago
        )
    ) or 0
    zone_violations_24h = await session.scalar(
        select(func.count()).select_from(Event).where(
            Event.event_type == "zone_violation", Event.occurred_at >= day_ago
        )
    ) or 0
    linked_events = await session.scalar(
        select(func.count()).select_from(Event).where(Event.person_id.isnot(None))
    ) or 0

    active_persons = await session.scalar(
        select(func.count()).select_from(Person).where(Person.status == "active")
    ) or 0
    active_zones = await session.scalar(
        select(func.count()).select_from(Zone).where(Zone.active.is_(True))
    ) or 0
    active_cameras = await session.scalar(
        select(func.count(func.distinct(Event.camera_id))).select_from(Event)
    ) or 0

    pending_alerts = await session.scalar(
        select(func.count()).select_from(Alert).where(Alert.status == "pending")
    ) or 0
    sent_alerts = await session.scalar(
        select(func.count()).select_from(Alert).where(Alert.status == "sent")
    ) or 0
    failed_alerts = await session.scalar(
        select(func.count()).select_from(Alert).where(Alert.status == "failed")
    ) or 0

    return {
        "total_events": total_events,
        "events_by_type": events_by_type,
        "events_24h": events_24h,
        "events_7d": events_7d,
        "events_30d": events_30d,
        "falls_24h": falls_24h,
        "zone_violations_24h": zone_violations_24h,
        "linked_events": linked_events,
        "active_persons": active_persons,
        "active_zones": active_zones,
        "active_cameras": active_cameras,
        "alerts": {
            "pending": pending_alerts,
            "sent": sent_alerts,
            "failed": failed_alerts,
            "total": pending_alerts + sent_alerts + failed_alerts,
        },
        "generated_at": now.isoformat(),
    }


async def get_events_timeseries(
    session: AsyncSession,
    granularity: str = "day",
    days_back: int = 7,
    event_type: str | None = None,
) -> list[dict]:
    """Return event counts bucketed by time (uses PostgreSQL date_trunc)."""
    if is_edge_mode():
        return await _edge_call(
            "get_events_timeseries", granularity, days_back, event_type
        )
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    bucket_expr = func.date_trunc(granularity, Event.occurred_at)

    stmt = (
        select(bucket_expr.label("bucket"), func.count().label("count"))
        .where(Event.occurred_at >= cutoff)
        .group_by(bucket_expr)
        .order_by(bucket_expr)
    )
    if event_type:
        stmt = stmt.where(Event.event_type == event_type)

    result = await session.execute(stmt)
    return [
        {"bucket": row.bucket.isoformat(), "count": row.count}
        for row in result.fetchall()
    ]


async def get_camera_analytics(session: AsyncSession) -> list[dict]:
    """Return per-camera aggregate statistics."""
    if is_edge_mode():
        return await _edge_call("get_camera_analytics")
    stmt = (
        select(
            Event.camera_id,
            func.count().label("total_events"),
            func.count(case((Event.event_type == "fall", 1))).label("falls"),
            func.count(case((Event.event_type == "zone_violation", 1))).label("zone_violations"),
            func.count(func.distinct(Event.track_id)).label("unique_tracks"),
            func.count(case((Event.person_id.isnot(None), 1))).label("linked_events"),
            func.max(Event.occurred_at).label("last_seen"),
        )
        .group_by(Event.camera_id)
        .order_by(func.count().desc())
    )
    result = await session.execute(stmt)
    return [
        {
            "camera_id": row.camera_id,
            "total_events": row.total_events,
            "falls": row.falls,
            "zone_violations": row.zone_violations,
            "unique_tracks": row.unique_tracks,
            "linked_events": row.linked_events,
            "last_seen": row.last_seen.isoformat() if row.last_seen else None,
        }
        for row in result.fetchall()
    ]


async def get_zone_analytics(session: AsyncSession) -> list[dict]:
    """Return per-zone violation counts (all zones, including inactive)."""
    if is_edge_mode():
        return await _edge_call("get_zone_analytics")
    stmt = (
        select(
            Zone.id.label("zone_id"),
            Zone.name,
            Zone.zone_type,
            Zone.camera_id,
            Zone.active,
            func.count(Event.id).label("violation_count"),
            func.max(Event.occurred_at).label("last_violation"),
        )
        .outerjoin(Event, Event.zone_id == Zone.id)
        .group_by(Zone.id, Zone.name, Zone.zone_type, Zone.camera_id, Zone.active)
        .order_by(func.count(Event.id).desc())
    )
    result = await session.execute(stmt)
    return [
        {
            "zone_id": str(row.zone_id),
            "name": row.name,
            "zone_type": row.zone_type,
            "camera_id": row.camera_id,
            "active": row.active,
            "violation_count": row.violation_count,
            "last_violation": row.last_violation.isoformat() if row.last_violation else None,
        }
        for row in result.fetchall()
    ]


async def get_alert_analytics(session: AsyncSession) -> dict:
    """Return alert delivery statistics."""
    if is_edge_mode():
        return await _edge_call("get_alert_analytics")
    status_rows = await session.execute(
        select(Alert.status, Alert.alert_type, func.count().label("n"))
        .group_by(Alert.status, Alert.alert_type)
        .order_by(Alert.status)
    )
    by_status: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for row in status_rows:
        by_status[row.status] = by_status.get(row.status, 0) + row.n
        by_type[row.alert_type] = by_type.get(row.alert_type, 0) + row.n

    total = sum(by_status.values())
    return {
        "total": total,
        "by_status": by_status,
        "by_channel": by_type,
    }
