"""Async wrappers around edge_store — mirrors db.dal without AsyncSession."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any

from . import edge_store as store


async def _run(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)


async def get_person(person_id: uuid.UUID):
    return await _run(store.get_person, person_id)


async def list_persons(status: str | None = None, limit: int = 100):
    return await _run(store.list_persons, status, limit)


async def create_person(**kwargs: Any):
    return await _run(store.create_person, **kwargs)


async def update_person(person_id: uuid.UUID, **kwargs: Any):
    return await _run(store.update_person, person_id, **kwargs)


async def delete_person(person_id: uuid.UUID) -> bool:
    return await _run(store.delete_person, person_id)


async def create_person_embedding(**kwargs: Any):
    return await _run(store.create_person_embedding, **kwargs)


async def list_person_embeddings(person_id: uuid.UUID):
    return await _run(store.list_person_embeddings, person_id)


async def get_person_embedding(emb_id: uuid.UUID):
    return await _run(store.get_person_embedding, emb_id)


async def delete_person_embedding(emb_id: uuid.UUID) -> bool:
    return await _run(store.delete_person_embedding, emb_id)


async def list_face_gallery():
    return await _run(store.list_face_gallery)


async def list_all_embeddings_for_matching():
    return await _run(store.list_all_embeddings_for_matching)


async def get_zone(zone_id: uuid.UUID):
    return await _run(store.get_zone, zone_id)


async def list_zones(camera_id: str | None = None, active_only: bool = True):
    return await _run(store.list_zones, camera_id, active_only)


async def create_zone(**kwargs: Any):
    return await _run(store.create_zone, **kwargs)


async def update_zone(zone_id: uuid.UUID, **kwargs: Any):
    return await _run(store.update_zone, zone_id, **kwargs)


async def delete_zone(zone_id: uuid.UUID) -> bool:
    return await _run(store.delete_zone, zone_id)


async def get_event(event_id: uuid.UUID):
    return await _run(store.get_event, event_id)


async def get_event_by_event_id(event_id: str):
    return await _run(store.get_event_by_event_id, event_id)


async def list_events(**kwargs: Any):
    return await _run(store.list_events, **kwargs)


async def link_event_to_person(event_uuid: uuid.UUID, person_id: uuid.UUID):
    return await _run(store.link_event_to_person, event_uuid, person_id)


async def delete_events_before(cutoff: datetime) -> int:
    return await _run(store.delete_events_before, cutoff)


async def upsert_event(event_id: str, **kwargs: Any):
    return await _run(store.upsert_event, event_id, **kwargs)


async def create_alert(**kwargs: Any):
    return await _run(store.create_alert, **kwargs)


async def update_alert(alert_id: uuid.UUID, **kwargs: Any):
    return await _run(store.update_alert, alert_id, **kwargs)


async def list_pending_alerts(limit: int = 50):
    return await _run(store.list_pending_alerts, limit)


async def get_camera_config(camera_id: str):
    return await _run(store.get_camera_config, camera_id)


async def upsert_camera_config(camera_id: str, **kwargs: Any):
    return await _run(store.upsert_camera_config, camera_id, **kwargs)


async def delete_camera_config(camera_id: str) -> int:
    return await _run(store.delete_camera_config, camera_id)


async def record_email_alert_attempt(
    db_event_pk: uuid.UUID,
    *,
    success: bool,
    attempted_at: datetime,
) -> None:
    alert = await _run(
        store.get_alert_by_event_and_type, db_event_pk, "email"
    )
    if alert is None:
        return
    attempts = (alert.attempts or 0) + 1
    updates: dict[str, Any] = {
        "attempts": attempts,
        "last_attempt_at": attempted_at,
        "status": "sent" if success else "failed",
    }
    if success:
        updates["sent_at"] = attempted_at
    await _run(store.update_alert, alert.id, **updates)


async def get_analytics_summary():
    return await _run(store.get_analytics_summary)


async def get_events_timeseries(
    granularity: str = "day",
    days_back: int = 7,
    event_type: str | None = None,
):
    return await _run(
        store.get_events_timeseries, granularity, days_back, event_type
    )


async def get_camera_analytics():
    return await _run(store.get_camera_analytics)


async def get_zone_analytics():
    return await _run(store.get_zone_analytics)


async def get_alert_analytics():
    return await _run(store.get_alert_analytics)
