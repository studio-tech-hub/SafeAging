"""Unit tests for edge SQLite store (DATABASE_URL empty mode)."""

from __future__ import annotations

import uuid

import pytest

from people_analytics_service.db import edge_store as store
from people_analytics_service.db import edge_dal


@pytest.fixture
def edge_db(tmp_path, monkeypatch):
    path = str(tmp_path / "edge_state.db")
    monkeypatch.setenv("EDGE_SQLITE_PATH", path)
    store.close_edge_store()
    store.init_edge_store(path)
    yield path
    store.close_edge_store()


@pytest.mark.asyncio
async def test_person_crud(edge_db):
    person = await edge_dal.create_person(name="Alice", room="101")
    assert person.name == "Alice"
    assert person.room == "101"

    fetched = await edge_dal.get_person(person.id)
    assert fetched is not None
    assert fetched.name == "Alice"

    people = await edge_dal.list_persons()
    assert len(people) == 1

    updated = await edge_dal.update_person(person.id, notes="VIP")
    assert updated is not None
    assert updated.notes == "VIP"


@pytest.mark.asyncio
async def test_face_gallery(edge_db):
    person = await edge_dal.create_person(name="Bob", status="active")
    emb_bytes = b"\x00" * 512
    await edge_dal.create_person_embedding(
        person_id=person.id,
        embedding=emb_bytes,
        embedding_type="face",
        source="test",
    )

    gallery = await edge_dal.list_face_gallery()
    assert len(gallery) == 1
    assert gallery[0]["name"] == "Bob"
    assert gallery[0]["embedding"] == emb_bytes


@pytest.mark.asyncio
async def test_upsert_event_idempotent(edge_db):
    from datetime import datetime, timezone

    occurred = datetime.now(timezone.utc)
    e1 = await edge_dal.upsert_event(
        "evt-1",
        camera_id="cam_a",
        event_type="fall",
        occurred_at=occurred,
    )
    e2 = await edge_dal.upsert_event(
        "evt-1",
        camera_id="cam_a",
        event_type="fall",
        occurred_at=occurred,
    )
    assert e1.id == e2.id

    events = await edge_dal.list_events(camera_id="cam_a")
    assert len(events) == 1


@pytest.mark.asyncio
async def test_camera_config_delete(edge_db):
    await edge_dal.upsert_camera_config("cam_x", confidence_threshold=0.5)
    deleted = await edge_dal.delete_camera_config("cam_x")
    assert deleted == 1
    assert await edge_dal.get_camera_config("cam_x") is None
