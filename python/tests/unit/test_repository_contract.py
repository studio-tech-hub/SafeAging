"""P2-2: contract tests for db.repository.

One set of assertions, run against BOTH backends db.dal already dispatches
between (edge SQLite and Postgres) — this is the concrete fix for the
"silent drift" risk called out in the P2-2 plan: the two backends implement
the same call surface completely independently under the hood, with
nothing today asserting they behave identically for the same inputs.

Edge-SQLite runs unconditionally (pure sqlite3, no external services,
exactly like tests/unit/test_edge_store.py). The Postgres side is real
opt-in integration coverage — it needs a live, migrated Postgres reachable
at DATABASE_URL — and is marked `@pytest.mark.integration` so it's skipped
by default (see tests/conftest.py's auto-skip) and additionally
self-skips with a clear reason if DATABASE_URL is unset or unreachable.
We deliberately do NOT fake the Postgres branch with a local SQLite engine:
db/models.py uses Postgres-only column types (UUID, JSONB) that don't
compile against SQLite, so a fake would test nothing real.

Run just the always-on edge-SQLite half:
    pytest tests/unit/test_repository_contract.py

Run both halves against a live Postgres:
    DATABASE_URL=postgresql://... pytest -m integration tests/unit/test_repository_contract.py
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from people_analytics_service.db import repository
from people_analytics_service.db import session as db_session

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(params=["edge", pytest.param("postgres", marks=pytest.mark.integration)])
async def repo(request, tmp_path, monkeypatch):
    """Yield (repository, session) for one backend, via the SAME startup
    path the real app uses (db.session.init_engine()) — not a shortcut
    straight into db.edge_store/db.dal internals."""
    backend = request.param

    if backend == "edge":
        edge_path = str(tmp_path / "contract_edge.db")
        monkeypatch.setenv("EDGE_SQLITE_PATH", edge_path)
        # init_engine() falls back to os.environ["DATABASE_URL"] whenever its
        # own argument is falsy (`database_url or os.getenv(...)`), so an
        # ambient DATABASE_URL from .env would silently defeat the "" we pass
        # below — clear it explicitly rather than relying on argument order.
        monkeypatch.setenv("DATABASE_URL", "")
        db_session.init_engine("")  # empty DATABASE_URL -> edge mode
        try:
            async with db_session.get_session() as session:
                yield repository.get_repository(), session
        finally:
            await db_session.dispose_engine()
        return

    # backend == "postgres"
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        pytest.skip("postgres contract test needs DATABASE_URL (opt-in, -m integration)")
    try:
        db_session.init_engine(url)
        async with db_session.get_session() as session:
            from sqlalchemy import text

            await session.execute(text("SELECT 1"))
            yield repository.get_repository(), session
    except Exception as exc:  # noqa: BLE001 - any connection/driver failure -> skip
        pytest.skip(f"postgres unreachable at DATABASE_URL: {exc}")
    finally:
        await db_session.dispose_engine()


# ── Persons ──────────────────────────────────────────────────────────────────

async def test_person_crud_contract(repo):
    repo_impl, session = repo
    assert isinstance(repo_impl, repository.Repository)

    created = await repo_impl.create_person(session, name="Contract Alice", room="101")
    assert created.name == "Contract Alice"
    assert created.room == "101"
    assert created.status == "active"  # default on both backends

    fetched = await repo_impl.get_person(session, created.id)
    assert fetched is not None
    assert fetched.name == "Contract Alice"

    listed = await repo_impl.list_persons(session, status="active")
    assert any(p.id == created.id for p in listed)

    updated = await repo_impl.update_person(session, created.id, notes="VIP")
    assert updated is not None
    assert updated.notes == "VIP"

    assert await repo_impl.delete_person(session, created.id) is True
    assert await repo_impl.get_person(session, created.id) is None
    # deleting again is a no-op, not an error, on both backends
    assert await repo_impl.delete_person(session, created.id) is False


async def test_person_embedding_contract(repo):
    repo_impl, session = repo
    person = await repo_impl.create_person(session, name="Contract Embed Person")

    emb = await repo_impl.create_person_embedding(
        session,
        person_id=person.id,
        embedding=b"\x00" * 512,
        embedding_type="face",
        source="contract-test",
    )
    assert emb.person_id == person.id

    listed = await repo_impl.list_person_embeddings(session, person.id)
    assert len(listed) == 1

    gallery = await repo_impl.list_face_gallery(session)
    assert any(row["person_id"] == person.id for row in gallery)

    assert await repo_impl.delete_person_embedding(session, emb.id) is True
    assert await repo_impl.delete_person_embedding(session, emb.id) is False


# ── Zones ────────────────────────────────────────────────────────────────────

async def test_zone_crud_contract(repo):
    repo_impl, session = repo
    camera_id = f"contract_cam_{uuid.uuid4().hex[:8]}"

    created = await repo_impl.create_zone(
        session,
        camera_id=camera_id,
        name="Contract Zone",
        zone_type="forbidden",
        geometry={"points": [[0, 0], [1, 0], [1, 1], [0, 1]]},
    )
    assert created.name == "Contract Zone"
    assert created.active is True

    fetched = await repo_impl.get_zone(session, created.id)
    assert fetched is not None

    listed = await repo_impl.list_zones(session, camera_id=camera_id)
    assert any(z.id == created.id for z in listed)

    updated = await repo_impl.update_zone(session, created.id, active=False)
    assert updated is not None
    assert updated.active is False

    # delete_zone's contract (pinned deliberately): truthy "deleted row"
    # value on success, falsy on a missing zone — on BOTH backends. db.dal
    # already normalises edge_dal's raw bool into this same shape; see
    # db/repository.py's module docstring for why that normalisation
    # matters and must not be bypassed.
    result = await repo_impl.delete_zone(session, created.id)
    assert bool(result)
    assert await repo_impl.get_zone(session, created.id) is None
    assert not (await repo_impl.delete_zone(session, created.id))


# ── Events ───────────────────────────────────────────────────────────────────

async def test_event_crud_contract(repo):
    repo_impl, session = repo
    camera_id = f"contract_cam_{uuid.uuid4().hex[:8]}"
    event_id = f"contract-{uuid.uuid4().hex[:8]}"
    occurred = datetime.now(timezone.utc)

    created = await repo_impl.upsert_event(
        session,
        event_id,
        camera_id=camera_id,
        event_type="detection",
        occurred_at=occurred,
    )
    assert created.event_id == event_id

    # upsert_event is idempotent by event_id: same key returns the existing
    # row, never a duplicate — on both backends.
    again = await repo_impl.upsert_event(
        session,
        event_id,
        camera_id=camera_id,
        event_type="detection",
        occurred_at=occurred,
    )
    assert again.id == created.id

    fetched = await repo_impl.get_event_by_event_id(session, event_id)
    assert fetched is not None
    assert fetched.id == created.id

    listed = await repo_impl.list_events(session, camera_id=camera_id)
    assert any(e.id == created.id for e in listed)
    assert len(listed) == 1  # the upsert above must not have duplicated

    person = await repo_impl.create_person(session, name="Contract Linked Person")
    linked = await repo_impl.link_event_to_person(session, created.id, person.id)
    assert linked is not None
    assert linked.person_id == person.id


# ── Camera configs ───────────────────────────────────────────────────────────

async def test_camera_config_crud_contract(repo):
    repo_impl, session = repo
    camera_id = f"contract_cam_{uuid.uuid4().hex[:8]}"

    created = await repo_impl.upsert_camera_config(session, camera_id, confidence_threshold=0.55)
    assert created.camera_id == camera_id
    assert created.confidence_threshold == 0.55

    fetched = await repo_impl.get_camera_config(session, camera_id)
    assert fetched is not None
    assert fetched.confidence_threshold == 0.55

    # upsert_camera_config's contract: second call updates in place, does
    # not create a second row — on both backends.
    updated = await repo_impl.upsert_camera_config(session, camera_id, confidence_threshold=0.75)
    assert updated.confidence_threshold == 0.75

    deleted_count = await repo_impl.delete_camera_config(session, camera_id)
    assert deleted_count == 1
    assert await repo_impl.get_camera_config(session, camera_id) is None
