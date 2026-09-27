"""P2-2: Repository interface over the Postgres / edge-SQLite persistence split.

Why this file exists
---------------------
db/dal.py (async, SQLAlchemy) and db/edge_store.py (sync sqlite3, wrapped
async by db/edge_dal.py) are two independent implementations of "the same"
data access surface (persons, zones, events, camera configs), selected at
startup by db/session.py's is_edge_mode() flag. Every new query had to be
written twice with no shared interface forcing them to stay in sync — a DRY
violation with real correctness risk (the two backends can silently drift).

This module does NOT rewrite either backend. It defines the Protocol
(structural interface) that api.py/admin_router.py already rely on today —
purely as a documentation/typing step, entirely inferred from the call
sites' existing usage of db.dal — plus a get_repository() factory so call
sites depend on `Repository`/`get_repository()`, not on `db.dal` or
`db.edge_store` directly.

Why PostgresRepository and EdgeSqliteRepository both delegate to db.dal
-------------------------------------------------------------------------
db.dal's functions are *already* the single, already-tested dispatch point
between the two backends: every function checks is_edge_mode() and either
runs its own SQLAlchemy/Postgres code or forwards to db.edge_dal. Standing
up a *second*, independent branch here (e.g. having EdgeSqliteRepository
call db.edge_dal directly, bypassing db.dal) would silently reintroduce the
exact problem this task exists to close — db.dal already contains small
but real normalisation on top of the raw edge_dal/edge_store calls (see
dal.delete_zone, which turns edge_dal's raw bool into the same `Zone |
ZoneRecord | None` shape the Postgres branch returns) that a naive
"EdgeSqliteRepository -> edge_dal" shortcut would silently drop, breaking
callers that rely on dal.py's existing contract.

So PostgresRepository / EdgeSqliteRepository are named, typed *markers*
over the one already-correct db.dal surface, not a second copy of it.
get_repository() picks between them by mirroring is_edge_mode() (decided
once at process startup by db.session.init_engine() and never changed
again), matching production behaviour exactly. Because both classes are
thin pass-throughs to the same functions, contract drift can only be
introduced inside db.dal itself — which is exactly what
tests/unit/test_repository_contract.py exists to catch, by exercising
db.dal's edge-SQLite branch and (opt-in, when a live Postgres is reachable)
its Postgres branch with the same inputs and asserting equivalent outputs.

Scope: matches the CRUD surface api.py/admin_router.py call today. Analytics
dashboards, retention sweeps, etc. use the same functions and are covered
too, since everything here is a pass-through to db.dal.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from . import dal
from .session import is_edge_mode

# ── Protocols (structural interfaces) ──────────────────────────────────────
# One Protocol per entity, matching the exact call signatures api.py /
# admin_router.py already use against db.dal today. Splitting them lets a
# call site that only touches, say, zones, type-hint against just
# ZoneRepository instead of the full Repository surface.


@runtime_checkable
class PersonRepository(Protocol):
    async def get_person(self, session, person_id): ...
    async def list_persons(self, session, status=None, limit=100): ...
    async def create_person(self, session, **kwargs): ...
    async def update_person(self, session, person_id, **kwargs): ...
    async def delete_person(self, session, person_id) -> bool: ...
    async def create_person_embedding(self, session, **kwargs): ...
    async def list_person_embeddings(self, session, person_id): ...
    async def get_person_embedding(self, session, emb_id): ...
    async def delete_person_embedding(self, session, emb_id) -> bool: ...
    async def list_face_gallery(self, session) -> list[dict]: ...
    async def list_all_embeddings_for_matching(self, session): ...


@runtime_checkable
class ZoneRepository(Protocol):
    async def get_zone(self, session, zone_id): ...
    async def list_zones(self, session, camera_id=None, active_only=True): ...
    async def create_zone(self, session, **kwargs): ...
    async def update_zone(self, session, zone_id, **kwargs): ...
    async def delete_zone(self, session, zone_id): ...


@runtime_checkable
class EventRepository(Protocol):
    async def get_event(self, session, event_id): ...
    async def get_event_by_event_id(self, session, event_id): ...
    async def list_events(self, session, **kwargs): ...
    async def link_event_to_person(self, session, event_uuid, person_id): ...
    async def delete_events_before(self, session, cutoff) -> int: ...
    async def upsert_event(self, session, event_id, **kwargs): ...


@runtime_checkable
class CameraConfigRepository(Protocol):
    async def get_camera_config(self, session, camera_id): ...
    async def upsert_camera_config(self, session, camera_id, **kwargs): ...
    async def delete_camera_config(self, session, camera_id) -> int: ...


@runtime_checkable
class AlertRepository(Protocol):
    async def create_alert(self, session, **kwargs): ...
    async def update_alert(self, session, alert_id, **kwargs): ...
    async def list_pending_alerts(self, session, limit=50): ...
    async def record_email_alert_attempt(
        self, session, db_event_pk, *, success, attempted_at
    ) -> None: ...


@runtime_checkable
class AnalyticsRepository(Protocol):
    async def get_analytics_summary(self, session) -> dict: ...
    async def get_events_timeseries(
        self, session, granularity="day", days_back=7, event_type=None
    ) -> list[dict]: ...
    async def get_camera_analytics(self, session) -> list[dict]: ...
    async def get_zone_analytics(self, session) -> list[dict]: ...
    async def get_alert_analytics(self, session) -> dict: ...


@runtime_checkable
class Repository(
    PersonRepository,
    ZoneRepository,
    EventRepository,
    CameraConfigRepository,
    AlertRepository,
    AnalyticsRepository,
    Protocol,
):
    """Full surface api.py / admin_router.py call today (via db.dal)."""


# ── Concrete implementations ────────────────────────────────────────────────

class _DalBackedRepository:
    """Every Repository method, 1:1 pass-through to the identically-named
    db.dal function. See the module docstring for why this is a pass-through
    rather than a reimplementation."""

    # Person
    get_person = staticmethod(dal.get_person)
    list_persons = staticmethod(dal.list_persons)
    create_person = staticmethod(dal.create_person)
    update_person = staticmethod(dal.update_person)
    delete_person = staticmethod(dal.delete_person)
    create_person_embedding = staticmethod(dal.create_person_embedding)
    list_person_embeddings = staticmethod(dal.list_person_embeddings)
    get_person_embedding = staticmethod(dal.get_person_embedding)
    delete_person_embedding = staticmethod(dal.delete_person_embedding)
    list_face_gallery = staticmethod(dal.list_face_gallery)
    list_all_embeddings_for_matching = staticmethod(dal.list_all_embeddings_for_matching)

    # Zone
    get_zone = staticmethod(dal.get_zone)
    list_zones = staticmethod(dal.list_zones)
    create_zone = staticmethod(dal.create_zone)
    update_zone = staticmethod(dal.update_zone)
    delete_zone = staticmethod(dal.delete_zone)

    # Event
    get_event = staticmethod(dal.get_event)
    get_event_by_event_id = staticmethod(dal.get_event_by_event_id)
    list_events = staticmethod(dal.list_events)
    link_event_to_person = staticmethod(dal.link_event_to_person)
    delete_events_before = staticmethod(dal.delete_events_before)
    upsert_event = staticmethod(dal.upsert_event)

    # Alert
    create_alert = staticmethod(dal.create_alert)
    update_alert = staticmethod(dal.update_alert)
    list_pending_alerts = staticmethod(dal.list_pending_alerts)
    record_email_alert_attempt = staticmethod(dal.record_email_alert_attempt)

    # CameraConfig
    get_camera_config = staticmethod(dal.get_camera_config)
    upsert_camera_config = staticmethod(dal.upsert_camera_config)
    delete_camera_config = staticmethod(dal.delete_camera_config)

    # Analytics
    get_analytics_summary = staticmethod(dal.get_analytics_summary)
    get_events_timeseries = staticmethod(dal.get_events_timeseries)
    get_camera_analytics = staticmethod(dal.get_camera_analytics)
    get_zone_analytics = staticmethod(dal.get_zone_analytics)
    get_alert_analytics = staticmethod(dal.get_alert_analytics)


class PostgresRepository(_DalBackedRepository):
    """Repository backed by db.dal's Postgres/SQLAlchemy code path.

    Only meaningfully distinct from EdgeSqliteRepository while the process
    is NOT in edge mode (db.session.is_edge_mode() is False) — get_repository()
    below only ever hands this out in that case, matching what db.dal itself
    would do anyway. See the module docstring for the full rationale.
    """


class EdgeSqliteRepository(_DalBackedRepository):
    """Repository backed by db.dal's edge-SQLite code path (db.edge_dal /
    db.edge_store). See PostgresRepository / the module docstring."""


def get_repository() -> Repository:
    """Return the Repository implementation matching the active backend.

    Mirrors db.session.is_edge_mode() — the single source of truth for which
    backend is active in this process, decided once at startup by
    db.session.init_engine() and never changed again. Call sites should
    depend on this factory (and the Repository Protocol above), not on
    db.dal / db.edge_store directly.
    """
    return EdgeSqliteRepository() if is_edge_mode() else PostgresRepository()
