"""SQLite edge store — persons, embeddings, zones, events when DATABASE_URL is empty.

Used on AI Box deployments where PostgreSQL is unavailable.  Separate from the
durable outbox SQLite (OUTBOX_DB_PATH); this file holds application state.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 1

_db_path: str | None = None
_lock = threading.Lock()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _dt_to_str(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _str_to_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _uuid_str(value: uuid.UUID | str | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _parse_uuid(value: str | None) -> uuid.UUID | None:
    if not value:
        return None
    return uuid.UUID(value)


def _json_dumps(obj: Any) -> str | None:
    if obj is None:
        return None
    return json.dumps(obj, separators=(",", ":"))


def _json_loads(raw: str | None) -> Any:
    if raw is None or raw == "":
        return None
    return json.loads(raw)


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS persons (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    age             INTEGER,
    date_of_birth   TEXT,
    gender          TEXT,
    notes      TEXT,
    room       TEXT,
    status     TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS person_embeddings (
    id              TEXT PRIMARY KEY,
    person_id       TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    embedding       BLOB NOT NULL,
    embedding_type  TEXT NOT NULL,
    source          TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_person_embeddings_person_id
    ON person_embeddings(person_id);

CREATE TABLE IF NOT EXISTS zones (
    id         TEXT PRIMARY KEY,
    camera_id  TEXT NOT NULL,
    name       TEXT NOT NULL,
    zone_type  TEXT NOT NULL,
    geometry   TEXT NOT NULL,
    active     INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_zones_camera_id ON zones(camera_id);

CREATE TABLE IF NOT EXISTS events (
    id          TEXT PRIMARY KEY,
    event_id    TEXT NOT NULL UNIQUE,
    camera_id   TEXT NOT NULL,
    track_id    INTEGER,
    event_type  TEXT NOT NULL,
    cls         TEXT,
    confidence  REAL,
    bbox_x      REAL,
    bbox_y      REAL,
    bbox_w      REAL,
    bbox_h      REAL,
    zone_id     TEXT REFERENCES zones(id),
    person_id   TEXT REFERENCES persons(id),
    payload     TEXT,
    occurred_at TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_camera_id ON events(camera_id);
CREATE INDEX IF NOT EXISTS ix_events_event_type ON events(event_type);
CREATE INDEX IF NOT EXISTS ix_events_occurred_at ON events(occurred_at);
CREATE INDEX IF NOT EXISTS ix_events_person_id ON events(person_id);

CREATE TABLE IF NOT EXISTS alerts (
    id              TEXT PRIMARY KEY,
    event_id        TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    alert_type      TEXT NOT NULL,
    recipient       TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',
    attempts        INTEGER NOT NULL DEFAULT 0,
    last_attempt_at TEXT,
    sent_at         TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS camera_configs (
    id                   TEXT PRIMARY KEY,
    camera_id            TEXT NOT NULL UNIQUE,
    confidence_threshold REAL,
    iou_threshold        REAL,
    frame_period         INTEGER,
    zone_ids             TEXT,
    extra                TEXT,
    updated_at           TEXT NOT NULL
);
"""


@dataclass
class PersonRecord:
    id: uuid.UUID
    name: str
    age: int | None
    date_of_birth: date | None
    gender: str | None
    notes: str | None
    room: str | None
    status: str
    created_at: datetime
    updated_at: datetime


@dataclass
class PersonEmbeddingRecord:
    id: uuid.UUID
    person_id: uuid.UUID
    embedding: bytes
    embedding_type: str
    source: str | None
    created_at: datetime


@dataclass
class ZoneRecord:
    id: uuid.UUID
    camera_id: str
    name: str
    zone_type: str
    geometry: dict
    active: bool
    created_at: datetime
    updated_at: datetime


@dataclass
class EventRecord:
    id: uuid.UUID
    event_id: str
    camera_id: str
    track_id: int | None
    event_type: str
    cls: str | None
    confidence: float | None
    bbox_x: float | None
    bbox_y: float | None
    bbox_w: float | None
    bbox_h: float | None
    zone_id: uuid.UUID | None
    person_id: uuid.UUID | None
    payload: dict | None
    occurred_at: datetime
    created_at: datetime


@dataclass
class AlertRecord:
    id: uuid.UUID
    event_id: uuid.UUID
    alert_type: str
    recipient: str | None
    status: str
    attempts: int
    last_attempt_at: datetime | None
    sent_at: datetime | None
    created_at: datetime


@dataclass
class CameraConfigRecord:
    id: uuid.UUID
    camera_id: str
    confidence_threshold: float | None
    iou_threshold: float | None
    frame_period: int | None
    zone_ids: list | None
    extra: dict | None
    updated_at: datetime


def init_edge_store(path: str) -> None:
    """Create or open the edge SQLite database and apply schema."""
    global _db_path
    db_file = Path(path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        _db_path = str(db_file)
        conn = _connect()
        try:
            conn.executescript(_SCHEMA_SQL)
            _migrate_schema(conn)
            conn.execute(
                "INSERT OR IGNORE INTO schema_meta(key, value) VALUES ('version', ?)",
                (str(_SCHEMA_VERSION),),
            )
            conn.commit()
        finally:
            conn.close()


def _migrate_schema(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(persons)")}
    if "date_of_birth" not in cols:
        conn.execute("ALTER TABLE persons ADD COLUMN date_of_birth TEXT")


def is_initialized() -> bool:
    return _db_path is not None


def close_edge_store() -> None:
    global _db_path
    _db_path = None


def _connect() -> sqlite3.Connection:
    if _db_path is None:
        raise RuntimeError("Edge SQLite store is not initialized")
    conn = sqlite3.connect(_db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _date_to_str(value: date | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10]


def _row_person(row: sqlite3.Row) -> PersonRecord:
    return PersonRecord(
        id=_parse_uuid(row["id"]),  # type: ignore[arg-type]
        name=row["name"],
        age=row["age"],
        date_of_birth=_parse_date(row["date_of_birth"]),
        gender=row["gender"],
        notes=row["notes"],
        room=row["room"],
        status=row["status"],
        created_at=_str_to_dt(row["created_at"]),  # type: ignore[arg-type]
        updated_at=_str_to_dt(row["updated_at"]),  # type: ignore[arg-type]
    )


def _row_embedding(row: sqlite3.Row) -> PersonEmbeddingRecord:
    return PersonEmbeddingRecord(
        id=_parse_uuid(row["id"]),  # type: ignore[arg-type]
        person_id=_parse_uuid(row["person_id"]),  # type: ignore[arg-type]
        embedding=row["embedding"],
        embedding_type=row["embedding_type"],
        source=row["source"],
        created_at=_str_to_dt(row["created_at"]),  # type: ignore[arg-type]
    )


def _row_zone(row: sqlite3.Row) -> ZoneRecord:
    return ZoneRecord(
        id=_parse_uuid(row["id"]),  # type: ignore[arg-type]
        camera_id=row["camera_id"],
        name=row["name"],
        zone_type=row["zone_type"],
        geometry=_json_loads(row["geometry"]) or {},
        active=bool(row["active"]),
        created_at=_str_to_dt(row["created_at"]),  # type: ignore[arg-type]
        updated_at=_str_to_dt(row["updated_at"]),  # type: ignore[arg-type]
    )


def _row_event(row: sqlite3.Row) -> EventRecord:
    return EventRecord(
        id=_parse_uuid(row["id"]),  # type: ignore[arg-type]
        event_id=row["event_id"],
        camera_id=row["camera_id"],
        track_id=row["track_id"],
        event_type=row["event_type"],
        cls=row["cls"],
        confidence=row["confidence"],
        bbox_x=row["bbox_x"],
        bbox_y=row["bbox_y"],
        bbox_w=row["bbox_w"],
        bbox_h=row["bbox_h"],
        zone_id=_parse_uuid(row["zone_id"]),
        person_id=_parse_uuid(row["person_id"]),
        payload=_json_loads(row["payload"]),
        occurred_at=_str_to_dt(row["occurred_at"]),  # type: ignore[arg-type]
        created_at=_str_to_dt(row["created_at"]),  # type: ignore[arg-type]
    )


def _row_alert(row: sqlite3.Row) -> AlertRecord:
    return AlertRecord(
        id=_parse_uuid(row["id"]),  # type: ignore[arg-type]
        event_id=_parse_uuid(row["event_id"]),  # type: ignore[arg-type]
        alert_type=row["alert_type"],
        recipient=row["recipient"],
        status=row["status"],
        attempts=row["attempts"],
        last_attempt_at=_str_to_dt(row["last_attempt_at"]),
        sent_at=_str_to_dt(row["sent_at"]),
        created_at=_str_to_dt(row["created_at"]),  # type: ignore[arg-type]
    )


def _row_camera_config(row: sqlite3.Row) -> CameraConfigRecord:
    return CameraConfigRecord(
        id=_parse_uuid(row["id"]),  # type: ignore[arg-type]
        camera_id=row["camera_id"],
        confidence_threshold=row["confidence_threshold"],
        iou_threshold=row["iou_threshold"],
        frame_period=row["frame_period"],
        zone_ids=_json_loads(row["zone_ids"]),
        extra=_json_loads(row["extra"]),
        updated_at=_str_to_dt(row["updated_at"]),  # type: ignore[arg-type]
    )


# ── Person ────────────────────────────────────────────────────────────────────

def get_person(person_id: uuid.UUID) -> PersonRecord | None:
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT * FROM persons WHERE id = ?", (_uuid_str(person_id),)
            ).fetchone()
            return _row_person(row) if row else None
        finally:
            conn.close()


def list_persons(status: str | None = None, limit: int = 100) -> list[PersonRecord]:
    with _lock:
        conn = _connect()
        try:
            if status:
                rows = conn.execute(
                    "SELECT * FROM persons WHERE status = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM persons ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [_row_person(r) for r in rows]
        finally:
            conn.close()


def create_person(**kwargs: Any) -> PersonRecord:
    now = _utcnow()
    person_id = kwargs.pop("id", None) or uuid.uuid4()
    row = {
        "id": _uuid_str(person_id),
        "name": kwargs["name"],
        "age": kwargs.get("age"),
        "date_of_birth": _date_to_str(kwargs.get("date_of_birth")),
        "gender": kwargs.get("gender"),
        "notes": kwargs.get("notes"),
        "room": kwargs.get("room"),
        "status": kwargs.get("status", "active"),
        "created_at": _dt_to_str(kwargs.get("created_at", now)),
        "updated_at": _dt_to_str(kwargs.get("updated_at", now)),
    }
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO persons "
                "(id, name, age, date_of_birth, gender, notes, room, status, created_at, updated_at) "
                "VALUES (:id, :name, :age, :date_of_birth, :gender, :notes, :room, :status, "
                ":created_at, :updated_at)",
                row,
            )
            conn.commit()
        finally:
            conn.close()
    return get_person(person_id)  # type: ignore[return-value]


def update_person(person_id: uuid.UUID, **kwargs: Any) -> PersonRecord | None:
    if not kwargs:
        return get_person(person_id)
    kwargs.setdefault("updated_at", _utcnow())
    sets = []
    params: list[Any] = []
    for key, value in kwargs.items():
        if key == "updated_at" and isinstance(value, datetime):
            value = _dt_to_str(value)
        elif key == "date_of_birth":
            value = _date_to_str(value)
        sets.append(f"{key} = ?")
        params.append(value)
    params.append(_uuid_str(person_id))
    with _lock:
        conn = _connect()
        try:
            cur = conn.execute(
                f"UPDATE persons SET {', '.join(sets)} WHERE id = ?", params
            )
            conn.commit()
            if cur.rowcount == 0:
                return None
        finally:
            conn.close()
    return get_person(person_id)


# ── PersonEmbedding ───────────────────────────────────────────────────────────

def create_person_embedding(**kwargs: Any) -> PersonEmbeddingRecord:
    now = _utcnow()
    emb_id = kwargs.pop("id", None) or uuid.uuid4()
    person_id = kwargs["person_id"]
    row = (
        _uuid_str(emb_id),
        _uuid_str(person_id),
        kwargs["embedding"],
        kwargs["embedding_type"],
        kwargs.get("source"),
        _dt_to_str(kwargs.get("created_at", now)),
    )
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO person_embeddings "
                "(id, person_id, embedding, embedding_type, source, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                row,
            )
            conn.commit()
        finally:
            conn.close()
    return get_person_embedding(emb_id)  # type: ignore[return-value]


def list_person_embeddings(person_id: uuid.UUID) -> list[PersonEmbeddingRecord]:
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT * FROM person_embeddings WHERE person_id = ? "
                "ORDER BY created_at",
                (_uuid_str(person_id),),
            ).fetchall()
            return [_row_embedding(r) for r in rows]
        finally:
            conn.close()


def get_person_embedding(emb_id: uuid.UUID) -> PersonEmbeddingRecord | None:
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT * FROM person_embeddings WHERE id = ?",
                (_uuid_str(emb_id),),
            ).fetchone()
            return _row_embedding(row) if row else None
        finally:
            conn.close()


def delete_person_embedding(emb_id: uuid.UUID) -> bool:
    with _lock:
        conn = _connect()
        try:
            cur = conn.execute(
                "DELETE FROM person_embeddings WHERE id = ?",
                (_uuid_str(emb_id),),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


def list_face_gallery() -> list[dict]:
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT pe.person_id, p.name, p.gender, p.date_of_birth, p.age, pe.embedding "
                "FROM person_embeddings pe "
                "JOIN persons p ON pe.person_id = p.id "
                "WHERE p.status = 'active' AND pe.embedding_type = 'face' "
                "ORDER BY pe.created_at"
            ).fetchall()
            return [
                {
                    "person_id": _parse_uuid(r["person_id"]),
                    "name": r["name"],
                    "gender": r["gender"],
                    "date_of_birth": _parse_date(r["date_of_birth"]),
                    "age": r["age"],
                    "embedding": r["embedding"],
                }
                for r in rows
            ]
        finally:
            conn.close()


def list_all_embeddings_for_matching() -> list[PersonEmbeddingRecord]:
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT pe.* FROM person_embeddings pe "
                "JOIN persons p ON pe.person_id = p.id "
                "WHERE p.status = 'active' "
                "ORDER BY pe.created_at"
            ).fetchall()
            return [_row_embedding(r) for r in rows]
        finally:
            conn.close()


# ── Zone ──────────────────────────────────────────────────────────────────────

def get_zone(zone_id: uuid.UUID) -> ZoneRecord | None:
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT * FROM zones WHERE id = ?", (_uuid_str(zone_id),)
            ).fetchone()
            return _row_zone(row) if row else None
        finally:
            conn.close()


def list_zones(
    camera_id: str | None = None, active_only: bool = True
) -> list[ZoneRecord]:
    clauses = []
    params: list[Any] = []
    if camera_id:
        clauses.append("camera_id = ?")
        params.append(camera_id)
    if active_only:
        clauses.append("active = 1")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                f"SELECT * FROM zones {where} ORDER BY created_at", params
            ).fetchall()
            return [_row_zone(r) for r in rows]
        finally:
            conn.close()


def create_zone(**kwargs: Any) -> ZoneRecord:
    now = _utcnow()
    zone_id = kwargs.pop("id", None) or uuid.uuid4()
    row = (
        _uuid_str(zone_id),
        kwargs["camera_id"],
        kwargs["name"],
        kwargs["zone_type"],
        _json_dumps(kwargs["geometry"]),
        1 if kwargs.get("active", True) else 0,
        _dt_to_str(kwargs.get("created_at", now)),
        _dt_to_str(kwargs.get("updated_at", now)),
    )
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO zones "
                "(id, camera_id, name, zone_type, geometry, active, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
            conn.commit()
        finally:
            conn.close()
    return get_zone(zone_id)  # type: ignore[return-value]


def update_zone(zone_id: uuid.UUID, **kwargs: Any) -> ZoneRecord | None:
    if not kwargs:
        return get_zone(zone_id)
    kwargs.setdefault("updated_at", _utcnow())
    sets = []
    params: list[Any] = []
    for key, value in kwargs.items():
        if key == "geometry":
            value = _json_dumps(value)
        elif key == "active":
            value = 1 if value else 0
        elif key == "updated_at" and isinstance(value, datetime):
            value = _dt_to_str(value)
        sets.append(f"{key} = ?")
        params.append(value)
    params.append(_uuid_str(zone_id))
    with _lock:
        conn = _connect()
        try:
            cur = conn.execute(
                f"UPDATE zones SET {', '.join(sets)} WHERE id = ?", params
            )
            conn.commit()
            if cur.rowcount == 0:
                return None
        finally:
            conn.close()
    return get_zone(zone_id)


# ── Event ─────────────────────────────────────────────────────────────────────

def get_event(event_id: uuid.UUID) -> EventRecord | None:
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT * FROM events WHERE id = ?", (_uuid_str(event_id),)
            ).fetchone()
            return _row_event(row) if row else None
        finally:
            conn.close()


def get_event_by_event_id(event_id: str) -> EventRecord | None:
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
            return _row_event(row) if row else None
        finally:
            conn.close()


def list_events(
    camera_id: str | None = None,
    event_type: str | None = None,
    track_id: int | None = None,
    person_id: uuid.UUID | None = None,
    zone_id: uuid.UUID | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[EventRecord]:
    clauses = []
    params: list[Any] = []
    if camera_id:
        clauses.append("camera_id = ?")
        params.append(camera_id)
    if event_type:
        clauses.append("event_type = ?")
        params.append(event_type)
    if track_id is not None:
        clauses.append("track_id = ?")
        params.append(track_id)
    if person_id is not None:
        clauses.append("person_id = ?")
        params.append(_uuid_str(person_id))
    if zone_id is not None:
        clauses.append("zone_id = ?")
        params.append(_uuid_str(zone_id))
    if start_time is not None:
        clauses.append("occurred_at >= ?")
        params.append(_dt_to_str(start_time))
    if end_time is not None:
        clauses.append("occurred_at <= ?")
        params.append(_dt_to_str(end_time))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.extend([limit, offset])
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                f"SELECT * FROM events {where} "
                f"ORDER BY occurred_at DESC LIMIT ? OFFSET ?",
                params,
            ).fetchall()
            return [_row_event(r) for r in rows]
        finally:
            conn.close()


def link_event_to_person(
    event_uuid: uuid.UUID, person_id: uuid.UUID
) -> EventRecord | None:
    with _lock:
        conn = _connect()
        try:
            cur = conn.execute(
                "UPDATE events SET person_id = ? WHERE id = ?",
                (_uuid_str(person_id), _uuid_str(event_uuid)),
            )
            conn.commit()
            if cur.rowcount == 0:
                return None
        finally:
            conn.close()
    return get_event(event_uuid)


def delete_events_before(cutoff: datetime) -> int:
    with _lock:
        conn = _connect()
        try:
            cur = conn.execute(
                "DELETE FROM events WHERE occurred_at < ?",
                (_dt_to_str(cutoff),),
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()


def upsert_event(event_id: str, **kwargs: Any) -> EventRecord:
    existing = get_event_by_event_id(event_id)
    if existing is not None:
        return existing
    now = _utcnow()
    pk = kwargs.pop("id", None) or uuid.uuid4()
    row = (
        _uuid_str(pk),
        event_id,
        kwargs["camera_id"],
        kwargs.get("track_id"),
        kwargs["event_type"],
        kwargs.get("cls"),
        kwargs.get("confidence"),
        kwargs.get("bbox_x"),
        kwargs.get("bbox_y"),
        kwargs.get("bbox_w"),
        kwargs.get("bbox_h"),
        _uuid_str(kwargs.get("zone_id")),
        _uuid_str(kwargs.get("person_id")),
        _json_dumps(kwargs.get("payload")),
        _dt_to_str(kwargs["occurred_at"]),
        _dt_to_str(kwargs.get("created_at", now)),
    )
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO events "
                "(id, event_id, camera_id, track_id, event_type, cls, confidence, "
                "bbox_x, bbox_y, bbox_w, bbox_h, zone_id, person_id, payload, "
                "occurred_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
            conn.commit()
        finally:
            conn.close()
    return get_event(pk)  # type: ignore[return-value]


# ── Alert ─────────────────────────────────────────────────────────────────────

def create_alert(**kwargs: Any) -> AlertRecord:
    now = _utcnow()
    alert_id = kwargs.pop("id", None) or uuid.uuid4()
    row = (
        _uuid_str(alert_id),
        _uuid_str(kwargs["event_id"]),
        kwargs["alert_type"],
        kwargs.get("recipient"),
        kwargs.get("status", "pending"),
        kwargs.get("attempts", 0),
        _dt_to_str(kwargs["last_attempt_at"]) if kwargs.get("last_attempt_at") else None,
        _dt_to_str(kwargs["sent_at"]) if kwargs.get("sent_at") else None,
        _dt_to_str(kwargs.get("created_at", now)),
    )
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO alerts "
                "(id, event_id, alert_type, recipient, status, attempts, "
                "last_attempt_at, sent_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
            conn.commit()
        finally:
            conn.close()
    return get_alert(alert_id)  # type: ignore[return-value]


def get_alert(alert_id: uuid.UUID) -> AlertRecord | None:
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT * FROM alerts WHERE id = ?", (_uuid_str(alert_id),)
            ).fetchone()
            return _row_alert(row) if row else None
        finally:
            conn.close()


def update_alert(alert_id: uuid.UUID, **kwargs: Any) -> AlertRecord | None:
    if not kwargs:
        return get_alert(alert_id)
    sets = []
    params: list[Any] = []
    for key, value in kwargs.items():
        if key in ("last_attempt_at", "sent_at") and isinstance(value, datetime):
            value = _dt_to_str(value)
        sets.append(f"{key} = ?")
        params.append(value)
    params.append(_uuid_str(alert_id))
    with _lock:
        conn = _connect()
        try:
            cur = conn.execute(
                f"UPDATE alerts SET {', '.join(sets)} WHERE id = ?", params
            )
            conn.commit()
            if cur.rowcount == 0:
                return None
        finally:
            conn.close()
    return get_alert(alert_id)


def get_alert_by_event_and_type(
    event_pk: uuid.UUID, alert_type: str
) -> AlertRecord | None:
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT * FROM alerts WHERE event_id = ? AND alert_type = ? "
                "ORDER BY created_at LIMIT 1",
                (_uuid_str(event_pk), alert_type),
            ).fetchone()
            return _row_alert(row) if row else None
        finally:
            conn.close()


def list_pending_alerts(limit: int = 50) -> list[AlertRecord]:
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT * FROM alerts WHERE status = 'pending' "
                "ORDER BY created_at LIMIT ?",
                (limit,),
            ).fetchall()
            return [_row_alert(r) for r in rows]
        finally:
            conn.close()


# ── CameraConfig ──────────────────────────────────────────────────────────────

def get_camera_config(camera_id: str) -> CameraConfigRecord | None:
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT * FROM camera_configs WHERE camera_id = ?",
                (camera_id,),
            ).fetchone()
            return _row_camera_config(row) if row else None
        finally:
            conn.close()


def upsert_camera_config(camera_id: str, **kwargs: Any) -> CameraConfigRecord:
    now = _utcnow()
    existing = get_camera_config(camera_id)
    if existing is not None:
        sets = ["updated_at = ?"]
        params: list[Any] = [_dt_to_str(now)]
        for key, value in kwargs.items():
            if key in ("zone_ids", "extra"):
                value = _json_dumps(value)
            sets.append(f"{key} = ?")
            params.append(value)
        params.append(camera_id)
        with _lock:
            conn = _connect()
            try:
                conn.execute(
                    f"UPDATE camera_configs SET {', '.join(sets)} WHERE camera_id = ?",
                    params,
                )
                conn.commit()
            finally:
                conn.close()
        return get_camera_config(camera_id)  # type: ignore[return-value]

    cfg_id = uuid.uuid4()
    row = (
        _uuid_str(cfg_id),
        camera_id,
        kwargs.get("confidence_threshold"),
        kwargs.get("iou_threshold"),
        kwargs.get("frame_period"),
        _json_dumps(kwargs.get("zone_ids")),
        _json_dumps(kwargs.get("extra")),
        _dt_to_str(now),
    )
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO camera_configs "
                "(id, camera_id, confidence_threshold, iou_threshold, frame_period, "
                "zone_ids, extra, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
            conn.commit()
        finally:
            conn.close()
    return get_camera_config(camera_id)  # type: ignore[return-value]


def delete_camera_config(camera_id: str) -> int:
    with _lock:
        conn = _connect()
        try:
            cur = conn.execute(
                "DELETE FROM camera_configs WHERE camera_id = ?",
                (camera_id,),
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()


# ── Analytics ─────────────────────────────────────────────────────────────────

def _scalar(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def get_analytics_summary() -> dict:
    now = _utcnow()
    day_ago = now - timedelta(days=1)
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)
    with _lock:
        conn = _connect()
        try:
            total_events = _scalar(conn, "SELECT COUNT(*) FROM events")
            type_rows = conn.execute(
                "SELECT event_type, COUNT(*) AS n FROM events "
                "GROUP BY event_type ORDER BY n DESC"
            ).fetchall()
            events_by_type = {r["event_type"]: r["n"] for r in type_rows}
            events_24h = _scalar(
                conn,
                "SELECT COUNT(*) FROM events WHERE occurred_at >= ?",
                (_dt_to_str(day_ago),),
            )
            events_7d = _scalar(
                conn,
                "SELECT COUNT(*) FROM events WHERE occurred_at >= ?",
                (_dt_to_str(week_ago),),
            )
            events_30d = _scalar(
                conn,
                "SELECT COUNT(*) FROM events WHERE occurred_at >= ?",
                (_dt_to_str(month_ago),),
            )
            falls_24h = _scalar(
                conn,
                "SELECT COUNT(*) FROM events WHERE event_type = 'fall' "
                "AND occurred_at >= ?",
                (_dt_to_str(day_ago),),
            )
            zone_violations_24h = _scalar(
                conn,
                "SELECT COUNT(*) FROM events WHERE event_type = 'zone_violation' "
                "AND occurred_at >= ?",
                (_dt_to_str(day_ago),),
            )
            linked_events = _scalar(
                conn, "SELECT COUNT(*) FROM events WHERE person_id IS NOT NULL"
            )
            active_persons = _scalar(
                conn,
                "SELECT COUNT(*) FROM persons WHERE status = 'active'",
            )
            active_zones = _scalar(
                conn, "SELECT COUNT(*) FROM zones WHERE active = 1"
            )
            active_cameras = _scalar(
                conn, "SELECT COUNT(DISTINCT camera_id) FROM events"
            )
            pending_alerts = _scalar(
                conn,
                "SELECT COUNT(*) FROM alerts WHERE status = 'pending'",
            )
            sent_alerts = _scalar(
                conn, "SELECT COUNT(*) FROM alerts WHERE status = 'sent'"
            )
            failed_alerts = _scalar(
                conn, "SELECT COUNT(*) FROM alerts WHERE status = 'failed'"
            )
        finally:
            conn.close()
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


def get_events_timeseries(
    granularity: str = "day",
    days_back: int = 7,
    event_type: str | None = None,
) -> list[dict]:
    cutoff = _utcnow() - timedelta(days=days_back)
    fmt = "%Y-%m-%d" if granularity == "day" else "%Y-%m-%dT%H:00:00"
    bucket_expr = f"strftime('{fmt}', occurred_at)"
    clauses = ["occurred_at >= ?"]
    params: list[Any] = [_dt_to_str(cutoff)]
    if event_type:
        clauses.append("event_type = ?")
        params.append(event_type)
    where = " AND ".join(clauses)
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                f"SELECT {bucket_expr} AS bucket, COUNT(*) AS count "
                f"FROM events WHERE {where} GROUP BY bucket ORDER BY bucket",
                params,
            ).fetchall()
        finally:
            conn.close()
    return [
        {"bucket": f"{row['bucket']}T00:00:00+00:00" if granularity == "day" else f"{row['bucket']}+00:00", "count": row["count"]}
        for row in rows
    ]


def get_camera_analytics() -> list[dict]:
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT camera_id, "
                "COUNT(*) AS total_events, "
                "SUM(CASE WHEN event_type = 'fall' THEN 1 ELSE 0 END) AS falls, "
                "SUM(CASE WHEN event_type = 'zone_violation' THEN 1 ELSE 0 END) "
                "AS zone_violations, "
                "COUNT(DISTINCT track_id) AS unique_tracks, "
                "SUM(CASE WHEN person_id IS NOT NULL THEN 1 ELSE 0 END) AS linked_events, "
                "MAX(occurred_at) AS last_seen "
                "FROM events GROUP BY camera_id ORDER BY total_events DESC"
            ).fetchall()
        finally:
            conn.close()
    return [
        {
            "camera_id": row["camera_id"],
            "total_events": row["total_events"],
            "falls": row["falls"],
            "zone_violations": row["zone_violations"],
            "unique_tracks": row["unique_tracks"],
            "linked_events": row["linked_events"],
            "last_seen": _str_to_dt(row["last_seen"]).isoformat()
            if row["last_seen"]
            else None,
        }
        for row in rows
    ]


def get_zone_analytics() -> list[dict]:
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT z.id AS zone_id, z.name, z.zone_type, z.camera_id, z.active, "
                "COUNT(e.id) AS violation_count, MAX(e.occurred_at) AS last_violation "
                "FROM zones z "
                "LEFT JOIN events e ON e.zone_id = z.id "
                "GROUP BY z.id, z.name, z.zone_type, z.camera_id, z.active "
                "ORDER BY violation_count DESC"
            ).fetchall()
        finally:
            conn.close()
    return [
        {
            "zone_id": row["zone_id"],
            "name": row["name"],
            "zone_type": row["zone_type"],
            "camera_id": row["camera_id"],
            "active": bool(row["active"]),
            "violation_count": row["violation_count"],
            "last_violation": _str_to_dt(row["last_violation"]).isoformat()
            if row["last_violation"]
            else None,
        }
        for row in rows
    ]


def get_alert_analytics() -> dict:
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT status, alert_type, COUNT(*) AS n "
                "FROM alerts GROUP BY status, alert_type ORDER BY status"
            ).fetchall()
        finally:
            conn.close()
    by_status: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + row["n"]
        by_type[row["alert_type"]] = by_type.get(row["alert_type"], 0) + row["n"]
    total = sum(by_status.values())
    return {"total": total, "by_status": by_status, "by_channel": by_type}
