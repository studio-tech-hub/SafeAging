"""S P1.2 — Admin REST API: persons, zones, events (read-only), admin reset.

All endpoints:
  - require the same X-API-Key auth as the main service (if API_KEY_REQUIRED=true)
  - return 503 if the database engine is not available (degraded mode)
  - are async and use the shared AsyncSession from db.session

Mount point: /admin  (included in api.py)
"""

import base64
import hmac
import threading
import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from .retention import get_status as _retention_status

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import API_KEY, API_KEY_REQUIRED
from .db import get_session
from .db import dal
from .config import logger
from .person_age import effective_age, parse_date_of_birth
from .zone_engine import invalidate_zone_cache
from .config_engine import invalidate_config_cache
from . import face_engine


# ── Auth dependency ────────────────────────────────────────────────────────────

def _auth(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> None:
    if not API_KEY_REQUIRED:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized: missing or invalid API key")


# ── DB availability guard ──────────────────────────────────────────────────────

def _require_db() -> None:
    """Raise 503 if neither Postgres nor edge SQLite is available."""
    from .db.session import is_ready

    if not is_ready():
        raise HTTPException(
            status_code=503,
            detail="Database not available — service running in degraded mode",
        )


# ── Pydantic schemas ───────────────────────────────────────────────────────────

def _prepare_person_write(payload: dict) -> dict:
    """Parse date_of_birth; do not persist client-supplied age (computed at read time)."""
    data = dict(payload)
    dob_raw = data.pop("date_of_birth", None)
    data.pop("age", None)
    if dob_raw:
        dob = parse_date_of_birth(dob_raw)
        if dob is None:
            raise HTTPException(
                status_code=422,
                detail="Invalid date_of_birth — use the date picker or YYYY-MM-DD",
            )
        data["date_of_birth"] = dob
    return data


def _person_to_out(person: Any) -> "PersonOut":
    dob = getattr(person, "date_of_birth", None)
    return PersonOut(
        id=person.id,
        name=person.name,
        gender=person.gender,
        notes=person.notes,
        room=person.room,
        status=person.status,
        created_at=person.created_at,
        updated_at=person.updated_at,
        date_of_birth=dob,
        age=effective_age(date_of_birth=dob, stored_age=getattr(person, "age", None)),
    )


class PersonCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    date_of_birth: Optional[str] = Field(
        None,
        description="Ngày sinh DD/MM/YYYY hoặc YYYY-MM-DD (tuổi tính tự động)",
    )
    gender: Optional[str] = Field(None, max_length=50)
    notes: Optional[str] = None
    room: Optional[str] = Field(None, max_length=255)
    status: str = Field("active", pattern="^(active|inactive|unknown)$")


class PersonUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    date_of_birth: Optional[str] = Field(
        None,
        description="Ngày sinh DD/MM/YYYY hoặc YYYY-MM-DD",
    )
    gender: Optional[str] = Field(None, max_length=50)
    notes: Optional[str] = None
    room: Optional[str] = Field(None, max_length=255)
    status: Optional[str] = Field(None, pattern="^(active|inactive|unknown)$")


class PersonOut(BaseModel):
    id: uuid.UUID
    name: str
    age: Optional[int] = Field(None, description="Tuổi tính từ ngày sinh (real-time)")
    date_of_birth: Optional[date] = None
    gender: Optional[str]
    notes: Optional[str]
    room: Optional[str]
    status: str
    created_at: datetime
    updated_at: datetime


class ZoneCreate(BaseModel):
    camera_id: str = Field(..., min_length=1, max_length=255)
    name: str = Field(..., min_length=1, max_length=255)
    zone_type: str = Field(..., pattern="^(forbidden|entry|exit|roi)$")
    geometry: Dict[str, Any] = Field(..., description='{"points": [[x,y], ...]}')
    active: bool = True


class ZoneUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    zone_type: Optional[str] = Field(None, pattern="^(forbidden|entry|exit|roi)$")
    geometry: Optional[Dict[str, Any]] = None
    active: Optional[bool] = None


class ZoneOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    camera_id: str
    name: str
    zone_type: str
    geometry: Dict[str, Any]
    active: bool
    created_at: datetime
    updated_at: datetime


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    event_id: str
    camera_id: str
    track_id: Optional[int]
    event_type: str
    cls: Optional[str]
    confidence: Optional[float]
    bbox_x: Optional[float]
    bbox_y: Optional[float]
    bbox_w: Optional[float]
    bbox_h: Optional[float]
    zone_id: Optional[uuid.UUID]
    person_id: Optional[uuid.UUID]
    payload: Optional[Dict[str, Any]]
    occurred_at: datetime
    created_at: datetime


class CameraConfigUpsert(BaseModel):
    confidence_threshold: Optional[float] = Field(None, ge=0.1, le=1.0)
    iou_threshold: Optional[float] = Field(None, ge=0.1, le=1.0)
    frame_period: Optional[int] = Field(None, ge=1, le=30)
    zone_ids: Optional[List[str]] = None
    extra: Optional[Dict[str, Any]] = None


class CameraConfigOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    camera_id: str
    confidence_threshold: Optional[float]
    iou_threshold: Optional[float]
    frame_period: Optional[int]
    zone_ids: Optional[List[Any]]
    extra: Optional[Dict[str, Any]]
    updated_at: datetime


class AdminResetOut(BaseModel):
    status: str
    cameras_reset: int
    total_persons_cleared: int
    db_events_deleted: Optional[int] = None


class LinkPersonBody(BaseModel):
    person_id: uuid.UUID


# ── ReID / Embedding schemas ───────────────────────────────────────────────────

class EmbeddingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    person_id: uuid.UUID
    embedding_type: str
    source: Optional[str]
    created_at: datetime


class EnrollFromImageBody(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded JPEG or PNG image of the person")
    embedding_type: str = Field("body", pattern="^(body|face)$")
    source: Optional[str] = Field(None, max_length=255)
    bbox_x: Optional[float] = Field(None, description="Bounding-box left (pixels)")
    bbox_y: Optional[float] = Field(None, description="Bounding-box top (pixels)")
    bbox_w: Optional[float] = Field(None, description="Bounding-box width (pixels)")
    bbox_h: Optional[float] = Field(None, description="Bounding-box height (pixels)")


class EnrollFaceBatchBody(BaseModel):
    """Enroll face templates from 5–30 images or a video clip (frames sampled at 3–5 fps)."""

    images_b64: Optional[List[str]] = Field(
        None,
        description="5–30 base64-encoded still images (JPEG/PNG)",
    )
    video_b64: Optional[str] = Field(
        None,
        description="Base64-encoded video clip; frames are sampled before face enrollment",
    )
    video_sample_fps: Optional[float] = Field(
        None,
        ge=3.0,
        le=5.0,
        description="Frame sampling rate when video_b64 is set (default 4 fps)",
    )
    source: Optional[str] = Field(None, max_length=255)

    @model_validator(mode="after")
    def _validate_batch_payload(self) -> "EnrollFaceBatchBody":
        from .config import ENROLL_FACE_MAX_IMAGES, ENROLL_FACE_MIN_IMAGES

        has_images = bool(self.images_b64)
        has_video = bool(self.video_b64 and self.video_b64.strip())
        if has_images == has_video:
            raise ValueError(
                "Provide exactly one of images_b64 (5–30 photos) or video_b64, not both"
            )
        if has_images:
            n = len(self.images_b64 or [])
            if n < ENROLL_FACE_MIN_IMAGES or n > ENROLL_FACE_MAX_IMAGES:
                raise ValueError(
                    f"images_b64 must contain {ENROLL_FACE_MIN_IMAGES}–"
                    f"{ENROLL_FACE_MAX_IMAGES} items (got {n})"
                )
        return self


class EnrollFaceBatchResult(BaseModel):
    person_id: uuid.UUID
    embeddings_created: int
    frames_processed: int
    frames_with_face: int
    embedding_ids: List[uuid.UUID]


class ReidMatchBody(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded JPEG or PNG image")
    threshold: float = Field(0.65, ge=0.0, le=1.0, description="Cosine similarity threshold")


class ReidMatchResult(BaseModel):
    matched: bool
    person_id: Optional[uuid.UUID]
    score: Optional[float]
    person_name: Optional[str] = None


class EnrollFromTrackBody(BaseModel):
    camera_id: str = Field(..., min_length=1, max_length=255)
    track_id: int = Field(..., description="Live track_id currently shown on the bounding box")
    # Provide person_id to add a face to an existing person, OR provide
    # name (+ optional fields) to create a new person on the fly.
    person_id: Optional[uuid.UUID] = Field(None, description="Existing person to attach the face to")
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    date_of_birth: Optional[str] = Field(
        None,
        description="Ngày sinh DD/MM/YYYY hoặc YYYY-MM-DD",
    )
    gender: Optional[str] = Field(None, max_length=50)
    room: Optional[str] = Field(None, max_length=255)
    notes: Optional[str] = None


class EnrollFromTrackResult(BaseModel):
    person_id: uuid.UUID
    person_name: str
    embedding_id: uuid.UUID
    created_person: bool


class BatchAutoLinkBody(BaseModel):
    event_types: Optional[List[str]] = Field(
        None, description="Event types to process (default: fall, zone_violation)"
    )
    limit: int = Field(200, ge=1, le=1000)
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(_auth)],
)


# ── Person endpoints ───────────────────────────────────────────────────────────

@router.get("/persons", response_model=List[PersonOut], summary="List persons")
async def list_persons(
    status: Optional[str] = Query(None, description="Filter by status (active/inactive/unknown)"),
    limit: int = Query(100, ge=1, le=1000),
):
    _require_db()
    async with get_session() as session:
        persons = await dal.list_persons(session, status=status, limit=limit)
    return [ _person_to_out(p) for p in persons ]


@router.post("/persons", response_model=PersonOut, status_code=201, summary="Create person")
async def create_person(body: PersonCreate):
    _require_db()
    payload = _prepare_person_write(body.model_dump())
    async with get_session() as session:
        person = await dal.create_person(session, **payload)
        out = _person_to_out(person)
        await session.commit()
    logger.info(f"[admin] Created person id={out.id} name={out.name!r}")
    return out


@router.get("/persons/{person_id}", response_model=PersonOut, summary="Get person by ID")
async def get_person(person_id: uuid.UUID):
    _require_db()
    async with get_session() as session:
        person = await dal.get_person(session, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
    return _person_to_out(person)


@router.put("/persons/{person_id}", response_model=PersonOut, summary="Update person")
async def update_person(person_id: uuid.UUID, body: PersonUpdate):
    _require_db()
    raw = {k: v for k, v in body.model_dump().items() if v is not None}
    if not raw:
        raise HTTPException(status_code=422, detail="No fields provided for update")
    payload = _prepare_person_write(raw)
    async with get_session() as session:
        person = await dal.update_person(session, person_id, **payload)
        if person is None:
            raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
        out = _person_to_out(person)
        await session.commit()
    logger.info(f"[admin] Updated person id={person_id} fields={list(payload.keys())}")
    return out


@router.delete("/persons/{person_id}", status_code=204, summary="Permanently delete person and face embeddings")
async def delete_person(person_id: uuid.UUID):
    _require_db()
    async with get_session() as session:
        deleted = await dal.delete_person(session, person_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
        await session.commit()
    face_engine.invalidate_gallery()
    logger.info(f"[admin] Deleted person id={person_id} (hard delete)")


@router.get(
    "/persons/{person_id}/history",
    response_model=List[EventOut],
    summary="Audit trail: all events linked to a person",
)
async def get_person_history(
    person_id: uuid.UUID,
    event_type: Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Return all Event records linked to this person, newest first.

    Events are linked either automatically (when person_id is set at infer time)
    or manually via POST /admin/events/{event_uuid}/link-person.
    """
    _require_db()
    async with get_session() as session:
        person = await dal.get_person(session, person_id)
        if person is None:
            raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
        events = await dal.list_events(
            session,
            person_id=person_id,
            event_type=event_type,
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset,
        )
    return [EventOut.model_validate(e) for e in events]


# ── Zone endpoints ─────────────────────────────────────────────────────────────

@router.get("/zones", response_model=List[ZoneOut], summary="List zones")
async def list_zones(
    camera_id: Optional[str] = Query(None),
    active_only: bool = Query(True),
):
    _require_db()
    async with get_session() as session:
        zones = await dal.list_zones(session, camera_id=camera_id, active_only=active_only)
    return [ZoneOut.model_validate(z) for z in zones]


@router.post("/zones", response_model=ZoneOut, status_code=201, summary="Create zone")
async def create_zone(body: ZoneCreate):
    _require_db()
    async with get_session() as session:
        zone = await dal.create_zone(session, **body.model_dump())
        out = ZoneOut.model_validate(zone)
        await session.commit()
    invalidate_zone_cache(out.camera_id)
    logger.info(f"[admin] Created zone id={out.id} name={out.name!r} camera={out.camera_id}")
    return out


@router.get("/zones/{zone_id}", response_model=ZoneOut, summary="Get zone by ID")
async def get_zone(zone_id: uuid.UUID):
    _require_db()
    async with get_session() as session:
        zone = await dal.get_zone(session, zone_id)
    if zone is None:
        raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found")
    return ZoneOut.model_validate(zone)


@router.put("/zones/{zone_id}", response_model=ZoneOut, summary="Update zone")
async def update_zone(zone_id: uuid.UUID, body: ZoneUpdate):
    _require_db()
    payload = {k: v for k, v in body.model_dump().items() if v is not None}
    if not payload:
        raise HTTPException(status_code=422, detail="No fields provided for update")
    async with get_session() as session:
        zone = await dal.update_zone(session, zone_id, **payload)
        if zone is None:
            raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found")
        out = ZoneOut.model_validate(zone)
        await session.commit()
    invalidate_zone_cache(out.camera_id)
    logger.info(f"[admin] Updated zone id={zone_id} fields={list(payload.keys())}")
    return out


@router.delete("/zones/{zone_id}", status_code=204, summary="Deactivate zone (active→false)")
async def delete_zone(zone_id: uuid.UUID):
    _require_db()
    async with get_session() as session:
        zone = await dal.update_zone(session, zone_id, active=False)
        if zone is None:
            raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found")
        out = ZoneOut.model_validate(zone)
        await session.commit()
    invalidate_zone_cache(out.camera_id)
    logger.info(f"[admin] Deactivated zone id={zone_id}")


# ── Event endpoints ─────────────────────────────────────────────────────────────

@router.get("/events", response_model=List[EventOut], summary="List events")
async def list_events(
    camera_id: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    track_id: Optional[int] = Query(None, description="Filter by tracker track_id"),
    person_id: Optional[uuid.UUID] = Query(None, description="Filter by linked person UUID"),
    zone_id: Optional[uuid.UUID] = Query(None, description="Filter by zone UUID"),
    start_time: Optional[datetime] = Query(None, description="ISO8601 lower bound for occurred_at"),
    end_time: Optional[datetime] = Query(None, description="ISO8601 upper bound for occurred_at"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    _require_db()
    async with get_session() as session:
        events = await dal.list_events(
            session,
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
    return [EventOut.model_validate(e) for e in events]


@router.get("/events/{event_id}", response_model=EventOut, summary="Get event by event_id string")
async def get_event(event_id: str):
    _require_db()
    async with get_session() as session:
        event = await dal.get_event_by_event_id(session, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found")
    return EventOut.model_validate(event)


@router.post(
    "/events/{event_uuid}/link-person",
    response_model=EventOut,
    summary="Link an event to a known person (manual audit labelling)",
)
async def link_event_to_person(event_uuid: uuid.UUID, body: LinkPersonBody):
    """Associate an event record with a person profile.

    Useful for building a person-level audit trail without full ReID:
    an operator reviews an event (e.g. a fall), identifies the person, and
    calls this endpoint to label the event.
    """
    _require_db()
    async with get_session() as session:
        # Verify person exists
        person = await dal.get_person(session, body.person_id)
        if person is None:
            raise HTTPException(status_code=404, detail=f"Person {body.person_id} not found")
        event = await dal.link_event_to_person(session, event_uuid, body.person_id)
        if event is None:
            raise HTTPException(status_code=404, detail=f"Event {event_uuid} not found")
        out = EventOut.model_validate(event)
        await session.commit()
    logger.info(f"[admin] Linked event={event_uuid} → person={body.person_id}")
    return out


# ── Admin reset ────────────────────────────────────────────────────────────────

@router.post(
    "/reset/{camera_id}",
    response_model=AdminResetOut,
    summary="Reset in-memory tracking state for a camera",
)
async def admin_reset_camera(camera_id: str):
    """Reset in-memory tracking state for one camera.

    Unlike `POST /reset/{camera_id}`, this endpoint is under `/admin`
    and provides a consistent JSON response. DB records are preserved.
    """
    from .tracking import camera_states, camera_states_lock

    with camera_states_lock:
        state = camera_states.get(camera_id)

    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"Camera '{camera_id}' not yet initialised",
        )

    with state["lock"]:
        old_count = len(state["seen_ids"])
        state["seen_ids"].clear()
        state["seen_ids_with_time"].clear()
        state["tracks"].clear()
        state["track_history"].clear()
        state["next_id"] = 1
        state["frame_idx"] = 0
        if state.get("fall_detector"):
            state["fall_detector"].reset_fall()

    logger.info(f"[admin] Reset camera={camera_id!r} cleared={old_count} persons")
    return AdminResetOut(
        status="reset",
        cameras_reset=1,
        total_persons_cleared=old_count,
    )


@router.post(
    "/reset_all",
    response_model=AdminResetOut,
    summary="Reset in-memory tracking state for ALL cameras",
)
async def admin_reset_all():
    from .tracking import camera_states, camera_states_lock

    with camera_states_lock:
        camera_items = list(camera_states.items())

    total_cleared = 0
    for _, state in camera_items:
        with state["lock"]:
            total_cleared += len(state["seen_ids"])
            state["seen_ids"].clear()
            state["seen_ids_with_time"].clear()
            state["tracks"].clear()
            state["track_history"].clear()
            state["next_id"] = 1
            state["frame_idx"] = 0
            if state.get("fall_detector"):
                state["fall_detector"].reset_fall()

    logger.info(f"[admin] Reset all {len(camera_items)} cameras, cleared={total_cleared}")
    return AdminResetOut(
        status="reset_all",
        cameras_reset=len(camera_items),
        total_persons_cleared=total_cleared,
    )


# ── Camera config endpoints (P2.2) ────────────────────────────────────────────

@router.get("/camera-configs/{camera_id}", response_model=CameraConfigOut, summary="Get per-camera inference config")
async def get_camera_config(camera_id: str):
    _require_db()
    async with get_session() as session:
        cfg = await dal.get_camera_config(session, camera_id)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"No config found for camera '{camera_id}'")
    return CameraConfigOut.model_validate(cfg)


@router.put(
    "/camera-configs/{camera_id}",
    response_model=CameraConfigOut,
    summary="Create or update per-camera inference config",
)
async def upsert_camera_config(camera_id: str, body: CameraConfigUpsert):
    _require_db()
    payload = {k: v for k, v in body.model_dump().items() if v is not None}
    async with get_session() as session:
        cfg = await dal.upsert_camera_config(session, camera_id=camera_id, **payload)
        out = CameraConfigOut.model_validate(cfg)
        await session.commit()
    invalidate_config_cache(camera_id)
    logger.info(f"[admin] Upserted camera config camera={camera_id!r} fields={list(payload.keys())}")
    return out


@router.delete("/camera-configs/{camera_id}", status_code=204, summary="Delete per-camera inference config")
async def delete_camera_config(camera_id: str):
    _require_db()
    async with get_session() as session:
        deleted = await dal.delete_camera_config(session, camera_id=camera_id)
    invalidate_config_cache(camera_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No config found for camera '{camera_id}'")
    logger.info(f"[admin] Deleted camera config camera={camera_id!r}")


# ── Person embedding endpoints (S P2.1) ───────────────────────────────────────

@router.get(
    "/persons/{person_id}/embeddings",
    response_model=List[EmbeddingOut],
    summary="List ReID embeddings for a person",
)
async def list_embeddings(person_id: uuid.UUID):
    """Return metadata for all stored body/face embeddings for this person.

    Raw embedding bytes are not included in the response.
    """
    _require_db()
    async with get_session() as session:
        person = await dal.get_person(session, person_id)
        if person is None:
            raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
        embs = await dal.list_person_embeddings(session, person_id)
    return [EmbeddingOut.model_validate(e) for e in embs]


@router.post(
    "/persons/{person_id}/embeddings",
    response_model=EmbeddingOut,
    status_code=201,
    summary="Enroll a person from an uploaded image (S P2.1)",
)
async def enroll_from_image(person_id: uuid.UUID, body: EnrollFromImageBody):
    """Compute a ReID embedding from an uploaded image and attach it to a person profile.

    The image should be a crop containing a single person.  If the person occupies
    only part of the image, supply bbox_x/y/w/h to crop before embedding.

    Use this to build a reference gallery for known residents.
    """
    _require_db()
    async with get_session() as session:
        person = await dal.get_person(session, person_id)
        if person is None:
            raise HTTPException(status_code=404, detail=f"Person {person_id} not found")

    try:
        jpeg_bytes = base64.b64decode(body.image_b64)
    except Exception:
        raise HTTPException(status_code=422, detail="image_b64 is not valid base64")

    from . import reid_service

    try:
        if body.embedding_type == "face":
            emb_id = await reid_service.enroll_face_from_jpeg(
                person_id,
                jpeg_bytes,
                bbox_x=body.bbox_x,
                bbox_y=body.bbox_y,
                bbox_w=body.bbox_w,
                bbox_h=body.bbox_h,
                source=body.source,
            )
        else:
            emb_id = await reid_service.enroll_person_from_jpeg(
                person_id,
                jpeg_bytes,
                bbox_x=body.bbox_x,
                bbox_y=body.bbox_y,
                bbox_w=body.bbox_w,
                bbox_h=body.bbox_h,
                embedding_type=body.embedding_type,
                source=body.source,
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    async with get_session() as session:
        emb = await dal.get_person_embedding(session, emb_id)
    logger.info(f"[admin] Enrolled embedding id={emb_id} for person={person_id}")
    return EmbeddingOut.model_validate(emb)


@router.post(
    "/persons/{person_id}/embeddings/face-batch",
    response_model=EnrollFaceBatchResult,
    status_code=201,
    summary="Enroll face from 5–30 photos or a video clip (3–5 fps sampling)",
)
async def enroll_face_batch(person_id: uuid.UUID, body: EnrollFaceBatchBody):
    """Build a face gallery from multiple views for more stable recognition.

    - **Photos:** send 5–30 base64 images; each frame is processed like a single upload.
    - **Video:** send one base64 video; frames are extracted at 3–5 fps (max 30 frames),
      then the same face pipeline runs. At least 5 frames must contain a detectable face.
    """
    _require_db()
    async with get_session() as session:
        person = await dal.get_person(session, person_id)
        if person is None:
            raise HTTPException(status_code=404, detail=f"Person {person_id} not found")

    from . import reid_service

    try:
        if body.images_b64:
            decoded: list[bytes] = []
            for i, b64 in enumerate(body.images_b64):
                try:
                    decoded.append(base64.b64decode(b64))
                except Exception:
                    raise HTTPException(
                        status_code=422,
                        detail=f"images_b64[{i}] is not valid base64",
                    )
            result = await reid_service.enroll_faces_from_image_bytes_list(
                person_id,
                decoded,
                source=body.source or "upload_batch",
            )
        else:
            try:
                video_bytes = base64.b64decode(body.video_b64)  # type: ignore[arg-type]
            except Exception:
                raise HTTPException(status_code=422, detail="video_b64 is not valid base64")
            result = await reid_service.enroll_faces_from_video_bytes(
                person_id,
                video_bytes,
                sample_fps=body.video_sample_fps,
                source=body.source or "upload_video",
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    logger.info(
        "[admin] Face batch enroll person=%s embeddings=%d frames=%d with_face=%d",
        person_id,
        len(result.embedding_ids),
        result.frames_processed,
        result.frames_with_face,
    )
    return EnrollFaceBatchResult(
        person_id=person_id,
        embeddings_created=len(result.embedding_ids),
        frames_processed=result.frames_processed,
        frames_with_face=result.frames_with_face,
        embedding_ids=result.embedding_ids,
    )


@router.post(
    "/persons/{person_id}/embeddings/from-event/{event_uuid}",
    response_model=EmbeddingOut,
    status_code=201,
    summary="Enroll a person using a stored event snapshot (S P2.1)",
)
async def enroll_from_event(person_id: uuid.UUID, event_uuid: uuid.UUID):
    """Download the snapshot from an existing fall/zone_violation event and enroll
    the person from it.

    Requires MinIO to be configured and the event to have a snapshot.
    """
    _require_db()
    async with get_session() as session:
        person = await dal.get_person(session, person_id)
        if person is None:
            raise HTTPException(status_code=404, detail=f"Person {person_id} not found")

    from . import reid_service

    try:
        emb_id = await reid_service.enroll_person_from_event(person_id, event_uuid)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    async with get_session() as session:
        emb = await dal.get_person_embedding(session, emb_id)
    logger.info(f"[admin] Enrolled embedding from event={event_uuid} for person={person_id}")
    return EmbeddingOut.model_validate(emb)


@router.delete(
    "/persons/{person_id}/embeddings/{emb_id}",
    status_code=204,
    summary="Delete a specific embedding",
)
async def delete_embedding(person_id: uuid.UUID, emb_id: uuid.UUID):
    _require_db()
    async with get_session() as session:
        emb = await dal.get_person_embedding(session, emb_id)
        if emb is None or emb.person_id != person_id:
            raise HTTPException(status_code=404, detail=f"Embedding {emb_id} not found for person {person_id}")
        await dal.delete_person_embedding(session, emb_id)
        await session.commit()
    logger.info(f"[admin] Deleted embedding id={emb_id} for person={person_id}")


# ── Live tracks (for the operator UI to pick an Unknown person) ───────────────

@router.get(
    "/live/tracks",
    response_model=Dict[str, Any],
    summary="Current live tracks per camera with identity status",
)
async def live_tracks(camera_id: Optional[str] = Query(None, description="Filter to one camera")):
    """Return the tracks currently held in memory, with resolved identity.

    The UI uses this to list people on screen and offer an 'Enroll' action for
    any track still labelled Unknown.
    """
    from .tracking import camera_states, camera_states_lock

    with camera_states_lock:
        items = [(cid, st) for cid, st in camera_states.items() if (camera_id is None or cid == camera_id)]

    cameras: Dict[str, Any] = {}
    for cid, state in items:
        with state["lock"]:
            identity = dict(state.get("identity", {}))
            tracks = [dict(tr) for tr in state.get("tracks", [])]
        track_list = []
        for tr in tracks:
            if tr.get("status") != "confirmed":
                continue
            tid = tr["id"]
            ident = identity.get(tid, {})
            bbox = tr.get("bbox") or (0, 0, 0, 0)
            track_list.append({
                "track_id": tid,
                "recognized": bool(ident.get("recognized")),
                "person_id": ident.get("person_id"),
                "person_name": ident.get("name") or ("Unknown" if ident else None),
                "person_no": ident.get("no"),
                "person_age": ident.get("age"),
                "gender": ident.get("gender"),
                "bbox": {"x": bbox[0], "y": bbox[1], "x2": bbox[2], "y2": bbox[3]},
                "has_crop": tid in (state.get("recent_crops") or {}),
            })
        cameras[cid] = track_list
    return {"cameras": cameras}


# ── Event snapshot proxy (serves images from MinIO to the UI) ─────────────────

@router.get(
    "/events/{event_uuid}/snapshot",
    summary="Fetch an event's snapshot image (proxied from object storage)",
)
async def event_snapshot(event_uuid: uuid.UUID):
    from fastapi import Response
    from . import reid_service

    _require_db()
    async with get_session() as session:
        event = await dal.get_event(session, event_uuid)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    payload = event.payload or {}
    key = payload.get("snapshot_key")
    if not key:
        raise HTTPException(status_code=404, detail="Event has no snapshot")
    jpeg = await reid_service._download_snapshot(key)
    if jpeg is None:
        raise HTTPException(status_code=404, detail="Snapshot not available (object storage not configured?)")
    return Response(content=jpeg, media_type="image/jpeg")


# ── Enroll from a live track (face capture from the frame) ────────────────────

def _get_recent_track_crop(camera_id: str, track_id: int):
    """Fetch the most recent head-region crop cached for a live track.

    Returns a BGR numpy array (copy) or None if the track is not present /
    has no cached crop yet.
    """
    from .tracking import camera_states, camera_states_lock

    with camera_states_lock:
        state = camera_states.get(camera_id)
    if state is None:
        return None
    with state["lock"]:
        crops = state.get("recent_crops") or {}
        crop = crops.get(track_id)
        if crop is None:
            return None
        return crop.copy()


@router.post(
    "/enroll/from-track",
    response_model=EnrollFromTrackResult,
    status_code=201,
    summary="Enroll a person's face from a live (possibly Unknown) track",
)
async def enroll_from_track(body: EnrollFromTrackBody):
    """Capture the most recent face crop of a live track and enroll it.

    Operator flow for an 'Unknown' bounding box:
      1. UI shows the live track_id as Unknown.
      2. Operator types the person's name/gender (or picks an existing person).
      3. This endpoint grabs the latest cached crop of that track, extracts a
         face embedding, creates/links the person, and refreshes the gallery so
         the next frames label the box as "(Tên, Giới, No.)".
    """
    _require_db()

    if body.person_id is None and not body.name:
        raise HTTPException(
            status_code=422,
            detail="Provide either person_id (existing) or name (to create a new person)",
        )

    crop = _get_recent_track_crop(body.camera_id, body.track_id)
    if crop is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No recent crop for track {body.track_id} on camera '{body.camera_id}'. "
                "Make sure the person is currently visible and face recognition is enabled."
            ),
        )

    created_person = False
    async with get_session() as session:
        if body.person_id is not None:
            person = await dal.get_person(session, body.person_id)
            if person is None:
                raise HTTPException(status_code=404, detail=f"Person {body.person_id} not found")
        else:
            create_payload = _prepare_person_write({
                "name": body.name,
                "date_of_birth": body.date_of_birth,
                "gender": body.gender,
                "room": body.room,
                "notes": body.notes,
                "status": "active",
            })
            person = await dal.create_person(session, **create_payload)
            created_person = True
        person_pk = person.id
        person_name = person.name
        await session.commit()

    from . import reid_service

    try:
        emb_id = await reid_service.enroll_face_from_bgr(
            person_pk, crop, source=f"track:{body.camera_id}:{body.track_id}"
        )
    except ValueError as exc:
        # If we just created the person but couldn't find a face, surface a clear error.
        raise HTTPException(status_code=422, detail=str(exc))

    logger.info(
        "[admin] Enrolled face from track camera=%s track=%s -> person=%s (created=%s)",
        body.camera_id, body.track_id, person_pk, created_person,
    )
    return EnrollFromTrackResult(
        person_id=person_pk,
        person_name=person_name,
        embedding_id=emb_id,
        created_person=created_person,
    )


# ── ReID match + batch endpoints (S P2.1) ─────────────────────────────────────

@router.post(
    "/reid/match",
    response_model=ReidMatchResult,
    summary="Match an image against all enrolled persons (S P2.1)",
)
async def reid_match(body: ReidMatchBody):
    """Upload an image (base64 JPEG/PNG) and find the best matching enrolled person.

    Returns the matched person's ID, name, and cosine similarity score,
    or matched=false if no person is above the similarity threshold.
    """
    _require_db()
    try:
        jpeg_bytes = base64.b64decode(body.image_b64)
    except Exception:
        raise HTTPException(status_code=422, detail="image_b64 is not valid base64")

    from . import reid_service

    result = await reid_service.match_jpeg_to_person(jpeg_bytes, threshold=body.threshold)

    if result is None:
        return ReidMatchResult(matched=False, person_id=None, score=None)

    person_id, score = result
    async with get_session() as session:
        person = await dal.get_person(session, person_id)
    return ReidMatchResult(
        matched=True,
        person_id=person_id,
        score=round(score, 4),
        person_name=person.name if person else None,
    )


@router.post(
    "/reid/auto-link",
    summary="Batch auto-link unlinked fall/zone events to persons via ReID (S P2.1)",
    response_model=Dict[str, Any],
)
async def reid_auto_link(body: BatchAutoLinkBody):
    """Run ReID on recent unlinked fall and zone_violation events and automatically
    link them to the best matching person profile.

    Useful after enrolling new persons to retroactively label past events.
    Returns counts of processed, linked, and skipped events.
    """
    _require_db()
    from . import reid_service

    summary = await reid_service.batch_auto_link_unlinked(
        event_types=body.event_types,
        limit=body.limit,
        threshold=body.threshold,
    )
    logger.info(f"[admin] ReID batch auto-link: {summary}")
    return summary


# ── Retention status (S P2.4) ─────────────────────────────────────────────────

@router.get(
    "/retention/status",
    summary="Retention policy status",
    response_model=Dict[str, Any],
)
async def retention_status():
    """Returns current retention worker configuration and running state."""
    return _retention_status()


# ── Analytics endpoints (S P2.5) ──────────────────────────────────────────────

@router.get(
    "/analytics/summary",
    response_model=Dict[str, Any],
    summary="Overall system analytics summary",
)
async def analytics_summary():
    """Return aggregate counts for events, persons, zones, cameras, and alerts.

    Useful for a top-level status dashboard — all figures are computed live
    from PostgreSQL.  Counts are broken down by event type and time window
    (last 24 h, 7 d, 30 d).
    """
    _require_db()
    async with get_session() as session:
        return await dal.get_analytics_summary(session)


@router.get(
    "/analytics/events/timeseries",
    response_model=List[Dict[str, Any]],
    summary="Event counts over time",
)
async def analytics_timeseries(
    granularity: str = Query(
        "day",
        pattern="^(hour|day|week)$",
        description="Time bucket size",
    ),
    days_back: int = Query(7, ge=1, le=365, description="How many days back to query"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
):
    """Return event counts bucketed by time for charting trends.

    Example: `?granularity=hour&days_back=1` returns hourly counts for the
    last 24 hours.  `?granularity=day&event_type=fall` returns daily fall counts.
    """
    _require_db()
    async with get_session() as session:
        return await dal.get_events_timeseries(
            session,
            granularity=granularity,
            days_back=days_back,
            event_type=event_type,
        )


@router.get(
    "/analytics/cameras",
    response_model=List[Dict[str, Any]],
    summary="Per-camera statistics",
)
async def analytics_cameras():
    """Return aggregate statistics for every camera that has produced events.

    Includes: total events, fall count, zone violation count, unique track
    count, number of events linked to persons via ReID, and last-seen time.
    """
    _require_db()
    async with get_session() as session:
        return await dal.get_camera_analytics(session)


@router.get(
    "/analytics/zones",
    response_model=List[Dict[str, Any]],
    summary="Per-zone violation statistics",
)
async def analytics_zones():
    """Return violation counts and last-violation timestamps for every zone."""
    _require_db()
    async with get_session() as session:
        return await dal.get_zone_analytics(session)


@router.get(
    "/analytics/alerts",
    response_model=Dict[str, Any],
    summary="Alert delivery statistics",
)
async def analytics_alerts():
    """Return alert counts broken down by delivery status and channel."""
    _require_db()
    async with get_session() as session:
        return await dal.get_alert_analytics(session)
