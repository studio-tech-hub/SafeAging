"""S P2.1 — Higher-level ReID operations.

Combines reid_engine (feature extraction) with DB and MinIO access to provide:
  - Person enrollment from an uploaded image or event snapshot
  - Crop-to-person matching against all stored embeddings
  - Automatic event→person linking (called from the outbox worker fan-out)

All async entry points offload CPU-bound work (model inference) to a thread via
`asyncio.to_thread()` so the event loop is never blocked.
"""

import asyncio
import io
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .reid_engine import (
    DEFAULT_MATCH_THRESHOLD,
    EMBEDDING_DIM,
    bytes_to_embedding,
    compute_embedding,
    embedding_to_bytes,
    find_best_match,
)

logger = logging.getLogger(__name__)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _crop_bgr(bgr: np.ndarray, bx: float, by: float, bw: float, bh: float) -> np.ndarray:
    """Crop a BGR frame to a bounding box; returns the full image if bbox is invalid."""
    h, w = bgr.shape[:2]
    x1, y1 = max(0, int(bx)), max(0, int(by))
    x2, y2 = min(w, int(bx + bw)), min(h, int(by + bh))
    if x2 > x1 and y2 > y1:
        return bgr[y1:y2, x1:x2]
    return bgr


def _jpeg_to_bgr(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Cannot decode image bytes (expected JPEG/PNG)")
    return bgr


def _subsample_list(items: list, max_n: int) -> list:
    """Evenly subsample *items* down to at most *max_n* entries."""
    if len(items) <= max_n:
        return items
    if max_n <= 0:
        return []
    step = len(items) / max_n
    return [items[int(i * step)] for i in range(max_n)]


def _video_bytes_to_bgr_frames(
    video_bytes: bytes,
    sample_fps: float,
    max_frames: int,
) -> list[np.ndarray]:
    """Decode a video clip and return BGR frames sampled at *sample_fps* (3–5 typical)."""
    if not video_bytes:
        raise ValueError("Empty video payload")

    suffix = ".mp4"
    tmp_path = None
    frames: list[np.ndarray] = []
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video file (unsupported format or corrupt file)")

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if native_fps is None or native_fps < 1.0 or native_fps > 240.0:
            native_fps = 25.0

        frame_interval = max(1, int(round(native_fps / max(sample_fps, 0.1))))
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if index % frame_interval == 0:
                frames.append(frame)
            index += 1
        cap.release()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if not frames:
        raise ValueError("No frames could be read from the video")

    return _subsample_list(frames, max_frames)


@dataclass
class FaceBatchEnrollResult:
    embedding_ids: list[uuid.UUID]
    frames_processed: int
    frames_with_face: int


async def enroll_faces_batch(
    person_id: uuid.UUID,
    bgr_frames: list[np.ndarray],
    *,
    source: Optional[str] = None,
    min_faces: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> FaceBatchEnrollResult:
    """Extract face embeddings from multiple BGR frames and persist each success.

    Requires at least *min_faces* successful detections (default from config).
    Processes at most *max_frames* inputs (subsampled evenly if more are supplied).
    """
    from . import face_engine
    from .config import ENROLL_FACE_MAX_IMAGES, ENROLL_FACE_MIN_IMAGES
    from .db import dal, get_session

    min_required = ENROLL_FACE_MIN_IMAGES if min_faces is None else min_faces
    max_allowed = ENROLL_FACE_MAX_IMAGES if max_frames is None else max_frames

    if not bgr_frames:
        raise ValueError("No frames to process")

    selected = _subsample_list(bgr_frames, max_allowed)
    extracted: list[tuple[int, np.ndarray]] = []

    for idx, bgr in enumerate(selected):
        emb = await asyncio.to_thread(face_engine.extract_embedding, bgr)
        if emb is not None:
            extracted.append((idx, emb))

    if len(extracted) < min_required:
        raise ValueError(
            f"Need at least {min_required} frames with a detectable face, "
            f"got {len(extracted)} from {len(selected)} frame(s)"
        )

    embedding_ids: list[uuid.UUID] = []
    async with get_session() as session:
        for idx, emb in extracted:
            e = await dal.create_person_embedding(
                session,
                person_id=person_id,
                embedding=face_engine.embedding_to_bytes(emb),
                embedding_type="face",
                source=source or f"batch:{idx}",
            )
            embedding_ids.append(e.id)
        await session.commit()

    face_engine.invalidate_gallery()
    logger.info(
        "[reid] Batch face enroll person=%s embeddings=%d frames=%d with_face=%d",
        person_id,
        len(embedding_ids),
        len(selected),
        len(extracted),
    )
    return FaceBatchEnrollResult(
        embedding_ids=embedding_ids,
        frames_processed=len(selected),
        frames_with_face=len(extracted),
    )


async def enroll_faces_from_image_bytes_list(
    person_id: uuid.UUID,
    image_bytes_list: list[bytes],
    source: Optional[str] = "upload_batch",
) -> FaceBatchEnrollResult:
    """Decode multiple uploaded images and enroll all detectable faces."""
    frames: list[np.ndarray] = []
    for raw in image_bytes_list:
        frames.append(await asyncio.to_thread(_jpeg_to_bgr, raw))
    return await enroll_faces_batch(person_id, frames, source=source)


async def enroll_faces_from_video_bytes(
    person_id: uuid.UUID,
    video_bytes: bytes,
    sample_fps: Optional[float] = None,
    source: Optional[str] = "upload_video",
) -> FaceBatchEnrollResult:
    """Sample frames from a video clip (3–5 fps) then batch-enroll faces."""
    from .config import ENROLL_FACE_MAX_IMAGES, ENROLL_VIDEO_SAMPLE_FPS

    fps = ENROLL_VIDEO_SAMPLE_FPS if sample_fps is None else sample_fps
    fps = max(3.0, min(5.0, float(fps)))

    frames = await asyncio.to_thread(
        _video_bytes_to_bgr_frames,
        video_bytes,
        fps,
        ENROLL_FACE_MAX_IMAGES,
    )
    return await enroll_faces_batch(person_id, frames, source=source)


async def _download_snapshot(key: str) -> Optional[bytes]:
    """Download an object from MinIO/S3. Returns bytes or None on any failure."""
    from .config import S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET_SNAPSHOTS

    if not S3_ENDPOINT or not S3_ACCESS_KEY:
        return None

    def _sync() -> Optional[bytes]:
        try:
            import boto3

            s3 = boto3.client(
                "s3",
                endpoint_url=S3_ENDPOINT,
                aws_access_key_id=S3_ACCESS_KEY,
                aws_secret_access_key=S3_SECRET_KEY,
            )
            buf = io.BytesIO()
            s3.download_fileobj(S3_BUCKET_SNAPSHOTS, key, buf)
            return buf.getvalue()
        except Exception as exc:
            logger.debug("[reid] MinIO download failed key=%s: %s", key, exc)
            return None

    return await asyncio.to_thread(_sync)


async def _load_all_candidates() -> list[tuple[uuid.UUID, np.ndarray]]:
    """Load all embeddings from DB and return as list of (person_id, np.ndarray)."""
    from .db import get_session
    from .db import dal

    async with get_session() as session:
        embs = await dal.list_all_embeddings_for_matching(session)
    return [(e.person_id, bytes_to_embedding(e.embedding)) for e in embs]


# ── Public API ────────────────────────────────────────────────────────────────

async def enroll_person_from_bgr(
    person_id: uuid.UUID,
    bgr: np.ndarray,
    embedding_type: str = "body",
    source: Optional[str] = None,
) -> uuid.UUID:
    """Compute embedding from a BGR image array and persist it to the DB.

    Returns the new embedding's UUID.
    """
    from .db import get_session
    from .db import dal

    vec = await asyncio.to_thread(compute_embedding, bgr)
    async with get_session() as session:
        emb = await dal.create_person_embedding(
            session,
            person_id=person_id,
            embedding=embedding_to_bytes(vec),
            embedding_type=embedding_type,
            source=source or "manual",
        )
        emb_id = emb.id
        await session.commit()
    logger.info("[reid] Enrolled embedding id=%s for person=%s type=%s", emb_id, person_id, embedding_type)
    return emb_id


async def enroll_person_from_jpeg(
    person_id: uuid.UUID,
    jpeg_bytes: bytes,
    bbox_x: Optional[float] = None,
    bbox_y: Optional[float] = None,
    bbox_w: Optional[float] = None,
    bbox_h: Optional[float] = None,
    embedding_type: str = "body",
    source: Optional[str] = None,
) -> uuid.UUID:
    """Decode JPEG, optionally crop to person bounding box, then enroll.

    Returns the new embedding's UUID.
    """
    bgr = await asyncio.to_thread(_jpeg_to_bgr, jpeg_bytes)
    if all(v is not None for v in (bbox_x, bbox_y, bbox_w, bbox_h)):
        bgr = _crop_bgr(bgr, bbox_x, bbox_y, bbox_w, bbox_h)  # type: ignore[arg-type]
    return await enroll_person_from_bgr(person_id, bgr, embedding_type, source)


async def enroll_face_from_bgr(
    person_id: uuid.UUID,
    bgr: np.ndarray,
    source: Optional[str] = None,
) -> uuid.UUID:
    """Detect a face in a BGR image, compute its ArcFace embedding, and persist it.

    Raises ValueError if no usable face is found.
    Invalidates the live face-recognition gallery so the new person is matched
    immediately on subsequent frames.
    """
    from . import face_engine
    from .db import get_session
    from .db import dal

    emb = await asyncio.to_thread(face_engine.extract_embedding, bgr)
    if emb is None:
        raise ValueError("No face detected in the provided image")

    async with get_session() as session:
        e = await dal.create_person_embedding(
            session,
            person_id=person_id,
            embedding=face_engine.embedding_to_bytes(emb),
            embedding_type="face",
            source=source or "manual",
        )
        emb_id = e.id
        await session.commit()

    face_engine.invalidate_gallery()
    logger.info("[reid] Enrolled FACE embedding id=%s for person=%s", emb_id, person_id)
    return emb_id


async def enroll_face_from_jpeg(
    person_id: uuid.UUID,
    jpeg_bytes: bytes,
    bbox_x: Optional[float] = None,
    bbox_y: Optional[float] = None,
    bbox_w: Optional[float] = None,
    bbox_h: Optional[float] = None,
    source: Optional[str] = None,
) -> uuid.UUID:
    """Decode JPEG/PNG, optionally crop to a bbox, then enroll a face embedding."""
    bgr = await asyncio.to_thread(_jpeg_to_bgr, jpeg_bytes)
    if all(v is not None for v in (bbox_x, bbox_y, bbox_w, bbox_h)):
        bgr = _crop_bgr(bgr, bbox_x, bbox_y, bbox_w, bbox_h)  # type: ignore[arg-type]
    return await enroll_face_from_bgr(person_id, bgr, source)


async def enroll_person_from_event(
    person_id: uuid.UUID,
    event_db_id: uuid.UUID,
    embedding_type: str = "body",
) -> uuid.UUID:
    """Download an event's snapshot from MinIO, crop to the person's bbox, and enroll.

    Raises ValueError if MinIO is not configured or the snapshot is missing.
    Returns the new embedding's UUID.
    """
    from .db import get_session
    from .db.models import Event as EventModel
    from sqlalchemy import select

    async with get_session() as session:
        result = await session.execute(
            select(EventModel).where(EventModel.id == event_db_id)
        )
        event = result.scalar_one_or_none()
        if event is None:
            raise ValueError(f"Event {event_db_id} not found")
        payload = event.payload or {}
        snapshot_key = payload.get("snapshot_key")
        bbox = (event.bbox_x, event.bbox_y, event.bbox_w, event.bbox_h)

    if not snapshot_key:
        raise ValueError(f"Event {event_db_id} has no snapshot (not a fall/zone event?)")

    jpeg_bytes = await _download_snapshot(snapshot_key)
    if jpeg_bytes is None:
        raise ValueError("MinIO not configured or snapshot download failed")

    return await enroll_person_from_jpeg(
        person_id,
        jpeg_bytes,
        bbox_x=bbox[0],
        bbox_y=bbox[1],
        bbox_w=bbox[2],
        bbox_h=bbox[3],
        embedding_type=embedding_type,
        source=f"event:{event_db_id}",
    )


async def match_jpeg_to_person(
    jpeg_bytes: bytes,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> tuple[uuid.UUID, float] | None:
    """Find the best matching person for a JPEG image.

    Returns (person_id, cosine_score) if a match is found above threshold, else None.
    """
    bgr = await asyncio.to_thread(_jpeg_to_bgr, jpeg_bytes)
    candidates = await _load_all_candidates()
    if not candidates:
        return None
    vec = await asyncio.to_thread(compute_embedding, bgr)
    return find_best_match(vec, candidates, threshold)


async def try_auto_link_event(
    event_db_id: uuid.UUID,
    threshold: Optional[float] = None,
) -> tuple[uuid.UUID, float] | None:
    """Download event snapshot, match against person embeddings, auto-link if found.

    Called from the outbox worker fan-out for fall/zone_violation events.
    Best-effort: any exception is caught and logged; returns None on failure.

    Returns (person_id, score) if successfully linked, else None.
    """
    from .config import REID_MATCH_THRESHOLD

    effective_threshold = threshold if threshold is not None else REID_MATCH_THRESHOLD

    try:
        from .db import get_session
        from .db import dal
        from .db.models import Event as EventModel
        from sqlalchemy import select

        async with get_session() as session:
            result = await session.execute(
                select(EventModel).where(EventModel.id == event_db_id)
            )
            event = result.scalar_one_or_none()
            if event is None:
                return None
            if event.person_id is not None:
                # Already linked — skip
                return None
            payload = event.payload or {}
            snapshot_key = payload.get("snapshot_key")
            bbox = (event.bbox_x, event.bbox_y, event.bbox_w, event.bbox_h)

        if not snapshot_key:
            return None

        jpeg_bytes = await _download_snapshot(snapshot_key)
        if jpeg_bytes is None:
            return None

        candidates = await _load_all_candidates()
        if not candidates:
            return None

        bgr = await asyncio.to_thread(_jpeg_to_bgr, jpeg_bytes)
        if all(v is not None for v in bbox):
            bgr = _crop_bgr(bgr, *bbox)  # type: ignore[arg-type]

        vec = await asyncio.to_thread(compute_embedding, bgr)
        match_result = find_best_match(vec, candidates, effective_threshold)

        if match_result is None:
            return None

        person_id, score = match_result
        async with get_session() as session:
            await dal.link_event_to_person(session, event_db_id, person_id)
            await session.commit()

        logger.info(
            "[reid] Auto-linked event=%s → person=%s score=%.3f",
            event_db_id,
            person_id,
            score,
        )
        try:
            from .metrics import REID_AUTO_LINKS
            # camera_id not directly available here; use "unknown" as label
            REID_AUTO_LINKS.labels(camera_id="auto").inc()
        except Exception:
            pass
        return person_id, score

    except Exception as exc:
        logger.debug("[reid] Auto-link failed for event=%s: %s", event_db_id, exc)
        return None


async def batch_auto_link_unlinked(
    event_types: list[str] | None = None,
    limit: int = 200,
    threshold: Optional[float] = None,
) -> dict:
    """Batch: try to link recent unlinked events to persons via ReID.

    Returns a summary dict with counts of processed, linked, and skipped events.
    """
    from .config import REID_MATCH_THRESHOLD
    from .db import get_session
    from .db import dal
    from .db.models import Event as EventModel
    from sqlalchemy import select

    effective_threshold = threshold if threshold is not None else REID_MATCH_THRESHOLD
    types = event_types or ["fall", "zone_violation"]

    async with get_session() as session:
        stmt = (
            select(EventModel)
            .where(
                EventModel.person_id.is_(None),
                EventModel.event_type.in_(types),
            )
            .order_by(EventModel.occurred_at.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        events = list(result.scalars().all())
        event_ids = [e.id for e in events]

    processed, linked, skipped = 0, 0, 0
    for eid in event_ids:
        processed += 1
        result = await try_auto_link_event(eid, threshold=effective_threshold)
        if result is not None:
            linked += 1
        else:
            skipped += 1

    logger.info(
        "[reid] Batch auto-link done: processed=%d linked=%d skipped=%d",
        processed,
        linked,
        skipped,
    )
    return {"processed": processed, "linked": linked, "skipped": skipped}
