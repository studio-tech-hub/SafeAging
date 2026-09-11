"""Real-time face recognition engine (ArcFace via insightface).

Responsibilities
----------------
- Lazily load an insightface model pack (SCRFD detector + ArcFace recogniser).
- Extract a 512-d L2-normalised face embedding from a person crop.
- Maintain an in-memory gallery of enrolled face embeddings (loaded from the
  DB in a background thread, like zone_engine) and match query embeddings
  against it via cosine similarity.

Design notes
------------
- All heavy work is synchronous and CPU/GPU-bound; it is called from the sync
  /infer endpoint (which already runs in a thread-pool worker), so it never
  blocks the asyncio event loop.
- If insightface is not installed or the model fails to load, the engine
  degrades gracefully: `available()` returns False and recognition is skipped,
  so the rest of the pipeline keeps running.
- The gallery is refreshed on a TTL and can be force-invalidated after enrollment.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Face embedding dimension (buffalo_l/s ArcFace — 512-d).
FACE_EMBEDDING_DIM: int = 512

# ── model state ───────────────────────────────────────────────────────────────
_model_lock = threading.Lock()
_app = None                      # insightface.app.FaceAnalysis instance
_model_load_failed = False


@dataclass
class FaceMatch:
    person_id: str
    name: str
    gender: Optional[str]
    age: Optional[int]
    score: float


# insightface's buffalo_s/buffalo_l packs bundle 5 ONNX models: detection,
# recognition (ArcFace embedding — what extract_embedding()/recognize_crop()
# actually use), and three we never read: genderage (face.gender/face.age —
# this codebase uses the DB-stored date_of_birth/gender instead, see
# person_age.effective_age()), landmark_3d_68, and landmark_2d_106 (extra
# high-res landmark sets; alignment for the recognition model uses face.kps,
# which comes from the *detection* model's own output, not these). Loading
# and running 3 unused ONNX models on every single face costs real CPU for
# zero benefit -- confirmed empirically during P1-6: restricting to just the
# two tasks actually consumed cut per-call latency ~5-8x on a CPU benchmark
# (see CPU_PRODUCTION_PROFILE.md's P1-6 section for the measured numbers).
_REQUIRED_FACE_TASKS: tuple[str, ...] = ("detection", "recognition")


def _build_app():
    """Create and prepare an insightface FaceAnalysis app. Returns None on failure."""
    from .config import FACE_DET_SIZE, FACE_MODEL_PACK
    from .face_providers import build_face_providers

    try:
        from insightface.app import FaceAnalysis
    except Exception as exc:  # pragma: no cover - import guard
        logger.warning("[face] insightface not available: %s", exc)
        return None

    providers, provider_options, backend_label = build_face_providers()
    ctx_id = -1

    try:
        kwargs: dict = {
            "name": FACE_MODEL_PACK,
            "providers": providers,
            "allowed_modules": list(_REQUIRED_FACE_TASKS),
        }
        if provider_options is not None:
            kwargs["provider_options"] = provider_options
        app = FaceAnalysis(**kwargs)
        app.prepare(ctx_id=ctx_id, det_size=(FACE_DET_SIZE, FACE_DET_SIZE))
        logger.info(
            "[face] insightface '%s' ready backend=%s providers=%s det_size=%d tasks=%s",
            FACE_MODEL_PACK,
            backend_label,
            providers,
            FACE_DET_SIZE,
            list(app.models.keys()),
        )
        return app
    except Exception as exc:
        if backend_label == "qnn_gpu":
            logger.warning("[face] QNN GPU load failed (%s); retrying CPU", exc)
            try:
                app = FaceAnalysis(
                    name=FACE_MODEL_PACK,
                    providers=["CPUExecutionProvider"],
                    allowed_modules=list(_REQUIRED_FACE_TASKS),
                )
                app.prepare(ctx_id=-1, det_size=(FACE_DET_SIZE, FACE_DET_SIZE))
                logger.info("[face] insightface '%s' ready backend=cpu (fallback)", FACE_MODEL_PACK)
                return app
            except Exception as cpu_exc:
                logger.error("[face] CPU fallback failed: %s", cpu_exc, exc_info=True)
                return None
        logger.error("[face] Failed to load insightface model: %s", exc, exc_info=True)
        return None


def _get_app():
    global _app, _model_load_failed
    if _app is not None or _model_load_failed:
        return _app
    with _model_lock:
        if _app is None and not _model_load_failed:
            _app = _build_app()
            if _app is None:
                _model_load_failed = True
    return _app


# ── adaptive detector input size (P1-6) ──────────────────────────────────────
# SCRFD's own detect() already accepts a per-call `input_size` override (see
# insightface.model_zoo.scrfd.SCRFD.detect) -- insightface's FaceAnalysis.get()
# just never passes one, always falling back to the app's fixed prepare()-time
# det_size (FACE_DET_SIZE, 640 by default). Running that full-size canvas on an
# already-tight, already-small person-head crop wastes compute the same way
# running YOLO at 1280px on a postage-stamp image would: a detector's conv-layer
# cost is driven by canvas size, not by how much of that canvas the actual face
# occupies. Small crops get a smaller canvas instead. Recognition/embedding is
# a separate, fixed-112x112 ONNX model (ArcFaceONNX.get(), see
# tools/_insightface_src for the vendored reference source used to verify this
# during P1-6) and is completely unaffected either way.
_DET_SIZE_BUCKETS: tuple[int, ...] = (320, 480)


def select_det_size(crop_w: int, crop_h: int, ceiling: Optional[int] = None) -> int:
    """Pick the smallest SCRFD detector input size that comfortably fits a
    crop of this size, never exceeding `ceiling` (defaults to the configured
    FACE_DET_SIZE).

    The buckets below the ceiling are a pure efficiency optimization for crops
    that are already small (e.g. a distant/small person on a wide-angle
    camera); a crop at or beyond the ceiling's own scale still gets the full
    configured size, exactly like before this change. Pure function of crop
    dimensions — no model/config access needed when `ceiling` is passed
    explicitly, so it is trivially unit-testable.
    """
    if ceiling is None:
        from .config import FACE_DET_SIZE

        ceiling = FACE_DET_SIZE
    long_side = max(int(crop_w), int(crop_h), 1)
    for bucket in _DET_SIZE_BUCKETS:
        if bucket >= ceiling:
            break
        if long_side <= bucket:
            return bucket
    return ceiling


def _detect_faces_with_size(app, bgr_crop: np.ndarray, det_size: int) -> list:
    """Equivalent of FaceAnalysis.get(img), but with an explicit per-call
    detector input size instead of the app's fixed prepare()-time size.

    Deliberately does NOT mutate `app.det_model.input_size` (the obvious
    one-line alternative): that attribute lives on the single shared `_app`
    singleton used by every caller in this process, including concurrent
    /infer request threads (FastAPI's sync-endpoint thread pool) and
    face_worker's own worker-thread pool -- two threads racing to set
    different sizes on the same shared object before either calls detect()
    would silently use the wrong size for one of them. Passing `input_size`
    straight into SCRFD.detect()'s already-supported parameter keeps each
    call's chosen size a local value, so concurrent calls never interfere.
    """
    from insightface.app.common import Face

    bboxes, kpss = app.det_model.detect(
        bgr_crop, input_size=(det_size, det_size), max_num=0, metric="default"
    )
    if bboxes.shape[0] == 0:
        return []
    faces = []
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i, 0:4]
        det_score = bboxes[i, 4]
        kps = kpss[i] if kpss is not None else None
        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        for taskname, model in app.models.items():
            if taskname == "detection":
                continue
            model.get(bgr_crop, face)
        faces.append(face)
    return faces


def _detect_faces(app, bgr_crop: np.ndarray, det_size: int) -> list:
    """_detect_faces_with_size(), falling back to the app's own app.get() (the
    original, fixed-size behavior) if anything about the adaptive-size path
    fails -- e.g. a future insightface upgrade changes these internals. Never
    lets an internal-API mismatch turn into a lost detection.
    """
    try:
        return _detect_faces_with_size(app, bgr_crop, det_size)
    except Exception as exc:
        logger.debug(
            "[face] adaptive det_size=%d path failed (%s); falling back to app.get()",
            det_size,
            exc,
        )
        return app.get(bgr_crop)


def available() -> bool:
    """True if face recognition can run (model loaded or loadable)."""
    from .config import ENABLE_FACE_RECOGNITION
    if not ENABLE_FACE_RECOGNITION:
        return False
    return _get_app() is not None


def warmup() -> bool:
    """Force model load at startup and preload the face gallery."""
    if not available():
        return False
    try:
        rows = _load_gallery_rows_sync()
        if rows:
            count = _apply_gallery_rows(rows)
            logger.info("[face] Gallery preloaded: %d embedding(s)", count)
        else:
            logger.info("[face] Gallery empty at startup (enroll faces via admin UI)")
    except Exception as exc:
        logger.warning("[face] Gallery preload failed: %s", exc)
    return True


# ── embedding extraction ───────────────────────────────────────────────────────

def extract_embedding(bgr_crop: np.ndarray) -> Optional[np.ndarray]:
    """Detect the most prominent face in a BGR crop and return its 512-d
    L2-normalised embedding, or None if no usable face is found.
    """
    app = _get_app()
    if app is None or bgr_crop is None or bgr_crop.size == 0:
        return None

    from .config import FACE_MIN_PIXELS

    # Upscale tiny crops so SCRFD can find faces on distant cameras.
    h, w = bgr_crop.shape[:2]
    min_side = min(h, w)
    if min_side < 160:
        scale = 160.0 / max(min_side, 1)
        bgr_crop = cv2.resize(
            bgr_crop,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_LINEAR,
        )
        h, w = bgr_crop.shape[:2]

    det_size = select_det_size(w, h)
    try:
        faces = _detect_faces(app, bgr_crop, det_size)
    except Exception as exc:
        logger.debug("[face] detection error: %s", exc)
        return None

    if not faces:
        return None

    # Pick the largest detected face.
    def _area(f) -> float:
        x1, y1, x2, y2 = f.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    face = max(faces, key=_area)
    x1, y1, x2, y2 = face.bbox
    if min(x2 - x1, y2 - y1) < FACE_MIN_PIXELS:
        return None

    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = getattr(face, "embedding", None)
        if emb is None:
            return None
        emb = np.asarray(emb, dtype=np.float32)
        n = np.linalg.norm(emb)
        if n > 1e-6:
            emb = emb / n
    return np.asarray(emb, dtype=np.float32)


def detect_face_bbox(bgr_crop: np.ndarray) -> Optional[tuple[float, float, float, float]]:
    """Return the largest face xyxy in the crop's pixel coordinates."""
    app = _get_app()
    if app is None or bgr_crop is None or bgr_crop.size == 0:
        return None

    h, w = bgr_crop.shape[:2]
    scale = 1.0
    work = bgr_crop
    if min(h, w) < 160:
        scale = 160.0 / max(min(h, w), 1)
        work = cv2.resize(
            bgr_crop,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_LINEAR,
        )

    work_h, work_w = work.shape[:2]
    det_size = select_det_size(work_w, work_h)
    try:
        faces = _detect_faces(app, work, det_size)
    except Exception as exc:
        logger.debug("[face] bbox detection error: %s", exc)
        return None

    if not faces:
        return None

    def _area(f) -> float:
        x1, y1, x2, y2 = f.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    face = max(faces, key=_area)
    x1, y1, x2, y2 = (float(v) for v in face.bbox)
    if scale != 1.0:
        inv = 1.0 / scale
        x1, y1, x2, y2 = x1 * inv, y1 * inv, x2 * inv, y2 * inv
    return (x1, y1, x2, y2)


def embedding_to_bytes(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def bytes_to_embedding(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32).copy()


# ── gallery cache (enrolled faces) ──────────────────────────────────────────────

_gallery_lock = threading.Lock()
# list of (person_id_str, name, gender, np.ndarray)
_gallery: list[tuple] = []
_gallery_loaded_at: float = 0.0
_gallery_refresh_in_progress = False
_gallery_guard_lock = threading.Lock()


def _gallery_is_stale() -> bool:
    from .config import FACE_GALLERY_REFRESH_SEC
    return (time.monotonic() - _gallery_loaded_at) >= FACE_GALLERY_REFRESH_SEC


def _schedule_gallery_refresh() -> None:
    global _gallery_refresh_in_progress
    with _gallery_guard_lock:
        if _gallery_refresh_in_progress:
            return
        _gallery_refresh_in_progress = True
    t = threading.Thread(target=_bg_load_gallery, daemon=True, name="face-gallery-refresh")
    t.start()


def _parse_gallery_rows(rows: list[dict]) -> list[tuple]:
    parsed: list[tuple] = []
    from .person_age import effective_age

    for r in rows:
        try:
            vec = bytes_to_embedding(r["embedding"])
            if vec.shape[0] != FACE_EMBEDDING_DIM:
                continue
            age = effective_age(
                date_of_birth=r.get("date_of_birth"),
                stored_age=r.get("age"),
            )
            parsed.append((str(r["person_id"]), r["name"], r.get("gender"), age, vec))
        except Exception:
            continue
    return parsed


def _load_gallery_rows_sync() -> list[dict]:
    """Load enrolled face rows without touching async_bridge (edge SQLite path)."""
    from .db.session import is_edge_mode

    if is_edge_mode():
        from .db import edge_store

        return edge_store.list_face_gallery()

    from .async_bridge import run_async

    return run_async(_async_load_gallery())


def _apply_gallery_rows(rows: list[dict]) -> int:
    global _gallery, _gallery_loaded_at
    parsed = _parse_gallery_rows(rows)
    with _gallery_lock:
        _gallery = parsed
        _gallery_loaded_at = time.monotonic()
    if parsed:
        logger.info("[face] Gallery loaded: %d face embedding(s)", len(parsed))
    else:
        logger.warning("[face] Gallery loaded but no usable face embeddings (rows=%d)", len(rows))
    return len(parsed)


def _bg_load_gallery() -> None:
    global _gallery_refresh_in_progress
    try:
        rows = _load_gallery_rows_sync()
        _apply_gallery_rows(rows)
    except Exception as exc:
        logger.warning("[face] Gallery load failed: %s", exc)
    finally:
        with _gallery_guard_lock:
            _gallery_refresh_in_progress = False


async def _async_load_gallery() -> list[dict]:
    from .db import get_session
    from .db import dal
    async with get_session() as session:
        return await dal.list_face_gallery(session)


def invalidate_gallery() -> None:
    """Force a gallery reload on the next match call (after enrollment)."""
    global _gallery_loaded_at
    with _gallery_lock:
        _gallery_loaded_at = 0.0
    _schedule_gallery_refresh()


def gallery_size() -> int:
    with _gallery_lock:
        return len(_gallery)


def match_embedding(query: np.ndarray, threshold: Optional[float] = None) -> Optional[FaceMatch]:
    """Match a query embedding against the gallery. Returns the best FaceMatch
    above threshold, or None. Triggers a background gallery refresh if stale.
    """
    from .config import FACE_MATCH_THRESHOLD
    thr = FACE_MATCH_THRESHOLD if threshold is None else threshold

    if _gallery_is_stale():
        _schedule_gallery_refresh()

    with _gallery_lock:
        gallery = list(_gallery)

    if not gallery and query is not None:
        # Edge box: background refresh may not have finished yet — load once inline.
        try:
            from .db.session import is_edge_mode

            if is_edge_mode():
                rows = _load_gallery_rows_sync()
                if rows:
                    _apply_gallery_rows(rows)
                    with _gallery_lock:
                        gallery = list(_gallery)
        except Exception as exc:
            logger.debug("[face] Inline gallery load skipped: %s", exc)

    if not gallery or query is None:
        return None

    best: Optional[FaceMatch] = None
    best_score = -1.0
    for pid, name, gender, age, ref in gallery:
        score = float(np.dot(query, ref))
        if score > best_score:
            best_score = score
            best = FaceMatch(person_id=pid, name=name, gender=gender, age=age, score=score)

    if best is not None and best_score >= thr:
        return best
    if best is not None:
        logger.info(
            "[face] Gallery miss: best=%.3f thr=%.3f person=%s",
            best_score,
            thr,
            best.name,
        )
    return None


def recognize_crop(bgr_crop: np.ndarray, threshold: Optional[float] = None) -> Optional[FaceMatch]:
    """Convenience: extract embedding from a crop and match against the gallery."""
    if bgr_crop is None or bgr_crop.size == 0:
        return None
    emb = extract_embedding(bgr_crop)
    if emb is None:
        h, w = bgr_crop.shape[:2]
        logger.info("[face] No face detected in crop %dx%d", w, h)
        return None
    return match_embedding(emb, threshold)
