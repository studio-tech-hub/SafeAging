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

import numpy as np

logger = logging.getLogger(__name__)

# Face embedding dimension produced by ArcFace r50 (buffalo_l).
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
    score: float


def _build_app():
    """Create and prepare an insightface FaceAnalysis app. Returns None on failure."""
    from .config import DEVICE, FACE_DET_SIZE, FACE_MODEL_PACK

    try:
        from insightface.app import FaceAnalysis
    except Exception as exc:  # pragma: no cover - import guard
        logger.warning("[face] insightface not available: %s", exc)
        return None

    use_cuda = str(DEVICE).startswith("cuda")
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_cuda
        else ["CPUExecutionProvider"]
    )
    ctx_id = 0 if use_cuda else -1

    try:
        app = FaceAnalysis(name=FACE_MODEL_PACK, providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=(FACE_DET_SIZE, FACE_DET_SIZE))
        logger.info(
            "[face] insightface '%s' ready (providers=%s, det_size=%d)",
            FACE_MODEL_PACK, providers, FACE_DET_SIZE,
        )
        return app
    except Exception as exc:
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


def available() -> bool:
    """True if face recognition can run (model loaded or loadable)."""
    from .config import ENABLE_FACE_RECOGNITION
    if not ENABLE_FACE_RECOGNITION:
        return False
    return _get_app() is not None


def warmup() -> bool:
    """Force model load at startup. Returns True if ready."""
    return available()


# ── embedding extraction ───────────────────────────────────────────────────────

def extract_embedding(bgr_crop: np.ndarray) -> Optional[np.ndarray]:
    """Detect the most prominent face in a BGR crop and return its 512-d
    L2-normalised embedding, or None if no usable face is found.
    """
    app = _get_app()
    if app is None or bgr_crop is None or bgr_crop.size == 0:
        return None

    from .config import FACE_MIN_PIXELS

    try:
        faces = app.get(bgr_crop)
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


def _bg_load_gallery() -> None:
    global _gallery, _gallery_loaded_at, _gallery_refresh_in_progress
    try:
        from .async_bridge import run_async

        rows = run_async(_async_load_gallery())
        parsed: list[tuple] = []
        for r in rows:
            try:
                vec = bytes_to_embedding(r["embedding"])
                if vec.shape[0] != FACE_EMBEDDING_DIM:
                    continue
                parsed.append((str(r["person_id"]), r["name"], r.get("gender"), vec))
            except Exception:
                continue
        with _gallery_lock:
            _gallery = parsed
            _gallery_loaded_at = time.monotonic()
        logger.debug("[face] Gallery loaded: %d face embedding(s)", len(parsed))
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

    if not gallery or query is None:
        return None

    best: Optional[FaceMatch] = None
    best_score = -1.0
    for pid, name, gender, ref in gallery:
        score = float(np.dot(query, ref))
        if score > best_score:
            best_score = score
            best = FaceMatch(person_id=pid, name=name, gender=gender, score=score)

    if best is not None and best_score >= thr:
        return best
    return None


def recognize_crop(bgr_crop: np.ndarray, threshold: Optional[float] = None) -> Optional[FaceMatch]:
    """Convenience: extract embedding from a crop and match against the gallery."""
    emb = extract_embedding(bgr_crop)
    if emb is None:
        return None
    return match_embedding(emb, threshold)
