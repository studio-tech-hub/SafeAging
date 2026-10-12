import asyncio
import base64
import hmac
import os
import struct
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict

from .config import (
    API_KEY,
    API_KEY_REQUIRED,
    ALLOW_INSECURE_NO_AUTH,
    AUTH_EFFECTIVELY_DISABLED,
    CONFIDENCE_THRESHOLD,
    CORS_ALLOW_ORIGINS,
    DEVICE,
    ENABLE_FACE_RECOGNITION,
    ENABLE_FACE_ASYNC,
    ENABLE_FALL_DETECTION,
    FACE_RECOG_INTERVAL_FRAMES,
    FACE_RECOG_UNKNOWN_RETRY_SEC,
    FACE_RECOG_USE_TIME_SCHEDULER,
    ENABLE_UNDISTORT,
    IOU_THRESHOLD,
    METRICS_WINDOW_SIZE,
    MODEL_PATH,
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_MAX_PER_CAMERA,
    RATE_LIMIT_MAX_PER_IP,
    RATE_LIMIT_SKIP_LOOPBACK,
    RATE_LIMIT_WINDOW_SECONDS,
    REQUIRE_HTTPS,
    CALIB_CAPTURE_DIR,
    CALIB_CAPTURE_MAX,
    SERVICE_HOST,
    SERVICE_PORT,
    TLS_CERT_FILE,
    TLS_KEY_FILE,
    YOLO_IMGSZ,
    get_config,
    log_config_summary,
    logger,
)
from .db import dispose_engine, init_engine
from .health import compute_status, dependency_snapshot, refresh_probes
from .admin_router import router as admin_router
from . import outbox_worker
from .metrics import (
    INFER_ERRORS,
    INFER_REQUESTS,
    SERVICE_INFO,
)
from .model import load_model
from . import yolo_backend
from . import face_engine
from . import face_worker
from .face_identity import apply_cached_identity_to_det, commit_face_match, commit_face_miss
from .preprocess import auto_adjust_brightness, load_calibration
from .tracking import (
    camera_states,
    camera_states_lock,
    get_camera_metrics_snapshot,
    get_camera_state,
    increment_camera_error,
    increment_camera_request,
)
from .pipeline import InferContext, PIPELINE_STAGES


_calib_capture_lock = threading.Lock()
_calib_capture_count = 0


def _maybe_capture_calib_frame(camera_id: str, frame: np.ndarray, seq: int) -> None:
    if not CALIB_CAPTURE_DIR or CALIB_CAPTURE_MAX <= 0:
        return
    global _calib_capture_count
    with _calib_capture_lock:
        if _calib_capture_count >= CALIB_CAPTURE_MAX:
            return
        _calib_capture_count += 1
        idx = _calib_capture_count
    try:
        out_dir = CALIB_CAPTURE_DIR
        os.makedirs(out_dir, exist_ok=True)
        safe_cam = "".join(c if c.isalnum() or c in "-_{}" else "_" for c in (camera_id or "cam"))
        path = os.path.join(out_dir, f"{safe_cam}_{seq:06d}_{idx:04d}.jpg")
        cv2.imwrite(path, frame)
    except Exception as exc:
        logger.warning("[%s] calib capture failed: %s", camera_id, exc)


log_config_summary()

app = FastAPI(title="YOLO26 Analytics Service")
if CORS_ALLOW_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ALLOW_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["*"],
    )

# S P1.2 – Admin CRUD API (persons, zones, events, reset)
app.include_router(admin_router)

# ── Operator web UI (static single-page app) ──────────────────────────────────
import os as _os
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

_STATIC_DIR = _os.path.join(_os.path.dirname(__file__), "static")
if _os.path.isdir(_STATIC_DIR):
    app.mount("/ui", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    async def _root_redirect():
        return RedirectResponse(url="/ui/")

class InferRequest(BaseModel):
    image: str                 # base64 (jpg/png/bmp)
    camera_id: Optional[str] = "default"

class Detection(BaseModel):
    cls: str
    score: float
    x: float
    y: float
    w: float
    h: float
    track_id: int
    fall_detected: bool = False
    stable: bool = True
    degraded: bool = False
    # P2.2 — Zone engine fields
    zone_id: Optional[str] = None
    zone_type: Optional[str] = None
    zone_name: Optional[str] = None
    zone_violation: bool = False
    # Face recognition / identity fields — consumed by the NX plugin to render
    # "(Ông A, Nam, No.1)" on the bounding box. Null/Unknown when not recognised.
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    person_gender: Optional[str] = None
    person_age: Optional[int] = None
    person_no: Optional[int] = None
    recognized: bool = False

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    timestamp: str
    service_uptime_seconds: float
    ready: bool
    startup_completed: bool
    model_loaded: bool
    device: str
    warmup_ok: bool
    warmup_error: Optional[str] = None
    model_load_error: Optional[str] = None
    auth_enabled: bool
    require_https: bool
    tls_enabled: bool
    last_ready_check: Optional[str] = None
    # Extended in S P1.5 — consumed by Plugin P1.1 for degraded-mode diagnostics
    reason_codes: List[str] = []
    dependencies: Dict[str, str] = {}
    # P P1.4 – pipeline stats for operator correlation with plugin queue/drop warnings
    pipeline: Dict[str, Any] = {}


class ReadinessState:
    def __init__(self):
        self._lock = threading.Lock()
        self.startup_completed = False
        self.model_loaded = False
        self.model_load_error: Optional[str] = None
        self.warmup_ok = False
        self.warmup_error: Optional[str] = None
        self.device = str(DEVICE)
        self.auth_enabled = API_KEY_REQUIRED
        self.require_https = REQUIRE_HTTPS
        self.tls_enabled = bool(TLS_CERT_FILE and TLS_KEY_FILE)
        self.last_ready_check: Optional[str] = None

    def reset_for_startup(self) -> None:
        self.update(
            startup_completed=False,
            model_loaded=False,
            model_load_error=None,
            warmup_ok=False,
            warmup_error=None,
            device=str(DEVICE),
            auth_enabled=API_KEY_REQUIRED,
            require_https=REQUIRE_HTTPS,
            tls_enabled=bool(TLS_CERT_FILE and TLS_KEY_FILE),
            last_ready_check=None,
        )

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def snapshot(self, mark_checked: bool = True) -> Dict[str, Any]:
        now = datetime.now().isoformat()
        uptime = time.time() - service_start_time

        with self._lock:
            if mark_checked:
                self.last_ready_check = now

            snapshot = {
                "startup_completed": self.startup_completed,
                "model_loaded": self.model_loaded,
                "device": self.device,
                "warmup_ok": self.warmup_ok,
                "warmup_error": self.warmup_error,
                "model_load_error": self.model_load_error,
                "auth_enabled": self.auth_enabled,
                "require_https": self.require_https,
                "tls_enabled": self.tls_enabled,
                "last_ready_check": self.last_ready_check,
            }

        ready = (
            snapshot["startup_completed"]
            and snapshot["model_loaded"]
            and snapshot["warmup_ok"]
        )
        if ready:
            status = "healthy"
        elif snapshot["startup_completed"] and snapshot["model_loaded"] and snapshot["warmup_error"]:
            status = "degraded"
        else:
            status = "not_ready"

        snapshot.update(
            {
                "status": status,
                "timestamp": now,
                "service_uptime_seconds": uptime,
                "ready": ready,
            }
        )
        return snapshot

# ============================
# Global State
# ============================
service_start_time = time.time()
request_counter = 0
error_counter = 0
counter_lock = threading.Lock()
rate_limit_lock = threading.RLock()
ip_request_buckets: Dict[str, deque[float]] = defaultdict(deque)
camera_request_buckets: Dict[str, deque[float]] = defaultdict(deque)
readiness_state = ReadinessState()


def _format_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _resolve_model_device(yolo_model=None) -> str:
    if yolo_model is None:
        return str(DEVICE)

    predictor = getattr(yolo_model, "predictor", None)
    predictor_device = getattr(predictor, "device", None)
    if predictor_device is not None:
        return str(predictor_device)

    inner_model = getattr(yolo_model, "model", None)
    if inner_model is not None:
        try:
            return str(next(inner_model.parameters()).device)
        except (AttributeError, StopIteration, TypeError):
            pass

    direct_device = getattr(yolo_model, "device", None)
    if direct_device is not None:
        return str(direct_device)

    return str(DEVICE)


def _warmup_model(yolo_model) -> str:
    # Warm a small square frame to initialize PyTorch/CUDA context without
    # paying the full cost of the largest production imgsz at startup.
    warmup_imgsz = int(YOLO_IMGSZ)
    warmup_frame = np.zeros((warmup_imgsz, warmup_imgsz, 3), dtype=np.uint8)
    yolo_backend.predict_yolo(
        yolo_model,
        warmup_frame,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        classes=[0],
        imgsz=warmup_imgsz,
        verbose=False,
        augment=False,
    )
    return _resolve_model_device(yolo_model)


def _record_camera_infer_error(state: Dict[str, Any], error_message: str) -> None:
    increment_error_counter()
    increment_camera_error(state, error_message=error_message)


def _decode_current_transport_image(image_payload_b64: str, camera_id: str) -> tuple[np.ndarray, str, int, int]:
    """Decode the current JSON + base64 transport kept for plugin compatibility.

    This remains the default transport forever, for any plugin build that
    doesn't opt into transport_mode=binary. See _decode_binary_transport_image()
    for the additive multipart/binary alternative added in P1-4 to remove the
    base64 encode/decode + ~33% payload-size overhead on hardware that opts in.
    """
    encoded_payload_bytes = len(image_payload_b64.encode("utf-8"))
    img_bytes = base64.b64decode(image_payload_b64)
    raw_payload_bytes = len(img_bytes)

    if img_bytes.startswith(b"BGR"):
        # Current plugin transport: base64(JSON) around a custom raw BGR blob.
        # Keep unchanged for compatibility until a real binary or multipart path exists.
        w, h = struct.unpack("<II", img_bytes[3:11])
        bgr_data = img_bytes[11:]
        frame = np.frombuffer(bgr_data, dtype=np.uint8).reshape((h, w, 3))
        frame = auto_adjust_brightness(frame, target_brightness=180.0)
        logger.debug(
            f"[{camera_id}] Decoded transport=bgr_base64_raw payload_bytes={raw_payload_bytes} "
            f"encoded_bytes={encoded_payload_bytes} size={w}x{h}"
        )
        return frame, "bgr_base64_raw", encoded_payload_bytes, raw_payload_bytes

    img_array = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    logger.debug(
        f"[{camera_id}] Decoded transport=image_base64 payload_bytes={raw_payload_bytes} "
        f"encoded_bytes={encoded_payload_bytes}"
    )
    return frame, "image_base64", encoded_payload_bytes, raw_payload_bytes


def _decode_binary_transport_image(image_bytes: bytes, camera_id: str) -> tuple[np.ndarray, str, int, int]:
    """Decode the additive multipart/binary transport (P1-4): raw JPEG bytes, no base64.

    Same (frame, transport_name, encoded_bytes, raw_bytes) contract as
    _decode_current_transport_image() so both transports can feed the shared
    _run_person_detection_pipeline() unchanged. There is no base64 "encoded size"
    concept for this transport (no inflation happens), so encoded_bytes and
    raw_bytes both report the actual wire size for straightforward /metrics
    comparison against the JSON+base64 transport.
    """
    raw_payload_bytes = len(image_bytes)
    img_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    logger.debug(
        f"[{camera_id}] Decoded transport=image_binary_multipart payload_bytes={raw_payload_bytes}"
    )
    return frame, "image_binary_multipart", raw_payload_bytes, raw_payload_bytes


def _crop_person_head_region(
    frame: np.ndarray, det: "Detection", W: int, H: int
) -> tuple[Optional[np.ndarray], tuple[int, int]]:
    """Crop the person bbox (with margin) for SCRFD face detection.

    InsightFace finds the face inside the full person box; using only the
    upper 60% often misses on wide-angle / Nx downscaled frames.
    Returns (crop_bgr, (origin_x, origin_y)) in frame pixel space.
    """
    x1 = det.x
    y1 = det.y
    bw = det.w
    bh = det.h
    margin_x = bw * 0.08
    margin_y = bh * 0.05
    cx1 = int(max(0, x1 - margin_x))
    cx2 = int(min(W, x1 + bw + margin_x))
    cy1 = int(max(0, y1 - margin_y))
    cy2 = int(min(H, y1 + bh + margin_y))
    if cx2 - cx1 < 16 or cy2 - cy1 < 16:
        return None, (0, 0)
    return frame[cy1:cy2, cx1:cx2], (cx1, cy1)


def _refine_person_bbox_from_face(
    det: "Detection",
    crop: Optional[np.ndarray],
    crop_origin: tuple[int, int],
    frame_w: int,
    frame_h: int,
    frame: Optional[np.ndarray] = None,
) -> None:
    """Anchor the display bbox on SCRFD face location when YOLO is horizontally off."""
    face_bbox = face_engine.detect_face_bbox(crop) if crop is not None and crop.size > 0 else None
    ox, oy = crop_origin
    if face_bbox is None and frame is not None:
        # YOLO often shifts left on wide-angle cameras — search a wider region.
        cx = det.x + det.w * 0.5
        cy = det.y + det.h * 0.5
        ew = min(float(frame_w), det.w * 1.8)
        eh = min(float(frame_h), det.h * 1.4)
        ex1 = max(0, int(cx - ew * 0.35))
        ey1 = max(0, int(cy - eh * 0.30))
        ex2 = min(frame_w, int(cx + ew * 0.65))
        ey2 = min(frame_h, int(cy + eh * 0.70))
        if ex2 - ex1 >= 32 and ey2 - ey1 >= 32:
            big_crop = frame[ey1:ey2, ex1:ex2]
            face_bbox = face_engine.detect_face_bbox(big_crop)
            if face_bbox is not None:
                ox, oy = ex1, ey1
                crop = big_crop
    if face_bbox is None:
        return
    fx1, fy1, fx2, fy2 = face_bbox
    ff_x1 = ox + fx1
    ff_y1 = oy + fy1
    ff_x2 = ox + fx2
    ff_y2 = oy + fy2
    fw = ff_x2 - ff_x1
    fh = ff_y2 - ff_y1
    if fw < 8.0 or fh < 8.0:
        return

    pad_x = fw * 0.85
    pad_up = fh * 0.35
    pad_down = fh * 3.8
    nx1 = max(0.0, ff_x1 - pad_x)
    ny1 = max(0.0, ff_y1 - pad_up)
    nx2 = min(float(frame_w), ff_x2 + pad_x)
    ny2 = min(float(frame_h), ff_y2 + pad_down)

    yx2 = det.x + det.w
    yy2 = det.y + det.h
    nx1 = min(nx1, det.x)
    ny1 = min(ny1, det.y)
    nx2 = max(nx2, yx2)
    ny2 = max(ny2, yy2)

    det.x = nx1
    det.y = ny1
    det.w = max(1.0, nx2 - nx1)
    det.h = max(1.0, ny2 - ny1)


def _apply_face_identity(
    camera_id: str,
    state: Dict[str, Any],
    detections: List["Detection"],
    frame: np.ndarray,
    camera_frame_idx: int,
    camera_lock: Any,
) -> None:
    """Resolve and attach person identity (name/gender/No.) to stable detections.

    Recognition is throttled per track and cached so the rendered label stays
    stable across frames. Unknown persons keep retrying every interval; once a
    track is recognised the identity sticks for that track's lifetime.
    """
    if not face_engine.available():
        return

    with camera_lock:
        identity: Dict[int, Dict[str, Any]] = state.setdefault("identity", {})
        recent_crops: Dict[int, np.ndarray] = state.setdefault("recent_crops", {})

    H, W = frame.shape[:2]

    for det in detections:
        if det.cls != "person":
            continue
        # QNO often renders degraded fallback YOLO boxes (stable=False) when tracking
        # has not confirmed yet — still run face on those boxes for identity labels.

        tid = det.track_id
        cached = identity.get(tid)

        crop, crop_origin = _crop_person_head_region(frame, det, W, H)
        if crop is not None and crop.size > 0:
            _refine_person_bbox_from_face(det, crop, crop_origin, W, H, frame)

        attempt = False
        now_mono = time.monotonic()
        if cached is None:
            attempt = True
        elif not cached.get("recognized"):
            if cached.get("pending") and ENABLE_FACE_ASYNC:
                attempt = False
            elif FACE_RECOG_USE_TIME_SCHEDULER:
                last_ts = cached.get("last_attempt_ts", 0.0)
                if now_mono - last_ts >= FACE_RECOG_UNKNOWN_RETRY_SEC:
                    attempt = True
            else:
                last = cached.get("last_attempt", -10_000)
                if camera_frame_idx - last >= FACE_RECOG_INTERVAL_FRAMES:
                    attempt = True

        if attempt:
            if crop is not None and crop.size > 0:
                with camera_lock:
                    recent_crops[tid] = crop.copy()

            pool = face_worker.get_pool() if ENABLE_FACE_ASYNC else None
            queue_busy = (
                pool is not None and pool.stats().get("queue_depth", 0) > 0
            )
            if pool is not None and queue_busy:
                queued = face_worker.submit_recognition(
                    camera_id=camera_id,
                    track_id=tid,
                    crop=crop,
                    state=state,
                    camera_lock=camera_lock,
                    frame_idx=camera_frame_idx,
                )
                if queued:
                    face_worker.mark_attempt_pending(
                        state, camera_lock, tid, camera_frame_idx
                    )
            else:
                match = face_engine.recognize_crop(crop) if crop is not None else None
                if match is not None:
                    commit_face_match(
                        state, camera_lock, tid, match, camera_frame_idx
                    )
                    logger.info(
                        "[%s] Face matched track=%s name=%s score=%.3f",
                        camera_id,
                        tid,
                        match.name,
                        match.score,
                    )
                else:
                    commit_face_miss(state, camera_lock, tid, camera_frame_idx)

        with camera_lock:
            cached = identity.get(tid)
        apply_cached_identity_to_det(det, cached)


def _get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "").strip()
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client and request.client.host else "unknown"

def _is_loopback_ip(ip: str) -> bool:
    normalized = (ip or "").strip().lower()
    return normalized in {"127.0.0.1", "::1", "localhost"}

def _extract_api_token(request: Request) -> str:
    api_key_header = request.headers.get("x-api-key", "").strip()
    if api_key_header:
        return api_key_header

    auth_header = request.headers.get("authorization", "").strip()
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return ""

def _enforce_api_key(request: Request) -> None:
    if not API_KEY_REQUIRED:
        return

    provided = _extract_api_token(request)
    if not provided or not hmac.compare_digest(provided, API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized: missing or invalid API key")

def _enforce_rate_limit(request: Request, camera_id: Optional[str] = None) -> None:
    if not RATE_LIMIT_ENABLED:
        return

    now_ts = time.time()
    cutoff = now_ts - RATE_LIMIT_WINDOW_SECONDS
    client_ip = _get_client_ip(request)

    if RATE_LIMIT_SKIP_LOOPBACK and _is_loopback_ip(client_ip):
        return

    with rate_limit_lock:
        ip_bucket = ip_request_buckets[client_ip]
        while ip_bucket and ip_bucket[0] <= cutoff:
            ip_bucket.popleft()
        if len(ip_bucket) >= RATE_LIMIT_MAX_PER_IP:
            retry_after = max(1, int(RATE_LIMIT_WINDOW_SECONDS - (now_ts - ip_bucket[0])))
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded for client IP",
                headers={"Retry-After": str(retry_after)},
            )
        ip_bucket.append(now_ts)

        if camera_id:
            cam_key = f"{client_ip}:{camera_id}"
            cam_bucket = camera_request_buckets[cam_key]
            while cam_bucket and cam_bucket[0] <= cutoff:
                cam_bucket.popleft()
            if len(cam_bucket) >= RATE_LIMIT_MAX_PER_CAMERA:
                retry_after = max(1, int(RATE_LIMIT_WINDOW_SECONDS - (now_ts - cam_bucket[0])))
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded for camera '{camera_id}'",
                    headers={"Retry-After": str(retry_after)},
                )
            cam_bucket.append(now_ts)

def enforce_security(request: Request, camera_id: Optional[str] = None, require_auth: bool = True) -> None:
    if REQUIRE_HTTPS:
        proto = request.headers.get("x-forwarded-proto", request.url.scheme).lower()
        if proto != "https":
            raise HTTPException(status_code=403, detail="HTTPS required")

    if require_auth:
        _enforce_api_key(request)

    _enforce_rate_limit(request, camera_id)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

    # The operator UI needs a relaxed CSP (self-hosted inline app + image fetches).
    # API endpoints keep the strict locked-down policy.
    path = request.url.path
    if path == "/" or path.startswith("/ui"):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'"
        )
    else:
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    return response

def increment_request_counter() -> int:
    global request_counter
    with counter_lock:
        request_counter += 1
        return request_counter

def increment_error_counter() -> int:
    global error_counter
    with counter_lock:
        error_counter += 1
        return error_counter

def get_counter_snapshot() -> tuple:
    with counter_lock:
        return request_counter, error_counter

@app.get("/health", response_model=HealthResponse)
async def health_check(response: Response):
    """Readiness endpoint for operators and the NX plugin.

    Returns status "healthy" | "degraded" | "not_ready" plus reason_codes
    and per-dependency status so Plugin P1.1 can emit specific diagnostic events.
    Dependency probes are cached (30 s TTL) to keep this endpoint fast.
    """
    snapshot = readiness_state.snapshot()

    # Probe dependencies (no-op if cache is fresh)
    await refresh_probes()
    deps = dependency_snapshot()

    model_ok = (
        snapshot["startup_completed"]
        and snapshot["model_loaded"]
        and snapshot["warmup_ok"]
    )

    # P P1.4 – aggregate per-camera pipeline stats for operator correlation
    with camera_states_lock:
        camera_ids = list(camera_states.keys())
    total_requests = 0
    total_errors = 0
    per_camera: Dict[str, Any] = {}
    for cam_id in camera_ids:
        try:
            cam_state = get_camera_state(cam_id)
            snap = get_camera_metrics_snapshot(cam_state)
            total_requests += snap.get("request_count", 0)
            total_errors += snap.get("error_count", 0)
            per_camera[cam_id] = {
                "requests": snap.get("request_count", 0),
                "errors": snap.get("error_count", 0),
                "avg_infer_ms": round(snap.get("avg_inference_ms", 0.0), 1),
                "p95_infer_ms": round(snap.get("p95_inference_ms", 0.0), 1),
                "tracks": snap.get("last_track_count", 0),
            }
        except Exception:
            pass
    pipeline = {
        "active_cameras": len(per_camera),
        "total_requests": total_requests,
        "total_errors": total_errors,
        "error_rate": round(total_errors / total_requests, 4) if total_requests > 0 else 0.0,
        "yolo_backend": yolo_backend.active_backend(),
        "qnn": yolo_backend.qnn_runtime_status(),
        "cameras": per_camera,
    }
    snapshot["pipeline"] = pipeline

    status, reason_codes = compute_status(
        model_ok=model_ok, deps=deps, pipeline=pipeline, auth_disabled=AUTH_EFFECTIVELY_DISABLED
    )

    snapshot["status"] = status
    snapshot["reason_codes"] = reason_codes
    snapshot["dependencies"] = deps

    health_payload = HealthResponse(**snapshot)
    if not health_payload.ready:
        response.status_code = 503
    return health_payload


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics scrape endpoint.

    Exposes all safeaging_* counters/gauges/histograms plus standard Python
    process metrics (CPU, memory, GC). Scraped by Prometheus every 15 s.
    No authentication required — restrict at the network/firewall level.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )

# ============================
# Inference Endpoint
# ============================
def _run_person_detection_pipeline(
    frame: np.ndarray,
    *,
    camera_id: str,
    state: Dict[str, Any],
    cam_seq: int,
    request_started_at: float,
    transport_name: str,
    transport_encoded_bytes: int,
    transport_raw_bytes: int,
    decode_time_ms: float,
    route_label: str = "/infer",
) -> List[Detection]:
    """Shared detection/tracking/fall/zone pipeline for an already-decoded BGR frame.

    P1-4: extracted from the body of /infer so the additive /infer/binary
    (multipart JPEG, no base64) route can share the exact same pipeline after
    its own transport-specific decode step. Both routes are byte-for-byte
    identical from this point on — only the transport-specific decode
    (_decode_current_transport_image vs _decode_binary_transport_image) and
    request bookkeeping done by each route handler differ.

    camera_request_count and cam_seq are the same per-camera sequence number;
    this function uses cam_seq for both roles (matches the pre-P1-4 code,
    which aliased them under two names within a single function).

    P2-1: this function is now a thin orchestrator over an ordered list of
    Stage objects (PreprocessStage, DetectStage, TrackStage, FallStage,
    FaceStage, ZoneStage, PersistStage — see the `pipeline` package)
    operating on a shared InferContext, replacing what used to be ~900
    lines of inline code sharing ~50 loose local variables. DecodeStage is
    not one of them: the already-decoded `frame` + transport metadata this
    function receives IS that stage's output, produced by the
    route-handler-level _decode_current_transport_image /
    _decode_binary_transport_image (P1-4).

    Behaviour is unchanged from before this refactor — every stage was
    extracted verbatim from this function's former body, one at a time,
    gated by python/tests/unit/test_infer_golden.py (the characterization
    test that must stay green, unmodified, through the whole decomposition).

    P2-3: `config` is fetched once per request via `get_config()` (an
    explicit accessor over the same process-wide AppConfig singleton the
    globals above are already derived from — not new state) and passed
    explicitly into every stage's `run()`, instead of each stage importing
    config.py's module-level globals directly. This is what makes stages
    unit-testable in isolation with a constructed AppConfig instead of
    `monkeypatch.setattr(config, ...)` on shared module globals.
    """
    config = get_config()
    ctx = InferContext(
        frame=frame,
        camera_id=camera_id,
        state=state,
        cam_seq=cam_seq,
        request_started_at=request_started_at,
        transport_name=transport_name,
        transport_encoded_bytes=transport_encoded_bytes,
        transport_raw_bytes=transport_raw_bytes,
        decode_time_ms=decode_time_ms,
        route_label=route_label,
    )
    try:
        for stage in PIPELINE_STAGES:
            stage.run(ctx, config)
            if ctx.stopped:
                return ctx.result
        return ctx.detections
    except Exception as e:
        error_message = f"Unexpected error in {route_label}: {type(e).__name__}: {e}"
        _record_camera_infer_error(state, error_message)
        logger.error(f"[{camera_id}] {error_message}", exc_info=True)
        INFER_REQUESTS.labels(camera_id=camera_id, status="error").inc()
        INFER_ERRORS.labels(camera_id=camera_id, kind="unknown").inc()
        return []



@app.post("/infer", response_model=List[Detection])
def infer(req: InferRequest, request: Request):
    """
    Main inference endpoint.
    
    Args:
        req: InferRequest with base64 encoded image
        
    Returns:
        List[Detection]: Detections with track_id
        
    Expected by C++ plugin:
    {
        "cls": "person",
        "score": 0.9,
        "x": 180.0,
        "y": 270.6,
        "w": 120.0,
        "h": 360.8,
        "track_id": 1
    }
    """
    increment_request_counter()   # global counter for /status total, not used for per-camera logic
    request_started_at = time.perf_counter()

    from .config import normalize_camera_id

    camera_id = normalize_camera_id(req.camera_id or "default")
    enforce_security(request, camera_id=camera_id, require_auth=True)
    state = get_camera_state(camera_id)
    camera_request_count = increment_camera_request(state)
    # Use per-camera counter for all debug/log frequency checks (replaces global req_seq)
    cam_seq = camera_request_count

    # ============================================
    # 1) Decode current transport layer
    # Keep JSON + base64 for plugin compatibility; isolated here so the
    # additive /infer/binary transport (P1-4) below shares everything
    # downstream via _run_person_detection_pipeline().
    # ============================================
    decode_start = time.perf_counter()
    try:
        frame, transport_name, transport_encoded_bytes, transport_raw_bytes = _decode_current_transport_image(
            req.image,
            camera_id,
        )
        decode_time_ms = (time.perf_counter() - decode_start) * 1000.0

        if frame is None:
            error_message = "Failed to decode image - got None"
            _record_camera_infer_error(state, error_message)
            logger.warning(f"[{camera_id}] {error_message}")
            return []

    except Exception as e:
        decode_time_ms = (time.perf_counter() - decode_start) * 1000.0
        error_message = f"Image decode error: {type(e).__name__}: {e}"
        _record_camera_infer_error(state, error_message)
        logger.warning(f"[{camera_id}] {error_message}")
        return []

    return _run_person_detection_pipeline(
        frame,
        camera_id=camera_id,
        state=state,
        cam_seq=cam_seq,
        request_started_at=request_started_at,
        transport_name=transport_name,
        transport_encoded_bytes=transport_encoded_bytes,
        transport_raw_bytes=transport_raw_bytes,
        decode_time_ms=decode_time_ms,
        route_label="/infer",
    )


@app.post("/infer/binary", response_model=List[Detection])
def infer_binary(
    request: Request,
    camera_id: str = Form("default"),
    image: UploadFile = File(...),
):
    """P1-4: additive binary/multipart transport for /infer.

    Accepts multipart/form-data with a `camera_id` form field and the raw
    JPEG bytes as a file part (`image`) — no base64 wrapping. Base64 inflates
    payload size ~33% and costs both sides an extra encode/decode pass per
    frame; this route removes that tax for any plugin build that opts in via
    the transport_mode=binary setting.

    Fully additive: /infer (JSON+base64) is completely unchanged and remains
    the default transport for every existing plugin build. This route shares
    the exact same detection/tracking/fall/zone pipeline via
    _run_person_detection_pipeline() — the only difference is how the JPEG
    bytes arrive on the wire.

    Deliberately a sync `def`, not `async def`: FastAPI dispatches sync route
    handlers to its worker thread pool automatically, matching /infer's
    behaviour. The pipeline below is CPU-bound (YOLO inference, OpenCV) and
    would otherwise block the asyncio event loop for the whole request if
    this were `async def`, starving /health, /metrics and every other
    in-flight request on the same event loop. UploadFile.file is the
    underlying SpooledTemporaryFile, so plain sync .read() is correct here
    (no event loop involved) — see FastAPI's own guidance on sync endpoints
    for CPU-bound file uploads.
    """
    increment_request_counter()
    request_started_at = time.perf_counter()

    from .config import normalize_camera_id

    resolved_camera_id = normalize_camera_id(camera_id or "default")
    enforce_security(request, camera_id=resolved_camera_id, require_auth=True)
    state = get_camera_state(resolved_camera_id)
    camera_request_count = increment_camera_request(state)

    decode_start = time.perf_counter()
    try:
        image_bytes = image.file.read()
        frame, transport_name, transport_encoded_bytes, transport_raw_bytes = _decode_binary_transport_image(
            image_bytes,
            resolved_camera_id,
        )
        decode_time_ms = (time.perf_counter() - decode_start) * 1000.0

        if frame is None:
            error_message = "Failed to decode image - got None"
            _record_camera_infer_error(state, error_message)
            logger.warning(f"[{resolved_camera_id}] {error_message}")
            return []

    except Exception as e:
        decode_time_ms = (time.perf_counter() - decode_start) * 1000.0
        error_message = f"Image decode error: {type(e).__name__}: {e}"
        _record_camera_infer_error(state, error_message)
        logger.warning(f"[{resolved_camera_id}] {error_message}")
        return []

    return _run_person_detection_pipeline(
        frame,
        camera_id=resolved_camera_id,
        state=state,
        cam_seq=camera_request_count,
        request_started_at=request_started_at,
        transport_name=transport_name,
        transport_encoded_bytes=transport_encoded_bytes,
        transport_raw_bytes=transport_raw_bytes,
        decode_time_ms=decode_time_ms,
        route_label="/infer/binary",
    )


# ============================
# Status Endpoint
# ============================
@app.get("/status")
def status(request: Request):
    """Get service status and statistics"""
    enforce_security(request, require_auth=True)
    uptime = time.time() - service_start_time
    cameras_info = {}

    with camera_states_lock:
        camera_items = list(camera_states.items())

    for cam_id, state in camera_items:
        with state["lock"]:
            metrics_snapshot = get_camera_metrics_snapshot(state)
            last_update_ts = metrics_snapshot["last_update_ts"]
            cam_info = {
                "frame_idx": int(state.get("frame_idx", 0)),
                "tracks": len(state["tracks"]),
                "unique_persons": len(state["seen_ids"]),
                "created_at": datetime.fromtimestamp(state["created_at"]).isoformat(),
                "request_count": metrics_snapshot["request_count"],
                "error_count": metrics_snapshot["error_count"],
                "last_infer_ms": metrics_snapshot["last_infer_ms"],
                "avg_inference_ms": metrics_snapshot["avg_inference_ms"],
                "p50_inference_ms": metrics_snapshot["p50_inference_ms"],
                "p95_inference_ms": metrics_snapshot["p95_inference_ms"],
                "avg_decode_ms": metrics_snapshot["avg_decode_ms"],
                "avg_preprocess_ms": metrics_snapshot["avg_preprocess_ms"],
                "avg_yolo_ms": metrics_snapshot["avg_yolo_ms"],
                "last_track_count": metrics_snapshot["last_track_count"],
                "avg_track_count": metrics_snapshot["avg_track_count"],
                "last_error": metrics_snapshot["last_error"],
                "last_update_ts": datetime.fromtimestamp(last_update_ts).isoformat() if last_update_ts > 0 else None,
                "metrics_window_size": metrics_snapshot["metrics_window_size"],
            }

            # Add fall detection stats if enabled
            if ENABLE_FALL_DETECTION and state["fall_detector"]:
                fall_stats = state["fall_detector"].get_stats()
                cam_info["fall_detection"] = {
                    "enabled": True,
                    "total_tracked": fall_stats["total_tracked"],
                    "total_fallen": fall_stats["total_fallen"],
                    "current_frame": fall_stats["current_frame"],
                }
            else:
                cam_info["fall_detection"] = {"enabled": False}

        cameras_info[cam_id] = cam_info

    total_requests, total_errors = get_counter_snapshot()

    return {
        "service": "YOLO26 People Analytics + Fall Detection",
        "status": "running",
        "uptime_seconds": uptime,
        "total_requests": total_requests,
        "total_errors": total_errors,
        "error_rate": (total_errors / total_requests * 100) if total_requests > 0 else 0.0,
        "active_cameras": len(camera_items),
        "metrics_window_size": METRICS_WINDOW_SIZE,
        "fall_detection": "ENABLED" if ENABLE_FALL_DETECTION else "DISABLED",
        "cameras": cameras_info,
        "model": MODEL_PATH,
        "timestamp": datetime.now().isoformat()
    }


# ============================
# Per-camera Config Endpoint (P2.2)
# ============================
@app.get("/config/{camera_id}")
async def get_camera_config_endpoint(camera_id: str, request: Request):
    """Return per-camera inference config from the database.

    The plugin can poll this endpoint on startup and periodically to
    receive camera-specific overrides (confidence threshold, frame period, etc.).
    Fields that are null mean "use service default".
    """
    from .db import get_session
    from .db.dal import get_camera_config, list_zones
    from .config import CONFIDENCE_THRESHOLD as _DEFAULT_CONF, IOU_THRESHOLD as _DEFAULT_IOU
    from .config import normalize_camera_id

    # Mirror /infer's own normalization so a config saved (with either braced
    # or unbraced camera_id) via the admin API is the same row the plugin
    # reads back here, regardless of which form it queries with.
    camera_id = normalize_camera_id(camera_id)
    cfg = None
    zones_summary: list = []
    if request.app.state.__dict__.get("db_ok", True):
        try:
            async with get_session() as session:
                cfg = await get_camera_config(session, camera_id)
                zones = await list_zones(session, camera_id=camera_id, active_only=True)
                zones_summary = [{"id": str(z.id), "name": z.name, "zone_type": z.zone_type} for z in zones]
        except Exception as _e:
            logger.debug("[config] DB unavailable for camera config: %s", _e)

    return {
        "camera_id": camera_id,
        "confidence_threshold": cfg.confidence_threshold if cfg else None,
        "iou_threshold": cfg.iou_threshold if cfg else None,
        "frame_period": cfg.frame_period if cfg else None,
        "zone_ids": cfg.zone_ids if cfg else [],
        "extra": cfg.extra if cfg else {},
        "active_zones": zones_summary,
        "service_defaults": {
            "confidence_threshold": _DEFAULT_CONF,
            "iou_threshold": _DEFAULT_IOU,
        },
        "updated_at": cfg.updated_at.isoformat() if cfg else None,
    }


# ============================
# Reset Count Endpoint
# ============================
@app.post("/reset/{camera_id}")
def reset_camera(camera_id: str, request: Request):
    """
    Reset count for a specific camera.
    Call: POST http://127.0.0.1:18000/reset/default
    """
    enforce_security(request, camera_id=camera_id, require_auth=True)
    with camera_states_lock:
        state = camera_states.get(camera_id)

    if state is None:
        return {
            "camera_id": camera_id,
            "status": "not_found",
            "message": f"Camera {camera_id} not yet initialized"
        }

    with state["lock"]:
        old_count = len(state["seen_ids"])
        state["seen_ids"].clear()
        state["seen_ids_with_time"].clear()
        state["tracks"].clear()
        state["track_history"].clear()
        state["next_id"] = 1
        state["frame_idx"] = 0
        if state.get("identity"):
            state["identity"].clear()
        if state.get("person_no_map"):
            state["person_no_map"].clear()
        state["next_person_no"] = 1
        if state["fall_detector"]:
            state["fall_detector"].reset_fall()

    logger.info(f"[{camera_id}] Reset: cleared {old_count} persons, count now = 0")
    return {
        "camera_id": camera_id,
        "status": "reset",
        "previous_count": old_count,
        "current_count": 0
    }

@app.post("/reset_all")
def reset_all(request: Request):
    """Reset count for ALL cameras"""
    enforce_security(request, require_auth=True)
    with camera_states_lock:
        camera_items = list(camera_states.items())

    total_persons = 0
    for cam_id, state in camera_items:
        with state["lock"]:
            total_persons += len(state["seen_ids"])
            state["seen_ids"].clear()
            state["seen_ids_with_time"].clear()
            state["tracks"].clear()
            state["track_history"].clear()
            state["next_id"] = 1
            state["frame_idx"] = 0
            if state.get("identity"):
                state["identity"].clear()
            if state.get("person_no_map"):
                state["person_no_map"].clear()
            state["next_person_no"] = 1
            if state["fall_detector"]:
                state["fall_detector"].reset_fall()

    logger.info(f"Reset all {len(camera_items)} cameras, cleared {total_persons} persons")
    return {
        "status": "reset_all",
        "cameras_reset": len(camera_items),
        "total_persons_cleared": total_persons
    }

@app.post("/reset_fall/{camera_id}")
def reset_fall_detection(camera_id: str, request: Request):
    """Reset fall detection state for a specific camera"""
    enforce_security(request, camera_id=camera_id, require_auth=True)
    with camera_states_lock:
        state = camera_states.get(camera_id)

    if state and state["fall_detector"]:
        with state["lock"]:
            state["fall_detector"].reset_fall()
        logger.info(f"[{camera_id}] Fall detection state reset")
        return {
            "camera_id": camera_id,
            "status": "fall_detection_reset",
            "fall_detection_enabled": ENABLE_FALL_DETECTION
        }
    else:
        return {
            "camera_id": camera_id,
            "status": "not_found",
            "message": f"Camera {camera_id} not found or fall detection disabled"
        }

@app.post("/reset_fall_all")
def reset_fall_all(request: Request):
    """Reset fall detection state for ALL cameras"""
    enforce_security(request, require_auth=True)
    reset_count = 0
    with camera_states_lock:
        camera_items = list(camera_states.items())

    for cam_id, state in camera_items:
        with state["lock"]:
            if state["fall_detector"]:
                state["fall_detector"].reset_fall()
                reset_count += 1
    logger.info(f"Fall detection state reset for {reset_count} camera(s)")
    return {
        "status": "fall_detection_reset_all",
        "cameras_reset": reset_count
    }


# ============================
# Startup & Shutdown Events
# ============================
@app.on_event("startup")
async def startup_event():
    """Called when service starts"""
    readiness_state.reset_for_startup()

    # Load camera calibration if undistort is enabled
    if ENABLE_UNDISTORT:
        load_calibration()
    
    logger.info(f"âœ… FastAPI app started on {SERVICE_HOST}:{SERVICE_PORT}")
    logger.info(f"Health check: http://{SERVICE_HOST}:{SERVICE_PORT}/health")
    logger.info(f"Inference: http://{SERVICE_HOST}:{SERVICE_PORT}/infer")
    logger.info(f"Status: http://{SERVICE_HOST}:{SERVICE_PORT}/status")
    logger.info(f"Auth API key: {'ENABLED' if API_KEY_REQUIRED else 'DISABLED'}")
    if API_KEY_REQUIRED:
        logger.info("Protected endpoints require header: X-API-Key: <API_KEY>")
    if AUTH_EFFECTIVELY_DISABLED:
        for _ in range(3):
            logger.critical(
                "!!! AUTHENTICATION DISABLED — /infer, /admin/*, /config/* are "
                "reachable WITHOUT an API key on this network. "
                f"ALLOW_INSECURE_NO_AUTH={ALLOW_INSECURE_NO_AUTH}. "
                "Set API_KEY_REQUIRED=true and API_KEY to secure this deployment. !!!"
            )
    logger.info(f"Rate limit: {'ENABLED' if RATE_LIMIT_ENABLED else 'DISABLED'}")
    if RATE_LIMIT_ENABLED:
        logger.info(
            f"Rate limit window={RATE_LIMIT_WINDOW_SECONDS}s ip={RATE_LIMIT_MAX_PER_IP}/window "
            f"camera={RATE_LIMIT_MAX_PER_CAMERA}/window"
        )
        logger.info(f"Rate limit skip loopback: {'YES' if RATE_LIMIT_SKIP_LOOPBACK else 'NO'}")
    logger.info(f"HTTPS required: {'YES' if REQUIRE_HTTPS else 'NO'}")
    logger.info(f"Direct TLS (uvicorn): {'ENABLED' if TLS_CERT_FILE and TLS_KEY_FILE else 'DISABLED'}")
    if CORS_ALLOW_ORIGINS:
        logger.info(f"CORS allow origins: {', '.join(CORS_ALLOW_ORIGINS)}")
    logger.info(f"Reset count for camera: POST http://{SERVICE_HOST}:{SERVICE_PORT}/reset/default")
    logger.info(f"Reset all cameras: POST http://{SERVICE_HOST}:{SERVICE_PORT}/reset_all")
    logger.info(f"Admin API docs:    http://{SERVICE_HOST}:{SERVICE_PORT}/docs  (persons/zones/events/reset)")
    if ENABLE_FALL_DETECTION:
        logger.info(f"Reset fall detection for camera: POST http://{SERVICE_HOST}:{SERVICE_PORT}/reset_fall/default")
        logger.info(f"Reset fall detection for all cameras: POST http://{SERVICE_HOST}:{SERVICE_PORT}/reset_fall_all")

    # ── Database engine ───────────────────────────────────────────────────────
    # Non-blocking: if the DB URL is missing or unreachable the service starts
    # anyway and reports "degraded" in /health until the connection recovers.
    try:
        from .config import DATABASE_URL as _DATABASE_URL

        # Pass the already-resolved value explicitly (P0-4: supports
        # DATABASE_URL_FILE) instead of letting init_engine() re-read the raw
        # env var itself, which would bypass the _FILE secrets convention.
        # P2-3: also pass the injected AppConfig so init_engine() reads
        # EDGE_SQLITE_PATH from it instead of a second, independent
        # os.getenv() call.
        _startup_config = get_config()
        init_engine(_DATABASE_URL, config=_startup_config)
        from .db.session import is_edge_mode

        if is_edge_mode():
            logger.info(f"Edge SQLite store ready ({_startup_config.edge_sqlite_path})")
        else:
            logger.info("Database engine initialised (pool ready)")
    except Exception as _db_exc:
        logger.warning(
            f"Database engine init skipped — service will run in degraded mode: {_db_exc}"
        )

    from .async_bridge import init_async_bridge
    init_async_bridge(asyncio.get_running_loop())

    # ── S P1.3 Outbox worker ──────────────────────────────────────────────────
    loop = asyncio.get_event_loop()
    outbox_worker.init_outbox(loop)
    await outbox_worker.start_worker()
    logger.info("Outbox worker started")

    # ── S P2.3 Alert engine config ────────────────────────────────────────────
    from . import alert_engine
    from .config import (
        ALERT_DEDUPE_SEC, ALERT_MAX_RETRIES,
        ALERT_RATE_LIMIT_MAX, ALERT_RATE_LIMIT_WINDOW_SEC,
    )
    alert_engine.configure(
        dedupe_sec=ALERT_DEDUPE_SEC,
        rate_limit_max=ALERT_RATE_LIMIT_MAX,
        rate_limit_window_sec=ALERT_RATE_LIMIT_WINDOW_SEC,
        max_retries=ALERT_MAX_RETRIES,
    )

    # ── S P2.4 Retention worker ────────────────────────────────────────────────
    from . import retention as _retention
    from .config import RETENTION_DAYS, RETENTION_CHECK_HOURS
    _retention.start_retention_worker(
        retention_days=RETENTION_DAYS,
        check_hours=RETENTION_CHECK_HOURS,
    )

    logger.info("Model load start")
    yolo_model = None
    try:
        yolo_model = load_model()
        runtime_device = _resolve_model_device(yolo_model)
        readiness_state.update(
            model_loaded=True,
            model_load_error=None,
            device=runtime_device,
        )
        logger.info(f"Model load success (device={runtime_device})")
    except Exception as exc:
        load_error = _format_exception(exc)
        readiness_state.update(
            model_loaded=False,
            model_load_error=load_error,
            warmup_ok=False,
            warmup_error="warmup skipped because model load failed",
            device=str(DEVICE),
        )
        logger.error(f"Model load fail: {load_error}", exc_info=True)
    else:
        logger.info("Warmup start")
        try:
            runtime_device = _warmup_model(yolo_model)
            readiness_state.update(
                warmup_ok=True,
                warmup_error=None,
                device=runtime_device,
            )
            logger.info(f"Warmup success (device={runtime_device})")
        except Exception as exc:
            warmup_error = _format_exception(exc)
            runtime_device = _resolve_model_device(yolo_model)
            readiness_state.update(
                warmup_ok=False,
                warmup_error=warmup_error,
                device=runtime_device,
            )
            logger.error(f"Warmup fail on {runtime_device}: {warmup_error}", exc_info=True)
    finally:
        readiness_state.update(startup_completed=True)
        startup_health = readiness_state.snapshot(mark_checked=False)
        logger.info(
            "Startup readiness: "
            f"status={startup_health['status']} "
            f"ready={startup_health['ready']} "
            f"model_loaded={startup_health['model_loaded']} "
            f"warmup_ok={startup_health['warmup_ok']} "
            f"device={startup_health['device']} "
            f"auth_enabled={startup_health['auth_enabled']}"
        )
        SERVICE_INFO.info({
            "version": "1.0.0",
            "model_path": str(MODEL_PATH),
            "device": str(startup_health["device"]),
        })
        logger.info("Metrics endpoint: GET /metrics (Prometheus scrape)")

    # Face models (buffalo_l ~280MB) download in background so /health and /ui
    # respond immediately after YOLO warmup — do not block application startup.
    if ENABLE_FACE_RECOGNITION:
        asyncio.create_task(_background_face_warmup(), name="face-warmup")


async def _background_face_warmup() -> None:
    try:
        from .config import ENABLE_FACE_ASYNC, FACE_ASYNC_QUEUE_SIZE, FACE_ASYNC_WORKERS

        if ENABLE_FACE_ASYNC:
            face_worker.ensure_started(FACE_ASYNC_WORKERS, FACE_ASYNC_QUEUE_SIZE)
        logger.info("Face recognition warmup started (background)")
        ready = await asyncio.to_thread(face_engine.warmup)
        if ready:
            face_engine.invalidate_gallery()
            logger.info("Face recognition ready (insightface loaded)")
        else:
            logger.warning(
                "Face recognition unavailable — insightface not installed or model load failed; "
                "detections will not be labelled with identities"
            )
    except Exception as _fexc:
        logger.warning("Face recognition warmup skipped: %s", _fexc)


@app.on_event("shutdown")
async def shutdown_event():
    """Called when service shuts down"""
    logger.info("Service shutting down")
    total_requests, total_errors = get_counter_snapshot()
    logger.info(f"Total requests: {total_requests}")
    logger.info(f"Total errors: {total_errors}")
    await outbox_worker.stop_worker(drain_timeout=5.0)
    logger.info("Outbox worker stopped")
    await dispose_engine()
    logger.info("Database engine disposed")


