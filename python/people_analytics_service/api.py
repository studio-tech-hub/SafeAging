import asyncio
import base64
import hmac
import struct
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict

from .config import (
    API_KEY,
    API_KEY_REQUIRED,
    BBOX_SMOOTHING,
    CONFIDENCE_THRESHOLD,
    CORS_ALLOW_ORIGINS,
    DEVICE,
    ENABLE_CLAHE,
    ENABLE_FACE_RECOGNITION,
    ENABLE_FALL_DETECTION,
    ENABLE_FRAME_ENHANCEMENT,
    ENABLE_HISTORY_REMATCH,
    ENABLE_MULTI_SCALE,
    FACE_RECOG_INTERVAL_FRAMES,
    ENABLE_POST_NMS,
    ENABLE_ROI,
    ENABLE_UNDISTORT,
    FLICKER_REUSE_TIME,
    HOLD_SUPPRESS_IOU,
    IOU_THRESHOLD,
    MIN_DETECTION_AREA,
    METRICS_LOG_INTERVAL,
    METRICS_WINDOW_SIZE,
    MODEL_PATH,
    NEW_TRACK_MIN_CONFIDENCE,
    OUTPUT_DEDUPE_IOU,
    OUTPUT_TENTATIVE_TRACKS,
    PERSON_MIN_HW_RATIO,
    POST_NMS_IOU,
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_MAX_PER_CAMERA,
    RATE_LIMIT_MAX_PER_IP,
    RATE_LIMIT_SKIP_LOOPBACK,
    RATE_LIMIT_WINDOW_SECONDS,
    REQUIRE_HTTPS,
    SAVE_DEBUG_SAMPLES,
    SERVICE_HOST,
    SERVICE_PORT,
    TLS_CERT_FILE,
    TLS_KEY_FILE,
    TRACK_DUPLICATE_IOU,
    TRACK_MAX_MISSES,
    TRACK_MIN_HITS,
    TRACK_OUTPUT_HOLD_TIME,
    TRACK_TTL,
    TENTATIVE_MAX_MISSES,
    UNIQUE_COUNT_MIN_HITS,
    USE_HALF,
    YOLO_IMGSZ,
    log_config_summary,
    logger,
)
from .db import dispose_engine, init_engine
from .health import compute_status, dependency_snapshot, refresh_probes
from .admin_router import router as admin_router
from . import outbox_worker
from .outbox_worker import OutboxEvent
from .metrics import (
    ACTIVE_TRACKS,
    DECODE_LATENCY,
    DETECTIONS_PER_FRAME,
    INFER_ERRORS,
    INFER_LATENCY,
    INFER_REQUESTS,
    PREPROCESS_LATENCY,
    SERVICE_INFO,
    YOLO_LATENCY,
)
from .model import load_model
from . import face_engine
from .zone_engine import check_zones, get_zones_for_camera_sync
from .config_engine import get_per_camera_config_sync
from .preprocess import (
    apply_roi,
    auto_adjust_brightness,
    load_calibration,
    multi_scale_inference_smart,
    preprocess_frame,
    remap_detections_to_input_space,
    undistort_frame,
)
from .tracking import (
    appearance_distance,
    camera_states,
    camera_states_lock,
    dedupe_output_tracks,
    get_camera_metrics_snapshot,
    get_camera_state,
    global_track_assignment,
    has_duplicate_track_overlap,
    increment_camera_error,
    increment_camera_request,
    iou,
    post_nms_dedupe,
    record_camera_metrics,
    smooth_bbox,
    suppress_overlapping_holds,
    update_track_motion,
)


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
    zone_violation: bool = False
    # Face recognition / identity fields — consumed by the NX plugin to render
    # "(Ông A, Nam, No.1)" on the bounding box. Null/Unknown when not recognised.
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    person_gender: Optional[str] = None
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
    warmup_imgsz = max(64, min(int(YOLO_IMGSZ), 640))
    warmup_frame = np.zeros((warmup_imgsz, warmup_imgsz, 3), dtype=np.uint8)
    yolo_model.predict(
        warmup_frame,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        classes=[0],
        imgsz=warmup_imgsz,
        verbose=False,
        augment=False,
        device=DEVICE,
        half=USE_HALF,
    )[0]
    return _resolve_model_device(yolo_model)


def _record_camera_infer_error(state: Dict[str, Any], error_message: str) -> None:
    increment_error_counter()
    increment_camera_error(state, error_message=error_message)


def _decode_current_transport_image(image_payload_b64: str, camera_id: str) -> tuple[np.ndarray, str, int, int]:
    """Decode the current JSON + base64 transport kept for plugin compatibility.

    TODO: Keep this path for the current C++ plugin, but migrate toward a
    multipart/form-data or raw binary upload path to remove base64 overhead.
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


def _crop_person_head_region(frame: np.ndarray, det: "Detection", W: int, H: int) -> Optional[np.ndarray]:
    """Crop the upper region of a person bbox (head + shoulders) for face detection.

    Faces are concentrated in the top portion of a standing person; cropping
    there gives the detector a larger, cleaner face and is cheaper than the
    full body. Coords are in the same input-frame space as the detection.
    """
    x1 = det.x
    y1 = det.y
    bw = det.w
    bh = det.h
    margin_x = bw * 0.12
    cx1 = int(max(0, x1 - margin_x))
    cx2 = int(min(W, x1 + bw + margin_x))
    cy1 = int(max(0, y1))
    cy2 = int(min(H, y1 + bh * 0.6))
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return None
    return frame[cy1:cy2, cx1:cx2]


def _apply_face_identity(
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
        person_no_map: Dict[str, int] = state.setdefault("person_no_map", {})
        recent_crops: Dict[int, np.ndarray] = state.setdefault("recent_crops", {})

    H, W = frame.shape[:2]

    for det in detections:
        if not det.stable or det.degraded or det.cls != "person":
            continue

        tid = det.track_id
        cached = identity.get(tid)

        attempt = False
        if cached is None:
            attempt = True
        elif not cached.get("recognized"):
            last = cached.get("last_attempt", -10_000)
            if camera_frame_idx - last >= FACE_RECOG_INTERVAL_FRAMES:
                attempt = True

        if attempt:
            crop = _crop_person_head_region(frame, det, W, H)
            if crop is not None and crop.size > 0:
                # Keep a recent crop so an operator can enroll this (possibly
                # Unknown) track from the live frame via the admin API.
                with camera_lock:
                    recent_crops[tid] = crop.copy()
            match = face_engine.recognize_crop(crop) if crop is not None else None
            if match is not None:
                no = person_no_map.get(match.person_id)
                if no is None:
                    with camera_lock:
                        no = person_no_map.get(match.person_id)
                        if no is None:
                            no = int(state.get("next_person_no", 1))
                            person_no_map[match.person_id] = no
                            state["next_person_no"] = no + 1
                cached = {
                    "recognized": True,
                    "person_id": match.person_id,
                    "name": match.name,
                    "gender": match.gender,
                    "no": no,
                    "score": match.score,
                    "last_attempt": camera_frame_idx,
                }
            else:
                cached = {
                    "recognized": False,
                    "person_id": None,
                    "name": None,
                    "gender": None,
                    "no": None,
                    "last_attempt": camera_frame_idx,
                }
            with camera_lock:
                identity[tid] = cached

        if cached and cached.get("recognized"):
            det.recognized = True
            det.person_id = cached.get("person_id")
            det.person_name = cached.get("name")
            det.person_gender = cached.get("gender")
            det.person_no = cached.get("no")
        else:
            det.recognized = False
            det.person_name = "Unknown"


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
    status, reason_codes = compute_status(model_ok=model_ok, deps=deps)

    snapshot["status"] = status
    snapshot["reason_codes"] = reason_codes
    snapshot["dependencies"] = deps

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
    snapshot["pipeline"] = {
        "active_cameras": len(per_camera),
        "total_requests": total_requests,
        "total_errors": total_errors,
        "error_rate": round(total_errors / total_requests, 4) if total_requests > 0 else 0.0,
        "cameras": per_camera,
    }

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

    camera_id = req.camera_id or "default"
    enforce_security(request, camera_id=camera_id, require_auth=True)
    state = get_camera_state(camera_id)
    camera_request_count = increment_camera_request(state)
    # Use per-camera counter for all debug/log frequency checks (replaces global req_seq)
    cam_seq = camera_request_count
    decode_time_ms = 0.0
    preprocess_time_ms = 0.0
    yolo_time_ms = 0.0
    transport_name = "unknown"
    transport_encoded_bytes = 0
    transport_raw_bytes = 0
    
    try:
        # ============================================
        # 1) Decode current transport layer
        # Keep JSON + base64 for plugin compatibility; isolate it here so a
        # future multipart/raw-binary transport can replace this cleanly.
        # ============================================
        try:
            decode_start = time.perf_counter()
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

        H, W = frame.shape[:2]

        # Keep a reference to the originally decoded full frame for event snapshots.
        # Detection coords are later remapped back to this input space, so snapshots
        # stored for review/enrollment match the bbox coordinates we persist.
        snapshot_source_frame = frame

        # Debug: Check if frame is mostly empty/dark
        mean_bgr = cv2.mean(frame)
        frame_mean = float((mean_bgr[0] + mean_bgr[1] + mean_bgr[2]) / 3.0)
        gray_for_stats = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        frame_min_val, frame_max_val, _, _ = cv2.minMaxLoc(gray_for_stats)
        frame_max = int(frame_max_val)
        frame_min = int(frame_min_val)
        if cam_seq % 20 == 0:
            logger.debug(f"[{camera_id}] Frame: {W}x{H}, mean={frame_mean:.1f}, min={frame_min}, max={frame_max}")
            logger.debug(
                f"[{camera_id}] Transport: {transport_name}, "
                f"encoded_bytes={transport_encoded_bytes}, raw_bytes={transport_raw_bytes}"
            )
            if SAVE_DEBUG_SAMPLES:
                try:
                    import os
                    os.makedirs('frame_samples', exist_ok=True)
                    sample_path = f'frame_samples/{camera_id}_{cam_seq:06d}.jpg'
                    cv2.imwrite(sample_path, frame)
                    logger.debug(f"[{camera_id}] Saved frame sample to {sample_path}")
                except Exception as e:
                    logger.warning(f"Failed to save frame sample: {e}")
        logger.debug(f"[{camera_id}] Frame size: {W}x{H}")

        # Store transform metadata for coordinate mapping (before undistort/ROI)
        undistort_crop_offset = (0, 0)
        
        # ============================================
        # 1.4) Undistortion (for wide-angle/fisheye cameras)
        # ============================================
        if ENABLE_UNDISTORT:
            undistort_start = time.perf_counter()
            frame, undistort_crop_offset = undistort_frame(frame)
            undistort_time_ms = (time.perf_counter() - undistort_start) * 1000.0
            preprocess_time_ms += undistort_time_ms
            if cam_seq % 20 == 0:
                logger.debug(f"[{camera_id}] Undistortion: {undistort_time_ms:.1f}ms")
        pre_roi_shape = frame.shape

        # ============================================
        # 1.5) ROI Crop (Region of Interest)
        # ============================================
        roi_box = None
        roi_type = None
        if ENABLE_ROI:
            roi_start = time.perf_counter()
            frame, roi_box, roi_type = apply_roi(frame)
            roi_time_ms = (time.perf_counter() - roi_start) * 1000.0
            preprocess_time_ms += roi_time_ms
            H_roi, W_roi = frame.shape[:2]
            if cam_seq % 20 == 0:
                logger.debug(f"[{camera_id}] ROI ({roi_type}): {H_roi}x{W_roi}, time={roi_time_ms:.1f}ms")
        else:
            roi_box = (0, 0, frame.shape[1], frame.shape[0])

        H, W = frame.shape[:2]

        # ============================================
        # 1.6) Preprocess frame for wide-angle optimization
        # ============================================
        if ENABLE_CLAHE or ENABLE_FRAME_ENHANCEMENT:
            preprocess_start = time.perf_counter()
            frame = preprocess_frame(frame)
            preprocess_step_ms = (time.perf_counter() - preprocess_start) * 1000.0
            preprocess_time_ms += preprocess_step_ms
            if cam_seq % 20 == 0:
                logger.debug(
                    f"[{camera_id}] Frame preprocessing: {preprocess_step_ms:.1f}ms "
                    f"(CLAHE={ENABLE_CLAHE}, Enhancement={ENABLE_FRAME_ENHANCEMENT})"
                )

        # ============================================
        # 1.7) Per-camera config override (P2.2)
        # ============================================
        cam_cfg = get_per_camera_config_sync(camera_id)
        _conf_threshold = (cam_cfg["confidence_threshold"] if cam_cfg and cam_cfg.get("confidence_threshold") is not None
                           else CONFIDENCE_THRESHOLD)
        _iou_threshold = (cam_cfg["iou_threshold"] if cam_cfg and cam_cfg.get("iou_threshold") is not None
                          else IOU_THRESHOLD)

        # ============================================
        # 2) Run YOLO inference (person class only)
        # ============================================
        try:
            # Startup preloads the model; keep this as a defensive fallback.
            yolo_model = load_model()

            # Save pre-inference frame (after ROI crop) - every 200 frames
            if SAVE_DEBUG_SAMPLES and cam_seq % 200 == 0:
                try:
                    import os
                    os.makedirs('frame_samples', exist_ok=True)
                    sample_pre = f'frame_samples/{camera_id}_{cam_seq:06d}_pre_yolo.jpg'
                    cv2.imwrite(sample_pre, frame)
                except Exception as e:
                    logger.debug(f"Failed to save pre-YOLO frame: {e}")
            
            yolo_start = time.perf_counter()
            
            # Choose inference strategy
            if ENABLE_MULTI_SCALE:
                # Smart multi-scale detection: only retry at larger scale if no detections
                r = multi_scale_inference_smart(yolo_model, frame, H, W)
            else:
                # Standard single-scale inference - PRODUCTION MODE
                r = yolo_model.predict(
                    frame,
                    conf=_conf_threshold,
                    iou=_iou_threshold,
                    classes=[0],  # person only
                    imgsz=YOLO_IMGSZ,
                    verbose=False,
                    augment=False,
                    device=DEVICE,
                    half=USE_HALF,
                )[0]
            
            yolo_time_ms = (time.perf_counter() - yolo_start) * 1000.0
            
            # Save post-inference frame with detections - every 200 frames
            if SAVE_DEBUG_SAMPLES and cam_seq % 200 == 0:
                try:
                    import os
                    os.makedirs('frame_samples', exist_ok=True)
                    frame_with_boxes = frame.copy()
                    if r.boxes is not None and len(r.boxes) > 0:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    sample_post = f'frame_samples/{camera_id}_{cam_seq:06d}_post_yolo.jpg'
                    cv2.imwrite(sample_post, frame_with_boxes)
                except Exception as e:
                    logger.debug(f"Failed to save post-YOLO frame: {e}")
            
            # Log detection details for debugging
            boxes_for_log = r.boxes if r.boxes is not None else []
            num_boxes = len(boxes_for_log)
            if cam_seq % 20 == 0:
                logger.debug(
                    f"[{camera_id}] YOLO: {num_boxes} objects "
                    f"(conf={_conf_threshold:.2f}, iou={_iou_threshold:.2f}, inference={yolo_time_ms:.1f}ms)"
                )
                if num_boxes > 0:
                    for i, box in enumerate(boxes_for_log[:3]):  # Show first 3
                        logger.debug(f"  Box {i}: cls={int(box.cls[0].item())}, conf={float(box.conf[0].item()):.2f}")
        except Exception as e:
            error_message = f"YOLO inference error: {type(e).__name__}: {e}"
            _record_camera_infer_error(state, error_message)
            logger.error(f"[{camera_id}] {error_message}")
            return []

        now = time.time()
        camera_lock = state["lock"]
        with camera_lock:
            state["frame_idx"] = int(state.get("frame_idx", 0)) + 1
            camera_frame_idx = state["frame_idx"]
            tracks = [dict(tr) for tr in state["tracks"]]
            track_history_snapshot = [dict(tr) for tr in state["track_history"]]
            next_id = state["next_id"]
            fall_detector = state["fall_detector"]

        detections: List[Detection] = []

        # ============================================
        # 3) Process YOLO outputs with tracking
        # ============================================
        # Build raw detections first (for optional extra NMS)
        raw_dets: List[Dict[str, Any]] = []
        boxes_iter = r.boxes if r.boxes is not None else []
        for box in boxes_iter:
            try:
                cls_id = int(box.cls[0].item())
                if cls_id != 0:  # Only person
                    continue

                score = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Clamp to frame boundaries
                x1 = max(0.0, min(x1, W - 1.0))
                y1 = max(0.0, min(y1, H - 1.0))
                x2 = max(0.0, min(x2, W - 1.0))
                y2 = max(0.0, min(y2, H - 1.0))

                w_box = x2 - x1
                h_box = y2 - y1
                area = w_box * h_box
                hw_ratio = h_box / max(1.0, w_box)

                # Filter by minimum area
                if w_box <= 1.0 or h_box <= 1.0 or area < MIN_DETECTION_AREA:
                    logger.debug(f"[{camera_id}] Skipping small detection: {w_box:.1f}x{h_box:.1f} (area={area:.0f} < {MIN_DETECTION_AREA})")
                    continue

                # Reject furniture-like wide boxes that YOLO occasionally confuses as people.
                if hw_ratio < PERSON_MIN_HW_RATIO:
                    logger.debug(
                        f"[{camera_id}] Skipping wide person-like detection: "
                        f"{w_box:.1f}x{h_box:.1f} ratio={hw_ratio:.2f} < {PERSON_MIN_HW_RATIO:.2f}"
                    )
                    continue

                det_box = (x1, y1, x2, y2)
                raw_dets.append({"bbox": det_box, "score": score})
            except Exception as e:
                logger.warning(f"[{camera_id}] Error parsing box: {type(e).__name__}: {e}")
                continue

        # Extra dedupe to avoid overlapping boxes for the same person
        if ENABLE_POST_NMS and len(raw_dets) > 1:
            raw_dets = post_nms_dedupe(raw_dets, POST_NMS_IOU)

        # Track state (persistent)
        track_by_id = {tr["id"]: tr for tr in tracks}
        for tr in track_by_id.values():
            tr.setdefault("status", "confirmed")
            tr.setdefault("hits", TRACK_MIN_HITS)
            tr.setdefault("misses", 0)
            tr.setdefault("vx", 0.0)
            tr.setdefault("vy", 0.0)
            tr.setdefault("vw", 0.0)
            tr.setdefault("vh", 0.0)
            tr.setdefault("measurement_bbox", tr.get("bbox"))

        matched_ids = set()
        output_track_ids = set()

        # Global optimal 1-1 matching (Hungarian) over gated det-track pairs.
        det_to_track, det_appearances = global_track_assignment(raw_dets, track_by_id, frame, now)

        for det_idx, track_id in det_to_track.items():
            det = raw_dets[det_idx]
            det_box = det["bbox"]
            score = det["score"]
            det_appearance = det_appearances[det_idx]

            tr = track_by_id.get(track_id)
            if not tr:
                continue

            prev_bbox = tr.get("bbox", det_box)
            prev_measurement_bbox = tr.get("measurement_bbox", prev_bbox)
            if BBOX_SMOOTHING > 0.0 and tr.get("bbox"):
                tr["bbox"] = smooth_bbox(prev_bbox, det_box, BBOX_SMOOTHING)
            else:
                tr["bbox"] = det_box

            update_track_motion(
                tr,
                prev_bbox=prev_measurement_bbox,
                new_bbox=det_box,
                now_ts=now
            )
            tr["last_seen"] = now
            tr["appearance"] = det_appearance
            tr["measurement_bbox"] = det_box
            tr["score"] = score
            tr["misses"] = 0
            tr["hits"] = int(tr.get("hits", 0)) + 1
            if tr["status"] != "confirmed" and tr["hits"] >= TRACK_MIN_HITS:
                tr["status"] = "confirmed"
            elif tr["status"] == "lost":
                tr["status"] = "confirmed"

            track_by_id[track_id] = tr
            matched_ids.add(track_id)

        for tr in track_by_id.values():
            if tr["id"] in matched_ids:
                continue
            tr["misses"] = int(tr.get("misses", 0)) + 1
            if tr.get("status") == "tentative":
                if tr["misses"] > TENTATIVE_MAX_MISSES:
                    tr["status"] = "deleted"
            else:
                tr["status"] = "lost"

        unmatched_det_idxs = [idx for idx in range(len(raw_dets)) if idx not in det_to_track]
        for det_idx in unmatched_det_idxs:
            try:
                det = raw_dets[det_idx]
                det_box = det["bbox"]
                score = det["score"]
                det_appearance = det_appearances[det_idx]
                track_id = None

                # Optional history rematch: appearance-only for returning person.
                if ENABLE_HISTORY_REMATCH and track_history_snapshot and det_appearance.get("color_hist") is not None:
                    best_hist_score = -1.0
                    best_hist = None
                    for hist_tr in track_history_snapshot:
                        if hist_tr.get("id") in matched_ids:
                            continue
                        if hist_tr.get("appearance") and hist_tr["appearance"].get("color_hist") is not None:
                            app_dist = appearance_distance(hist_tr["appearance"]["color_hist"], det_appearance["color_hist"])
                            if app_dist < 0.4:
                                hist_score = 1.0 - app_dist
                                if hist_score > best_hist_score:
                                    best_hist_score = hist_score
                                    best_hist = hist_tr
                    if best_hist is not None:
                        track_id = int(best_hist["id"])
                        track_by_id[track_id] = {
                            "id": track_id,
                            "bbox": det_box,
                            "measurement_bbox": det_box,
                            "last_seen": now,
                            "created_at": best_hist.get("created_at", now),
                            "appearance": det_appearance,
                            "score": score,
                            "status": "confirmed",
                            "hits": max(TRACK_MIN_HITS, int(best_hist.get("hits", TRACK_MIN_HITS))),
                            "misses": 0,
                            "vx": 0.0,
                            "vy": 0.0,
                            "vw": 0.0,
                            "vh": 0.0,
                        }
                        logger.debug(f"[{camera_id}] Re-matched track ID={track_id} from history")

                if track_id is None:
                    if score < NEW_TRACK_MIN_CONFIDENCE:
                        continue

                    if has_duplicate_track_overlap(
                        det_box=det_box,
                        track_by_id=track_by_id,
                        now_ts=now,
                        overlap_iou=TRACK_DUPLICATE_IOU,
                        recent_only_sec=max(TRACK_OUTPUT_HOLD_TIME, 0.5),
                    ):
                        continue

                    track_id = next_id
                    next_id += 1
                    track_by_id[track_id] = {
                        "id": track_id,
                        "bbox": det_box,
                        "measurement_bbox": det_box,
                        "last_seen": now,
                        "created_at": now,
                        "appearance": det_appearance,
                        "score": score,
                        "status": "tentative",
                        "hits": 1,
                        "misses": 0,
                        "vx": 0.0,
                        "vy": 0.0,
                        "vw": 0.0,
                        "vh": 0.0,
                    }

                matched_ids.add(track_id)
            except Exception as e:
                logger.warning(f"[{camera_id}] Error processing unmatched detection: {type(e).__name__}: {e}")
                continue

        # Output selection: confirmed tracks by default; optional tentative output via config.
        for tr in track_by_id.values():
            if tr.get("status") == "deleted":
                continue

            is_confirmed = tr.get("status") == "confirmed"
            is_tentative = tr.get("status") == "tentative"

            if tr["id"] in matched_ids:
                if is_confirmed or (OUTPUT_TENTATIVE_TRACKS and is_tentative):
                    output_track_ids.add(tr["id"])
                continue

            if not is_confirmed:
                continue
            if int(tr.get("misses", 0)) > TRACK_MAX_MISSES:
                continue
            if now - tr.get("last_seen", 0.0) > TRACK_OUTPUT_HOLD_TIME:
                continue
            if suppress_overlapping_holds(tr, output_track_ids, track_by_id, HOLD_SUPPRESS_IOU):
                continue
            output_track_ids.add(tr["id"])

        # Build final detections from stable track state (one bbox per track_id).
        detections = []
        for track_id in sorted(output_track_ids):
            tr = track_by_id.get(track_id)
            if not tr:
                continue
            is_confirmed = tr.get("status") == "confirmed"
            x1, y1, x2, y2 = tr["bbox"]
            w_box = max(0.0, x2 - x1)
            h_box = max(0.0, y2 - y1)
            if w_box <= 1.0 or h_box <= 1.0:
                continue
            detections.append(Detection(
                cls="person",
                score=float(tr.get("score", 0.0)),
                x=float(x1),
                y=float(y1),
                w=float(w_box),
                h=float(h_box),
                track_id=int(track_id),
                stable=bool(is_confirmed),
                degraded=False,
            ))

        detections = dedupe_output_tracks(detections, OUTPUT_DEDUPE_IOU)

        if raw_dets and not detections:
            best_score = max(det["score"] for det in raw_dets)
            logger.warning(
                f"[{camera_id}] YOLO raw persons={len(raw_dets)} but output detections=0 "
                f"(best_score={best_score:.2f}, track_min_hits={TRACK_MIN_HITS}, "
                f"output_tentative={OUTPUT_TENTATIVE_TRACKS}, "
                f"new_track_min_confidence={NEW_TRACK_MIN_CONFIDENCE:.2f})"
            )

            # Degraded mode: if stable service-side tracking filters everything out but YOLO still
            # sees a person, fall back to raw YOLO boxes so Nx can still render a bbox. The plugin
            # must treat these as render-only detections, not lifecycle events.
            fallback_detections: List[Detection] = []
            used_track_ids: set[int] = set()
            reuse_window_sec = max(TRACK_OUTPUT_HOLD_TIME, 0.5)

            for det in sorted(raw_dets, key=lambda item: item["score"], reverse=True):
                det_box = det["bbox"]
                best_track_id = None
                best_iou_score = 0.0

                for tr in track_by_id.values():
                    if tr["id"] in used_track_ids:
                        continue
                    if now - tr.get("last_seen", 0.0) > reuse_window_sec:
                        continue

                    overlap = iou(det_box, tr["bbox"])
                    if overlap > best_iou_score:
                        best_iou_score = overlap
                        best_track_id = int(tr["id"])

                if best_track_id is None or best_iou_score < 0.10:
                    best_track_id = next_id
                    next_id += 1
                    track_by_id[best_track_id] = {
                        "id": best_track_id,
                        "bbox": det_box,
                        "measurement_bbox": det_box,
                        "last_seen": now,
                        "created_at": now,
                        "appearance": None,
                        "score": det["score"],
                        "status": "tentative",
                        "hits": 1,
                        "misses": 0,
                        "vx": 0.0,
                        "vy": 0.0,
                        "vw": 0.0,
                        "vh": 0.0,
                    }
                else:
                    tr = track_by_id[best_track_id]
                    tr["bbox"] = det_box
                    tr["measurement_bbox"] = det_box
                    tr["last_seen"] = now
                    tr["score"] = det["score"]
                    tr["misses"] = 0
                    if tr.get("status") == "lost":
                        tr["status"] = "tentative"
                    track_by_id[best_track_id] = tr

                used_track_ids.add(best_track_id)

                x1, y1, x2, y2 = det_box
                fallback_detections.append(Detection(
                    cls="person",
                    score=float(det["score"]),
                    x=float(x1),
                    y=float(y1),
                    w=float(max(0.0, x2 - x1)),
                    h=float(max(0.0, y2 - y1)),
                    track_id=int(best_track_id),
                    stable=False,
                    degraded=True,
                ))

            if fallback_detections:
                detections = fallback_detections
                logger.warning(
                    f"[{camera_id}] Fallback to raw YOLO detections for plugin continuity: "
                    f"{len(detections)} person(s)"
                )

        # ============================================
        # 3.2) Fall Detection (NEW)
        # ============================================
        if ENABLE_FALL_DETECTION and fall_detector:
            try:
                # Prepare detection data for fall detection
                fall_input_detections = []
                for det in detections:
                    if not det.stable or det.degraded:
                        continue
                    fall_input_detections.append({
                        'track_id': det.track_id,
                        'bbox': (det.x, det.y, det.x + det.w, det.y + det.h),
                        'confidence': det.score,
                    })
                
                # Update fall detector with current frame detections
                with camera_lock:
                    fall_results = fall_detector.update(fall_input_detections, camera_frame_idx)
                    fall_stats = fall_detector.get_stats()
                
                # Mark detections with fall status
                for det in detections:
                    det.fall_detected = fall_results.get(det.track_id, False)
                    if det.fall_detected:
                        logger.warning(f"[{camera_id}] FALL DETECTED: Person {det.track_id} (score={det.score:.2f})")

                if fall_stats['total_fallen'] > 0:
                    logger.info(f"[{camera_id}] Fall Status: {fall_stats['total_fallen']} person(s) fallen out of {fall_stats['total_tracked']} tracked")
                    
            except Exception as e:
                error_message = f"Fall detection error: {type(e).__name__}: {e}"
                _record_camera_infer_error(state, error_message)
                logger.error(f"[{camera_id}] {error_message}")
                # Fall detection errors don't stop inference, just log and continue

        # ============================================
        # 3.5) Map detections back to input frame coordinates
        # ============================================
        remap_detections_to_input_space(
            detections=detections,
            roi_box=roi_box,
            roi_type=roi_type,
            pre_roi_shape=pre_roi_shape,
            undistort_crop_offset=undistort_crop_offset,
        )

        # ============================================
        # 3.6) Face recognition — real-time identity on the bounding box
        # Runs after remap so crop coords match the snapshot frame space.
        # Throttled per track; identity cached for stable labels.
        # ============================================
        if ENABLE_FACE_RECOGNITION and detections:
            try:
                _apply_face_identity(
                    state=state,
                    detections=detections,
                    frame=snapshot_source_frame,
                    camera_frame_idx=camera_frame_idx,
                    camera_lock=camera_lock,
                )
            except Exception as _fe:
                logger.debug("[%s] Face recognition error: %s", camera_id, _fe)

        # ============================================
        # 3.7) Zone check (P2.2)
        # Runs after remap so zone geometry and detection coords
        # are both in the full input-frame pixel space.
        # ============================================
        zone_violations_this_frame: list = []
        if detections:
            try:
                active_zones = get_zones_for_camera_sync(camera_id)
                if active_zones:
                    zone_violations_this_frame = check_zones(detections, active_zones, camera_id)
            except Exception as _ze:
                logger.warning("[%s] Zone check error: %s", camera_id, _ze)

        # ============================================
        # 4) Anti-flicker: reuse last output if empty
        # ============================================
        with camera_lock:
            if not detections and state["last_output"]:
                time_since_last = now - state["last_time"]
                if time_since_last < FLICKER_REUSE_TIME:
                    logger.debug(f"[{camera_id}] Reusing last output (anti-flicker, {time_since_last*1000:.0f}ms)")
                    detections = state["last_output"]
            else:
                state["last_output"] = detections
                state["last_time"] = now

            # ============================================
            # 5) Track TTL cleanup with history buffer
            # ============================================
            old_count = len(state["tracks"])

            active_tracks = []
            expired_tracks = []
            for tr in track_by_id.values():
                if tr.get("status") == "deleted":
                    expired_tracks.append(tr)
                    continue

                if tr.get("status") == "tentative" and int(tr.get("misses", 0)) > TENTATIVE_MAX_MISSES:
                    tr["status"] = "deleted"
                    expired_tracks.append(tr)
                    continue

                if now - tr.get("last_seen", 0.0) <= TRACK_TTL:
                    active_tracks.append(tr)
                else:
                    expired_tracks.append(tr)

            if ENABLE_HISTORY_REMATCH:
                state["track_history"] = [tr for tr in state["track_history"] if now - tr.get("last_seen", now) <= 30.0]
                history_candidates = [
                    tr for tr in expired_tracks
                    if tr.get("status") in {"confirmed", "lost"} and tr.get("appearance")
                ]
                state["track_history"].extend(history_candidates)
            else:
                state["track_history"] = []

            state["tracks"] = active_tracks
            removed = old_count - len(state["tracks"])
            if removed > 0:
                logger.debug(f"[{camera_id}] Removed {removed} stale tracks from active state")

            state["next_id"] = next_id

            # Purge cached face identity / crops for tracks that are no longer active
            active_track_ids = {tr["id"] for tr in active_tracks}
            identity_cache = state.get("identity")
            if identity_cache:
                for dead_tid in [t for t in identity_cache if t not in active_track_ids]:
                    identity_cache.pop(dead_tid, None)
            recent_crops_cache = state.get("recent_crops")
            if recent_crops_cache:
                for dead_tid in [t for t in recent_crops_cache if t not in active_track_ids]:
                    recent_crops_cache.pop(dead_tid, None)

            # ============================================
            # 6) Count tracking with proper ID deduplication
            # ============================================
            # Track IDs with their last_seen timestamp to properly age out old IDs
            if "seen_ids_with_time" not in state:
                state["seen_ids_with_time"] = {}  # {track_id: last_seen_timestamp}
            
            # Add confirmed tracks with enough hits
            stable_unique_ids = {
                int(track_id)
                for track_id in output_track_ids
                if track_id in track_by_id
                and track_by_id[track_id].get("status") == "confirmed"
                and int(track_by_id[track_id].get("hits", 0)) >= UNIQUE_COUNT_MIN_HITS
            }
            
            # Update timestamps for currently-active confirmed tracks
            for track_id in stable_unique_ids:
                state["seen_ids_with_time"][track_id] = now
            
            # Expire old IDs that haven't been seen recently (older than TRACK_TTL + grace period)
            # Grace period allows re-detection of same person after brief exit
            id_expiry_threshold = TRACK_TTL + 30.0  # Keep IDs for up to 45s after last detection
            expired_ids = [
                vid for vid, last_seen in state["seen_ids_with_time"].items()
                if now - last_seen > id_expiry_threshold
            ]
            for vid in expired_ids:
                del state["seen_ids_with_time"][vid]
            
            # Maintain backward compatibility with seen_ids set for any external code
            state["seen_ids"] = set(state["seen_ids_with_time"].keys())

            # S P1.3: track which IDs have already been reported to the outbox
            _reported_ids: set = state.setdefault("outbox_reported_ids", set())
            new_outbox_ids: set = stable_unique_ids - _reported_ids
            _reported_ids.update(stable_unique_ids)

            tracks_count = len(state["tracks"])
            unique_count = len(state["seen_ids"])

        total_infer_ms = (time.perf_counter() - request_started_at) * 1000.0
        metrics_snapshot = record_camera_metrics(
            state,
            infer_ms=total_infer_ms,
            decode_ms=decode_time_ms,
            preprocess_ms=preprocess_time_ms,
            yolo_ms=yolo_time_ms,
            track_count=tracks_count,
        )

        # Prometheus observations (success path)
        INFER_REQUESTS.labels(camera_id=camera_id, status="success").inc()
        INFER_LATENCY.labels(camera_id=camera_id).observe(total_infer_ms / 1000.0)
        DECODE_LATENCY.labels(camera_id=camera_id).observe(decode_time_ms / 1000.0)
        PREPROCESS_LATENCY.labels(camera_id=camera_id).observe(preprocess_time_ms / 1000.0)
        YOLO_LATENCY.labels(camera_id=camera_id).observe(yolo_time_ms / 1000.0)
        DETECTIONS_PER_FRAME.labels(camera_id=camera_id).observe(float(len(detections)))
        ACTIVE_TRACKS.labels(camera_id=camera_id).set(float(tracks_count))

        if camera_request_count == 1 or camera_request_count % METRICS_LOG_INTERVAL == 0:
            logger.info(
                f"[{camera_id}] Summary: detections={len(detections)} "
                f"tracks={tracks_count} unique={unique_count} "
                f"infer_ms={total_infer_ms:.1f} decode_ms={decode_time_ms:.1f} "
                f"preprocess_ms={preprocess_time_ms:.1f} yolo_ms={yolo_time_ms:.1f} "
                f"p50_ms={metrics_snapshot['p50_inference_ms']:.1f} "
                f"p95_ms={metrics_snapshot['p95_inference_ms']:.1f} "
                f"errors={metrics_snapshot['error_count']}"
            )

        # ── S P1.3: non-blocking outbox enqueue ──────────────────────────────
        _now_utc = datetime.now(tz=timezone.utc)

        fall_dets = [d for d in detections if d.fall_detected]
        has_new_detections = any(d.track_id in new_outbox_ids for d in detections)
        needs_snapshot = bool(fall_dets or zone_violations_this_frame or has_new_detections)

        # Encode the event snapshot JPEG at most once per frame, only when an event fires.
        _snap_jpeg: Optional[bytes] = None
        if needs_snapshot:
            try:
                ok, _snap_buf = cv2.imencode(
                    ".jpg", snapshot_source_frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
                )
                if ok:
                    _snap_jpeg = bytes(_snap_buf)
            except Exception as _enc_exc:
                logger.debug("[%s] Snapshot encode failed: %s", camera_id, _enc_exc)

        def _det_person_fields(det: "Detection") -> dict:
            return {
                "person_id": det.person_id,
                "person_name": det.person_name,
            }

        # Fall events — always carry a snapshot
        for _det in fall_dets:
            outbox_worker.enqueue_event(OutboxEvent(
                camera_id=camera_id,
                event_type="fall",
                occurred_at=_now_utc,
                track_id=_det.track_id,
                cls=_det.cls,
                confidence=_det.score,
                bbox_x=_det.x,
                bbox_y=_det.y,
                bbox_w=_det.w,
                bbox_h=_det.h,
                snapshot_jpeg=_snap_jpeg,
                **_det_person_fields(_det),
            ))

        # New confirmed-track events — first appearance, now with a snapshot for review/enrollment
        for _det in detections:
            if _det.track_id in new_outbox_ids:
                outbox_worker.enqueue_event(OutboxEvent(
                    camera_id=camera_id,
                    event_type="detection",
                    occurred_at=_now_utc,
                    track_id=_det.track_id,
                    cls=_det.cls,
                    confidence=_det.score,
                    bbox_x=_det.x,
                    bbox_y=_det.y,
                    bbox_w=_det.w,
                    bbox_h=_det.h,
                    snapshot_jpeg=_snap_jpeg,
                    **_det_person_fields(_det),
                ))

        # Zone violation events (P2.2) — one event per violation per entry, now with a snapshot
        for _zv in zone_violations_this_frame:
            _zv_det = next((d for d in detections if d.track_id == _zv.track_id), None)
            outbox_worker.enqueue_event(OutboxEvent(
                camera_id=camera_id,
                event_type="zone_violation",
                occurred_at=_now_utc,
                track_id=_zv.track_id,
                cls="person",
                confidence=_zv_det.score if _zv_det else 0.0,
                bbox_x=_zv_det.x if _zv_det else 0.0,
                bbox_y=_zv_det.y if _zv_det else 0.0,
                bbox_w=_zv_det.w if _zv_det else 0.0,
                bbox_h=_zv_det.h if _zv_det else 0.0,
                zone_id=_zv.zone_id,
                snapshot_jpeg=_snap_jpeg,
                **(_det_person_fields(_zv_det) if _zv_det else {}),
            ))

        return detections

    except Exception as e:
        error_message = f"Unexpected error in /infer: {type(e).__name__}: {e}"
        _record_camera_infer_error(state, error_message)
        logger.error(f"[{camera_id}] {error_message}", exc_info=True)
        INFER_REQUESTS.labels(camera_id=camera_id, status="error").inc()
        INFER_ERRORS.labels(camera_id=camera_id, kind="unknown").inc()
        return []


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
        init_engine()
        from .db.session import is_edge_mode

        if is_edge_mode():
            from .config import EDGE_SQLITE_PATH
            logger.info(f"Edge SQLite store ready ({EDGE_SQLITE_PATH})")
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


