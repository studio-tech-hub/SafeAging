import base64
import hmac
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import (
    API_KEY,
    API_KEY_REQUIRED,
    BBOX_SMOOTHING,
    CONFIDENCE_THRESHOLD,
    CORS_ALLOW_ORIGINS,
    DEVICE,
    ENABLE_CLAHE,
    ENABLE_FALL_DETECTION,
    ENABLE_FRAME_ENHANCEMENT,
    ENABLE_HISTORY_REMATCH,
    ENABLE_MULTI_SCALE,
    ENABLE_POST_NMS,
    ENABLE_ROI,
    ENABLE_UNDISTORT,
    FLICKER_REUSE_TIME,
    HOLD_SUPPRESS_IOU,
    IOU_THRESHOLD,
    MIN_DETECTION_AREA,
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
from .model import load_model
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
    get_camera_state,
    global_track_assignment,
    has_duplicate_track_overlap,
    iou,
    post_nms_dedupe,
    smooth_bbox,
    suppress_overlapping_holds,
    update_track_motion,
)


log_config_summary()

app = FastAPI(title="YOLOv8 Analytics Service")
if CORS_ALLOW_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ALLOW_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

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
    fall_detected: bool = False  # NEW: Fall detection flag
    stable: bool = True
    degraded: bool = False

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    service_uptime_seconds: float

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
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
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
def health_check():
    """Health check endpoint for NX plugin"""
    uptime = time.time() - service_start_time
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        service_uptime_seconds=uptime
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
    req_seq = increment_request_counter()
    start_time = time.time()
    
    camera_id = req.camera_id or "default"
    enforce_security(request, camera_id=camera_id, require_auth=True)
    
    try:
        # ============================================
        # 1) Decode base64 image with auto brightness fix
        # ============================================
        try:
            img_bytes = base64.b64decode(req.image)
            
            # Check if it's BGR format (new C++ format) or JPEG/PNG (old format)
            if img_bytes.startswith(b'BGR'):
                # New BGR format: "BGR" + width(4 bytes) + height(4 bytes) + raw BGR data
                import struct
                header = img_bytes[:11]  # 3 (BGR) + 4 (width) + 4 (height)
                w, h = struct.unpack('<II', img_bytes[3:11])
                bgr_data = img_bytes[11:]
                
                # Reconstruct Mat from raw BGR data
                frame = np.frombuffer(bgr_data, dtype=np.uint8).reshape((h, w, 3))
                logger.debug(f"[{camera_id}] Decoded BGR format: {w}x{h}")
                
                # AUTO BRIGHTNESS ADJUSTMENT (always applied for BGR frames)
                frame = auto_adjust_brightness(frame, target_brightness=180.0)
            else:
                # Old format: JPEG/PNG encoded image
                img_array = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                logger.debug(f"[{camera_id}] Decoded JPEG/PNG format")
            
            if frame is None:
                increment_error_counter()
                logger.warning(f"[{camera_id}] Failed to decode image - got None")
                return []
                 
        except Exception as e:
            increment_error_counter()
            logger.warning(f"[{camera_id}] Image decode error: {type(e).__name__}: {e}")
            return []

        H, W = frame.shape[:2]
        
        # Debug: Check if frame is mostly empty/dark
        mean_bgr = cv2.mean(frame)
        frame_mean = float((mean_bgr[0] + mean_bgr[1] + mean_bgr[2]) / 3.0)
        gray_for_stats = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        frame_min_val, frame_max_val, _, _ = cv2.minMaxLoc(gray_for_stats)
        frame_max = int(frame_max_val)
        frame_min = int(frame_min_val)
        if req_seq % 20 == 0:
            logger.info(f"[{camera_id}] Frame: {W}x{H}, mean={frame_mean:.1f}, min={frame_min}, max={frame_max}")
            if SAVE_DEBUG_SAMPLES:
                # Save frame to disk only when explicitly enabled via env var.
                try:
                    import os
                    os.makedirs('frame_samples', exist_ok=True)
                    sample_path = f'frame_samples/frame_{req_seq:06d}.jpg'
                    cv2.imwrite(sample_path, frame)
                    logger.info(f"[{camera_id}] Saved frame sample to {sample_path}")
                except Exception as e:
                    logger.warning(f"Failed to save frame sample: {e}")
        logger.debug(f"[{camera_id}] Frame size: {W}x{H}")

        # Store transform metadata for coordinate mapping (before undistort/ROI)
        undistort_crop_offset = (0, 0)
        
        # ============================================
        # 1.4) Undistortion (for wide-angle/fisheye cameras)
        # ============================================
        if ENABLE_UNDISTORT:
            undistort_start = time.time()
            frame, undistort_crop_offset = undistort_frame(frame)
            undistort_time = time.time() - undistort_start
            if req_seq % 20 == 0:
                logger.info(f"[{camera_id}] Undistortion: {undistort_time*1000:.1f}ms")
        pre_roi_shape = frame.shape

        # ============================================
        # 1.5) ROI Crop (Region of Interest)
        # ============================================
        roi_box = None
        roi_type = None
        if ENABLE_ROI:
            roi_start = time.time()
            frame, roi_box, roi_type = apply_roi(frame)
            roi_time = time.time() - roi_start
            H_roi, W_roi = frame.shape[:2]
            if req_seq % 20 == 0:
                logger.info(f"[{camera_id}] ROI ({roi_type}): {H_roi}x{W_roi}, time={roi_time*1000:.1f}ms")
        else:
            roi_box = (0, 0, frame.shape[1], frame.shape[0])

        H, W = frame.shape[:2]

        # ============================================
        # 1.6) Preprocess frame for wide-angle optimization
        # ============================================
        if ENABLE_CLAHE or ENABLE_FRAME_ENHANCEMENT:
            inference_start = time.time()
            frame = preprocess_frame(frame)
            preprocess_time = time.time() - inference_start
            if req_seq % 20 == 0:
                logger.info(f"[{camera_id}] Frame preprocessing: {preprocess_time*1000:.1f}ms (CLAHE={ENABLE_CLAHE}, Enhancement={ENABLE_FRAME_ENHANCEMENT})")

        # ============================================
        # 2) Run YOLO inference (person class only)
        # ============================================
        try:
            # Load model if needed (lazy load)
            yolo_model = load_model()
            
            # Save pre-inference frame (after ROI crop) - every 200 frames
            if SAVE_DEBUG_SAMPLES and req_seq % 200 == 0:
                try:
                    import os
                    os.makedirs('frame_samples', exist_ok=True)
                    sample_pre = f'frame_samples/frame_{req_seq:06d}_pre_yolo.jpg'
                    cv2.imwrite(sample_pre, frame)
                except Exception as e:
                    logger.debug(f"Failed to save pre-YOLO frame: {e}")
            
            inference_start = time.time()
            
            # Choose inference strategy
            if ENABLE_MULTI_SCALE:
                # Smart multi-scale detection: only retry at larger scale if no detections
                r = multi_scale_inference_smart(yolo_model, frame, H, W)
            else:
                # Standard single-scale inference - PRODUCTION MODE
                r = yolo_model.predict(
                    frame, 
                    conf=CONFIDENCE_THRESHOLD,  # 0.45 by default - catches people reliably
                    iou=IOU_THRESHOLD,  # 0.45 - standard NMS
                    classes=[0],  # person only
                    imgsz=YOLO_IMGSZ,  # 960 for better small object detection (after ROI crop)
                    verbose=False,
                    augment=False,  # No test-time augmentation in production
                    device=DEVICE,
                    half=USE_HALF,
                )[0]
            
            inference_time_ms = (time.time() - inference_start) * 1000
            
            # Save post-inference frame with detections - every 200 frames
            if SAVE_DEBUG_SAMPLES and req_seq % 200 == 0:
                try:
                    import os
                    os.makedirs('frame_samples', exist_ok=True)
                    frame_with_boxes = frame.copy()
                    if r.boxes is not None and len(r.boxes) > 0:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    sample_post = f'frame_samples/frame_{req_seq:06d}_post_yolo.jpg'
                    cv2.imwrite(sample_post, frame_with_boxes)
                except Exception as e:
                    logger.debug(f"Failed to save post-YOLO frame: {e}")
            
            # Log detection details for debugging
            boxes_for_log = r.boxes if r.boxes is not None else []
            num_boxes = len(boxes_for_log)
            if req_seq % 20 == 0:
                logger.info(f"[{camera_id}] YOLO: {num_boxes} objects (conf={CONFIDENCE_THRESHOLD}, inference={inference_time_ms:.1f}ms)")
                if num_boxes > 0:
                    for i, box in enumerate(boxes_for_log[:3]):  # Show first 3
                        logger.info(f"  Box {i}: cls={int(box.cls[0].item())}, conf={float(box.conf[0].item()):.2f}")
        except Exception as e:
            increment_error_counter()
            logger.error(f"[{camera_id}] YOLO inference error: {type(e).__name__}: {e}")
            return []

        now = time.time()
        state = get_camera_state(camera_id)
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
                logger.error(f"[{camera_id}] Fall detection error: {type(e).__name__}: {e}")
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

            inference_time = time.time() - start_time
            state["inference_times"].append(inference_time)
            if len(state["inference_times"]) > 30:
                state["inference_times"] = state["inference_times"][-30:]

            avg_inference_time = sum(state["inference_times"]) / len(state["inference_times"])
            max_inference_time = max(state["inference_times"])
            tracks_count = len(state["tracks"])
            unique_count = len(state["seen_ids"])
        
        logger.info(
            f"[{camera_id}] Detections: {len(detections)} | "
            f"Tracks: {tracks_count} | "
            f"Unique: {unique_count} | "
            f"Time: {inference_time*1000:.1f}ms (avg: {avg_inference_time*1000:.1f}ms)"
        )

        return detections

    except Exception as e:
        increment_error_counter()
        logger.error(f"[{camera_id}] Unexpected error in /infer: {type(e).__name__}: {e}", exc_info=True)
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
            cam_info = {
                "frame_idx": int(state.get("frame_idx", 0)),
                "tracks": len(state["tracks"]),
                "unique_persons": len(state["seen_ids"]),
                "created_at": datetime.fromtimestamp(state["created_at"]).isoformat(),
                "avg_inference_ms": (sum(state["inference_times"]) / len(state["inference_times"]) * 1000)
                                   if state["inference_times"] else 0.0
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
        "service": "YOLOv8 People Analytics + Fall Detection",
        "status": "running",
        "uptime_seconds": uptime,
        "total_requests": total_requests,
        "total_errors": total_errors,
        "error_rate": (total_errors / total_requests * 100) if total_requests > 0 else 0.0,
        "active_cameras": len(camera_items),
        "fall_detection": "ENABLED" if ENABLE_FALL_DETECTION else "DISABLED",
        "cameras": cameras_info,
        "model": MODEL_PATH,
        "timestamp": datetime.now().isoformat()
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
    if ENABLE_FALL_DETECTION:
        logger.info(f"Reset fall detection for camera: POST http://{SERVICE_HOST}:{SERVICE_PORT}/reset_fall/default")
        logger.info(f"Reset fall detection for all cameras: POST http://{SERVICE_HOST}:{SERVICE_PORT}/reset_fall_all")

@app.on_event("shutdown")
async def shutdown_event():
    """Called when service shuts down"""
    logger.info("Service shutting down")
    total_requests, total_errors = get_counter_snapshot()
    logger.info(f"Total requests: {total_requests}")
    logger.info(f"Total errors: {total_errors}")


