import base64
import hmac
import json
import time
import logging
import os
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import torch

# Import fall detection module
from fall_detection import FallDetectionManager

# ============================
# Logging Configuration
# ============================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Fix PyTorch 2.6 compatibility issue with YOLO weights
if hasattr(torch.serialization, 'add_safe_globals'):
    try:
        torch.serialization.add_safe_globals([
            torch.nn.modules.container.Sequential,
            torch.nn.modules.linear.Linear,
            torch.nn.modules.activation.ReLU,
            torch.nn.modules.activation.SiLU,
            torch.nn.modules.batchnorm.BatchNorm2d,
            torch.nn.modules.conv.Conv2d,
        ])
    except Exception as e:
        logger.warning(f"Could not add safe globals: {e}")
import uvicorn

# ============================
# PyTorch 2.6+ Compatibility Fix
# ============================
if hasattr(torch.serialization, 'add_safe_globals'):
    try:
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
    except Exception:
        pass  # Silently continue if fix not needed

# ============================
# Configuration
# ============================
def _load_local_env() -> None:
    """Load KEY=VALUE pairs from .env without overwriting existing env vars."""
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    env_path = next((p for p in candidates if p.exists()), None)
    if env_path is None:
        return

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception as e:
        logger.warning(f"Could not load .env file: {e}")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"Invalid int for {name}='{raw}', using default {default}")
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"Invalid float for {name}='{raw}', using default {default}")
        return default


def _clamp(value: float, low: float, high: float, name: str) -> float:
    if value < low:
        logger.warning(f"{name}={value} is below {low}, clamping to {low}")
        return low
    if value > high:
        logger.warning(f"{name}={value} is above {high}, clamping to {high}")
        return high
    return value


class AppConfig:
    def __init__(self) -> None:
        cuda_available = torch.cuda.is_available()

        self.service_port = _env_int("SERVICE_PORT", 18000)
        self.service_host = os.getenv("SERVICE_HOST", "127.0.0.1")
        self.model_path = os.getenv("MODEL_PATH", "yolov8n.pt")
        self.confidence_threshold = _clamp(_env_float("CONFIDENCE_THRESHOLD", 0.35), 0.0, 1.0, "CONFIDENCE_THRESHOLD")  # FIXED: Was 0.15 → Industry standard 0.35
        self.iou_threshold = _clamp(_env_float("IOU_THRESHOLD", 0.45), 0.0, 1.0, "IOU_THRESHOLD")
        self.min_detection_area = max(1, _env_int("MIN_DETECTION_AREA", 20))
        self.person_min_hw_ratio = _clamp(
            _env_float("PERSON_MIN_HW_RATIO", 0.80), 0.1, 10.0, "PERSON_MIN_HW_RATIO"
        )
        self.track_ttl = max(1.0, _env_float("TRACK_TTL", 15.0))
        self.ioa_threshold = _clamp(_env_float("IOA_THRESHOLD", 0.05), 0.0, 1.0, "IOA_THRESHOLD")
        # Keep only a very short reuse window. The service is already fast enough that
        # longer reuse makes bbox disappearance feel sticky and late.
        self.flicker_reuse_time = max(0.0, _env_float("FLICKER_REUSE_TIME", 0.25))

        # Defaults fixed for your workflow: run service directly without env setup.
        self.bbox_smoothing = _clamp(_env_float("BBOX_SMOOTHING", 0.35), 0.0, 1.0, "BBOX_SMOOTHING")
        self.enable_post_nms = _env_bool("ENABLE_POST_NMS", True)
        self.post_nms_iou = _clamp(_env_float("POST_NMS_IOU", 0.6), 0.0, 1.0, "POST_NMS_IOU")
        self.match_iou_threshold = _clamp(_env_float("MATCH_IOU_THRESHOLD", 0.20), 0.0, 1.0, "MATCH_IOU_THRESHOLD")
        self.max_center_distance_ratio = _clamp(
            _env_float("MAX_CENTER_DISTANCE_RATIO", 1.2), 0.1, 10.0, "MAX_CENTER_DISTANCE_RATIO"
        )
        # Keep track creation threshold aligned with YOLO confidence by default.
        self.new_track_min_confidence = _clamp(
            _env_float("NEW_TRACK_MIN_CONFIDENCE", self.confidence_threshold),
            0.0,
            1.0,
            "NEW_TRACK_MIN_CONFIDENCE"
        )
        self.track_duplicate_iou = _clamp(
            _env_float("TRACK_DUPLICATE_IOU", 0.65), 0.0, 1.0, "TRACK_DUPLICATE_IOU"
        )
        self.track_output_hold_time = max(0.0, _env_float("TRACK_OUTPUT_HOLD_TIME", 0.5))
        self.enable_history_rematch = _env_bool("ENABLE_HISTORY_REMATCH", True)
        # Match age must be longer than real-world inference cadence; otherwise every slow frame
        # spawns new IDs and the downstream plugin sees heavy bbox flicker.
        default_match_age = max(self.track_output_hold_time, 2.5)
        self.max_match_age = max(0.0, _env_float("MAX_MATCH_AGE", default_match_age))
        self.track_min_hits = max(1, _env_int("TRACK_MIN_HITS", 1))
        self.unique_count_min_hits = max(
            1,
            _env_int("UNIQUE_COUNT_MIN_HITS", max(3, self.track_min_hits))
        )
        self.track_max_misses = max(1, _env_int("TRACK_MAX_MISSES", 6))
        self.tentative_max_misses = max(0, _env_int("TENTATIVE_MAX_MISSES", 1))
        self.track_match_min_score = _clamp(
            _env_float("TRACK_MATCH_MIN_SCORE", 0.25), 0.0, 1.0, "TRACK_MATCH_MIN_SCORE"
        )
        self.output_tentative_tracks = _env_bool("OUTPUT_TENTATIVE_TRACKS", True)
        self.output_dedupe_iou = _clamp(_env_float("OUTPUT_DEDUPE_IOU", 0.55), 0.0, 1.0, "OUTPUT_DEDUPE_IOU")
        self.hold_suppress_iou = _clamp(_env_float("HOLD_SUPPRESS_IOU", 0.4), 0.0, 1.0, "HOLD_SUPPRESS_IOU")

        self.enable_fall_detection = _env_bool("ENABLE_FALL_DETECTION", True)
        self.fall_velocity_threshold = max(0.0, _env_float("FALL_VELOCITY_THRESHOLD", 20.0))
        self.fall_angle_change_threshold = max(0.0, _env_float("FALL_ANGLE_CHANGE_THRESHOLD", 45.0))
        self.fall_aspect_ratio_threshold = max(0.0, _env_float("FALL_ASPECT_RATIO_THRESHOLD", 1.5))
        self.fall_confidence_threshold = _clamp(_env_float("FALL_CONFIDENCE_THRESHOLD", 0.8), 0.0, 1.0, "FALL_CONFIDENCE_THRESHOLD")

        self.enable_clahe = _env_bool("ENABLE_CLAHE", False)  # FIXED: Was True → Disable to speed up (was bottleneck!)
        self.enable_multi_scale = _env_bool("ENABLE_MULTI_SCALE", False)  # FIXED: Was True → Disable to speed up (was bottleneck!)
        self.enable_frame_enhancement = _env_bool("ENABLE_FRAME_ENHANCEMENT", False)
        self.clahe_clip_limit = max(0.1, _env_float("CLAHE_CLIP_LIMIT", 2.0))
        self.clahe_tile_size = max(2, _env_int("CLAHE_TILE_SIZE", 16))
        self.save_debug_samples = _env_bool("SAVE_DEBUG_SAMPLES", False)

        self.enable_roi = _env_bool("ENABLE_ROI", True)
        self.roi_type = os.getenv("ROI_TYPE", "rect").strip().lower()
        if self.roi_type not in {"rect", "polygon"}:
            logger.warning(f"Invalid ROI_TYPE='{self.roi_type}', fallback to 'rect'")
            self.roi_type = "rect"
        self.roi_x_min = _clamp(_env_float("ROI_X_MIN", 0.0), 0.0, 1.0, "ROI_X_MIN")
        self.roi_x_max = _clamp(_env_float("ROI_X_MAX", 1.0), 0.0, 1.0, "ROI_X_MAX")
        self.roi_y_min = _clamp(_env_float("ROI_Y_MIN", 0.3), 0.0, 1.0, "ROI_Y_MIN")
        self.roi_y_max = _clamp(_env_float("ROI_Y_MAX", 1.0), 0.0, 1.0, "ROI_Y_MAX")
        self.roi_polygon_json = os.getenv("ROI_POLYGON_JSON", "")

        if self.roi_x_min >= self.roi_x_max:
            logger.warning("ROI_X_MIN must be < ROI_X_MAX, fallback to [0.0, 1.0]")
            self.roi_x_min, self.roi_x_max = 0.0, 1.0
        if self.roi_y_min >= self.roi_y_max:
            logger.warning("ROI_Y_MIN must be < ROI_Y_MAX, fallback to [0.3, 1.0]")
            self.roi_y_min, self.roi_y_max = 0.3, 1.0

        self.enable_undistort = _env_bool("ENABLE_UNDISTORT", True)
        self.camera_matrix_json = os.getenv("CAMERA_MATRIX_JSON", "")
        self.distortion_coeffs_json = os.getenv("DISTORTION_COEFFS_JSON", "")
        self.calibration_file = os.getenv("CALIBRATION_FILE", "camera_calibration.json")

        self.yolo_imgsz = max(64, _env_int("YOLO_IMGSZ", 1280))

        self.device = os.getenv("DEVICE", "cuda:0" if cuda_available else "cpu")
        self.use_half = _env_bool("USE_HALF", False)
        if not self.device.startswith("cuda"):
            self.use_half = False
        self.torch_cudnn_benchmark = _env_bool("TORCH_CUDNN_BENCHMARK", True)

        # Security controls (MVP)
        self.api_key_required = _env_bool("API_KEY_REQUIRED", True)
        self.api_key = os.getenv("API_KEY", "").strip()
        if self.api_key_required and not self.api_key:
            logger.warning("API_KEY_REQUIRED=true but API_KEY is empty. Authentication is disabled until API_KEY is set.")
            self.api_key_required = False

        self.require_https = _env_bool("REQUIRE_HTTPS", False)
        self.tls_cert_file = os.getenv("TLS_CERT_FILE", "").strip()
        self.tls_key_file = os.getenv("TLS_KEY_FILE", "").strip()
        if bool(self.tls_cert_file) != bool(self.tls_key_file):
            logger.warning("TLS_CERT_FILE and TLS_KEY_FILE must be set together. Direct TLS will be disabled.")
            self.tls_cert_file = ""
            self.tls_key_file = ""
        self.rate_limit_enabled = _env_bool("RATE_LIMIT_ENABLED", True)
        self.rate_limit_window_seconds = max(1, _env_int("RATE_LIMIT_WINDOW_SECONDS", 60))
        self.rate_limit_max_per_ip = max(1, _env_int("RATE_LIMIT_MAX_PER_IP", 600))
        self.rate_limit_max_per_camera = max(1, _env_int("RATE_LIMIT_MAX_PER_CAMERA", 300))
        self.rate_limit_skip_loopback = _env_bool("RATE_LIMIT_SKIP_LOOPBACK", True)
        cors_raw = os.getenv("CORS_ALLOW_ORIGINS", "")
        self.cors_allow_origins = [v.strip() for v in cors_raw.split(",") if v.strip()]


_load_local_env()
CONFIG = AppConfig()

# Keep old names to avoid touching downstream inference logic.
SERVICE_PORT = CONFIG.service_port
SERVICE_HOST = CONFIG.service_host
MODEL_PATH = CONFIG.model_path
CONFIDENCE_THRESHOLD = CONFIG.confidence_threshold
IOU_THRESHOLD = CONFIG.iou_threshold
MIN_DETECTION_AREA = CONFIG.min_detection_area
PERSON_MIN_HW_RATIO = CONFIG.person_min_hw_ratio
TRACK_TTL = CONFIG.track_ttl
IOA_THRESHOLD = CONFIG.ioa_threshold
FLICKER_REUSE_TIME = CONFIG.flicker_reuse_time
BBOX_SMOOTHING = CONFIG.bbox_smoothing
ENABLE_POST_NMS = CONFIG.enable_post_nms
POST_NMS_IOU = CONFIG.post_nms_iou
MATCH_IOU_THRESHOLD = CONFIG.match_iou_threshold
MAX_CENTER_DISTANCE_RATIO = CONFIG.max_center_distance_ratio
NEW_TRACK_MIN_CONFIDENCE = CONFIG.new_track_min_confidence
TRACK_DUPLICATE_IOU = CONFIG.track_duplicate_iou
TRACK_OUTPUT_HOLD_TIME = CONFIG.track_output_hold_time
ENABLE_HISTORY_REMATCH = CONFIG.enable_history_rematch
MAX_MATCH_AGE = CONFIG.max_match_age
TRACK_MIN_HITS = CONFIG.track_min_hits
UNIQUE_COUNT_MIN_HITS = CONFIG.unique_count_min_hits
TRACK_MAX_MISSES = CONFIG.track_max_misses
TENTATIVE_MAX_MISSES = CONFIG.tentative_max_misses
TRACK_MATCH_MIN_SCORE = CONFIG.track_match_min_score
OUTPUT_TENTATIVE_TRACKS = CONFIG.output_tentative_tracks
OUTPUT_DEDUPE_IOU = CONFIG.output_dedupe_iou
HOLD_SUPPRESS_IOU = CONFIG.hold_suppress_iou
ENABLE_FALL_DETECTION = CONFIG.enable_fall_detection
FALL_VELOCITY_THRESHOLD = CONFIG.fall_velocity_threshold
FALL_ANGLE_CHANGE_THRESHOLD = CONFIG.fall_angle_change_threshold
FALL_ASPECT_RATIO_THRESHOLD = CONFIG.fall_aspect_ratio_threshold
FALL_CONFIDENCE_THRESHOLD = CONFIG.fall_confidence_threshold
ENABLE_CLAHE = CONFIG.enable_clahe
ENABLE_MULTI_SCALE = CONFIG.enable_multi_scale
ENABLE_FRAME_ENHANCEMENT = CONFIG.enable_frame_enhancement
CLAHE_CLIP_LIMIT = CONFIG.clahe_clip_limit
CLAHE_TILE_SIZE = CONFIG.clahe_tile_size
SAVE_DEBUG_SAMPLES = CONFIG.save_debug_samples
ENABLE_ROI = CONFIG.enable_roi
ROI_TYPE = CONFIG.roi_type
ROI_X_MIN = CONFIG.roi_x_min
ROI_X_MAX = CONFIG.roi_x_max
ROI_Y_MIN = CONFIG.roi_y_min
ROI_Y_MAX = CONFIG.roi_y_max
ROI_POLYGON_JSON = CONFIG.roi_polygon_json
ENABLE_UNDISTORT = CONFIG.enable_undistort
CAMERA_MATRIX_JSON = CONFIG.camera_matrix_json
DISTORTION_COEFFS_JSON = CONFIG.distortion_coeffs_json
CALIBRATION_FILE = CONFIG.calibration_file
YOLO_IMGSZ = CONFIG.yolo_imgsz
DEVICE = CONFIG.device
USE_HALF = CONFIG.use_half
TORCH_CUDNN_BENCHMARK = CONFIG.torch_cudnn_benchmark
API_KEY_REQUIRED = CONFIG.api_key_required
API_KEY = CONFIG.api_key
REQUIRE_HTTPS = CONFIG.require_https
TLS_CERT_FILE = CONFIG.tls_cert_file
TLS_KEY_FILE = CONFIG.tls_key_file
RATE_LIMIT_ENABLED = CONFIG.rate_limit_enabled
RATE_LIMIT_WINDOW_SECONDS = CONFIG.rate_limit_window_seconds
RATE_LIMIT_MAX_PER_IP = CONFIG.rate_limit_max_per_ip
RATE_LIMIT_MAX_PER_CAMERA = CONFIG.rate_limit_max_per_camera
RATE_LIMIT_SKIP_LOOPBACK = CONFIG.rate_limit_skip_loopback
CORS_ALLOW_ORIGINS = CONFIG.cors_allow_origins

if DEVICE.startswith("cuda"):
    try:
        torch.backends.cudnn.benchmark = TORCH_CUDNN_BENCHMARK
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

logger.info(f"="*60)
logger.info(f"YOLOv8 People Analytics Service")
logger.info(f"="*60)
logger.info(f"Port: {SERVICE_PORT}")
logger.info(f"Host: {SERVICE_HOST}")
logger.info(f"Model: {MODEL_PATH}")
logger.info(f"Confidence: {CONFIDENCE_THRESHOLD}")
logger.info(f"IOU: {IOU_THRESHOLD}")
logger.info(f"Person min h/w ratio: {PERSON_MIN_HW_RATIO}")
logger.info(f"ImgSize: {YOLO_IMGSZ}")
logger.info(f"Device: {DEVICE} | FP16: {USE_HALF}")
logger.info(f"="*60)
logger.info(f"CLAHE: {ENABLE_CLAHE}")
logger.info(f"Multi-Scale: {ENABLE_MULTI_SCALE}")
logger.info(f"Frame Enhancement: {ENABLE_FRAME_ENHANCEMENT}")
logger.info(f"ROI: {ENABLE_ROI} (type={ROI_TYPE})")
logger.info(f"Undistort: {ENABLE_UNDISTORT}")
logger.info(f"="*60)
logger.info(
    f"Tracking: post_nms={ENABLE_POST_NMS} iou={POST_NMS_IOU} "
    f"match_iou={MATCH_IOU_THRESHOLD} dup_iou={TRACK_DUPLICATE_IOU} "
    f"hold={TRACK_OUTPUT_HOLD_TIME}s match_age={MAX_MATCH_AGE}s "
    f"min_hits={TRACK_MIN_HITS} max_misses={TRACK_MAX_MISSES} "
    f"unique_min_hits={UNIQUE_COUNT_MIN_HITS} "
    f"tentative_misses={TENTATIVE_MAX_MISSES} min_match_score={TRACK_MATCH_MIN_SCORE} "
    f"out_tentative={OUTPUT_TENTATIVE_TRACKS} "
    f"out_dedupe_iou={OUTPUT_DEDUPE_IOU} hold_suppress_iou={HOLD_SUPPRESS_IOU} "
    f"history_rematch={ENABLE_HISTORY_REMATCH}"
)
logger.info(f"="*60)
logger.info(f"Fall Detection: {ENABLE_FALL_DETECTION}")
if ENABLE_FALL_DETECTION:
    logger.info(f"  Velocity Threshold: {FALL_VELOCITY_THRESHOLD}px/frame")
    logger.info(f"  Angle Change Threshold: {FALL_ANGLE_CHANGE_THRESHOLD}Â°")
    logger.info(f"  Aspect Ratio Threshold: {FALL_ASPECT_RATIO_THRESHOLD}")
    logger.info(f"  Confidence Threshold: {FALL_CONFIDENCE_THRESHOLD}")
logger.info(f"="*60)

# ============================
# Load YOLO Model
# ============================
model = None

def load_model():
    """Load YOLO model with PyTorch 2.6 compatibility"""
    global model
    if model is not None:
        return model
    
    try:
        logger.info(f"Loading YOLO model from: {MODEL_PATH}")
        
        # Monkey-patch torch.load to use weights_only=False for PyTorch 2.6+ compatibility
        original_torch_load = torch.load
        def patched_torch_load(f, *args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_torch_load(f, *args, **kwargs)
        torch.load = patched_torch_load
        
        try:
            model = YOLO(MODEL_PATH)
            try:
                model.fuse()  # fuse Conv+BN for speed
            except Exception:
                pass
            try:
                model.to(DEVICE)
            except Exception as e:
                logger.warning(f"Failed to move model to {DEVICE}: {e}")
            logger.info(f"âœ… YOLO model loaded successfully")
        finally:
            # Restore original torch.load
            torch.load = original_torch_load
            
        return model
    except Exception as e:
        logger.error(f"âŒ Failed to load YOLO model: {e}")
        raise

app = FastAPI(title="YOLOv8 Analytics Service")
if CORS_ALLOW_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ALLOW_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

# ============================
# Calibration Data (Camera-specific)
# ============================
camera_matrix = None
dist_coeffs = None

def load_calibration():
    """Load camera calibration from file or environment variables"""
    global camera_matrix, dist_coeffs
    
    # Try loading from environment variables first
    if CAMERA_MATRIX_JSON and DISTORTION_COEFFS_JSON:
        try:
            camera_matrix = np.array(json.loads(CAMERA_MATRIX_JSON))
            dist_coeffs = np.array(json.loads(DISTORTION_COEFFS_JSON))
            logger.info("âœ… Camera calibration loaded from environment variables")
            return True
        except Exception as e:
            logger.warning(f"Failed to load calibration from env: {e}")
    
    # Try loading from file
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                cal_data = json.load(f)
                camera_matrix = np.array(cal_data.get("camera_matrix", []))
                dist_coeffs = np.array(cal_data.get("distortion_coefficients", []))
                logger.info(f"âœ… Camera calibration loaded from {CALIBRATION_FILE}")
                return True
        except Exception as e:
            logger.warning(f"Failed to load calibration from file: {e}")
    
    if ENABLE_UNDISTORT:
        logger.warning("âš ï¸ Undistort enabled but no calibration data found. Create camera_calibration.json or set env variables.")
    return False

# ============================
# ROI & Undistortion Functions
# ============================
def apply_roi_rect(frame: np.ndarray) -> tuple:
    """
    Apply rectangular ROI crop.
    Returns: (cropped_frame, roi_box) where roi_box = (x1, y1, x2, y2) in original coordinates
    """
    h, w = frame.shape[:2]
    x1 = int(w * ROI_X_MIN)
    y1 = int(h * ROI_Y_MIN)
    x2 = int(w * ROI_X_MAX)
    y2 = int(h * ROI_Y_MAX)
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    cropped = frame[y1:y2, x1:x2]
    roi_box = (x1, y1, x2, y2)
    
    return cropped, roi_box

def apply_roi_polygon(frame: np.ndarray) -> tuple:
    """
    Apply polygon ROI using mask.
    Returns: (masked_frame, roi_box) where roi_box is bounding box of polygon
    """
    try:
        if not ROI_POLYGON_JSON:
            logger.warning("Polygon ROI enabled but ROI_POLYGON_JSON not provided")
            return frame, (0, 0, frame.shape[1], frame.shape[0])
        
        polygon = json.loads(ROI_POLYGON_JSON)
        h, w = frame.shape[:2]
        
        # Convert percentages to pixels
        points = np.array([
            [int(p[0] * w), int(p[1] * h)] for p in polygon
        ], dtype=np.int32)
        
        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], (255,))
        
        # Apply mask
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Get bounding box
        x_min = int(min(p[0] for p in polygon) * w)
        y_min = int(min(p[1] for p in polygon) * h)
        x_max = int(max(p[0] for p in polygon) * w)
        y_max = int(max(p[1] for p in polygon) * h)
        roi_box = (max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max))
        
        return result, roi_box
    except Exception as e:
        logger.warning(f"Polygon ROI error: {e}, using full frame")
        return frame, (0, 0, frame.shape[1], frame.shape[0])

def apply_roi(frame: np.ndarray) -> tuple:
    """
    Apply ROI (Region of Interest) crop.
    Returns: (cropped_frame, roi_box, roi_type)
    """
    if not ENABLE_ROI:
        h, w = frame.shape[:2]
        return frame, (0, 0, w, h), None
    
    if ROI_TYPE == "polygon":
        return apply_roi_polygon(frame) + (ROI_TYPE,)
    else:  # rect (default)
        return apply_roi_rect(frame) + (ROI_TYPE,)

def undistort_frame(frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Apply undistortion to wide-angle frame using camera calibration.
    Straightens curved lines at edges caused by wide-angle lens.
    Returns:
      - processed frame
      - crop offset (x, y) relative to undistorted full-size canvas
    """
    if not ENABLE_UNDISTORT or camera_matrix is None or dist_coeffs is None:
        return frame, (0, 0)
    
    try:
        h, w = frame.shape[:2]
        
        # Get optimal camera matrix (slightly crop to avoid black borders)
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 0.9
        )
        
        # Undistort
        result = cv2.undistort(frame, camera_matrix, dist_coeffs, newCameraMatrix=new_camera_matrix)
        
        # Crop to ROI if needed (remove black borders)
        x, y, w_roi, h_roi = roi
        crop_offset = (0, 0)
        if x > 0 or y > 0 or w_roi < w or h_roi < h:
            result = result[y:y+h_roi, x:x+w_roi]
            crop_offset = (int(x), int(y))

        return result, crop_offset
    except Exception as e:
        logger.warning(f"Undistort error: {e}, using original frame")
        return frame, (0, 0)

def remap_detections_to_input_space(
    detections: List["Detection"],
    roi_box: Optional[Tuple[float, float, float, float]],
    roi_type: Optional[str],
    pre_roi_shape: Tuple[int, ...],
    undistort_crop_offset: Tuple[int, int],
) -> None:
    """
    Map detections from ROI/undistort-processed frame back to input frame coordinates.
    Notes:
      - ROI offset is applied only for rectangular ROI crop.
      - Polygon ROI keeps full-frame size, so no ROI offset is applied.
      - Undistort crop offset is added when undistort removed black borders.
    """
    x_add, y_add = 0.0, 0.0

    if roi_box and roi_type == "rect":
        full_roi_box = (0, 0, pre_roi_shape[1], pre_roi_shape[0])
        if roi_box != full_roi_box:
            x_add += float(roi_box[0])
            y_add += float(roi_box[1])

    x_add += float(undistort_crop_offset[0])
    y_add += float(undistort_crop_offset[1])

    if x_add == 0.0 and y_add == 0.0:
        return

    for det in detections:
        det.x += x_add
        det.y += y_add

def scale_detections_to_original(detections: List[Dict], roi_box: tuple, original_shape: tuple) -> List[Dict]:
    """
    Scale detection coordinates from ROI-cropped frame back to original frame.
    roi_box = (x1, y1, x2, y2) in original frame coordinates
    """
    if roi_box == (0, 0, original_shape[1], original_shape[0]):
        # No ROI applied, no scaling needed
        return detections
    
    x_offset, y_offset, _, _ = roi_box
    
    for det in detections:
        det["x"] += x_offset
        det["y"] += y_offset
    
    return detections

# ============================
# Helper Functions
# ============================
def extract_appearance(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> dict:
    """Extract appearance features (color histogram) from detection region"""
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
    
    if x2 <= x1 or y2 <= y1:
        return {"color_hist": None}
    
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return {"color_hist": None}
    
    # Compute color histogram
    hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return {"color_hist": hist}

def appearance_distance(hist1, hist2) -> float:
    """
    Compute distance between two color histograms (0-1, lower = more similar)
    Using Bhattacharyya distance
    """
    if hist1 is None or hist2 is None:
        return 1.0
    try:
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    except:
        return 1.0

def combined_track_score(iou_score: float, appearance_dist: float) -> float:
    """
    Combine IoU and appearance distance into single score (0-1, higher = better match)
    When IoU is low (person far from last position), rely more on appearance
    """
    # Appearance distance to similarity (inverse)
    app_similarity = 1.0 - appearance_dist
    
    # Adaptive weighting: when IoU is low, rely more on appearance
    if iou_score < 0.1:
        # Person walked away: 50% IoU, 50% appearance
        combined = 0.5 * iou_score + 0.5 * app_similarity
    elif iou_score < 0.3:
        # Person moved: 60% IoU, 40% appearance
        combined = 0.6 * iou_score + 0.4 * app_similarity
    else:
        # Person nearby: 70% IoU, 30% appearance
        combined = 0.7 * iou_score + 0.3 * app_similarity
    
    return combined

# ============================
# Auto Brightness Adjustment
# ============================
def auto_adjust_brightness(frame: np.ndarray, target_brightness: float = 190.0) -> np.ndarray:
    """
    Automatically adjust frame brightness to optimal level for detection.
    Uses aggressive CLAHE + direct scaling for dark BGR frames from NX plugin.
    
    Args:
        frame: Input BGR frame
        target_brightness: Target mean brightness (default 190 for YOLO)
    
    Returns:
        Brightness-adjusted frame
    """
    try:
        current_brightness = np.mean(frame)
        
        # Aggressive adjustment: NX plugin sends dark frames (mean ~130)
        # We need to boost to ~190 for optimal YOLO detection
        if current_brightness < target_brightness:
            # Step 1: Apply aggressive CLAHE for contrast
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Very aggressive CLAHE for dark frames
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            l_clahe = clahe.apply(l)
            
            # Step 2: Additional direct scaling on L channel
            l_mean = float(np.mean(np.asarray(l_clahe, dtype=np.float32)))
            if l_mean < target_brightness:
                scale_factor = target_brightness / max(l_mean, 5)
                scale_factor = min(scale_factor, 3.0)  # Cap at 3.0x
                l_clahe = (l_clahe.astype(np.float32) * scale_factor).clip(0, 255).astype(np.uint8)
            
            # Merge back to BGR
            lab_adjusted = cv2.merge([l_clahe, a, b])
            frame_adjusted = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)
            
            new_brightness = float(np.mean(np.asarray(frame_adjusted, dtype=np.float32)))
            if new_brightness >= target_brightness * 0.95:  # Close enough to target
                logger.debug(f"Brightness corrected: {current_brightness:.1f} â†’ {new_brightness:.1f}")
                return frame_adjusted
        
        return frame
    except Exception as e:
        logger.warning(f"Brightness adjustment error: {e}")
        return frame
def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Great for wide-angle cameras with uneven lighting.
    """
    try:
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE only to L channel
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE)
        )
        l_clahe = clahe.apply(l)

        # Merge back
        lab_clahe = cv2.merge([l_clahe, a, b])
        frame_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        return frame_clahe
    except Exception as e:
        logger.warning(f"CLAHE failed: {e}, returning original frame")
        return frame

def apply_frame_enhancement(frame: np.ndarray) -> np.ndarray:
    """
    Enhance frame for better detection:
    - Slight blur to reduce noise
    - Gamma correction for brightness
    - Sharpening
    """
    try:
        # Noise reduction
        frame = cv2.bilateralFilter(frame, 9, 75, 75)
        
        # Gamma correction (brighten dark areas slightly)
        gamma = 1.1
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        frame = cv2.LUT(frame, table)
        
        # Unsharp mask for sharpening
        gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
        frame = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
        
        return frame
    except Exception as e:
        logger.warning(f"Frame enhancement failed: {e}, returning original frame")
        return frame

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Preprocess frame for better detection on wide-angle cameras.
    Apply optional CLAHE and enhancement based on config.
    """
    if ENABLE_CLAHE:
        frame = apply_clahe(frame)
    
    if ENABLE_FRAME_ENHANCEMENT:
        frame = apply_frame_enhancement(frame)
    
    return frame

def multi_scale_inference_smart(yolo_model, frame: np.ndarray, original_h: int, original_w: int):
    """
    SMART multi-scale inference: Only retry at larger scale if no detections found.
    
    Strategy:
    - Normal: Run at 1.0x scale (fast, good for nearby people)
    - If NO detections: Retry at 1.25x scale (catches small/distant people)
    - Goal: Keep speed while catching small objects
    
    This avoids running 3 scales every frame (3x slower).
    """
    class FilteredResult:
        def __init__(self, boxes=None):
            self.boxes = boxes

    class ScaledBox:
        """Minimal box wrapper compatible with downstream parsing logic."""
        def __init__(self, xyxy: torch.Tensor, conf: torch.Tensor, cls: torch.Tensor):
            self.xyxy = xyxy
            self.conf = conf
            self.cls = cls
    
    # Primary inference at 1.0x scale
    r = yolo_model.predict(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        classes=[0],  # person only
        imgsz=YOLO_IMGSZ,
        verbose=False,
        augment=False,
        device=DEVICE,
        half=USE_HALF,
    )[0]
    
    # If detections found, return them
    if r.boxes is not None and len(r.boxes) > 0:
        return FilteredResult(r.boxes)
    
    # No detections: Retry at larger scale to catch small/distant people
    logger.debug(f"No detections at 1.0x, retrying at 1.25x scale")
    h, w = int(original_h * 1.25), int(original_w * 1.25)
    scaled_frame = cv2.resize(frame, (w, h))
    
    r_scaled = yolo_model.predict(
        scaled_frame,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        classes=[0],  # person only
        imgsz=YOLO_IMGSZ,
        verbose=False,
        augment=False,
        device=DEVICE,
        half=USE_HALF,
    )[0]
    
    # Scale boxes back to original size (avoid in-place writes on inference tensors)
    if r_scaled.boxes is not None and len(r_scaled.boxes) > 0:
        scaled_boxes: List[ScaledBox] = []
        for box in r_scaled.boxes:
            xyxy = box.xyxy.detach().clone() / 1.25
            conf = box.conf.detach().clone()
            cls = box.cls.detach().clone()
            scaled_boxes.append(ScaledBox(xyxy=xyxy, conf=conf, cls=cls))
        return FilteredResult(scaled_boxes)
    
    return FilteredResult()

# ============================
# Pydantic Models
# ============================
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

# ============================
# STATE THEO CAMERA (Multi-camera support)
# ============================
camera_states: Dict[str, Dict[str, Any]] = {}
camera_states_lock = threading.RLock()

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

def get_camera_state(camera_id: str) -> Dict[str, Any]:
    """Get or create camera state"""
    with camera_states_lock:
        if camera_id not in camera_states:
            logger.info(f"[CAMERA] Initializing new camera: {camera_id}")
            camera_states[camera_id] = {
                "lock": threading.RLock(),
                "tracks": [],
                "track_history": [],  # NEW: Keep history of old tracks for re-matching
                "next_id": 1,
                "seen_ids": set(),
                "total_count": 0,
                "last_output": [],
                "last_time": 0.0,
                "inference_times": [],  # Track inference performance
                "created_at": time.time(),
                "fall_detector": FallDetectionManager(  # NEW: Fall detection manager
                    velocity_threshold=FALL_VELOCITY_THRESHOLD,
                    angle_change_threshold=FALL_ANGLE_CHANGE_THRESHOLD,
                    aspect_ratio_threshold=FALL_ASPECT_RATIO_THRESHOLD,
                    confidence_threshold=FALL_CONFIDENCE_THRESHOLD,
                ) if ENABLE_FALL_DETECTION else None,
            }
        return camera_states[camera_id]

def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    return inter / (area_a + area_b - inter + 1e-6)

def center_distance_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """
    Center distance normalized by average bbox diagonal.
    Lower is better (0 = same center).
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    acx = 0.5 * (ax1 + ax2)
    acy = 0.5 * (ay1 + ay2)
    bcx = 0.5 * (bx1 + bx2)
    bcy = 0.5 * (by1 + by2)
    dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5

    aw = max(1.0, ax2 - ax1)
    ah = max(1.0, ay2 - ay1)
    bw = max(1.0, bx2 - bx1)
    bh = max(1.0, by2 - by1)
    diag_a = (aw * aw + ah * ah) ** 0.5
    diag_b = (bw * bw + bh * bh) ** 0.5
    norm = max(1.0, 0.5 * (diag_a + diag_b))
    return dist / norm

def smooth_bbox(
    old_bbox: tuple[float, float, float, float],
    new_bbox: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    """Exponential smoothing with adaptive responsiveness for moving targets."""
    if alpha <= 0.0:
        return old_bbox
    if alpha >= 1.0:
        return new_bbox

    move_ratio = center_distance_ratio(old_bbox, new_bbox)
    adaptive_alpha = min(0.97, alpha + min(0.50, move_ratio * 0.8))

    ox1, oy1, ox2, oy2 = old_bbox
    nx1, ny1, nx2, ny2 = new_bbox
    return (
        ox1 * (1.0 - adaptive_alpha) + nx1 * adaptive_alpha,
        oy1 * (1.0 - adaptive_alpha) + ny1 * adaptive_alpha,
        ox2 * (1.0 - adaptive_alpha) + nx2 * adaptive_alpha,
        oy2 * (1.0 - adaptive_alpha) + ny2 * adaptive_alpha,
    )

def post_nms_dedupe(dets: List[Dict[str, Any]], iou_thresh: float) -> List[Dict[str, Any]]:
    """
    Extra IoU-based dedupe after YOLO NMS to avoid overlapping boxes for the same person.
    Keeps highest-score boxes.
    """
    if len(dets) <= 1:
        return dets
    dets_sorted = sorted(dets, key=lambda d: d["score"], reverse=True)
    kept: List[Dict[str, Any]] = []
    for det in dets_sorted:
        if all(iou(det["bbox"], k["bbox"]) < iou_thresh for k in kept):
            kept.append(det)
    return kept

def has_duplicate_track_overlap(
    det_box: tuple[float, float, float, float],
    track_by_id: Dict[int, Dict[str, Any]],
    now_ts: float,
    overlap_iou: float,
    recent_only_sec: float,
) -> bool:
    """Prevent creating a new track for the same person when detections are duplicated."""
    for tr in track_by_id.values():
        last_seen = tr.get("last_seen", 0.0)
        if now_ts - last_seen > recent_only_sec:
            continue
        if iou(det_box, tr["bbox"]) >= overlap_iou:
            return True
    return False

def predict_track_bbox(track: Dict[str, Any], now_ts: float) -> tuple[float, float, float, float]:
    """Predict next bbox from constant velocity model in xyxy space."""
    bbox = track.get("measurement_bbox") or track.get("bbox")
    if not bbox:
        return (0.0, 0.0, 0.0, 0.0)

    dt = max(0.0, now_ts - float(track.get("last_seen", now_ts)))
    if dt <= 1e-6:
        return bbox

    dt = min(dt, 1.0)
    vx = float(track.get("vx", 0.0))
    vy = float(track.get("vy", 0.0))
    vw = float(track.get("vw", 0.0))
    vh = float(track.get("vh", 0.0))

    x1, y1, x2, y2 = bbox
    pred = (x1 + vx * dt, y1 + vy * dt, x2 + vw * dt, y2 + vh * dt)
    px1, py1, px2, py2 = pred
    if px2 <= px1 or py2 <= py1:
        return bbox
    return pred


def update_track_motion(track: Dict[str, Any], prev_bbox: tuple[float, float, float, float], new_bbox: tuple[float, float, float, float], now_ts: float) -> None:
    """Update track velocity estimates for next-frame prediction."""
    prev_ts = float(track.get("last_seen", now_ts))
    dt = max(1e-3, now_ts - prev_ts)

    px1, py1, px2, py2 = prev_bbox
    nx1, ny1, nx2, ny2 = new_bbox

    track["vx"] = (nx1 - px1) / dt
    track["vy"] = (ny1 - py1) / dt
    track["vw"] = (nx2 - px2) / dt
    track["vh"] = (ny2 - py2) / dt


def hungarian_minimize(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """Solve rectangular minimum-cost assignment using Hungarian algorithm."""
    if cost_matrix.size == 0:
        return []

    transposed = False
    cost = cost_matrix
    if cost.shape[0] > cost.shape[1]:
        cost = cost.T
        transposed = True

    n_rows, n_cols = cost.shape
    u = np.zeros(n_rows + 1, dtype=np.float64)
    v = np.zeros(n_cols + 1, dtype=np.float64)
    p = np.zeros(n_cols + 1, dtype=np.int64)
    way = np.zeros(n_cols + 1, dtype=np.int64)

    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, np.inf, dtype=np.float64)
        used = np.zeros(n_cols + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0

            for j in range(1, n_cols + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j in range(0, n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignments: List[Tuple[int, int]] = []
    for j in range(1, n_cols + 1):
        if p[j] == 0:
            continue
        row = p[j] - 1
        col = j - 1
        if transposed:
            assignments.append((col, row))
        else:
            assignments.append((row, col))
    return assignments


def global_track_assignment(
    raw_dets: List[Dict[str, Any]],
    track_by_id: Dict[int, Dict[str, Any]],
    frame: np.ndarray,
    now_ts: float,
) -> tuple:
    """
    Global 1-1 assignment with motion prediction + Hungarian optimization.
    Returns det_idx -> track_id for accepted pairs only.
    """
    det_appearances: List[Dict[str, Any]] = [extract_appearance(frame, det["bbox"]) for det in raw_dets]

    if not raw_dets or not track_by_id:
        return {}, det_appearances

    track_candidates: List[Dict[str, Any]] = []
    for tr in track_by_id.values():
        if now_ts - tr.get("last_seen", 0.0) > MAX_MATCH_AGE:
            continue
        if int(tr.get("misses", 0)) > TRACK_MAX_MISSES:
            continue
        tr["pred_bbox"] = predict_track_bbox(tr, now_ts)
        track_candidates.append(tr)

    if not track_candidates:
        return {}, det_appearances

    invalid_cost = 1e6
    cost = np.full((len(raw_dets), len(track_candidates)), invalid_cost, dtype=np.float64)

    for det_idx, det in enumerate(raw_dets):
        det_box = det["bbox"]
        det_app = det_appearances[det_idx]

        for track_idx, tr in enumerate(track_candidates):
            pred_bbox_raw = tr.get("pred_bbox") or tr.get("bbox")
            if not pred_bbox_raw or len(pred_bbox_raw) != 4:
                continue
            pred_bbox: tuple[float, float, float, float] = (
                float(pred_bbox_raw[0]),
                float(pred_bbox_raw[1]),
                float(pred_bbox_raw[2]),
                float(pred_bbox_raw[3]),
            )
            iou_score = iou(det_box, pred_bbox)
            center_ratio = center_distance_ratio(det_box, pred_bbox)

            if iou_score < MATCH_IOU_THRESHOLD or center_ratio > MAX_CENTER_DISTANCE_RATIO:
                continue

            app_dist = 1.0
            if tr.get("appearance") and det_app.get("color_hist") is not None:
                tr_hist = tr["appearance"].get("color_hist")
                if tr_hist is not None:
                    app_dist = appearance_distance(tr_hist, det_app["color_hist"])

            match_score = combined_track_score(iou_score, app_dist)
            if match_score < TRACK_MATCH_MIN_SCORE:
                continue

            cost[det_idx, track_idx] = 1.0 - match_score

    assignments = hungarian_minimize(cost)

    matches: Dict[int, int] = {}
    for det_idx, track_idx in assignments:
        if det_idx >= cost.shape[0] or track_idx >= cost.shape[1]:
            continue
        if cost[det_idx, track_idx] >= invalid_cost:
            continue
        track_id = int(track_candidates[track_idx]["id"])
        matches[det_idx] = track_id

    return matches, det_appearances

def suppress_overlapping_holds(
    hold_track: Dict[str, Any],
    output_track_ids: set,
    track_by_id: Dict[int, Dict[str, Any]],
    overlap_iou: float,
) -> bool:
    """Skip held tracks that heavily overlap tracks already selected for current output."""
    for tid in output_track_ids:
        tr = track_by_id.get(tid)
        if not tr or tr.get("id") == hold_track.get("id"):
            continue
        if iou(hold_track["bbox"], tr["bbox"]) >= overlap_iou:
            return True
    return False

def dedupe_output_tracks(detections: List["Detection"], iou_thresh: float) -> List["Detection"]:
    """Final output dedupe to reduce overlapping duplicate boxes."""
    if len(detections) <= 1:
        return detections

    sorted_dets = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: List["Detection"] = []
    for det in sorted_dets:
        det_box = (det.x, det.y, det.x + det.w, det.y + det.h)
        if all(iou(det_box, (k.x, k.y, k.x + k.w, k.y + k.h)) < iou_thresh for k in kept):
            kept.append(det)
    return kept

# ============================
# Health Check Endpoint
# ============================
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
                track_id=int(track_id)
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

            # Plugin-local tracking is now authoritative for bbox continuity. If the service-side
            # tracker filters everything out but YOLO still sees a person, fall back to raw YOLO
            # boxes so Nx still receives a usable person bbox.
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
                    track_id=int(best_track_id)
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
                    fall_input_detections.append({
                        'track_id': det.track_id,
                        'bbox': (det.x, det.y, det.x + det.w, det.y + det.h),
                        'confidence': det.score,
                    })
                
                # Update fall detector with current frame detections
                with camera_lock:
                    fall_results = fall_detector.update(fall_input_detections, req_seq)
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
            # 6) Count tracking
            # ============================================
            stable_unique_ids = {
                int(track_id)
                for track_id in output_track_ids
                if track_id in track_by_id
                and track_by_id[track_id].get("status") == "confirmed"
                and int(track_by_id[track_id].get("hits", 0)) >= UNIQUE_COUNT_MIN_HITS
            }
            state["seen_ids"].update(stable_unique_ids)

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
        state["tracks"].clear()
        state["track_history"].clear()
        state["next_id"] = 1
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
            state["tracks"].clear()
            state["track_history"].clear()
            state["next_id"] = 1
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

# ============================
# Main
# ============================
if __name__ == "__main__":
    logger.info(f"Starting service on {SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn_kwargs: Dict[str, Any] = {
        "app": app,
        "host": SERVICE_HOST,
        "port": SERVICE_PORT,
        "log_level": "info",
        "access_log": True,
    }
    if TLS_CERT_FILE and TLS_KEY_FILE:
        uvicorn_kwargs["ssl_certfile"] = TLS_CERT_FILE
        uvicorn_kwargs["ssl_keyfile"] = TLS_KEY_FILE
        logger.info("Using direct TLS via uvicorn ssl_certfile/ssl_keyfile")
    uvicorn.run(**uvicorn_kwargs)







