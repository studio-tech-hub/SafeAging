import base64
import json
import time
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
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
        self.confidence_threshold = _clamp(_env_float("CONFIDENCE_THRESHOLD", 0.20), 0.0, 1.0, "CONFIDENCE_THRESHOLD")
        self.iou_threshold = _clamp(_env_float("IOU_THRESHOLD", 0.45), 0.0, 1.0, "IOU_THRESHOLD")
        self.min_detection_area = max(1, _env_int("MIN_DETECTION_AREA", 20))
        self.track_ttl = max(1.0, _env_float("TRACK_TTL", 15.0))
        self.ioa_threshold = _clamp(_env_float("IOA_THRESHOLD", 0.05), 0.0, 1.0, "IOA_THRESHOLD")
        self.flicker_reuse_time = max(0.0, _env_float("FLICKER_REUSE_TIME", 1.0))

        # Defaults fixed for your workflow: run service directly without env setup.
        self.bbox_smoothing = _clamp(_env_float("BBOX_SMOOTHING", 0.6), 0.0, 1.0, "BBOX_SMOOTHING")
        self.enable_post_nms = _env_bool("ENABLE_POST_NMS", True)
        self.post_nms_iou = _clamp(_env_float("POST_NMS_IOU", 0.6), 0.0, 1.0, "POST_NMS_IOU")
        self.match_iou_threshold = _clamp(_env_float("MATCH_IOU_THRESHOLD", 0.20), 0.0, 1.0, "MATCH_IOU_THRESHOLD")
        self.max_center_distance_ratio = _clamp(
            _env_float("MAX_CENTER_DISTANCE_RATIO", 1.2), 0.1, 10.0, "MAX_CENTER_DISTANCE_RATIO"
        )
        self.new_track_min_confidence = _clamp(
            _env_float("NEW_TRACK_MIN_CONFIDENCE", 0.35), 0.0, 1.0, "NEW_TRACK_MIN_CONFIDENCE"
        )
        self.track_duplicate_iou = _clamp(
            _env_float("TRACK_DUPLICATE_IOU", 0.65), 0.0, 1.0, "TRACK_DUPLICATE_IOU"
        )
        self.track_output_hold_time = max(0.0, _env_float("TRACK_OUTPUT_HOLD_TIME", 0.8))
        self.enable_history_rematch = _env_bool("ENABLE_HISTORY_REMATCH", False)

        self.enable_fall_detection = _env_bool("ENABLE_FALL_DETECTION", True)
        self.fall_velocity_threshold = max(0.0, _env_float("FALL_VELOCITY_THRESHOLD", 20.0))
        self.fall_angle_change_threshold = max(0.0, _env_float("FALL_ANGLE_CHANGE_THRESHOLD", 45.0))
        self.fall_aspect_ratio_threshold = max(0.0, _env_float("FALL_ASPECT_RATIO_THRESHOLD", 1.5))
        self.fall_confidence_threshold = _clamp(_env_float("FALL_CONFIDENCE_THRESHOLD", 0.8), 0.0, 1.0, "FALL_CONFIDENCE_THRESHOLD")

        self.enable_clahe = _env_bool("ENABLE_CLAHE", True)
        self.enable_multi_scale = _env_bool("ENABLE_MULTI_SCALE", True)
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


_load_local_env()
CONFIG = AppConfig()

# Keep old names to avoid touching downstream inference logic.
SERVICE_PORT = CONFIG.service_port
SERVICE_HOST = CONFIG.service_host
MODEL_PATH = CONFIG.model_path
CONFIDENCE_THRESHOLD = CONFIG.confidence_threshold
IOU_THRESHOLD = CONFIG.iou_threshold
MIN_DETECTION_AREA = CONFIG.min_detection_area
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
    f"hold={TRACK_OUTPUT_HOLD_TIME}s history_rematch={ENABLE_HISTORY_REMATCH}"
)
logger.info(f"="*60)
logger.info(f"Fall Detection: {ENABLE_FALL_DETECTION}")
if ENABLE_FALL_DETECTION:
    logger.info(f"  Velocity Threshold: {FALL_VELOCITY_THRESHOLD}px/frame")
    logger.info(f"  Angle Change Threshold: {FALL_ANGLE_CHANGE_THRESHOLD}°")
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
            logger.info(f"✅ YOLO model loaded successfully")
        finally:
            # Restore original torch.load
            torch.load = original_torch_load
            
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load YOLO model: {e}")
        raise

app = FastAPI(title="YOLOv8 Analytics Service")

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
            logger.info("✅ Camera calibration loaded from environment variables")
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
                logger.info(f"✅ Camera calibration loaded from {CALIBRATION_FILE}")
                return True
        except Exception as e:
            logger.warning(f"Failed to load calibration from file: {e}")
    
    if ENABLE_UNDISTORT:
        logger.warning("⚠️ Undistort enabled but no calibration data found. Create camera_calibration.json or set env variables.")
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
        cv2.fillPoly(mask, [points], 255)
        
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

def undistort_frame(frame: np.ndarray) -> np.ndarray:
    """
    Apply undistortion to wide-angle frame using camera calibration.
    Straightens curved lines at edges caused by wide-angle lens.
    """
    if not ENABLE_UNDISTORT or camera_matrix is None or dist_coeffs is None:
        return frame
    
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
        if x > 0 or y > 0 or w_roi < w or h_roi < h:
            result = result[y:y+h_roi, x:x+w_roi]
        
        return result
    except Exception as e:
        logger.warning(f"Undistort error: {e}, using original frame")
        return frame

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
def extract_appearance(frame: np.ndarray, bbox: tuple) -> dict:
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
            l_mean = np.mean(l_clahe)
            if l_mean < target_brightness:
                scale_factor = target_brightness / max(l_mean, 5)
                scale_factor = min(scale_factor, 3.0)  # Cap at 3.0x
                l_clahe = (l_clahe.astype(np.float32) * scale_factor).clip(0, 255).astype(np.uint8)
            
            # Merge back to BGR
            lab_adjusted = cv2.merge([l_clahe, a, b])
            frame_adjusted = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)
            
            new_brightness = np.mean(frame_adjusted)
            if new_brightness >= target_brightness * 0.95:  # Close enough to target
                logger.debug(f"Brightness corrected: {current_brightness:.1f} → {new_brightness:.1f}")
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
    
    # Scale boxes back to original size
    if r_scaled.boxes is not None:
        for box in r_scaled.boxes:
            box.xyxy[0] = box.xyxy[0] / 1.25
        return FilteredResult(r_scaled.boxes)
    
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

# ============================
# STATE THEO CAMERA (Multi-camera support)
# ============================
camera_states: Dict[str, Dict[str, Any]] = {}

def get_camera_state(camera_id: str) -> Dict[str, Any]:
    """Get or create camera state"""
    if camera_id not in camera_states:
        logger.info(f"[CAMERA] Initializing new camera: {camera_id}")
        camera_states[camera_id] = {
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

def iou(a, b) -> float:
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

def center_distance_ratio(a, b) -> float:
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

def smooth_bbox(old_bbox: tuple, new_bbox: tuple, alpha: float) -> tuple:
    """Exponential smoothing for bbox to reduce jitter (alpha = weight of new bbox)."""
    if alpha <= 0.0:
        return old_bbox
    if alpha >= 1.0:
        return new_bbox
    ox1, oy1, ox2, oy2 = old_bbox
    nx1, ny1, nx2, ny2 = new_bbox
    return (
        ox1 * (1.0 - alpha) + nx1 * alpha,
        oy1 * (1.0 - alpha) + ny1 * alpha,
        ox2 * (1.0 - alpha) + nx2 * alpha,
        oy2 * (1.0 - alpha) + ny2 * alpha,
    )

def post_nms_dedupe(dets: List[Dict[str, float]], iou_thresh: float) -> List[Dict[str, float]]:
    """
    Extra IoU-based dedupe after YOLO NMS to avoid overlapping boxes for the same person.
    Keeps highest-score boxes.
    """
    if len(dets) <= 1:
        return dets
    dets_sorted = sorted(dets, key=lambda d: d["score"], reverse=True)
    kept: List[Dict[str, float]] = []
    for det in dets_sorted:
        if all(iou(det["bbox"], k["bbox"]) < iou_thresh for k in kept):
            kept.append(det)
    return kept

def has_duplicate_track_overlap(
    det_box: tuple,
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
def infer(req: InferRequest):
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
    global request_counter, error_counter
    request_counter += 1
    start_time = time.time()
    
    camera_id = req.camera_id or "default"
    
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
                error_counter += 1
                logger.warning(f"[{camera_id}] Failed to decode image - got None")
                return []
                
        except Exception as e:
            error_counter += 1
            logger.warning(f"[{camera_id}] Image decode error: {type(e).__name__}: {e}")
            return []

        H, W = frame.shape[:2]
        
        # Debug: Check if frame is mostly empty/dark
        frame_mean = np.mean(frame)
        frame_max = np.max(frame)
        frame_min = np.min(frame)
        if request_counter % 20 == 0:
            logger.info(f"[{camera_id}] Frame: {W}x{H}, mean={frame_mean:.1f}, min={frame_min}, max={frame_max}")
            if SAVE_DEBUG_SAMPLES:
                # Save frame to disk only when explicitly enabled via env var.
                try:
                    import os
                    os.makedirs('frame_samples', exist_ok=True)
                    sample_path = f'frame_samples/frame_{request_counter:06d}.jpg'
                    cv2.imwrite(sample_path, frame)
                    logger.info(f"[{camera_id}] Saved frame sample to {sample_path}")
                except Exception as e:
                    logger.warning(f"Failed to save frame sample: {e}")
        logger.debug(f"[{camera_id}] Frame size: {W}x{H}")

        # Store original frame for coordinate mapping
        frame_original = frame.copy()
        original_shape = frame.shape
        
        # ============================================
        # 1.4) Undistortion (for wide-angle/fisheye cameras)
        # ============================================
        if ENABLE_UNDISTORT:
            undistort_start = time.time()
            frame = undistort_frame(frame)
            undistort_time = time.time() - undistort_start
            if request_counter % 20 == 0:
                logger.info(f"[{camera_id}] Undistortion: {undistort_time*1000:.1f}ms")

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
            if request_counter % 20 == 0:
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
            if request_counter % 20 == 0:
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
            num_boxes = len(r.boxes) if r.boxes is not None else 0
            if request_counter % 20 == 0:
                logger.info(f"[{camera_id}] YOLO: {num_boxes} objects (conf={CONFIDENCE_THRESHOLD}, inference={inference_time_ms:.1f}ms)")
                if num_boxes > 0:
                    for i, box in enumerate(r.boxes[:3]):  # Show first 3
                        logger.info(f"  Box {i}: cls={int(box.cls[0].item())}, conf={float(box.conf[0].item()):.2f}")
        except Exception as e:
            error_counter += 1
            logger.error(f"[{camera_id}] YOLO inference error: {type(e).__name__}: {e}")
            return []

        now = time.time()
        state = get_camera_state(camera_id)
        tracks = state["tracks"]
        next_id = state["next_id"]

        detections: List[Detection] = []

        # ============================================
        # 3) Process YOLO outputs with tracking
        # ============================================
        # Build raw detections first (for optional extra NMS)
        raw_dets: List[Dict[str, float]] = []
        for box in r.boxes:
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

                # Filter by minimum area
                if w_box <= 1.0 or h_box <= 1.0 or area < MIN_DETECTION_AREA:
                    logger.debug(f"[{camera_id}] Skipping small detection: {w_box:.1f}x{h_box:.1f} (area={area:.0f} < {MIN_DETECTION_AREA})")
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
        matched_ids = set()
        output_track_ids = set()

        for det in raw_dets:
            try:
                det_box = det["bbox"]
                score = det["score"]
                x1, y1, x2, y2 = det_box
                w_box = x2 - x1
                h_box = y2 - y1

                # ============================================
                # Track matching: find best match using IoU + Appearance
                # ============================================
                det_appearance = extract_appearance(frame, det_box)
                best_score, best_tr = -1.0, None

                # FIRST: Try to match against active tracks
                for tr in track_by_id.values():
                    if tr.get("id") in matched_ids:
                        continue
                    iou_score = iou(det_box, tr["bbox"])
                    center_ratio = center_distance_ratio(det_box, tr["bbox"])

                    # Hard gating to avoid cross-matching close objects with wrong IDs.
                    if iou_score < MATCH_IOU_THRESHOLD and center_ratio > MAX_CENTER_DISTANCE_RATIO:
                        continue

                    # Always try to compare appearance if we have it
                    if tr.get("appearance") and det_appearance["color_hist"] is not None:
                        app_dist = appearance_distance(tr["appearance"]["color_hist"], det_appearance["color_hist"])
                        app_similarity = 1.0 - app_dist
                        match_score = 0.85 * iou_score + 0.15 * app_similarity
                    else:
                        # Fall back to IoU only
                        match_score = iou_score

                    if match_score > best_score:
                        best_score, best_tr = match_score, tr

                # SECOND: If no good active track match, search track history
                # Disabled by default for stability (can be enabled by env if needed).
                if ENABLE_HISTORY_REMATCH and (best_tr is None or best_score < MATCH_IOU_THRESHOLD) and state["track_history"]:
                    for hist_tr in state["track_history"]:
                        if hist_tr.get("id") in matched_ids:
                            continue
                        # Only match if appearance is similar enough (less reliance on position)
                        if hist_tr.get("appearance") and det_appearance["color_hist"] is not None:
                            app_dist = appearance_distance(hist_tr["appearance"]["color_hist"], det_appearance["color_hist"])
                            # Person-like appearance match = likely same person returning
                            if app_dist < 0.4:  # More lenient: similar appearance
                                hist_score = 1.0 - app_dist  # Appearance-based score (0-1)
                                if hist_score > best_score:
                                    best_score, best_tr = hist_score, hist_tr
                                    logger.debug(f"[{camera_id}] Re-matched track ID={hist_tr['id']} from history (app_dist={app_dist:.2f})")

                match_threshold = MATCH_IOU_THRESHOLD

                if best_tr is not None and best_score >= match_threshold and best_tr.get("id") not in matched_ids:
                    # Existing track: update position and appearance (with smoothing)
                    track_id = best_tr["id"]
                    if BBOX_SMOOTHING > 0.0 and best_tr.get("bbox"):
                        best_tr["bbox"] = smooth_bbox(best_tr["bbox"], det_box, BBOX_SMOOTHING)
                    else:
                        best_tr["bbox"] = det_box
                    best_tr["last_seen"] = now
                    best_tr["appearance"] = det_appearance
                    best_tr["score"] = score
                    track_by_id[track_id] = best_tr
                else:
                    # New track
                    if score < NEW_TRACK_MIN_CONFIDENCE:
                        continue

                    if has_duplicate_track_overlap(
                        det_box=det_box,
                        track_by_id=track_by_id,
                        now_ts=now,
                        overlap_iou=TRACK_DUPLICATE_IOU,
                        recent_only_sec=max(TRACK_OUTPUT_HOLD_TIME, 0.5),
                    ):
                        # This detection is likely a duplicate of an already tracked person.
                        continue

                    track_id = next_id
                    next_id += 1
                    best_tr = {
                        "id": track_id,
                        "bbox": det_box,
                        "last_seen": now,
                        "created_at": now,
                        "appearance": det_appearance,
                        "score": score,
                    }
                    track_by_id[track_id] = best_tr

                matched_ids.add(track_id)
                output_track_ids.add(track_id)
            except Exception as e:
                logger.warning(f"[{camera_id}] Error processing detection: {type(e).__name__}: {e}")
                continue

        # Keep recent tracks for a short time to avoid flicker on temporary misses.
        for tr in track_by_id.values():
            if now - tr.get("last_seen", 0.0) <= TRACK_OUTPUT_HOLD_TIME:
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

        # ============================================
        # 3.2) Fall Detection (NEW)
        # ============================================
        if ENABLE_FALL_DETECTION and state["fall_detector"]:
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
                fall_results = state["fall_detector"].update(fall_input_detections, request_counter)
                
                # Mark detections with fall status
                for det in detections:
                    det.fall_detected = fall_results.get(det.track_id, False)
                    if det.fall_detected:
                        logger.warning(f"[{camera_id}] FALL DETECTED: Person {det.track_id} (score={det.score:.2f})")
                
                # Get current fall statistics
                fall_stats = state["fall_detector"].get_stats()
                if fall_stats['total_fallen'] > 0:
                    logger.info(f"[{camera_id}] Fall Status: {fall_stats['total_fallen']} person(s) fallen out of {fall_stats['total_tracked']} tracked")
                    
            except Exception as e:
                logger.error(f"[{camera_id}] Fall detection error: {type(e).__name__}: {e}")
                # Fall detection errors don't stop inference, just log and continue

        # ============================================
        # 3.5) Scale detections back to original frame coordinates (if ROI was applied)
        # ============================================
        if ENABLE_ROI and roi_box and roi_box != (0, 0, original_shape[1], original_shape[0]):
            x_offset, y_offset, _, _ = roi_box
            for det in detections:
                det.x += x_offset
                det.y += y_offset

        # ============================================
        # 4) Anti-flicker: reuse last output if empty
        # ============================================
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
        # Keep active tracks for ~15 seconds, move very old ones to history for re-matching
        old_count = len(state["tracks"])

        # Separate active tracks from expired ones
        active_tracks = []
        expired_tracks = []
        for tr in track_by_id.values():
            if now - tr["last_seen"] <= TRACK_TTL:  # 15 seconds
                active_tracks.append(tr)
            else:
                expired_tracks.append(tr)
        
        # Move expired tracks to history only when history rematch is enabled.
        if ENABLE_HISTORY_REMATCH:
            state["track_history"] = [tr for tr in state["track_history"] if now - tr.get("last_seen", now) <= 30.0]
            state["track_history"].extend(expired_tracks)
        else:
            state["track_history"] = []
        
        state["tracks"] = active_tracks
        removed = old_count - len(state["tracks"])
        if removed > 0:
            logger.debug(f"[{camera_id}] Moved {removed} tracks to history (will re-match if person returns)")
        
        state["next_id"] = next_id

        # ============================================
        # 6) Count tracking
        # ============================================
        ids = {d.track_id for d in detections}
        state["seen_ids"].update(ids)
        
        # Track inference time
        inference_time = time.time() - start_time
        state["inference_times"].append(inference_time)
        if len(state["inference_times"]) > 30:
            state["inference_times"] = state["inference_times"][-30:]
        
        avg_inference_time = sum(state["inference_times"]) / len(state["inference_times"])
        max_inference_time = max(state["inference_times"])
        
        logger.info(
            f"[{camera_id}] Detections: {len(detections)} | "
            f"Tracks: {len(state['tracks'])} | "
            f"Unique: {len(state['seen_ids'])} | "
            f"Time: {inference_time*1000:.1f}ms (avg: {avg_inference_time*1000:.1f}ms)"
        )

        return detections

    except Exception as e:
        error_counter += 1
        logger.error(f"[{camera_id}] Unexpected error in /infer: {type(e).__name__}: {e}", exc_info=True)
        return []


# ============================
# Status Endpoint
# ============================
@app.get("/status")
def status():
    """Get service status and statistics"""
    uptime = time.time() - service_start_time
    cameras_info = {}
    
    for cam_id, state in camera_states.items():
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
    
    return {
        "service": "YOLOv8 People Analytics + Fall Detection",
        "status": "running",
        "uptime_seconds": uptime,
        "total_requests": request_counter,
        "total_errors": error_counter,
        "error_rate": (error_counter / request_counter * 100) if request_counter > 0 else 0.0,
        "active_cameras": len(camera_states),
        "fall_detection": "ENABLED" if ENABLE_FALL_DETECTION else "DISABLED",
        "cameras": cameras_info,
        "model": MODEL_PATH,
        "timestamp": datetime.now().isoformat()
    }


# ============================
# Reset Count Endpoint
# ============================
@app.post("/reset/{camera_id}")
def reset_camera(camera_id: str):
    """
    Reset count for a specific camera.
    Call: POST http://127.0.0.1:18000/reset/default
    """
    global camera_states
    if camera_id in camera_states:
        old_count = len(camera_states[camera_id]["seen_ids"])
        camera_states[camera_id]["seen_ids"].clear()
        camera_states[camera_id]["tracks"].clear()
        camera_states[camera_id]["track_history"].clear()
        camera_states[camera_id]["next_id"] = 1
        logger.info(f"[{camera_id}] Reset: cleared {old_count} persons, count now = 0")
        return {
            "camera_id": camera_id,
            "status": "reset",
            "previous_count": old_count,
            "current_count": 0
        }
    else:
        return {
            "camera_id": camera_id,
            "status": "not_found",
            "message": f"Camera {camera_id} not yet initialized"
        }

@app.post("/reset_all")
def reset_all():
    """Reset count for ALL cameras"""
    global camera_states
    total_persons = sum(len(state["seen_ids"]) for state in camera_states.values())
    for cam_id in camera_states:
        camera_states[cam_id]["seen_ids"].clear()
        camera_states[cam_id]["tracks"].clear()
        camera_states[cam_id]["track_history"].clear()
        camera_states[cam_id]["next_id"] = 1
        # Reset fall detection too
        if camera_states[cam_id]["fall_detector"]:
            camera_states[cam_id]["fall_detector"].reset_fall()
    logger.info(f"Reset all {len(camera_states)} cameras, cleared {total_persons} persons")
    return {
        "status": "reset_all",
        "cameras_reset": len(camera_states),
        "total_persons_cleared": total_persons
    }

@app.post("/reset_fall/{camera_id}")
def reset_fall_detection(camera_id: str):
    """Reset fall detection state for a specific camera"""
    if camera_id in camera_states and camera_states[camera_id]["fall_detector"]:
        state = camera_states[camera_id]
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
def reset_fall_all():
    """Reset fall detection state for ALL cameras"""
    reset_count = 0
    for cam_id, state in camera_states.items():
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
    
    logger.info(f"✅ FastAPI app started on {SERVICE_HOST}:{SERVICE_PORT}")
    logger.info(f"Health check: http://{SERVICE_HOST}:{SERVICE_PORT}/health")
    logger.info(f"Inference: http://{SERVICE_HOST}:{SERVICE_PORT}/infer")
    logger.info(f"Status: http://{SERVICE_HOST}:{SERVICE_PORT}/status")
    logger.info(f"Reset count for camera: POST http://{SERVICE_HOST}:{SERVICE_PORT}/reset/default")
    logger.info(f"Reset all cameras: POST http://{SERVICE_HOST}:{SERVICE_PORT}/reset_all")
    if ENABLE_FALL_DETECTION:
        logger.info(f"Reset fall detection for camera: POST http://{SERVICE_HOST}:{SERVICE_PORT}/reset_fall/default")
        logger.info(f"Reset fall detection for all cameras: POST http://{SERVICE_HOST}:{SERVICE_PORT}/reset_fall_all")

@app.on_event("shutdown")
async def shutdown_event():
    """Called when service shuts down"""
    logger.info("🛑 Service shutting down")
    logger.info(f"Total requests: {request_counter}")
    logger.info(f"Total errors: {error_counter}")

# ============================
# Main
# ============================
if __name__ == "__main__":
    logger.info(f"Starting service on {SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(
        app,
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        log_level="info",
        access_log=True
    )
