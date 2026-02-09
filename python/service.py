import base64
import json
import time
import logging
import os
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
import torch
if hasattr(torch.serialization, 'add_safe_globals'):
    try:
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
    except Exception as e:
        pass  # Silently continue if fix not needed

# ============================
# Logging Configuration
# ============================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================
# Configuration
# ============================
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "18000"))
SERVICE_HOST = os.getenv("SERVICE_HOST", "127.0.0.1")
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))  # Optimized: 35% catches small people better
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))  # IoU for NMS
MIN_DETECTION_AREA = int(os.getenv("MIN_DETECTION_AREA", "20"))  # Catch even small people at distance
TRACK_TTL = float(os.getenv("TRACK_TTL", "15.0"))  # Keep tracks for 15 seconds
IOA_THRESHOLD = float(os.getenv("IOA_THRESHOLD", "0.05"))  # Track matching threshold
FLICKER_REUSE_TIME = float(os.getenv("FLICKER_REUSE_TIME", "1.0"))  # Anti-flicker reuse time

# ============================
# Fall Detection Configuration
# ============================
ENABLE_FALL_DETECTION = os.getenv("ENABLE_FALL_DETECTION", "true").lower() == "true"
FALL_VELOCITY_THRESHOLD = float(os.getenv("FALL_VELOCITY_THRESHOLD", "20.0"))  # Pixels per frame
FALL_ANGLE_CHANGE_THRESHOLD = float(os.getenv("FALL_ANGLE_CHANGE_THRESHOLD", "45.0"))  # Degrees
FALL_ASPECT_RATIO_THRESHOLD = float(os.getenv("FALL_ASPECT_RATIO_THRESHOLD", "1.5"))  # Width/Height ratio
FALL_CONFIDENCE_THRESHOLD = float(os.getenv("FALL_CONFIDENCE_THRESHOLD", "0.8"))  # 80% confidence for fall

# ============================
# Production Settings (Optimized for wide-angle cameras - ENABLED by default)
# ============================
ENABLE_CLAHE = os.getenv("ENABLE_CLAHE", "true").lower() == "true"
ENABLE_MULTI_SCALE = os.getenv("ENABLE_MULTI_SCALE", "true").lower() == "true"
ENABLE_FRAME_ENHANCEMENT = os.getenv("ENABLE_FRAME_ENHANCEMENT", "false").lower() == "true"
CLAHE_CLIP_LIMIT = float(os.getenv("CLAHE_CLIP_LIMIT", "2.0"))
CLAHE_TILE_SIZE = int(os.getenv("CLAHE_TILE_SIZE", "16"))

# ============================
# Wide-Angle Optimization: ROI (Region of Interest) - ENABLED by default
# ============================
# Crop to relevant area only (e.g., doorway, walkway) to catch smaller people
ENABLE_ROI = os.getenv("ENABLE_ROI", "true").lower() == "true"
ROI_TYPE = os.getenv("ROI_TYPE", "rect")  # "rect" or "polygon"
# Rectangle ROI: specify as percentage of frame (0.0-1.0)
ROI_X_MIN = float(os.getenv("ROI_X_MIN", "0.0"))    # Left edge percentage
ROI_X_MAX = float(os.getenv("ROI_X_MAX", "1.0"))    # Right edge percentage
ROI_Y_MIN = float(os.getenv("ROI_Y_MIN", "0.3"))    # Top edge percentage (skip ceiling/roof)
ROI_Y_MAX = float(os.getenv("ROI_Y_MAX", "1.0"))    # Bottom edge percentage (full height below)
# Polygon ROI: JSON array of [x,y] points as percentages
ROI_POLYGON_JSON = os.getenv("ROI_POLYGON_JSON", "")

# ============================
# Wide-Angle Optimization: Undistortion - ENABLED by default
# ============================
# For wide-angle/fisheye cameras, undistort first to normalize geometry
ENABLE_UNDISTORT = os.getenv("ENABLE_UNDISTORT", "true").lower() == "true"
CAMERA_MATRIX_JSON = os.getenv("CAMERA_MATRIX_JSON", "")  # 3x3 intrinsic matrix as JSON
DISTORTION_COEFFS_JSON = os.getenv("DISTORTION_COEFFS_JSON", "")  # k1,k2,p1,p2,k3... as JSON
CALIBRATION_FILE = os.getenv("CALIBRATION_FILE", "camera_calibration.json")

# ============================
# Inference Image Size
# ============================
# Increased from 640 to 960 for better small object detection (after ROI crop)
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "960"))

logger.info(f"="*60)
logger.info(f"YOLOv8 People Analytics Service")
logger.info(f"="*60)
logger.info(f"Port: {SERVICE_PORT}")
logger.info(f"Host: {SERVICE_HOST}")
logger.info(f"Model: {MODEL_PATH}")
logger.info(f"Confidence: {CONFIDENCE_THRESHOLD}")
logger.info(f"IOU: {IOU_THRESHOLD}")
logger.info(f"ImgSize: {YOLO_IMGSZ}")
logger.info(f"="*60)
logger.info(f"CLAHE: {ENABLE_CLAHE}")
logger.info(f"Multi-Scale: {ENABLE_MULTI_SCALE}")
logger.info(f"Frame Enhancement: {ENABLE_FRAME_ENHANCEMENT}")
logger.info(f"ROI: {ENABLE_ROI} (type={ROI_TYPE})")
logger.info(f"Undistort: {ENABLE_UNDISTORT}")
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
            model.to('cpu')
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
        device='cpu',
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
        device='cpu',
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
            # Save frame to disk for inspection
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
                    device='cpu',  # Force CPU for stability
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

        new_tracks = []
        detections: List[Detection] = []

        # ============================================
        # 3) Process YOLO outputs with tracking
        # ============================================
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

                # ============================================
                # Track matching: find best match using IoU + Appearance
                # ============================================
                det_appearance = extract_appearance(frame, det_box)
                best_score, best_tr = -1.0, None
                
                # FIRST: Try to match against active tracks
                for tr in tracks:
                    iou_score = iou(det_box, tr["bbox"])
                    
                    # Always try to compare appearance if we have it
                    if tr.get("appearance") and det_appearance["color_hist"] is not None:
                        app_dist = appearance_distance(tr["appearance"]["color_hist"], det_appearance["color_hist"])
                        match_score = combined_track_score(iou_score, app_dist)
                    else:
                        # Fall back to IoU only
                        match_score = iou_score
                    
                    if match_score > best_score:
                        best_score, best_tr = match_score, tr

                # SECOND: If no good active track match, search track history
                # This allows re-matching people who left and came back
                if (best_tr is None or best_score < IOA_THRESHOLD) and state["track_history"]:
                    for hist_tr in state["track_history"]:
                        # Only match if appearance is similar enough (less reliance on position)
                        if hist_tr.get("appearance") and det_appearance["color_hist"] is not None:
                            app_dist = appearance_distance(hist_tr["appearance"]["color_hist"], det_appearance["color_hist"])
                            # Person-like appearance match = likely same person returning
                            if app_dist < 0.4:  # More lenient: similar appearance
                                hist_score = 1.0 - app_dist  # Appearance-based score (0-1)
                                if hist_score > best_score:
                                    best_score, best_tr = hist_score, hist_tr
                                    logger.debug(f"[{camera_id}] Re-matched track ID={hist_tr['id']} from history (app_dist={app_dist:.2f})")

                # Track matching threshold: adjust based on appearance quality
                match_threshold = IOA_THRESHOLD
                if best_tr is not None and det_appearance["color_hist"] is not None and best_tr.get("appearance"):
                    app_dist = appearance_distance(best_tr["appearance"]["color_hist"], det_appearance["color_hist"])
                    # If appearance is very similar, lower threshold significantly
                    if app_dist < 0.3:
                        match_threshold = 0.05  # More forgiving if colors match
                
                if best_tr is not None and best_score >= match_threshold:
                    # Existing track: update position and appearance
                    track_id = best_tr["id"]
                    best_tr["bbox"] = det_box
                    best_tr["last_seen"] = now
                    best_tr["appearance"] = det_appearance
                    # If this was from history, bring it back to active tracks
                    if best_tr not in tracks:
                        tracks.append(best_tr)
                else:
                    # New track
                    track_id = next_id
                    next_id += 1
                    best_tr = {
                        "id": track_id, 
                        "bbox": det_box, 
                        "last_seen": now,
                        "created_at": now,
                        "appearance": det_appearance
                    }

                new_tracks.append(best_tr)

                # Create detection object for C++ plugin
                detections.append(Detection(
                    cls="person",
                    score=score,
                    x=float(x1),
                    y=float(y1),
                    w=float(w_box),
                    h=float(h_box),
                    track_id=int(track_id)
                ))
                
            except Exception as e:
                logger.warning(f"[{camera_id}] Error processing box: {type(e).__name__}: {e}")
                continue

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
        for tr in new_tracks:
            if now - tr["last_seen"] <= TRACK_TTL:  # 15 seconds
                active_tracks.append(tr)
            else:
                expired_tracks.append(tr)
        
        # Move expired tracks to history (keep for 30 seconds for re-matching)
        state["track_history"] = [tr for tr in state["track_history"] if now - tr.get("last_seen", now) <= 30.0]
        state["track_history"].extend(expired_tracks)
        
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
