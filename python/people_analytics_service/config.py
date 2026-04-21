import logging
import os
from pathlib import Path

import torch


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def _register_torch_safe_globals() -> None:
    if not hasattr(torch.serialization, "add_safe_globals"):
        return

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

    try:
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
    except Exception:
        pass


_register_torch_safe_globals()


def _load_local_env() -> None:
    """Load KEY=VALUE pairs from .env without overwriting existing env vars."""
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
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
        self.confidence_threshold = _clamp(
            _env_float("CONFIDENCE_THRESHOLD", 0.35), 0.0, 1.0, "CONFIDENCE_THRESHOLD"
        )
        self.iou_threshold = _clamp(_env_float("IOU_THRESHOLD", 0.45), 0.0, 1.0, "IOU_THRESHOLD")
        self.min_detection_area = max(1, _env_int("MIN_DETECTION_AREA", 20))
        self.person_min_hw_ratio = _clamp(
            _env_float("PERSON_MIN_HW_RATIO", 0.80), 0.1, 10.0, "PERSON_MIN_HW_RATIO"
        )
        self.track_ttl = max(1.0, _env_float("TRACK_TTL", 15.0))
        self.ioa_threshold = _clamp(_env_float("IOA_THRESHOLD", 0.05), 0.0, 1.0, "IOA_THRESHOLD")
        self.flicker_reuse_time = max(0.0, _env_float("FLICKER_REUSE_TIME", 0.1))

        self.bbox_smoothing = _clamp(_env_float("BBOX_SMOOTHING", 0.85), 0.0, 1.0, "BBOX_SMOOTHING")
        self.enable_post_nms = _env_bool("ENABLE_POST_NMS", True)
        self.post_nms_iou = _clamp(_env_float("POST_NMS_IOU", 0.6), 0.0, 1.0, "POST_NMS_IOU")
        self.match_iou_threshold = _clamp(
            _env_float("MATCH_IOU_THRESHOLD", 0.12), 0.0, 1.0, "MATCH_IOU_THRESHOLD"
        )
        self.max_center_distance_ratio = _clamp(
            _env_float("MAX_CENTER_DISTANCE_RATIO", 1.6), 0.1, 10.0, "MAX_CENTER_DISTANCE_RATIO"
        )
        self.new_track_min_confidence = _clamp(
            _env_float("NEW_TRACK_MIN_CONFIDENCE", self.confidence_threshold),
            0.0,
            1.0,
            "NEW_TRACK_MIN_CONFIDENCE",
        )
        self.track_duplicate_iou = _clamp(
            _env_float("TRACK_DUPLICATE_IOU", 0.65), 0.0, 1.0, "TRACK_DUPLICATE_IOU"
        )
        self.track_output_hold_time = max(0.0, _env_float("TRACK_OUTPUT_HOLD_TIME", 0.2))
        self.enable_history_rematch = _env_bool("ENABLE_HISTORY_REMATCH", True)
        default_match_age = max(self.track_output_hold_time, 2.5)
        self.max_match_age = max(0.0, _env_float("MAX_MATCH_AGE", default_match_age))
        self.track_min_hits = max(1, _env_int("TRACK_MIN_HITS", 1))
        self.unique_count_min_hits = max(
            1,
            _env_int("UNIQUE_COUNT_MIN_HITS", max(3, self.track_min_hits)),
        )
        self.track_max_misses = max(1, _env_int("TRACK_MAX_MISSES", 8))
        self.tentative_max_misses = max(0, _env_int("TENTATIVE_MAX_MISSES", 1))
        self.track_match_min_score = _clamp(
            _env_float("TRACK_MATCH_MIN_SCORE", 0.35), 0.0, 1.0, "TRACK_MATCH_MIN_SCORE"
        )
        self.output_tentative_tracks = _env_bool("OUTPUT_TENTATIVE_TRACKS", False)
        self.output_dedupe_iou = _clamp(
            _env_float("OUTPUT_DEDUPE_IOU", 0.55), 0.0, 1.0, "OUTPUT_DEDUPE_IOU"
        )
        self.hold_suppress_iou = _clamp(
            _env_float("HOLD_SUPPRESS_IOU", 0.4), 0.0, 1.0, "HOLD_SUPPRESS_IOU"
        )

        self.enable_fall_detection = _env_bool("ENABLE_FALL_DETECTION", True)
        self.fall_velocity_threshold = max(0.0, _env_float("FALL_VELOCITY_THRESHOLD", 20.0))
        self.fall_angle_change_threshold = max(0.0, _env_float("FALL_ANGLE_CHANGE_THRESHOLD", 45.0))
        self.fall_aspect_ratio_threshold = max(0.0, _env_float("FALL_ASPECT_RATIO_THRESHOLD", 1.5))
        self.fall_confidence_threshold = _clamp(
            _env_float("FALL_CONFIDENCE_THRESHOLD", 0.8), 0.0, 1.0, "FALL_CONFIDENCE_THRESHOLD"
        )

        self.enable_clahe = _env_bool("ENABLE_CLAHE", False)
        self.enable_multi_scale = _env_bool("ENABLE_MULTI_SCALE", False)
        self.enable_frame_enhancement = _env_bool("ENABLE_FRAME_ENHANCEMENT", False)
        self.clahe_clip_limit = max(0.1, _env_float("CLAHE_CLIP_LIMIT", 2.0))
        self.clahe_tile_size = max(2, _env_int("CLAHE_TILE_SIZE", 16))
        self.save_debug_samples = _env_bool("SAVE_DEBUG_SAMPLES", True)

        self.enable_roi = _env_bool("ENABLE_ROI", True)
        self.roi_type = os.getenv("ROI_TYPE", "rect").strip().lower()
        if self.roi_type not in {"rect", "polygon"}:
            logger.warning(f"Invalid ROI_TYPE='{self.roi_type}', fallback to 'rect'")
            self.roi_type = "rect"
        self.roi_x_min = _clamp(_env_float("ROI_X_MIN", 0.0), 0.0, 1.0, "ROI_X_MIN")
        self.roi_x_max = _clamp(_env_float("ROI_X_MAX", 1.0), 0.0, 1.0, "ROI_X_MAX")
        self.roi_y_min = _clamp(_env_float("ROI_Y_MIN", 0.0), 0.0, 1.0, "ROI_Y_MIN")
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

        self.api_key_required = _env_bool("API_KEY_REQUIRED", True)
        self.api_key = os.getenv("API_KEY", "").strip()
        if self.api_key_required and not self.api_key:
            logger.warning(
                "API_KEY_REQUIRED=true but API_KEY is empty. Authentication is disabled until API_KEY is set."
            )
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


def log_config_summary() -> None:
    logger.info("=" * 60)
    logger.info("YOLOv8 People Analytics Service")
    logger.info("=" * 60)
    logger.info(f"Port: {SERVICE_PORT}")
    logger.info(f"Host: {SERVICE_HOST}")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Confidence: {CONFIDENCE_THRESHOLD}")
    logger.info(f"IOU: {IOU_THRESHOLD}")
    logger.info(f"Person min h/w ratio: {PERSON_MIN_HW_RATIO}")
    logger.info(f"ImgSize: {YOLO_IMGSZ}")
    logger.info(f"Device: {DEVICE} | FP16: {USE_HALF}")
    logger.info("=" * 60)
    logger.info(f"CLAHE: {ENABLE_CLAHE}")
    logger.info(f"Multi-Scale: {ENABLE_MULTI_SCALE}")
    logger.info(f"Frame Enhancement: {ENABLE_FRAME_ENHANCEMENT}")
    logger.info(f"ROI: {ENABLE_ROI} (type={ROI_TYPE})")
    logger.info(f"Undistort: {ENABLE_UNDISTORT}")
    logger.info("=" * 60)
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
    logger.info("=" * 60)
    logger.info(f"Fall Detection: {ENABLE_FALL_DETECTION}")
    if ENABLE_FALL_DETECTION:
        logger.info(f"  Velocity Threshold: {FALL_VELOCITY_THRESHOLD}px/frame")
        logger.info(f"  Angle Change Threshold: {FALL_ANGLE_CHANGE_THRESHOLD}°")
        logger.info(f"  Aspect Ratio Threshold: {FALL_ASPECT_RATIO_THRESHOLD}")
        logger.info(f"  Confidence Threshold: {FALL_CONFIDENCE_THRESHOLD}")
    logger.info("=" * 60)
