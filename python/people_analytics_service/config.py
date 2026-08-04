import logging
import os
import re
from pathlib import Path

import torch


# Minimal bootstrap — formatter is replaced by configure_logging() below
logging.basicConfig(level=logging.INFO)
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


# Minimum square inference input (px) for YOLO, face detection, and ONNX exports.
MIN_INPUT_RESOLUTION = max(640, _env_int("MIN_INPUT_RESOLUTION", 640))

# Bumped whenever a recommended-*default* tuning change ships (e.g. P1-2's
# CONFIDENCE_THRESHOLD 0.75->0.55 and ROI invalid-range fallback 0.3->0.0), so
# support/ops can tell which documented defaults a given deployment was built
# against from `log_config_summary()` alone, without needing its full .env.
# This is NOT an API/DB schema version and has no effect on behavior by
# itself — it never gates logic, only appears in logs.
#   1 = pre-P1-2 baseline (CONFIDENCE_THRESHOLD=0.75, ROI fallback=[0.3, 1.0])
#   2 = P1-2 (CONFIDENCE_THRESHOLD=0.55, ROI fallback=[0.0, 1.0], track-state-aware
#       PERSON_MIN_HW_RATIO)
CONFIG_SCHEMA_VERSION = 2


def _clamp_min_square_resolution(env_name: str, default: int) -> int:
    """Clamp a square input size env var to MIN_INPUT_RESOLUTION (default 640)."""
    raw = _env_int(env_name, default)
    if raw < MIN_INPUT_RESOLUTION:
        logger.warning(
            "%s=%d is below minimum %d; using %d",
            env_name,
            raw,
            MIN_INPUT_RESOLUTION,
            MIN_INPUT_RESOLUTION,
        )
        return MIN_INPUT_RESOLUTION
    return raw


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"Invalid float for {name}='{raw}', using default {default}")
        return default


def _env_or_file(name: str, default: str = "") -> str:
    """Read a secret from `{name}_FILE` (Docker/Kubernetes secrets convention,
    e.g. API_KEY_FILE=/run/secrets/api_key) if set, else fall back to the
    plain `{name}` env var, else `default` (P0-4).

    `_FILE` takes precedence when both are set, matching the common
    "Docker secrets" convention (e.g. the official postgres/mysql images) so
    an operator can point `*_FILE` at a mounted secret without also having
    to remember to unset the inline var. Purely additive: unset `_FILE` and
    every existing plain-env-var deployment behaves exactly as before.
    """
    file_path = os.getenv(f"{name}_FILE", "").strip()
    if not file_path:
        return os.getenv(name, default).strip()
    try:
        return Path(file_path).read_text(encoding="utf-8").strip()
    except OSError as e:
        logger.warning(
            f"Could not read {name}_FILE={file_path!r} ({e}); "
            f"falling back to {name} env var."
        )
        return os.getenv(name, default).strip()


# Known placeholder/default API key values (case-insensitive) that must never
# be accepted as a "real" key when API_KEY_REQUIRED=true (P0-2).
_INSECURE_API_KEY_VALUES = {
    "",
    "changeme",
    "change-me",
    "change_me",
    "change-me-to-a-strong-secret",
    "your-api-key-here",
    "secret",
    "password",
    "admin",
    "test",
    "apikey",
}


# ── Secrets hygiene (P0-4) ────────────────────────────────────────────────
# No diagnostic/support-bundle tool exists yet (planned for P1 observability
# work), but any future one — and any ad hoc "dump config for a support
# ticket" debugging — MUST redact through these first. Established now as a
# convention so licensing keys / remote-update tokens (P4) follow the same
# pattern from day one instead of inventing their own.
_SECRET_KEY_PATTERN = re.compile(r"(API_KEY|PASSWORD|SECRET|TOKEN)", re.IGNORECASE)
_REDACTED = "***REDACTED***"


def is_secret_env_var(name: str) -> bool:
    """True if an env-var-style name looks like it holds a secret (API_KEY,
    *_PASSWORD, *_SECRET, *_TOKEN — including their *_FILE variants, since
    e.g. API_KEY_FILE still matches "API_KEY"). Never dump os.environ or a
    parsed .env file for logs/support bundles without filtering through
    this (or redact_env_dict() below) first."""
    return bool(_SECRET_KEY_PATTERN.search(name))


def redact_env_dict(env: dict) -> dict:
    """Return a copy of an env-var dict with secret-looking values replaced
    by a fixed marker. Intended for any future diagnostic/support-bundle
    tool (P1) that dumps configuration — e.g. `redact_env_dict(dict(os.environ))`."""
    return {k: (_REDACTED if v and is_secret_env_var(k) else v) for k, v in env.items()}


def redact_url_credentials(text: str) -> str:
    """Mask `user:password@` credentials embedded in a URL-like string (e.g.
    a Postgres DATABASE_URL) so it is safe to include in logs/diagnostics."""
    return re.sub(r"://([^:/@\s]+):([^@/\s]+)@", rf"://\1:{_REDACTED}@", text)


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
        self.model_path = os.getenv("MODEL_PATH", "yolo26n.pt")
        # P1-2: default lowered 0.75 -> 0.55. The nano model at 640px with the old
        # 0.75 default was tuned purely to suppress furniture false-positives, at
        # the direct cost of missing small/far people — a much worse trade-off for
        # a fall-detection/safety product than an occasional extra low-confidence
        # box (which downstream tracking's TRACK_MIN_HITS/NMS/dedupe already
        # filters before it becomes a "stable" output). Purely a documented-default
        # change: any deployment with an explicit CONFIDENCE_THRESHOLD in its .env
        # is completely unaffected.
        self.confidence_threshold = _clamp(
            _env_float("CONFIDENCE_THRESHOLD", 0.55), 0.0, 1.0, "CONFIDENCE_THRESHOLD"
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
        self.fall_confirm_frames = max(1, _env_int("FALL_CONFIRM_FRAMES", 3))
        self.fall_velocity_ref_height = max(20, _env_int("FALL_VELOCITY_REF_HEIGHT", 120))

        # Optional YOLO-pose overlay for fall (runs every N frames; adds latency).
        self.enable_pose_fall = _env_bool("ENABLE_POSE_FALL", False)
        self.pose_model_path = os.getenv("POSE_MODEL_PATH", "yolo11n-pose.pt").strip()
        self.pose_imgsz = _clamp_min_square_resolution("POSE_IMGSZ", 640)
        self.pose_interval_frames = max(1, _env_int("POSE_INTERVAL_FRAMES", 6))
        self.pose_keypoint_conf = _clamp(
            _env_float("POSE_KEYPOINT_CONF", 0.5), 0.0, 1.0, "POSE_KEYPOINT_CONF"
        )
        self.pose_torso_angle_threshold = max(
            10.0, _env_float("POSE_TORSO_ANGLE_THRESHOLD", 55.0)
        )

        # Upscale decoded frames when short side < MIN_INPUT_RESOLUTION before YOLO only.
        self.enable_input_upscale = _env_bool("ENABLE_INPUT_UPSCALE", True)
        self.enable_clahe = _env_bool("ENABLE_CLAHE", False)
        self.enable_multi_scale = _env_bool("ENABLE_MULTI_SCALE", False)
        self.enable_frame_enhancement = _env_bool("ENABLE_FRAME_ENHANCEMENT", False)
        self.clahe_clip_limit = max(0.1, _env_float("CLAHE_CLIP_LIMIT", 2.0))
        self.clahe_tile_size = max(2, _env_int("CLAHE_TILE_SIZE", 16))
        self.save_debug_samples = _env_bool("SAVE_DEBUG_SAMPLES", False)
        self.calib_capture_dir = os.getenv("CALIB_CAPTURE_DIR", "").strip()
        self.calib_capture_max = max(0, _env_int("CALIB_CAPTURE_MAX", 0))

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
            # P1-2: fall back to the full frame (0.0-1.0), not a guessed crop. The
            # previous [0.3, 1.0] fallback was tuned for one specific camera framing
            # and silently generalized as "the" default — for any other camera, it
            # crops out the top of frame where a standing person's head appears. An
            # invalid config should degrade to "no crop," never to a guessed one.
            logger.warning("ROI_Y_MIN must be < ROI_Y_MAX, fallback to [0.0, 1.0] (full frame)")
            self.roi_y_min, self.roi_y_max = 0.0, 1.0

        self.enable_undistort = _env_bool("ENABLE_UNDISTORT", False)
        self.camera_matrix_json = os.getenv("CAMERA_MATRIX_JSON", "")
        self.distortion_coeffs_json = os.getenv("DISTORTION_COEFFS_JSON", "")
        self.calibration_file = os.getenv("CALIBRATION_FILE", "camera_calibration.json")

        self.yolo_imgsz = _clamp_min_square_resolution("YOLO_IMGSZ", 640)
        if "_256" in self.model_path or "_192" in self.model_path or "_160" in self.model_path or "_320" in self.model_path or "_416" in self.model_path:
            logger.warning(
                "MODEL_PATH=%s appears to be a sub-%d ONNX export; "
                "use yolo26s.onnx or yolo26n.onnx (exported at %d) for production",
                self.model_path,
                MIN_INPUT_RESOLUTION,
                MIN_INPUT_RESOLUTION,
            )

        # Qualcomm QNN (HTP NPU / Adreno GPU) — not NVIDIA CUDA
        self.yolo_backend = os.getenv("YOLO_BACKEND", "cpu").strip().lower()
        self.qnn_backend_path = os.getenv("QNN_BACKEND_PATH", "").strip()
        self.qnn_htp_performance_mode = os.getenv("QNN_HTP_PERFORMANCE_MODE", "burst").strip()
        self.qnn_htp_fp16 = _env_bool("QNN_HTP_FP16", True)
        self.qnn_htp_arch = os.getenv("QNN_HTP_ARCH", "68").strip()
        self.qnn_soc_model = os.getenv("QNN_SOC_MODEL", "").strip()
        self.qnn_vtcm_mb = os.getenv("QNN_VTCM_MB", "").strip()
        self.qnn_context_cache = _env_bool("QNN_CONTEXT_CACHE", True)
        self.qnn_context_cache_dir = os.getenv("QNN_CONTEXT_CACHE_DIR", "/app/runtime/qnn_context").strip()
        self.qnn_disable_cpu_fallback = _env_bool("QNN_DISABLE_CPU_FALLBACK", False)
        # P1-3: coordinate space of end2end (NMS-embedded) ONNX box outputs.
        # "auto" infers per-detection from overflow past frame bounds (documented
        # fallback for already-exported models with no recorded contract); "canvas"
        # forces letterbox-canvas rescaling; "frame" forces original-frame passthrough.
        # Set explicitly once an export is known to always emit one or the other.
        self.end2end_coord_space = os.getenv("END2END_COORD_SPACE", "auto").strip().lower()
        if self.end2end_coord_space not in {"auto", "canvas", "frame"}:
            logger.warning(
                f"Invalid END2END_COORD_SPACE='{self.end2end_coord_space}', fallback to 'auto'"
            )
            self.end2end_coord_space = "auto"

        self.device = os.getenv("DEVICE", "cuda:0" if cuda_available else "cpu")
        self.use_half = _env_bool("USE_HALF", False)
        if not self.device.startswith("cuda"):
            self.use_half = False
        self.torch_cudnn_benchmark = _env_bool("TORCH_CUDNN_BENCHMARK", True)

        self.api_key_required = _env_bool("API_KEY_REQUIRED", True)
        # P0-4: API_KEY_FILE (Docker secrets convention) takes precedence over
        # the inline API_KEY env var when both are set.
        self.api_key = _env_or_file("API_KEY")
        # Explicit, opt-in-only escape hatch (P0-2). Default False so nobody
        # accidentally ships an unauthenticated deployment. Never set this on
        # a network-reachable box — see tools/generate_api_key.py instead.
        self.allow_insecure_no_auth = _env_bool("ALLOW_INSECURE_NO_AUTH", False)

        api_key_is_placeholder = self.api_key.lower() in _INSECURE_API_KEY_VALUES
        if self.api_key_required and (not self.api_key or api_key_is_placeholder):
            if self.allow_insecure_no_auth:
                logger.critical(
                    "!!! SECURITY WARNING: API_KEY_REQUIRED=true but API_KEY is "
                    "empty or a known placeholder, and ALLOW_INSECURE_NO_AUTH=true "
                    "is set. Starting WITHOUT authentication — /infer, /admin/*, "
                    "and /config/* are UNPROTECTED on the network. This must ONLY "
                    "be used for an isolated lab/dev box. Never expose this "
                    "configuration on a shared network or the internet. !!!"
                )
                self.api_key_required = False
            else:
                # Fail-closed (P0-2): a missing/placeholder key used to silently
                # disable auth. It no longer does — refuse to start instead, so
                # an insecure deployment can never happen by accident.
                logger.critical(
                    "FATAL: API_KEY_REQUIRED=true but API_KEY is empty or a "
                    "known placeholder value. Refusing to start with "
                    "authentication silently disabled. Fix with ONE of:\n"
                    "  1) Set API_KEY to a strong random value:\n"
                    "     python tools/generate_api_key.py\n"
                    "  2) Set API_KEY_REQUIRED=false to intentionally run "
                    "without auth (only for a fully isolated/trusted network).\n"
                    "  3) Set ALLOW_INSECURE_NO_AUTH=true to explicitly "
                    "acknowledge running without authentication (lab/dev only)."
                )
                raise RuntimeError(
                    "API_KEY_REQUIRED=true but API_KEY is empty or a placeholder "
                    "value. Set a real API_KEY (see tools/generate_api_key.py), "
                    "set API_KEY_REQUIRED=false, or set ALLOW_INSECURE_NO_AUTH=true."
                )

        # True whenever protected endpoints end up unauthenticated, regardless
        # of *why* (explicit API_KEY_REQUIRED=false or the escape hatch above).
        # Surfaced in /health reason_codes so monitoring can alert on it.
        self.auth_effectively_disabled = not self.api_key_required
        if self.auth_effectively_disabled and not self.allow_insecure_no_auth:
            logger.warning(
                "API_KEY_REQUIRED=false — /infer, /admin/*, and /config/* are "
                "UNPROTECTED. Only use this on a fully isolated/trusted network."
            )

        self.require_https = _env_bool("REQUIRE_HTTPS", False)
        self.tls_cert_file = os.getenv("TLS_CERT_FILE", "").strip()
        self.tls_key_file = os.getenv("TLS_KEY_FILE", "").strip()
        if bool(self.tls_cert_file) != bool(self.tls_key_file):
            logger.warning("TLS_CERT_FILE and TLS_KEY_FILE must be set together. Direct TLS will be disabled.")
            self.tls_cert_file = ""
            self.tls_key_file = ""
        self.rate_limit_enabled = _env_bool("RATE_LIMIT_ENABLED", True)
        self.rate_limit_window_seconds = max(1, _env_int("RATE_LIMIT_WINDOW_SECONDS", 60))
        self.rate_limit_max_per_ip = max(1, _env_int("RATE_LIMIT_MAX_PER_IP", 2400))
        self.rate_limit_max_per_camera = max(1, _env_int("RATE_LIMIT_MAX_PER_CAMERA", 1200))
        self.rate_limit_skip_loopback = _env_bool("RATE_LIMIT_SKIP_LOOPBACK", True)
        self.metrics_window_size = max(10, _env_int("METRICS_WINDOW_SIZE", 100))
        self.metrics_log_interval = max(1, _env_int("METRICS_LOG_INTERVAL", 20))
        cors_raw = os.getenv("CORS_ALLOW_ORIGINS", "")
        self.cors_allow_origins = [v.strip() for v in cors_raw.split(",") if v.strip()]

        # ── Observability (S P1.5) ────────────────────────────────────────────
        raw_log_format = os.getenv("LOG_FORMAT", "text").strip().lower()
        self.log_format = raw_log_format if raw_log_format in {"text", "json"} else "text"

        # ── External services (consumed by health probes + S P1.1/P1.3) ──────
        # P0-4: each of these accepts a `_FILE` variant (Docker secrets
        # convention) that takes precedence over the inline value when set —
        # e.g. DATABASE_URL_FILE=/run/secrets/database_url.
        self.database_url = _env_or_file("DATABASE_URL")
        self.s3_endpoint = os.getenv("S3_ENDPOINT", "").strip()
        self.s3_access_key = _env_or_file("S3_ACCESS_KEY")
        self.s3_secret_key = _env_or_file("S3_SECRET_KEY")
        self.s3_bucket_snapshots = os.getenv("S3_BUCKET_SNAPSHOTS", "safeaging-snapshots").strip()
        self.s3_region = os.getenv("S3_REGION", "us-east-1").strip()

        # ── S P1.3 – Alert / email config ────────────────────────────────────
        self.smtp_host = os.getenv("SMTP_HOST", "").strip()
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "").strip()
        self.smtp_password = _env_or_file("SMTP_PASSWORD")  # P0-4: SMTP_PASSWORD_FILE supported
        self.smtp_from = os.getenv("SMTP_FROM", "alerts@safeaging.local").strip()
        self.alert_email_to = os.getenv("ALERT_EMAIL_TO", "").strip()
        self.alert_fall_enabled = os.getenv("ALERT_FALL_ENABLED", "true").lower() == "true"
        # S P2.3 – alert engine config
        self.alert_zone_violation_enabled = os.getenv("ALERT_ZONE_VIOLATION_ENABLED", "true").lower() == "true"
        self.alert_dedupe_sec = float(os.getenv("ALERT_DEDUPE_SEC", "60"))
        self.alert_max_retries = int(os.getenv("ALERT_MAX_RETRIES", "3"))
        self.alert_rate_limit_max = int(os.getenv("ALERT_RATE_LIMIT_MAX", "10"))
        self.alert_rate_limit_window_sec = float(os.getenv("ALERT_RATE_LIMIT_WINDOW_SEC", "600"))

        # S P2.4 – retention policy
        self.retention_days = int(os.getenv("RETENTION_DAYS", "30"))
        self.retention_check_hours = float(os.getenv("RETENTION_CHECK_HOURS", "6"))

        # ── Edge durable outbox (survives restart/crash) ──────────────────────
        self.outbox_db_path = os.getenv("OUTBOX_DB_PATH", "edge_outbox.db").strip()
        self.outbox_max_attempts = max(1, _env_int("OUTBOX_MAX_ATTEMPTS", 8))
        self.edge_sqlite_path = os.getenv(
            "EDGE_SQLITE_PATH", "/app/runtime/edge_state.db"
        ).strip()

        # S P2.1 – ReID / person re-identification
        self.reid_match_threshold = float(os.getenv("REID_MATCH_THRESHOLD", "0.65"))
        self.reid_auto_link_enabled = os.getenv("REID_AUTO_LINK_ENABLED", "true").lower() == "true"

        # ── Face recognition (real-time identity on the bounding box) ──────────
        self.enable_face_recognition = _env_bool("ENABLE_FACE_RECOGNITION", True)
        # Cosine similarity threshold for ArcFace normed embeddings (0..1).
        self.face_match_threshold = _clamp(
            _env_float("FACE_MATCH_THRESHOLD", 0.45), 0.0, 1.0, "FACE_MATCH_THRESHOLD"
        )
        # Run face on stable tracks immediately; retry Unknown on a time schedule.
        self.face_recog_unknown_retry_sec = max(
            1.0, _env_float("FACE_RECOG_UNKNOWN_RETRY_SEC", 8.0)
        )
        # Legacy frame interval (used only when FACE_RECOG_USE_TIME_SCHEDULER=false).
        self.face_recog_interval_frames = max(1, _env_int("FACE_RECOG_INTERVAL_FRAMES", 12))
        self.face_recog_use_time_scheduler = _env_bool("FACE_RECOG_USE_TIME_SCHEDULER", True)
        # Minimum face bbox side (pixels) to attempt recognition.
        self.face_min_pixels = max(8, _env_int("FACE_MIN_PIXELS", 28))
        # insightface detector input size (square).
        self.face_det_size = _clamp_min_square_resolution("FACE_DET_SIZE", 640)
        # insightface model pack (buffalo_s = faster; buffalo_l = heavier / slightly more accurate).
        self.face_model_pack = os.getenv("FACE_MODEL_PACK", "buffalo_s").strip()
        # cpu | qnn_gpu (Adreno via onnxruntime-qnn; falls back to CPU)
        self.face_backend = os.getenv("FACE_BACKEND", "cpu").strip().lower()
        self.enable_face_async = _env_bool("ENABLE_FACE_ASYNC", True)
        self.face_async_workers = max(1, _env_int("FACE_ASYNC_WORKERS", 1))
        self.face_async_queue_size = max(4, _env_int("FACE_ASYNC_QUEUE_SIZE", 32))
        # Gallery (enrolled face embeddings) cache refresh interval.
        self.face_gallery_refresh_sec = max(5.0, _env_float("FACE_GALLERY_REFRESH_SEC", 30.0))

        # Face enrollment from multiple images or sampled video frames.
        self.enroll_face_min_images = max(1, _env_int("ENROLL_FACE_MIN_IMAGES", 5))
        self.enroll_face_max_images = max(
            self.enroll_face_min_images,
            _env_int("ENROLL_FACE_MAX_IMAGES", 30),
        )
        self.enroll_video_sample_fps = _clamp(
            _env_float("ENROLL_VIDEO_SAMPLE_FPS", 4.0), 3.0, 5.0, "ENROLL_VIDEO_SAMPLE_FPS"
        )

        # Health: mark service degraded when pipeline latency exceeds thresholds.
        self.health_p95_degraded_ms = max(100.0, _env_float("HEALTH_P95_DEGRADED_MS", 800.0))
        self.health_avg_degraded_ms = max(100.0, _env_float("HEALTH_AVG_DEGRADED_MS", 650.0))


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
FALL_CONFIRM_FRAMES = CONFIG.fall_confirm_frames
FALL_VELOCITY_REF_HEIGHT = CONFIG.fall_velocity_ref_height
ENABLE_POSE_FALL = CONFIG.enable_pose_fall
POSE_MODEL_PATH = CONFIG.pose_model_path
POSE_IMGSZ = CONFIG.pose_imgsz
POSE_INTERVAL_FRAMES = CONFIG.pose_interval_frames
POSE_KEYPOINT_CONF = CONFIG.pose_keypoint_conf
POSE_TORSO_ANGLE_THRESHOLD = CONFIG.pose_torso_angle_threshold
ENABLE_INPUT_UPSCALE = CONFIG.enable_input_upscale
ENABLE_CLAHE = CONFIG.enable_clahe
ENABLE_MULTI_SCALE = CONFIG.enable_multi_scale
ENABLE_FRAME_ENHANCEMENT = CONFIG.enable_frame_enhancement
CLAHE_CLIP_LIMIT = CONFIG.clahe_clip_limit
CLAHE_TILE_SIZE = CONFIG.clahe_tile_size
SAVE_DEBUG_SAMPLES = CONFIG.save_debug_samples
CALIB_CAPTURE_DIR = CONFIG.calib_capture_dir
CALIB_CAPTURE_MAX = CONFIG.calib_capture_max
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
YOLO_BACKEND = CONFIG.yolo_backend
QNN_BACKEND_PATH = CONFIG.qnn_backend_path
QNN_HTP_PERFORMANCE_MODE = CONFIG.qnn_htp_performance_mode
QNN_HTP_FP16 = CONFIG.qnn_htp_fp16
QNN_HTP_ARCH = CONFIG.qnn_htp_arch
QNN_SOC_MODEL = CONFIG.qnn_soc_model
QNN_VTCM_MB = CONFIG.qnn_vtcm_mb
QNN_CONTEXT_CACHE = CONFIG.qnn_context_cache
QNN_CONTEXT_CACHE_DIR = CONFIG.qnn_context_cache_dir
QNN_DISABLE_CPU_FALLBACK = CONFIG.qnn_disable_cpu_fallback
END2END_COORD_SPACE = CONFIG.end2end_coord_space
DEVICE = CONFIG.device
USE_HALF = CONFIG.use_half
TORCH_CUDNN_BENCHMARK = CONFIG.torch_cudnn_benchmark
API_KEY_REQUIRED = CONFIG.api_key_required
API_KEY = CONFIG.api_key
ALLOW_INSECURE_NO_AUTH = CONFIG.allow_insecure_no_auth
AUTH_EFFECTIVELY_DISABLED = CONFIG.auth_effectively_disabled
REQUIRE_HTTPS = CONFIG.require_https
TLS_CERT_FILE = CONFIG.tls_cert_file
TLS_KEY_FILE = CONFIG.tls_key_file
RATE_LIMIT_ENABLED = CONFIG.rate_limit_enabled
RATE_LIMIT_WINDOW_SECONDS = CONFIG.rate_limit_window_seconds
RATE_LIMIT_MAX_PER_IP = CONFIG.rate_limit_max_per_ip
RATE_LIMIT_MAX_PER_CAMERA = CONFIG.rate_limit_max_per_camera
RATE_LIMIT_SKIP_LOOPBACK = CONFIG.rate_limit_skip_loopback
METRICS_WINDOW_SIZE = CONFIG.metrics_window_size
METRICS_LOG_INTERVAL = CONFIG.metrics_log_interval
CORS_ALLOW_ORIGINS = CONFIG.cors_allow_origins

# Observability
LOG_FORMAT = CONFIG.log_format

# External services
DATABASE_URL = CONFIG.database_url
S3_ENDPOINT = CONFIG.s3_endpoint
S3_ACCESS_KEY = CONFIG.s3_access_key
S3_SECRET_KEY = CONFIG.s3_secret_key
S3_BUCKET_SNAPSHOTS = CONFIG.s3_bucket_snapshots
S3_REGION = CONFIG.s3_region

# Alert / email (S P1.3)
SMTP_HOST = CONFIG.smtp_host
SMTP_PORT = CONFIG.smtp_port
SMTP_USER = CONFIG.smtp_user
SMTP_PASSWORD = CONFIG.smtp_password
SMTP_FROM = CONFIG.smtp_from
ALERT_EMAIL_TO = CONFIG.alert_email_to
ALERT_FALL_ENABLED = CONFIG.alert_fall_enabled
# S P2.3
ALERT_ZONE_VIOLATION_ENABLED = CONFIG.alert_zone_violation_enabled
ALERT_DEDUPE_SEC = CONFIG.alert_dedupe_sec
ALERT_MAX_RETRIES = CONFIG.alert_max_retries
ALERT_RATE_LIMIT_MAX = CONFIG.alert_rate_limit_max
ALERT_RATE_LIMIT_WINDOW_SEC = CONFIG.alert_rate_limit_window_sec
# S P2.4
RETENTION_DAYS = CONFIG.retention_days
RETENTION_CHECK_HOURS = CONFIG.retention_check_hours
# Edge durable outbox
OUTBOX_DB_PATH = CONFIG.outbox_db_path
OUTBOX_MAX_ATTEMPTS = CONFIG.outbox_max_attempts
EDGE_SQLITE_PATH = CONFIG.edge_sqlite_path


def normalize_camera_id(camera_id: str) -> str:
    """Nx plugin uses ``{uuid}``; admin UI may omit braces — normalize for lookups."""
    cid = (camera_id or "").strip()
    if not cid or cid == "default":
        return cid
    if cid.startswith("{") and cid.endswith("}"):
        return cid
    core = cid.strip("{}")
    return "{" + core + "}"
# S P2.1
REID_MATCH_THRESHOLD = CONFIG.reid_match_threshold
REID_AUTO_LINK_ENABLED = CONFIG.reid_auto_link_enabled
# Face recognition
ENABLE_FACE_RECOGNITION = CONFIG.enable_face_recognition
FACE_MATCH_THRESHOLD = CONFIG.face_match_threshold
FACE_RECOG_UNKNOWN_RETRY_SEC = CONFIG.face_recog_unknown_retry_sec
FACE_RECOG_INTERVAL_FRAMES = CONFIG.face_recog_interval_frames
FACE_RECOG_USE_TIME_SCHEDULER = CONFIG.face_recog_use_time_scheduler
FACE_MIN_PIXELS = CONFIG.face_min_pixels
FACE_DET_SIZE = CONFIG.face_det_size
FACE_MODEL_PACK = CONFIG.face_model_pack
FACE_BACKEND = CONFIG.face_backend
ENABLE_FACE_ASYNC = CONFIG.enable_face_async
FACE_ASYNC_WORKERS = CONFIG.face_async_workers
FACE_ASYNC_QUEUE_SIZE = CONFIG.face_async_queue_size
FACE_GALLERY_REFRESH_SEC = CONFIG.face_gallery_refresh_sec
ENROLL_FACE_MIN_IMAGES = CONFIG.enroll_face_min_images
ENROLL_FACE_MAX_IMAGES = CONFIG.enroll_face_max_images
ENROLL_VIDEO_SAMPLE_FPS = CONFIG.enroll_video_sample_fps
HEALTH_P95_DEGRADED_MS = CONFIG.health_p95_degraded_ms
HEALTH_AVG_DEGRADED_MS = CONFIG.health_avg_degraded_ms


if DEVICE.startswith("cuda"):
    try:
        torch.backends.cudnn.benchmark = TORCH_CUDNN_BENCHMARK
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# Apply log format now that CONFIG is fully initialised
from .logging_setup import configure_logging  # noqa: E402 (late import to avoid circular deps)
configure_logging(LOG_FORMAT)


def log_config_summary() -> None:
    logger.info("=" * 60)
    logger.info("YOLO26 People Analytics Service")
    logger.info(f"Config schema version: {CONFIG_SCHEMA_VERSION}")
    logger.info("=" * 60)
    logger.info(f"Port: {SERVICE_PORT}")
    logger.info(f"Host: {SERVICE_HOST}")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Confidence: {CONFIDENCE_THRESHOLD}")
    logger.info(f"IOU: {IOU_THRESHOLD}")
    logger.info(f"Person min h/w ratio: {PERSON_MIN_HW_RATIO}")
    logger.info(f"ImgSize: {YOLO_IMGSZ} (min {MIN_INPUT_RESOLUTION})")
    logger.info(f"Face det size: {FACE_DET_SIZE} (min {MIN_INPUT_RESOLUTION})")
    logger.info(f"YOLO backend: {YOLO_BACKEND} (QNN=Qualcomm HTP/GPU, not CUDA)")
    if YOLO_BACKEND.startswith("qnn"):
        logger.info(
            f"QNN: backend_path={QNN_BACKEND_PATH or '(auto)'} "
            f"htp_mode={QNN_HTP_PERFORMANCE_MODE} htp_fp16={QNN_HTP_FP16} "
            f"htp_arch={QNN_HTP_ARCH} soc_model={QNN_SOC_MODEL or '(auto)'} "
            f"context_cache={QNN_CONTEXT_CACHE} "
            f"end2end_coord_space={END2END_COORD_SPACE}"
        )
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
        logger.info(f"  Velocity Threshold: {FALL_VELOCITY_THRESHOLD}px/frame (ref_h={FALL_VELOCITY_REF_HEIGHT})")
        logger.info(f"  Angle Change Threshold: {FALL_ANGLE_CHANGE_THRESHOLD}°")
        logger.info(f"  Aspect Ratio Threshold: {FALL_ASPECT_RATIO_THRESHOLD}")
        logger.info(f"  Confidence Threshold: {FALL_CONFIDENCE_THRESHOLD}")
        logger.info(f"  Confirm frames: {FALL_CONFIRM_FRAMES}")
        logger.info(
            f"  Pose overlay: {ENABLE_POSE_FALL} "
            f"(model={POSE_MODEL_PATH}, every {POSE_INTERVAL_FRAMES}f, "
            f"torso>{POSE_TORSO_ANGLE_THRESHOLD:.0f}°)"
        )
    logger.info(f"Face model pack: {FACE_MODEL_PACK}")
    logger.info(f"Face backend: {FACE_BACKEND} async={ENABLE_FACE_ASYNC} workers={FACE_ASYNC_WORKERS}")
    logger.info(f"Input upscale before YOLO: {ENABLE_INPUT_UPSCALE} (min {MIN_INPUT_RESOLUTION}px)")
    logger.info(f"Metrics: window={METRICS_WINDOW_SIZE} log_interval={METRICS_LOG_INTERVAL}")
    logger.info("=" * 60)
    logger.info(f"Log format: {LOG_FORMAT}")
    logger.info(f"DB configured: {'postgres' if DATABASE_URL else 'edge-sqlite' if EDGE_SQLITE_PATH else 'no'}")
    if not DATABASE_URL and EDGE_SQLITE_PATH:
        logger.info(f"Edge SQLite path: {EDGE_SQLITE_PATH}")
    logger.info(f"Object storage configured: {'yes' if S3_ENDPOINT else 'no'}")
    logger.info(f"Email alerts: {'yes' if SMTP_HOST and ALERT_EMAIL_TO else 'no'}")
    logger.info("=" * 60)
    if AUTH_EFFECTIVELY_DISABLED:
        logger.warning("!" * 60)
        logger.warning("! AUTHENTICATION IS DISABLED — /infer, /admin/*, /config/* are UNPROTECTED !")
        logger.warning("!" * 60)
    else:
        logger.info(f"Authentication: ENABLED (API_KEY_REQUIRED={API_KEY_REQUIRED})")
    logger.info("=" * 60)
