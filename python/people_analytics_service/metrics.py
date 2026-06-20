"""Prometheus metrics registry.

Single source of truth for every metric in the service.
Import from here; never create duplicate metrics in other modules.

Populated by:
  - api.py              → INFER_*, INFER_ERRORS, PREPROCESS_LATENCY,
                          ACTIVE_TRACKS, DETECTIONS_PER_FRAME
  - health.py           → DB_UP, OBJECT_STORAGE_UP
  - outbox_worker.py    → OUTBOX_PENDING, OUTBOX_ENQUEUED, OUTBOX_PROCESSED,
                          OUTBOX_DROPPED, OUTBOX_ERRORS
"""
from prometheus_client import Counter, Gauge, Histogram, Info

# ── Service metadata ──────────────────────────────────────────────────────────
SERVICE_INFO = Info(
    "safeaging_service",
    "Service metadata (version, model, device)",
)

# ── Inference counters ────────────────────────────────────────────────────────
INFER_REQUESTS = Counter(
    "safeaging_infer_requests_total",
    "Total /infer requests",
    ["camera_id", "status"],   # status: "success" | "error"
)

# ── Inference latency histograms ─────────────────────────────────────────────
INFER_LATENCY = Histogram(
    "safeaging_infer_latency_seconds",
    "End-to-end /infer latency (decode + preprocess + model + tracking)",
    ["camera_id"],
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
)

DECODE_LATENCY = Histogram(
    "safeaging_decode_latency_seconds",
    "Image decode (base64 + cv2) latency within /infer",
    ["camera_id"],
    buckets=[0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)

YOLO_LATENCY = Histogram(
    "safeaging_yolo_latency_seconds",
    "YOLO model inference latency within /infer",
    ["camera_id"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

PREPROCESS_LATENCY = Histogram(
    "safeaging_preprocess_latency_seconds",
    "Image preprocessing (undistortion, ROI crop, resize) latency within /infer",
    ["camera_id"],
    buckets=[0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1],
)

# ── Per-camera error counter ───────────────────────────────────────────────────
INFER_ERRORS = Counter(
    "safeaging_infer_errors_total",
    "Total /infer processing errors per camera",
    ["camera_id", "kind"],   # kind: "decode" | "preprocess" | "yolo" | "tracking" | "unknown"
)

# ── Per-frame detection counts ────────────────────────────────────────────────
DETECTIONS_PER_FRAME = Histogram(
    "safeaging_detections_per_frame",
    "Number of detection objects returned per /infer call",
    ["camera_id"],
    buckets=[0, 1, 2, 3, 5, 8, 13, 21],
)

# ── Track state ───────────────────────────────────────────────────────────────
ACTIVE_TRACKS = Gauge(
    "safeaging_active_tracks",
    "Number of currently active tracks per camera",
    ["camera_id"],
)

# ── Dependency health (1 = up, 0 = down) ─────────────────────────────────────
DB_UP = Gauge(
    "safeaging_db_up",
    "1 if PostgreSQL is reachable, 0 otherwise",
)

OBJECT_STORAGE_UP = Gauge(
    "safeaging_object_storage_up",
    "1 if MinIO/S3 object storage is reachable, 0 otherwise",
)

# ── Outbox (Service P1.3) ─────────────────────────────────────────────────────
OUTBOX_PENDING = Gauge(
    "safeaging_outbox_pending",
    "Current outbox queue depth (items waiting to be processed)",
)

OUTBOX_ENQUEUED = Counter(
    "safeaging_outbox_enqueued_total",
    "Total events enqueued to the outbox",
)

OUTBOX_DROPPED = Counter(
    "safeaging_outbox_dropped_total",
    "Total events dropped because the outbox queue was full",
)

OUTBOX_PROCESSED = Counter(
    "safeaging_outbox_processed_total",
    "Total events processed by the outbox worker",
)

OUTBOX_ERRORS = Counter(
    "safeaging_outbox_errors_total",
    "Total outbox processing errors",
    ["kind"],   # kind: "db" | "s3" | "email"
)

# ── Safety events (S P2.5) ────────────────────────────────────────────────────
EVENTS_PERSISTED = Counter(
    "safeaging_events_persisted_total",
    "Total safety events persisted to DB",
    ["event_type", "camera_id"],
)

REID_AUTO_LINKS = Counter(
    "safeaging_reid_auto_links_total",
    "Total events automatically linked to a person via ReID",
    ["camera_id"],
)

ALERTS_SENT = Counter(
    "safeaging_alerts_sent_total",
    "Total alert emails successfully sent",
    ["event_type"],
)

