"""Load and cache the YOLO model (CPU or Qualcomm QNN backend)."""

from .config import MODEL_PATH, YOLO_BACKEND, logger
from . import yolo_backend

model = None


def load_model():
    """Load YOLO model with backend from YOLO_BACKEND env (cpu | qnn_htp | qnn_gpu)."""
    global model
    if model is not None:
        return model

    try:
        logger.info(
            "Loading YOLO from %s (YOLO_BACKEND=%s)",
            MODEL_PATH,
            YOLO_BACKEND,
        )
        model = yolo_backend.load_yolo_model(MODEL_PATH)
        logger.info("YOLO ready (active_backend=%s)", yolo_backend.active_backend())
        return model
    except Exception as e:
        logger.error("Failed to load YOLO model: %s", e)
        raise
