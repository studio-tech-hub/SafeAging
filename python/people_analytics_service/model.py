from ultralytics import YOLO
import torch

from .config import DEVICE, MODEL_PATH, logger


model = None


def load_model():
    """Load YOLO model with PyTorch 2.6 compatibility."""
    global model
    if model is not None:
        return model

    try:
        logger.info(f"Loading YOLO model from: {MODEL_PATH}")

        original_torch_load = torch.load

        def patched_torch_load(f, *args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return original_torch_load(f, *args, **kwargs)

        torch.load = patched_torch_load

        try:
            model = YOLO(MODEL_PATH)
            try:
                model.fuse()
            except Exception:
                pass
            try:
                model.to(DEVICE)
            except Exception as e:
                logger.warning(f"Failed to move model to {DEVICE}: {e}")
            logger.info("✅ YOLO model loaded successfully")
        finally:
            torch.load = original_torch_load

        return model
    except Exception as e:
        logger.error(f"❌ Failed to load YOLO model: {e}")
        raise
