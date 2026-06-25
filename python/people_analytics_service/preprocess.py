import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .config import (
    CALIBRATION_FILE,
    CAMERA_MATRIX_JSON,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_SIZE,
    CONFIDENCE_THRESHOLD,
    DEVICE,
    DISTORTION_COEFFS_JSON,
    ENABLE_CLAHE,
    ENABLE_FRAME_ENHANCEMENT,
    ENABLE_MULTI_SCALE,
    ENABLE_ROI,
    ENABLE_UNDISTORT,
    IOU_THRESHOLD,
    ROI_POLYGON_JSON,
    ROI_TYPE,
    ROI_X_MAX,
    ROI_X_MIN,
    ROI_Y_MAX,
    ROI_Y_MIN,
    USE_HALF,
    YOLO_IMGSZ,
    logger,
)


camera_matrix = None
dist_coeffs = None


def load_calibration() -> bool:
    """Load camera calibration from file or environment variables."""
    global camera_matrix, dist_coeffs

    if CAMERA_MATRIX_JSON and DISTORTION_COEFFS_JSON:
        try:
            camera_matrix = np.array(json.loads(CAMERA_MATRIX_JSON))
            dist_coeffs = np.array(json.loads(DISTORTION_COEFFS_JSON))
            logger.info("✅ Camera calibration loaded from environment variables")
            return True
        except Exception as e:
            logger.warning(f"Failed to load calibration from env: {e}")

    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
                cal_data = json.load(f)
                camera_matrix = np.array(cal_data.get("camera_matrix", []))
                dist_coeffs = np.array(cal_data.get("distortion_coefficients", []))
                logger.info(f"✅ Camera calibration loaded from {CALIBRATION_FILE}")
                return True
        except Exception as e:
            logger.warning(f"Failed to load calibration from file: {e}")

    if ENABLE_UNDISTORT:
        logger.warning(
            "⚠️ Undistort enabled but no calibration data found. Create camera_calibration.json or set env variables."
        )
    return False


def apply_roi_rect(frame: np.ndarray, roi: Optional[Dict[str, float]] = None) -> tuple:
    h, w = frame.shape[:2]
    x_min = roi.get("x_min", ROI_X_MIN) if roi else ROI_X_MIN
    y_min = roi.get("y_min", ROI_Y_MIN) if roi else ROI_Y_MIN
    x_max = roi.get("x_max", ROI_X_MAX) if roi else ROI_X_MAX
    y_max = roi.get("y_max", ROI_Y_MAX) if roi else ROI_Y_MAX
    x1 = int(w * x_min)
    y1 = int(h * y_min)
    x2 = int(w * x_max)
    y2 = int(h * y_max)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    cropped = frame[y1:y2, x1:x2]
    roi_box = (x1, y1, x2, y2)
    return cropped, roi_box


def apply_roi_polygon(frame: np.ndarray, polygon_json: Optional[str] = None) -> tuple:
    try:
        raw = polygon_json if polygon_json is not None else ROI_POLYGON_JSON
        if not raw:
            logger.warning("Polygon ROI enabled but polygon JSON not provided")
            return frame, (0, 0, frame.shape[1], frame.shape[0])

        polygon = json.loads(raw)
        h, w = frame.shape[:2]

        points = np.array([[int(p[0] * w), int(p[1] * h)] for p in polygon], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], (255,))

        result = cv2.bitwise_and(frame, frame, mask=mask)

        x_min = int(min(p[0] for p in polygon) * w)
        y_min = int(min(p[1] for p in polygon) * h)
        x_max = int(max(p[0] for p in polygon) * w)
        y_max = int(max(p[1] for p in polygon) * h)
        roi_box = (max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max))

        return result, roi_box
    except Exception as e:
        logger.warning(f"Polygon ROI error: {e}, using full frame")
        return frame, (0, 0, frame.shape[1], frame.shape[0])


def apply_roi(frame: np.ndarray, roi_override: Optional[Dict[str, Any]] = None) -> tuple:
    if not ENABLE_ROI:
        h, w = frame.shape[:2]
        return frame, (0, 0, w, h), None

    roi_type = (roi_override or {}).get("type", ROI_TYPE)
    if roi_type == "polygon":
        poly = (roi_override or {}).get("polygon_json") or ROI_POLYGON_JSON
        return apply_roi_polygon(frame, polygon_json=poly) + ("polygon",)
    rect = (roi_override or {}).get("rect") if roi_override else None
    return apply_roi_rect(frame, roi=rect) + ("rect",)


def undistort_frame(frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    if not ENABLE_UNDISTORT or camera_matrix is None or dist_coeffs is None:
        return frame, (0, 0)

    try:
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 0.9)
        result = cv2.undistort(frame, camera_matrix, dist_coeffs, newCameraMatrix=new_camera_matrix)

        x, y, w_roi, h_roi = roi
        crop_offset = (0, 0)
        if x > 0 or y > 0 or w_roi < w or h_roi < h:
            result = result[y:y + h_roi, x:x + w_roi]
            crop_offset = (int(x), int(y))

        return result, crop_offset
    except Exception as e:
        logger.warning(f"Undistort error: {e}, using original frame")
        return frame, (0, 0)


def remap_detections_to_input_space(
    detections: List[Any],
    roi_box: Optional[Tuple[float, float, float, float]],
    roi_type: Optional[str],
    pre_roi_shape: Tuple[int, ...],
    undistort_crop_offset: Tuple[int, int],
) -> None:
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
    if roi_box == (0, 0, original_shape[1], original_shape[0]):
        return detections

    x_offset, y_offset, _, _ = roi_box
    for det in detections:
        det["x"] += x_offset
        det["y"] += y_offset
    return detections


def auto_adjust_brightness(frame: np.ndarray, target_brightness: float = 190.0) -> np.ndarray:
    try:
        current_brightness = np.mean(frame)
        if current_brightness < target_brightness:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            l_clahe = clahe.apply(l)

            l_mean = float(np.mean(np.asarray(l_clahe, dtype=np.float32)))
            if l_mean < target_brightness:
                scale_factor = target_brightness / max(l_mean, 5)
                scale_factor = min(scale_factor, 3.0)
                l_clahe = (l_clahe.astype(np.float32) * scale_factor).clip(0, 255).astype(np.uint8)

            lab_adjusted = cv2.merge([l_clahe, a, b])
            frame_adjusted = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)

            new_brightness = float(np.mean(np.asarray(frame_adjusted, dtype=np.float32)))
            if new_brightness >= target_brightness * 0.95:
                logger.debug(f"Brightness corrected: {current_brightness:.1f} → {new_brightness:.1f}")
                return frame_adjusted

        return frame
    except Exception as e:
        logger.warning(f"Brightness adjustment error: {e}")
        return frame


def apply_clahe(frame: np.ndarray) -> np.ndarray:
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE),
        )
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    except Exception as e:
        logger.warning(f"CLAHE failed: {e}, returning original frame")
        return frame


def apply_frame_enhancement(frame: np.ndarray) -> np.ndarray:
    try:
        frame = cv2.bilateralFilter(frame, 9, 75, 75)
        gamma = 1.1
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        frame = cv2.LUT(frame, table)
        gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
        frame = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
        return frame
    except Exception as e:
        logger.warning(f"Frame enhancement failed: {e}, returning original frame")
        return frame


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    if ENABLE_CLAHE:
        frame = apply_clahe(frame)
    if ENABLE_FRAME_ENHANCEMENT:
        frame = apply_frame_enhancement(frame)
    return frame


def ensure_min_input_side(frame: np.ndarray, min_side: int) -> Tuple[np.ndarray, float]:
    """Upscale frame so min(height, width) >= min_side.

    Returns (frame, scale) where pixel coordinates in the returned frame are
  ``original_coord * scale``. Map detections back with ``coord / scale``.
    """
    h, w = frame.shape[:2]
    short = min(h, w)
    if short >= min_side:
        return frame, 1.0
    scale = float(min_side) / float(short)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    upscaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return upscaled, scale


def multi_scale_inference_smart(yolo_model, frame: np.ndarray, original_h: int, original_w: int):
    class FilteredResult:
        def __init__(self, boxes=None):
            self.boxes = boxes

    class ScaledBox:
        def __init__(self, xyxy: torch.Tensor, conf: torch.Tensor, cls: torch.Tensor):
            self.xyxy = xyxy
            self.conf = conf
            self.cls = cls

    r = yolo_model.predict(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        classes=[0],
        imgsz=YOLO_IMGSZ,
        verbose=False,
        augment=False,
        device=DEVICE,
        half=USE_HALF,
    )[0]

    if r.boxes is not None and len(r.boxes) > 0:
        return FilteredResult(r.boxes)

    logger.debug("No detections at 1.0x, retrying at 1.25x scale")
    h, w = int(original_h * 1.25), int(original_w * 1.25)
    scaled_frame = cv2.resize(frame, (w, h))

    r_scaled = yolo_model.predict(
        scaled_frame,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        classes=[0],
        imgsz=YOLO_IMGSZ,
        verbose=False,
        augment=False,
        device=DEVICE,
        half=USE_HALF,
    )[0]

    if r_scaled.boxes is not None and len(r_scaled.boxes) > 0:
        scaled_boxes: List[ScaledBox] = []
        for box in r_scaled.boxes:
            xyxy = box.xyxy.detach().clone() / 1.25
            conf = box.conf.detach().clone()
            cls = box.cls.detach().clone()
            scaled_boxes.append(ScaledBox(xyxy=xyxy, conf=conf, cls=cls))
        return FilteredResult(scaled_boxes)

    return FilteredResult()
