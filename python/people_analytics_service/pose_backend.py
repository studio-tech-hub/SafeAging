"""Optional YOLO-pose overlay for fall detection (Phase 2).

Runs a lightweight pose model on the inference frame and matches detected
person poses to stable track bboxes. Keypoints feed torso-angle heuristics
in ``fall_detection`` when ``ENABLE_POSE_FALL=true``.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# COCO body keypoints used for torso angle.
_LEFT_SHOULDER = 5
_RIGHT_SHOULDER = 6
_LEFT_HIP = 11
_RIGHT_HIP = 12

_model_lock = threading.Lock()
_pose_model: Any = None
_pose_load_failed = False


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _torso_angle_from_vertical(keypoints: np.ndarray, conf_thresh: float) -> Optional[float]:
    """Return angle (degrees) of shoulder→hip vector from vertical; 90° = horizontal."""
    try:
        ls = keypoints[_LEFT_SHOULDER]
        rs = keypoints[_RIGHT_SHOULDER]
        lh = keypoints[_LEFT_HIP]
        rh = keypoints[_RIGHT_HIP]
    except (IndexError, TypeError):
        return None

    if float(ls[2]) < conf_thresh or float(rs[2]) < conf_thresh:
        return None
    if float(lh[2]) < conf_thresh or float(rh[2]) < conf_thresh:
        return None

    sx = (float(ls[0]) + float(rs[0])) / 2.0
    sy = (float(ls[1]) + float(rs[1])) / 2.0
    hx = (float(lh[0]) + float(rh[0])) / 2.0
    hy = (float(lh[1]) + float(rh[1])) / 2.0
    dx, dy = hx - sx, hy - sy
    length = math.hypot(dx, dy)
    if length < 8.0:
        return None
    # Vertical in image coords is +Y; acos(|dy|/len) → 0° upright, ~90° horizontal.
    return math.degrees(math.acos(min(1.0, abs(dy) / length)))


def _load_pose_model():
    global _pose_model, _pose_load_failed
    if _pose_model is not None or _pose_load_failed:
        return _pose_model
    with _model_lock:
        if _pose_model is None and not _pose_load_failed:
            from .config import POSE_IMGSZ, POSE_MODEL_PATH

            try:
                from ultralytics import YOLO

                _pose_model = YOLO(POSE_MODEL_PATH)
                logger.info(
                    "[pose] YOLO pose ready: model=%s imgsz=%d",
                    POSE_MODEL_PATH,
                    POSE_IMGSZ,
                )
            except Exception as exc:
                logger.error("[pose] Failed to load pose model: %s", exc, exc_info=True)
                _pose_load_failed = True
    return _pose_model


def available() -> bool:
    from .config import ENABLE_POSE_FALL

    if not ENABLE_POSE_FALL:
        return False
    return _load_pose_model() is not None


def match_pose_to_tracks(
    frame: np.ndarray,
    track_detections: List[Dict],
    *,
    conf: float,
    imgsz: int,
) -> Dict[int, Dict]:
    """Run pose once and match instances to track bboxes by IoU.

    ``track_detections`` items: ``track_id``, ``bbox`` as (x1,y1,x2,y2).
    """
    model = _load_pose_model()
    if model is None or not track_detections:
        return {}

    try:
        result = model.predict(
            frame,
            imgsz=imgsz,
            conf=conf,
            classes=[0],
            verbose=False,
        )[0]
    except Exception as exc:
        logger.warning("[pose] inference failed: %s", exc)
        return {}

    if result.keypoints is None or result.boxes is None:
        return {}

    kpts_data = result.keypoints.data
    if kpts_data is None or len(kpts_data) == 0:
        return {}

    pose_boxes = result.boxes.xyxy.cpu().numpy()
    hints: Dict[int, Dict] = {}

    for det in track_detections:
        track_id = int(det.get("track_id", -1))
        if track_id < 0:
            continue
        tb = det.get("bbox", (0, 0, 0, 0))
        track_box = tuple(float(v) for v in tb)

        best_iou, best_idx = 0.0, -1
        for idx, pbox in enumerate(pose_boxes):
            iou = _iou(track_box, tuple(float(v) for v in pbox))
            if iou > best_iou:
                best_iou, best_idx = iou, idx

        if best_idx < 0 or best_iou < 0.25:
            continue

        kpts = kpts_data[best_idx].cpu().numpy()
        torso_angle = _torso_angle_from_vertical(kpts, conf)
        if torso_angle is None:
            continue
        hints[track_id] = {
            "torso_angle_from_vertical": torso_angle,
            "pose_iou": best_iou,
        }

    return hints


def match_pose_to_crops(
    frame: np.ndarray,
    track_detections: List[Dict],
    candidate_track_ids: List[int],
    *,
    conf: float,
    imgsz: int,
) -> Dict[int, Dict]:
    """Tier-2 pose: run pose only on person crops when bbox fall is suspected."""
    model = _load_pose_model()
    if model is None or not candidate_track_ids:
        return {}

    wanted = set(int(t) for t in candidate_track_ids)
    hints: Dict[int, Dict] = {}
    fh, fw = frame.shape[:2]

    for det in track_detections:
        track_id = int(det.get("track_id", -1))
        if track_id not in wanted:
            continue
        x1, y1, x2, y2 = [int(v) for v in det.get("bbox", (0, 0, 0, 0))]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if x2 - x1 < 32 or y2 - y1 < 32:
            continue
        crop = frame[y1:y2, x1:x2]
        crop_sz = min(imgsz, max(crop.shape[0], crop.shape[1]))
        try:
            result = model.predict(
                crop,
                imgsz=crop_sz,
                conf=conf,
                verbose=False,
            )[0]
        except Exception as exc:
            logger.debug("[pose] crop infer failed tid=%s: %s", track_id, exc)
            continue
        if result.keypoints is None or result.keypoints.data is None:
            continue
        if len(result.keypoints.data) < 1:
            continue
        kpts = result.keypoints.data[0].cpu().numpy()
        torso_angle = _torso_angle_from_vertical(kpts, conf)
        if torso_angle is None:
            continue
        hints[track_id] = {
            "torso_angle_from_vertical": torso_angle,
            "pose_source": "crop",
        }
    return hints


def match_pose_to_crops(
    frame: np.ndarray,
    track_detections: List[Dict],
    candidate_track_ids: List[int],
    *,
    conf: float,
    imgsz: int,
) -> Dict[int, Dict]:
    """Tier-2 pose: run pose only on person crops when bbox fall is suspected."""
    model = _load_pose_model()
    if model is None or not candidate_track_ids:
        return {}

    wanted = set(int(t) for t in candidate_track_ids)
    hints: Dict[int, Dict] = {}
    fh, fw = frame.shape[:2]

    for det in track_detections:
        track_id = int(det.get("track_id", -1))
        if track_id not in wanted:
            continue
        x1, y1, x2, y2 = [int(v) for v in det.get("bbox", (0, 0, 0, 0))]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if x2 - x1 < 32 or y2 - y1 < 32:
            continue
        crop = frame[y1:y2, x1:x2]
        crop_sz = min(imgsz, max(crop.shape[0], crop.shape[1]))
        try:
            result = model.predict(
                crop,
                imgsz=crop_sz,
                conf=conf,
                verbose=False,
            )[0]
        except Exception as exc:
            logger.debug("[pose] crop infer failed tid=%s: %s", track_id, exc)
            continue
        if result.keypoints is None or result.keypoints.data is None:
            continue
        if len(result.keypoints.data) < 1:
            continue
        kpts = result.keypoints.data[0].cpu().numpy()
        torso_angle = _torso_angle_from_vertical(kpts, conf)
        if torso_angle is None:
            continue
        hints[track_id] = {
            "torso_angle_from_vertical": torso_angle,
            "pose_source": "crop",
        }
    return hints
