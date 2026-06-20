import threading
import time
from collections import deque
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .config import (
    FALL_ANGLE_CHANGE_THRESHOLD,
    FALL_ASPECT_RATIO_THRESHOLD,
    FALL_CONFIDENCE_THRESHOLD,
    FALL_VELOCITY_THRESHOLD,
    ENABLE_FALL_DETECTION,
    HOLD_SUPPRESS_IOU,
    MATCH_IOU_THRESHOLD,
    MAX_CENTER_DISTANCE_RATIO,
    MAX_MATCH_AGE,
    METRICS_WINDOW_SIZE,
    OUTPUT_DEDUPE_IOU,
    PERSON_MIN_HW_RATIO,
    POST_NMS_IOU,
    TRACK_DUPLICATE_IOU,
    TRACK_MATCH_MIN_SCORE,
    TRACK_MAX_MISSES,
    TRACK_MIN_HITS,
    TRACK_OUTPUT_HOLD_TIME,
    TRACK_TTL,
    UNIQUE_COUNT_MIN_HITS,
    logger,
)
from .fall import FallDetectionManager


camera_states: Dict[str, Dict[str, Any]] = {}
camera_states_lock = threading.RLock()


def _new_metrics_window():
    return deque(maxlen=METRICS_WINDOW_SIZE)


def _mean_or_zero(samples) -> float:
    if not samples:
        return 0.0
    return float(sum(samples) / len(samples))


def compute_percentile(samples, percentile: float) -> float:
    if not samples:
        return 0.0
    return float(np.percentile(np.asarray(list(samples), dtype=np.float64), percentile))


def _camera_metrics_snapshot_unlocked(state: Dict[str, Any]) -> Dict[str, Any]:
    infer_times_ms = list(state.get("infer_times_ms", ()))
    decode_times_ms = list(state.get("decode_times_ms", ()))
    preprocess_times_ms = list(state.get("preprocess_times_ms", ()))
    yolo_times_ms = list(state.get("yolo_times_ms", ()))
    track_counts = list(state.get("track_counts", ()))

    return {
        "request_count": int(state.get("request_count", 0)),
        "error_count": int(state.get("error_count", 0)),
        "last_error": state.get("last_error"),
        "last_infer_ms": float(state.get("last_infer_ms", 0.0)),
        "last_decode_ms": float(state.get("last_decode_ms", 0.0)),
        "last_preprocess_ms": float(state.get("last_preprocess_ms", 0.0)),
        "last_yolo_ms": float(state.get("last_yolo_ms", 0.0)),
        "last_track_count": int(state.get("last_track_count", 0)),
        "avg_inference_ms": _mean_or_zero(infer_times_ms),
        "p50_inference_ms": compute_percentile(infer_times_ms, 50),
        "p95_inference_ms": compute_percentile(infer_times_ms, 95),
        "avg_decode_ms": _mean_or_zero(decode_times_ms),
        "avg_preprocess_ms": _mean_or_zero(preprocess_times_ms),
        "avg_yolo_ms": _mean_or_zero(yolo_times_ms),
        "avg_track_count": _mean_or_zero(track_counts),
        "metrics_window_size": int(state.get("metrics_window_size", METRICS_WINDOW_SIZE)),
        "last_update_ts": float(state.get("last_update_ts", 0.0)),
    }


def get_camera_state(camera_id: str) -> Dict[str, Any]:
    with camera_states_lock:
        if camera_id not in camera_states:
            logger.info(f"[CAMERA] Initializing new camera: {camera_id}")
            camera_states[camera_id] = {
                "lock": threading.RLock(),
                # Per-camera frame counter. This must stay camera-local so TTL/stale logic
                # for fall detection is not affected by traffic from other cameras.
                "frame_idx": 0,
                "tracks": [],
                "track_history": [],
                "next_id": 1,
                "seen_ids": set(),
                "seen_ids_with_time": {},  # {track_id: last_seen_timestamp}
                "total_count": 0,
                "last_output": [],
                "last_time": 0.0,
                "metrics_window_size": METRICS_WINDOW_SIZE,
                "infer_times_ms": _new_metrics_window(),
                "decode_times_ms": _new_metrics_window(),
                "preprocess_times_ms": _new_metrics_window(),
                "yolo_times_ms": _new_metrics_window(),
                "track_counts": _new_metrics_window(),
                "request_count": 0,
                "error_count": 0,
                "last_error": None,
                "last_infer_ms": 0.0,
                "last_decode_ms": 0.0,
                "last_preprocess_ms": 0.0,
                "last_yolo_ms": 0.0,
                "last_track_count": 0,
                "last_update_ts": 0.0,
                "created_at": time.time(),
                "fall_detector": FallDetectionManager(
                    velocity_threshold=FALL_VELOCITY_THRESHOLD,
                    angle_change_threshold=FALL_ANGLE_CHANGE_THRESHOLD,
                    aspect_ratio_threshold=FALL_ASPECT_RATIO_THRESHOLD,
                    confidence_threshold=FALL_CONFIDENCE_THRESHOLD,
                ) if ENABLE_FALL_DETECTION else None,
            }
        return camera_states[camera_id]


def increment_camera_request(state: Dict[str, Any]) -> int:
    with state["lock"]:
        state["request_count"] = int(state.get("request_count", 0)) + 1
        state["last_update_ts"] = time.time()
        return state["request_count"]


def increment_camera_error(state: Dict[str, Any], error_message: str | None = None) -> int:
    with state["lock"]:
        state["error_count"] = int(state.get("error_count", 0)) + 1
        if error_message:
            state["last_error"] = error_message
        state["last_update_ts"] = time.time()
        return state["error_count"]


def record_camera_metrics(
    state: Dict[str, Any],
    *,
    infer_ms: float,
    decode_ms: float,
    preprocess_ms: float,
    yolo_ms: float,
    track_count: int,
) -> Dict[str, Any]:
    with state["lock"]:
        state["infer_times_ms"].append(float(infer_ms))
        state["decode_times_ms"].append(float(decode_ms))
        state["preprocess_times_ms"].append(float(preprocess_ms))
        state["yolo_times_ms"].append(float(yolo_ms))
        state["track_counts"].append(int(track_count))
        state["last_infer_ms"] = float(infer_ms)
        state["last_decode_ms"] = float(decode_ms)
        state["last_preprocess_ms"] = float(preprocess_ms)
        state["last_yolo_ms"] = float(yolo_ms)
        state["last_track_count"] = int(track_count)
        state["last_update_ts"] = time.time()
        return _camera_metrics_snapshot_unlocked(state)


def get_camera_metrics_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    with state["lock"]:
        return _camera_metrics_snapshot_unlocked(state)


def extract_appearance(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> dict:
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))

    if x2 <= x1 or y2 <= y1:
        return {"color_hist": None}

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return {"color_hist": None}

    hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return {"color_hist": hist}


def appearance_distance(hist1, hist2) -> float:
    if hist1 is None or hist2 is None:
        return 1.0
    try:
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    except Exception:
        return 1.0


def combined_track_score(iou_score: float, appearance_dist: float) -> float:
    app_similarity = 1.0 - appearance_dist

    if iou_score < 0.1:
        return 0.5 * iou_score + 0.5 * app_similarity
    if iou_score < 0.3:
        return 0.6 * iou_score + 0.4 * app_similarity
    return 0.7 * iou_score + 0.3 * app_similarity


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
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
    for tr in track_by_id.values():
        last_seen = tr.get("last_seen", 0.0)
        if now_ts - last_seen > recent_only_sec:
            continue
        if iou(det_box, tr["bbox"]) >= overlap_iou:
            return True
    return False


def predict_track_bbox(track: Dict[str, Any], now_ts: float) -> tuple[float, float, float, float]:
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


def update_track_motion(
    track: Dict[str, Any],
    prev_bbox: tuple[float, float, float, float],
    new_bbox: tuple[float, float, float, float],
    now_ts: float,
) -> None:
    prev_ts = float(track.get("last_seen", now_ts))
    dt = max(1e-3, now_ts - prev_ts)

    px1, py1, px2, py2 = prev_bbox
    nx1, ny1, nx2, ny2 = new_bbox

    track["vx"] = (nx1 - px1) / dt
    track["vy"] = (ny1 - py1) / dt
    track["vw"] = (nx2 - px2) / dt
    track["vh"] = (ny2 - py2) / dt


def hungarian_minimize(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
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
            pred_bbox = (
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
    for tid in output_track_ids:
        tr = track_by_id.get(tid)
        if not tr or tr.get("id") == hold_track.get("id"):
            continue
        if iou(hold_track["bbox"], tr["bbox"]) >= overlap_iou:
            return True
    return False


def dedupe_output_tracks(detections: List[Any], iou_thresh: float) -> List[Any]:
    if len(detections) <= 1:
        return detections

    sorted_dets = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: List[Any] = []
    for det in sorted_dets:
        det_box = (det.x, det.y, det.x + det.w, det.y + det.h)
        if all(iou(det_box, (k.x, k.y, k.x + k.w, k.y + k.h)) < iou_thresh for k in kept):
            kept.append(det)
    return kept
