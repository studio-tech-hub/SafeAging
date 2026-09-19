"""PreprocessStage: calib capture, debug frame stats, undistort, ROI crop,
CLAHE/enhancement, and per-camera config resolution (confidence/IOU
thresholds + face-recognition toggle).

Extracted verbatim (no behaviour change) from the top ~110 lines of the
former `_run_person_detection_pipeline` — inline comment sections "1)"
through "1.7)". Any exception here is intentionally NOT caught: it
propagates to the orchestrator's outer try/except in api.py, exactly as it
did when this was inline code inside that same outer try block.
"""
from __future__ import annotations

import os
import time

import cv2

from ...config import (
    CONFIDENCE_THRESHOLD,
    ENABLE_CLAHE,
    ENABLE_FACE_RECOGNITION,
    ENABLE_FRAME_ENHANCEMENT,
    ENABLE_ROI,
    ENABLE_UNDISTORT,
    IOU_THRESHOLD,
    SAVE_DEBUG_SAMPLES,
    logger,
)
from ...config_engine import get_per_camera_config_sync
from ...preprocess import apply_roi, preprocess_frame, undistort_frame
from .base import Stage


class PreprocessStage(Stage):
    def run(self, ctx) -> None:
        # Lazy import: api.py imports this package at module load, so a
        # top-level `from ..api import ...` here would be circular.
        from ... import api as _api

        camera_id = ctx.camera_id
        cam_seq = ctx.cam_seq
        frame = ctx.frame

        ctx.camera_request_count = cam_seq
        ctx.preprocess_time_ms = 0.0

        H, W = frame.shape[:2]
        _api._maybe_capture_calib_frame(camera_id, frame, cam_seq)

        # Keep a reference to the originally decoded full frame for event
        # snapshots. Detection coords are later remapped back to this input
        # space, so snapshots stored for review/enrollment match the bbox
        # coordinates we persist.
        ctx.snapshot_source_frame = frame

        # Debug: check if frame is mostly empty/dark
        mean_bgr = cv2.mean(frame)
        frame_mean = float((mean_bgr[0] + mean_bgr[1] + mean_bgr[2]) / 3.0)
        gray_for_stats = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        frame_min_val, frame_max_val, _, _ = cv2.minMaxLoc(gray_for_stats)
        frame_max = int(frame_max_val)
        frame_min = int(frame_min_val)
        if cam_seq % 20 == 0:
            logger.debug(f"[{camera_id}] Frame: {W}x{H}, mean={frame_mean:.1f}, min={frame_min}, max={frame_max}")
            logger.debug(
                f"[{camera_id}] Transport: {ctx.transport_name}, "
                f"encoded_bytes={ctx.transport_encoded_bytes}, raw_bytes={ctx.transport_raw_bytes}"
            )
            if SAVE_DEBUG_SAMPLES:
                try:
                    os.makedirs('frame_samples', exist_ok=True)
                    sample_path = f'frame_samples/{camera_id}_{cam_seq:06d}.jpg'
                    cv2.imwrite(sample_path, frame)
                    logger.debug(f"[{camera_id}] Saved frame sample to {sample_path}")
                except Exception as e:
                    logger.warning(f"Failed to save frame sample: {e}")
        logger.debug(f"[{camera_id}] Frame size: {W}x{H}")

        # Store transform metadata for coordinate mapping (before undistort/ROI)
        undistort_crop_offset = (0, 0)

        # 1.4) Undistortion (for wide-angle/fisheye cameras)
        if ENABLE_UNDISTORT:
            undistort_start = time.perf_counter()
            frame, undistort_crop_offset = undistort_frame(frame)
            undistort_time_ms = (time.perf_counter() - undistort_start) * 1000.0
            ctx.preprocess_time_ms += undistort_time_ms
            if cam_seq % 20 == 0:
                logger.debug(f"[{camera_id}] Undistortion: {undistort_time_ms:.1f}ms")
        pre_roi_shape = frame.shape

        cam_cfg = get_per_camera_config_sync(camera_id)
        roi_override = None
        if cam_cfg and isinstance(cam_cfg.get("extra"), dict):
            roi_override = cam_cfg["extra"].get("roi")

        # 1.5) ROI Crop (Region of Interest)
        roi_box = None
        roi_type = None
        if ENABLE_ROI:
            roi_start = time.perf_counter()
            frame, roi_box, roi_type = apply_roi(frame, roi_override=roi_override)
            roi_time_ms = (time.perf_counter() - roi_start) * 1000.0
            ctx.preprocess_time_ms += roi_time_ms
            H_roi, W_roi = frame.shape[:2]
            if cam_seq % 20 == 0:
                logger.debug(f"[{camera_id}] ROI ({roi_type}): {H_roi}x{W_roi}, time={roi_time_ms:.1f}ms")
        else:
            roi_box = (0, 0, frame.shape[1], frame.shape[0])

        H, W = frame.shape[:2]

        # 1.6) Preprocess frame for wide-angle optimization
        if ENABLE_CLAHE or ENABLE_FRAME_ENHANCEMENT:
            preprocess_start = time.perf_counter()
            frame = preprocess_frame(frame)
            preprocess_step_ms = (time.perf_counter() - preprocess_start) * 1000.0
            ctx.preprocess_time_ms += preprocess_step_ms
            if cam_seq % 20 == 0:
                logger.debug(
                    f"[{camera_id}] Frame preprocessing: {preprocess_step_ms:.1f}ms "
                    f"(CLAHE={ENABLE_CLAHE}, Enhancement={ENABLE_FRAME_ENHANCEMENT})"
                )

        # 1.7) Per-camera config override (P2.2)
        conf_threshold = (
            cam_cfg["confidence_threshold"]
            if cam_cfg and cam_cfg.get("confidence_threshold") is not None
            else CONFIDENCE_THRESHOLD
        )
        iou_threshold = (
            cam_cfg["iou_threshold"]
            if cam_cfg and cam_cfg.get("iou_threshold") is not None
            else IOU_THRESHOLD
        )

        # Per-camera face-recognition toggle (P1-6). Lives under
        # extra.enable_face_recognition (a bool) so it needs no schema
        # migration, same as the roi override above. Absence
        # (None/missing/non-bool) means "inherit the global
        # ENABLE_FACE_RECOGNITION setting" — existing deployments that never
        # set this are completely unaffected.
        enable_face_recognition = ENABLE_FACE_RECOGNITION
        if cam_cfg and isinstance(cam_cfg.get("extra"), dict):
            face_recog_override = cam_cfg["extra"].get("enable_face_recognition")
            if isinstance(face_recog_override, bool):
                enable_face_recognition = face_recog_override

        ctx.frame = frame
        ctx.H, ctx.W = H, W
        ctx.undistort_crop_offset = undistort_crop_offset
        ctx.pre_roi_shape = pre_roi_shape
        ctx.cam_cfg = cam_cfg
        ctx.roi_box = roi_box
        ctx.roi_type = roi_type
        ctx.conf_threshold = conf_threshold
        ctx.iou_threshold = iou_threshold
        ctx.enable_face_recognition = enable_face_recognition
