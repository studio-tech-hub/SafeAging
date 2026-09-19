"""DetectStage: load model, optional input upscale, run YOLO (single-scale
or smart multi-scale), and light debug-frame dumping.

Extracted verbatim (no behaviour change) from former section "2)" of
`_run_person_detection_pipeline`. This is the one stage with its own
try/except that short-circuits the whole pipeline (matches the original's
early `return []` on a YOLO inference error) — implemented here via
`ctx.stopped` / `ctx.result`, checked by the orchestrator in api.py after
every stage.
"""
from __future__ import annotations

import os
import time

import cv2

from ...config import (
    ENABLE_INPUT_UPSCALE,
    ENABLE_MULTI_SCALE,
    MIN_INPUT_RESOLUTION,
    SAVE_DEBUG_SAMPLES,
    YOLO_IMGSZ,
    logger,
)
from ...model import load_model
from ...preprocess import ensure_min_input_side, multi_scale_inference_smart
from ... import yolo_backend
from .base import Stage


class DetectStage(Stage):
    def run(self, ctx) -> None:
        from ... import api as _api

        camera_id = ctx.camera_id
        cam_seq = ctx.cam_seq
        frame = ctx.frame

        try:
            # Startup preloads the model; keep this as a defensive fallback.
            yolo_model = load_model()

            if SAVE_DEBUG_SAMPLES and cam_seq % 200 == 0:
                try:
                    os.makedirs('frame_samples', exist_ok=True)
                    sample_pre = f'frame_samples/{camera_id}_{cam_seq:06d}_pre_yolo.jpg'
                    cv2.imwrite(sample_pre, frame)
                except Exception as e:
                    logger.debug(f"Failed to save pre-YOLO frame: {e}")

            yolo_start = time.perf_counter()

            yolo_frame = frame
            yolo_inv_scale = 1.0
            if ENABLE_INPUT_UPSCALE:
                yolo_frame, yolo_scale = ensure_min_input_side(frame, MIN_INPUT_RESOLUTION)
                if yolo_scale != 1.0:
                    yolo_inv_scale = 1.0 / yolo_scale

            # Choose inference strategy
            if ENABLE_MULTI_SCALE:
                # Smart multi-scale detection: only retry at larger scale if no detections
                r = multi_scale_inference_smart(yolo_model, yolo_frame, ctx.H, ctx.W)
            else:
                # Standard single-scale inference - PRODUCTION MODE
                r = yolo_backend.predict_yolo(
                    yolo_model,
                    yolo_frame,
                    conf=ctx.conf_threshold,
                    iou=ctx.iou_threshold,
                    classes=[0],  # person only
                    imgsz=YOLO_IMGSZ,
                    verbose=False,
                    augment=False,
                )

            yolo_time_ms = (time.perf_counter() - yolo_start) * 1000.0

            if SAVE_DEBUG_SAMPLES and cam_seq % 200 == 0:
                try:
                    os.makedirs('frame_samples', exist_ok=True)
                    frame_with_boxes = frame.copy()
                    if r.boxes is not None and len(r.boxes) > 0:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    sample_post = f'frame_samples/{camera_id}_{cam_seq:06d}_post_yolo.jpg'
                    cv2.imwrite(sample_post, frame_with_boxes)
                except Exception as e:
                    logger.debug(f"Failed to save post-YOLO frame: {e}")

            # Log detection details for debugging
            boxes_for_log = r.boxes if r.boxes is not None else []
            num_boxes = len(boxes_for_log)
            if cam_seq % 20 == 0:
                logger.debug(
                    f"[{camera_id}] YOLO: {num_boxes} objects "
                    f"(conf={ctx.conf_threshold:.2f}, iou={ctx.iou_threshold:.2f}, inference={yolo_time_ms:.1f}ms)"
                )
                if num_boxes > 0:
                    for i, box in enumerate(boxes_for_log[:3]):  # Show first 3
                        logger.debug(f"  Box {i}: cls={int(box.cls[0].item())}, conf={float(box.conf[0].item()):.2f}")
        except Exception as e:
            error_message = f"YOLO inference error: {type(e).__name__}: {e}"
            _api._record_camera_infer_error(ctx.state, error_message)
            logger.error(f"[{camera_id}] {error_message}")
            ctx.stopped = True
            ctx.result = []
            return

        ctx.yolo_model = yolo_model
        ctx.yolo_frame = yolo_frame
        ctx.yolo_inv_scale = yolo_inv_scale
        ctx.yolo_result = r
        ctx.yolo_time_ms = yolo_time_ms
