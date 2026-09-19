"""FallStage: bbox-based fall detection (Tier 1, every frame) plus optional
pose-based refinement (Tier 2, only on bbox-flagged candidates).

Extracted verbatim (no behaviour change) from former section "3.2)" of
`_run_person_detection_pipeline`. Keeps its own try/except exactly as
before: a fall-detection error is logged and swallowed, never failing the
whole /infer request.
"""
from __future__ import annotations

from ...config import ENABLE_FALL_DETECTION, ENABLE_POSE_FALL, POSE_IMGSZ, POSE_KEYPOINT_CONF, logger
from ... import pose_backend
from .base import Stage


class FallStage(Stage):
    def run(self, ctx) -> None:
        from ... import api as _api

        camera_id = ctx.camera_id
        detections = ctx.detections
        track_by_id = ctx.track_by_id
        fall_detector = ctx.fall_detector
        camera_lock = ctx.camera_lock
        camera_frame_idx = ctx.camera_frame_idx
        yolo_frame = ctx.yolo_frame

        if not (ENABLE_FALL_DETECTION and fall_detector):
            return

        try:
            # Prepare detection data for fall detection
            fall_input_detections = []
            for det in detections:
                if not det.stable or det.degraded:
                    continue
                # P1-1: use the raw YOLO measurement (measurement_bbox), not
                # the rendered detection box — det.x/y/w/h now reflect the
                # smoothed track box, and fall-detection's velocity/aspect
                # ratio math needs true measurements so smoothing doesn't
                # damp or lag the real fall signal.
                tr = track_by_id.get(det.track_id)
                raw_bbox = tr.get("measurement_bbox") if tr else None
                bbox = raw_bbox or (det.x, det.y, det.x + det.w, det.y + det.h)
                fall_input_detections.append({
                    'track_id': det.track_id,
                    'bbox': bbox,
                    'confidence': det.score,
                })

            # Tier 1: bbox fall every frame
            with camera_lock:
                fall_results = fall_detector.update(
                    fall_input_detections,
                    camera_frame_idx,
                    pose_hints=None,
                )
                fall_stats = fall_detector.get_stats()

            # Tier 2: pose on person crop only when bbox signals pending fall
            if ENABLE_POSE_FALL and fall_input_detections:
                pose_candidates = fall_detector.get_pose_candidate_track_ids()
                if pose_candidates:
                    pose_hints = pose_backend.match_pose_to_crops(
                        yolo_frame,
                        fall_input_detections,
                        pose_candidates,
                        conf=POSE_KEYPOINT_CONF,
                        imgsz=POSE_IMGSZ,
                    )
                    if pose_hints:
                        with camera_lock:
                            refined = fall_detector.apply_pose_hints(pose_hints)
                            for tid, fell in refined.items():
                                if fell:
                                    fall_results[tid] = True
                            fall_stats = fall_detector.get_stats()

            # Mark detections with fall status
            for det in detections:
                det.fall_detected = fall_results.get(det.track_id, False)
                if det.fall_detected:
                    logger.warning(f"[{camera_id}] FALL DETECTED: Person {det.track_id} (score={det.score:.2f})")

            if fall_stats['total_fallen'] > 0:
                logger.info(f"[{camera_id}] Fall Status: {fall_stats['total_fallen']} person(s) fallen out of {fall_stats['total_tracked']} tracked")

            ctx.fall_stats = fall_stats

        except Exception as e:
            error_message = f"Fall detection error: {type(e).__name__}: {e}"
            _api._record_camera_infer_error(ctx.state, error_message)
            logger.error(f"[{camera_id}] {error_message}")
            # Fall detection errors don't stop inference, just log and continue
