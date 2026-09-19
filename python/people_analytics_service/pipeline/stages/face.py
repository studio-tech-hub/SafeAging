"""FaceStage: remaps detection coordinates back to full input-frame space,
then (if enabled, globally or per-camera) runs real-time face identity.

Extracted verbatim (no behaviour change) from former sections "3.5)" and
"3.6)" of `_run_person_detection_pipeline`. The coordinate remap runs
UNCONDITIONALLY here, exactly as it did originally — it is not gated by the
face-recognition toggle, because ZoneStage (which runs after this one) also
needs detections already remapped into full input-frame pixel space.
"""
from __future__ import annotations

from ...config import logger
from ...preprocess import remap_detections_to_input_space
from .base import Stage


class FaceStage(Stage):
    def run(self, ctx) -> None:
        from ... import api as _api

        camera_id = ctx.camera_id
        detections = ctx.detections

        # 3.5) Map detections back to input frame coordinates. Runs after
        # remap so crop coords match the snapshot frame space.
        remap_detections_to_input_space(
            detections=detections,
            roi_box=ctx.roi_box,
            roi_type=ctx.roi_type,
            pre_roi_shape=ctx.pre_roi_shape,
            undistort_crop_offset=ctx.undistort_crop_offset,
        )

        # 3.6) Face recognition — real-time identity on the bounding box.
        # Throttled per track; identity cached for stable labels.
        # ctx.enable_face_recognition (global default, per-camera-overridable
        # — see PreprocessStage) gates the *entire* call, not just the
        # recognition attempt inside it: a camera with the override off
        # never runs face detection/bbox-refinement or queues an async
        # recognition job for this frame, not merely "skips using the
        # result" (P1-6).
        if ctx.enable_face_recognition and detections:
            try:
                _api._apply_face_identity(
                    camera_id=camera_id,
                    state=ctx.state,
                    detections=detections,
                    frame=ctx.snapshot_source_frame,
                    camera_frame_idx=ctx.camera_frame_idx,
                    camera_lock=ctx.camera_lock,
                )
            except Exception as _fe:
                logger.warning("[%s] Face recognition error: %s", camera_id, _fe)
