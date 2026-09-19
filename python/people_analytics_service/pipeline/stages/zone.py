"""ZoneStage: per-camera forbidden/monitored zone violation checks (P2.2).

Extracted verbatim (no behaviour change) from former section "3.7)" of
`_run_person_detection_pipeline`. Runs after FaceStage's remap so zone
geometry and detection coords are both in the full input-frame pixel space.
Keeps its own try/except exactly as before: a zone-check error is logged
and swallowed, never failing the whole /infer request.
"""
from __future__ import annotations

from ...config import logger
from ...zone_engine import check_zones, get_zones_for_camera_sync
from .base import Stage


class ZoneStage(Stage):
    def run(self, ctx) -> None:
        camera_id = ctx.camera_id
        detections = ctx.detections

        zone_violations_this_frame: list = []
        if detections:
            try:
                active_zones = get_zones_for_camera_sync(camera_id)
                if active_zones:
                    zone_violations_this_frame = check_zones(detections, active_zones, camera_id)
            except Exception as _ze:
                logger.warning("[%s] Zone check error: %s", camera_id, _ze)

        ctx.zone_violations_this_frame = zone_violations_this_frame
