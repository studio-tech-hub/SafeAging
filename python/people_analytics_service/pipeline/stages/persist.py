"""PersistStage: anti-flicker output reuse, track TTL cleanup + history
buffer, unique-count/outbox-reported-ids bookkeeping, Prometheus metrics,
and non-blocking outbox enqueue (fall / new-detection / zone-violation
events, each carrying at most one shared snapshot JPEG per frame).

Extracted verbatim (no behaviour change) from former sections "4)" through
"6)" plus the metrics/outbox tail of `_run_person_detection_pipeline`. No
stage-local try/except here (same as the original): an unhandled exception
propagates to the orchestrator's outer try/except in api.py.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import cv2

from ...config import logger
from ...metrics import (
    ACTIVE_TRACKS,
    DECODE_LATENCY,
    DETECTIONS_PER_FRAME,
    INFER_LATENCY,
    INFER_REQUESTS,
    PREPROCESS_LATENCY,
    YOLO_LATENCY,
)
from ...tracking import record_camera_metrics
from ... import outbox_worker
from ...outbox_worker import OutboxEvent
from .base import Stage

if TYPE_CHECKING:
    from ...config import AppConfig
    from ..context import InferContext


def _det_person_fields(det) -> dict:
    return {
        "person_id": det.person_id,
        "person_name": det.person_name,
    }


class PersistStage(Stage):
    def run(self, ctx: "InferContext", config: "AppConfig") -> None:
        camera_id = ctx.camera_id
        state = ctx.state
        now = ctx.now
        track_by_id = ctx.track_by_id
        output_track_ids = ctx.output_track_ids
        detections = ctx.detections
        zone_violations_this_frame = ctx.zone_violations_this_frame

        with ctx.camera_lock:
            # 4) Anti-flicker: reuse last output if empty
            if not detections and state["last_output"]:
                time_since_last = now - state["last_time"]
                if time_since_last < config.flicker_reuse_time:
                    logger.debug(f"[{camera_id}] Reusing last output (anti-flicker, {time_since_last*1000:.0f}ms)")
                    detections = state["last_output"]
            else:
                state["last_output"] = detections
                state["last_time"] = now

            # 5) Track TTL cleanup with history buffer
            old_count = len(state["tracks"])

            active_tracks = []
            expired_tracks = []
            for tr in track_by_id.values():
                if tr.get("status") == "deleted":
                    expired_tracks.append(tr)
                    continue

                if tr.get("status") == "tentative" and int(tr.get("misses", 0)) > config.tentative_max_misses:
                    tr["status"] = "deleted"
                    expired_tracks.append(tr)
                    continue

                if now - tr.get("last_seen", 0.0) <= config.track_ttl:
                    active_tracks.append(tr)
                else:
                    expired_tracks.append(tr)

            if config.enable_history_rematch:
                state["track_history"] = [tr for tr in state["track_history"] if now - tr.get("last_seen", now) <= 30.0]
                history_candidates = [
                    tr for tr in expired_tracks
                    if tr.get("status") in {"confirmed", "lost"} and tr.get("appearance")
                ]
                state["track_history"].extend(history_candidates)
            else:
                state["track_history"] = []

            state["tracks"] = active_tracks
            removed = old_count - len(state["tracks"])
            if removed > 0:
                logger.debug(f"[{camera_id}] Removed {removed} stale tracks from active state")

            state["next_id"] = ctx.next_id

            # Purge cached face identity / crops for tracks that are no longer active
            active_track_ids = {tr["id"] for tr in active_tracks}
            identity_cache = state.get("identity")
            if identity_cache:
                for dead_tid in [t for t in identity_cache if t not in active_track_ids]:
                    identity_cache.pop(dead_tid, None)
            recent_crops_cache = state.get("recent_crops")
            if recent_crops_cache:
                for dead_tid in [t for t in recent_crops_cache if t not in active_track_ids]:
                    recent_crops_cache.pop(dead_tid, None)

            # 6) Count tracking with proper ID deduplication
            # Track IDs with their last_seen timestamp to properly age out old IDs
            if "seen_ids_with_time" not in state:
                state["seen_ids_with_time"] = {}  # {track_id: last_seen_timestamp}

            # Add confirmed tracks with enough hits
            stable_unique_ids = {
                int(track_id)
                for track_id in output_track_ids
                if track_id in track_by_id
                and track_by_id[track_id].get("status") == "confirmed"
                and int(track_by_id[track_id].get("hits", 0)) >= config.unique_count_min_hits
            }

            # Update timestamps for currently-active confirmed tracks
            for track_id in stable_unique_ids:
                state["seen_ids_with_time"][track_id] = now

            # Expire old IDs that haven't been seen recently (older than
            # config.track_ttl + grace period). Grace period allows re-detection of
            # same person after brief exit.
            id_expiry_threshold = config.track_ttl + 30.0  # Keep IDs for up to 45s after last detection
            expired_ids = [
                vid for vid, last_seen in state["seen_ids_with_time"].items()
                if now - last_seen > id_expiry_threshold
            ]
            for vid in expired_ids:
                del state["seen_ids_with_time"][vid]

            # Maintain backward compatibility with seen_ids set for any external code
            state["seen_ids"] = set(state["seen_ids_with_time"].keys())

            # S P1.3: track which IDs have already been reported to the outbox
            _reported_ids: set = state.setdefault("outbox_reported_ids", set())
            new_outbox_ids: set = stable_unique_ids - _reported_ids
            _reported_ids.update(stable_unique_ids)

            tracks_count = len(state["tracks"])
            unique_count = len(state["seen_ids"])

        total_infer_ms = (time.perf_counter() - ctx.request_started_at) * 1000.0
        metrics_snapshot = record_camera_metrics(
            state,
            infer_ms=total_infer_ms,
            decode_ms=ctx.decode_time_ms,
            preprocess_ms=ctx.preprocess_time_ms,
            yolo_ms=ctx.yolo_time_ms,
            track_count=tracks_count,
        )

        # Prometheus observations (success path)
        INFER_REQUESTS.labels(camera_id=camera_id, status="success").inc()
        INFER_LATENCY.labels(camera_id=camera_id).observe(total_infer_ms / 1000.0)
        DECODE_LATENCY.labels(camera_id=camera_id).observe(ctx.decode_time_ms / 1000.0)
        PREPROCESS_LATENCY.labels(camera_id=camera_id).observe(ctx.preprocess_time_ms / 1000.0)
        YOLO_LATENCY.labels(camera_id=camera_id).observe(ctx.yolo_time_ms / 1000.0)
        DETECTIONS_PER_FRAME.labels(camera_id=camera_id).observe(float(len(detections)))
        ACTIVE_TRACKS.labels(camera_id=camera_id).set(float(tracks_count))

        if ctx.camera_request_count == 1 or ctx.camera_request_count % config.metrics_log_interval == 0:
            logger.info(
                f"[{camera_id}] Summary: detections={len(detections)} "
                f"tracks={tracks_count} unique={unique_count} "
                f"infer_ms={total_infer_ms:.1f} decode_ms={ctx.decode_time_ms:.1f} "
                f"preprocess_ms={ctx.preprocess_time_ms:.1f} yolo_ms={ctx.yolo_time_ms:.1f} "
                f"p50_ms={metrics_snapshot['p50_inference_ms']:.1f} "
                f"p95_ms={metrics_snapshot['p95_inference_ms']:.1f} "
                f"errors={metrics_snapshot['error_count']}"
            )

        # ── S P1.3: non-blocking outbox enqueue ──────────────────────────────
        _now_utc = datetime.now(tz=timezone.utc)

        fall_dets = [d for d in detections if d.fall_detected]
        has_new_detections = any(d.track_id in new_outbox_ids for d in detections)
        needs_snapshot = bool(fall_dets or zone_violations_this_frame or has_new_detections)

        # Encode the event snapshot JPEG at most once per frame, only when an event fires.
        _snap_jpeg: Optional[bytes] = None
        if needs_snapshot:
            try:
                ok, _snap_buf = cv2.imencode(
                    ".jpg", ctx.snapshot_source_frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
                )
                if ok:
                    _snap_jpeg = bytes(_snap_buf)
            except Exception as _enc_exc:
                logger.debug("[%s] Snapshot encode failed: %s", camera_id, _enc_exc)

        # Fall events — always carry a snapshot
        for _det in fall_dets:
            outbox_worker.enqueue_event(OutboxEvent(
                camera_id=camera_id,
                event_type="fall",
                occurred_at=_now_utc,
                track_id=_det.track_id,
                cls=_det.cls,
                confidence=_det.score,
                bbox_x=_det.x,
                bbox_y=_det.y,
                bbox_w=_det.w,
                bbox_h=_det.h,
                snapshot_jpeg=_snap_jpeg,
                **_det_person_fields(_det),
            ))

        # New confirmed-track events — first appearance, now with a snapshot for review/enrollment
        for _det in detections:
            if _det.track_id in new_outbox_ids:
                outbox_worker.enqueue_event(OutboxEvent(
                    camera_id=camera_id,
                    event_type="detection",
                    occurred_at=_now_utc,
                    track_id=_det.track_id,
                    cls=_det.cls,
                    confidence=_det.score,
                    bbox_x=_det.x,
                    bbox_y=_det.y,
                    bbox_w=_det.w,
                    bbox_h=_det.h,
                    snapshot_jpeg=_snap_jpeg,
                    **_det_person_fields(_det),
                ))

        # Zone violation events (P2.2) — one event per violation per entry, now with a snapshot
        for _zv in zone_violations_this_frame:
            _zv_det = next((d for d in detections if d.track_id == _zv.track_id), None)
            outbox_worker.enqueue_event(OutboxEvent(
                camera_id=camera_id,
                event_type="zone_violation",
                occurred_at=_now_utc,
                track_id=_zv.track_id,
                cls="person",
                confidence=_zv_det.score if _zv_det else 0.0,
                bbox_x=_zv_det.x if _zv_det else 0.0,
                bbox_y=_zv_det.y if _zv_det else 0.0,
                bbox_w=_zv_det.w if _zv_det else 0.0,
                bbox_h=_zv_det.h if _zv_det else 0.0,
                zone_id=_zv.zone_id,
                snapshot_jpeg=_snap_jpeg,
                **(_det_person_fields(_zv_det) if _zv_det else {}),
            ))

        ctx.detections = detections
        ctx.tracks_count = tracks_count
        ctx.unique_count = unique_count
        ctx.total_infer_ms = total_infer_ms
        ctx.metrics_snapshot = metrics_snapshot
        ctx.new_outbox_ids = new_outbox_ids
