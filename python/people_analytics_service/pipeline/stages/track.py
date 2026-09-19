"""TrackStage: snapshot per-camera track state, turn raw YOLO boxes into
filtered detections, run global (Hungarian) track association, age/expire
unmatched tracks, and select the final stable output boxes — including the
"degraded" raw-YOLO fallback when service-side tracking filters out
everything but YOLO still sees a person.

Extracted verbatim (no behaviour change) from former section "3)" of
`_run_person_detection_pipeline`. No stage-local try/except here (same as
the original): an unhandled exception propagates to the orchestrator's
outer try/except in api.py, exactly as it did inline.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from ...config import (
    BBOX_SMOOTHING,
    ENABLE_HISTORY_REMATCH,
    ENABLE_POST_NMS,
    HOLD_SUPPRESS_IOU,
    MATCH_IOU_THRESHOLD,
    MIN_DETECTION_AREA,
    NEW_TRACK_MIN_CONFIDENCE,
    OUTPUT_DEDUPE_IOU,
    OUTPUT_TENTATIVE_TRACKS,
    PERSON_MIN_HW_RATIO,
    POST_NMS_IOU,
    TENTATIVE_MAX_MISSES,
    TRACK_DUPLICATE_IOU,
    TRACK_MAX_MISSES,
    TRACK_MIN_HITS,
    TRACK_OUTPUT_HOLD_TIME,
    logger,
)
from ...tracking import (
    appearance_distance,
    dedupe_output_tracks,
    global_track_assignment,
    has_duplicate_track_overlap,
    iou,
    overlaps_confirmed_track,
    post_nms_dedupe,
    select_output_bbox,
    smooth_bbox,
    suppress_overlapping_holds,
    update_track_motion,
)
from .base import Stage


class TrackStage(Stage):
    def run(self, ctx) -> None:
        from ... import api as _api  # Detection model lives in api.py

        Detection = _api.Detection
        camera_id = ctx.camera_id
        state = ctx.state
        frame = ctx.frame
        H, W = ctx.H, ctx.W
        yolo_inv_scale = ctx.yolo_inv_scale
        r = ctx.yolo_result

        now = time.time()
        camera_lock = state["lock"]
        with camera_lock:
            state["frame_idx"] = int(state.get("frame_idx", 0)) + 1
            camera_frame_idx = state["frame_idx"]
            tracks = [dict(tr) for tr in state["tracks"]]
            track_history_snapshot = [dict(tr) for tr in state["track_history"]]
            next_id = state["next_id"]
            fall_detector = state["fall_detector"]

        detections: List[Any] = []

        # Track state (persistent) — built *before* raw-detection filtering
        # so the aspect-ratio filter below can be track-state-aware (P1-2):
        # it needs to know which confirmed tracks already exist to decide
        # whether a wide box is furniture (new candidate) or a person who
        # has fallen (existing track).
        track_by_id: Dict[int, Dict[str, Any]] = {tr["id"]: tr for tr in tracks}
        for tr in track_by_id.values():
            tr.setdefault("status", "confirmed")
            tr.setdefault("hits", TRACK_MIN_HITS)
            tr.setdefault("misses", 0)
            tr.setdefault("vx", 0.0)
            tr.setdefault("vy", 0.0)
            tr.setdefault("vw", 0.0)
            tr.setdefault("vh", 0.0)
            tr.setdefault("measurement_bbox", tr.get("bbox"))

        # Build raw detections first (for optional extra NMS)
        raw_dets: List[Dict[str, Any]] = []
        boxes_iter = r.boxes if r.boxes is not None else []
        for box in boxes_iter:
            try:
                cls_id = int(box.cls[0].item())
                if cls_id != 0:  # Only person
                    continue

                score = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                if yolo_inv_scale != 1.0:
                    x1 *= yolo_inv_scale
                    y1 *= yolo_inv_scale
                    x2 *= yolo_inv_scale
                    y2 *= yolo_inv_scale

                # Clamp to frame boundaries
                x1 = max(0.0, min(x1, W - 1.0))
                y1 = max(0.0, min(y1, H - 1.0))
                x2 = max(0.0, min(x2, W - 1.0))
                y2 = max(0.0, min(y2, H - 1.0))

                w_box = x2 - x1
                h_box = y2 - y1
                area = w_box * h_box
                hw_ratio = h_box / max(1.0, w_box)

                # Filter by minimum area
                if w_box <= 1.0 or h_box <= 1.0 or area < MIN_DETECTION_AREA:
                    logger.debug(f"[{camera_id}] Skipping small detection: {w_box:.1f}x{h_box:.1f} (area={area:.0f} < {MIN_DETECTION_AREA})")
                    continue

                det_box = (x1, y1, x2, y2)

                # Reject furniture-like wide boxes that YOLO occasionally
                # confuses as people — but only for brand-new track
                # candidates. A wide box that overlaps an already-confirmed
                # track is far more likely that same person now lying down
                # (a fall) than furniture, so the filter must never drop a
                # person mid-track (P1-2).
                if hw_ratio < PERSON_MIN_HW_RATIO:
                    if not overlaps_confirmed_track(det_box, track_by_id, MATCH_IOU_THRESHOLD):
                        logger.debug(
                            f"[{camera_id}] Skipping wide person-like detection: "
                            f"{w_box:.1f}x{h_box:.1f} ratio={hw_ratio:.2f} < {PERSON_MIN_HW_RATIO:.2f}"
                        )
                        continue
                    logger.debug(
                        f"[{camera_id}] Wide box kept — overlaps a confirmed track "
                        f"(possible fall posture): {w_box:.1f}x{h_box:.1f} ratio={hw_ratio:.2f}"
                    )

                raw_dets.append({"bbox": det_box, "score": score})
            except Exception as e:
                logger.warning(f"[{camera_id}] Error parsing box: {type(e).__name__}: {e}")
                continue

        # Extra dedupe to avoid overlapping boxes for the same person
        if ENABLE_POST_NMS and len(raw_dets) > 1:
            raw_dets = post_nms_dedupe(raw_dets, POST_NMS_IOU)

        matched_ids = set()
        output_track_ids = set()

        # Global optimal 1-1 matching (Hungarian) over gated det-track pairs.
        det_to_track, det_appearances = global_track_assignment(raw_dets, track_by_id, frame, now)

        for det_idx, track_id in det_to_track.items():
            det = raw_dets[det_idx]
            det_box = det["bbox"]
            score = det["score"]
            det_appearance = det_appearances[det_idx]

            tr = track_by_id.get(track_id)
            if not tr:
                continue

            prev_bbox = tr.get("bbox", det_box)
            prev_measurement_bbox = tr.get("measurement_bbox", prev_bbox)
            if BBOX_SMOOTHING > 0.0 and tr.get("bbox"):
                tr["bbox"] = smooth_bbox(prev_bbox, det_box, BBOX_SMOOTHING)
            else:
                tr["bbox"] = det_box

            update_track_motion(
                tr,
                prev_bbox=prev_measurement_bbox,
                new_bbox=det_box,
                now_ts=now
            )
            tr["last_seen"] = now
            tr["appearance"] = det_appearance
            tr["measurement_bbox"] = det_box
            tr["score"] = score
            tr["misses"] = 0
            tr["hits"] = int(tr.get("hits", 0)) + 1
            if tr["status"] != "confirmed" and tr["hits"] >= TRACK_MIN_HITS:
                tr["status"] = "confirmed"
            elif tr["status"] == "lost":
                tr["status"] = "confirmed"

            track_by_id[track_id] = tr
            matched_ids.add(track_id)

        for tr in track_by_id.values():
            if tr["id"] in matched_ids:
                continue
            tr["misses"] = int(tr.get("misses", 0)) + 1
            if tr.get("status") == "tentative":
                if tr["misses"] > TENTATIVE_MAX_MISSES:
                    tr["status"] = "deleted"
            else:
                tr["status"] = "lost"

        unmatched_det_idxs = [idx for idx in range(len(raw_dets)) if idx not in det_to_track]
        for det_idx in unmatched_det_idxs:
            try:
                det = raw_dets[det_idx]
                det_box = det["bbox"]
                score = det["score"]
                det_appearance = det_appearances[det_idx]
                track_id = None

                # Optional history rematch: appearance-only for returning person.
                if ENABLE_HISTORY_REMATCH and track_history_snapshot and det_appearance.get("color_hist") is not None:
                    best_hist_score = -1.0
                    best_hist = None
                    for hist_tr in track_history_snapshot:
                        if hist_tr.get("id") in matched_ids:
                            continue
                        if hist_tr.get("appearance") and hist_tr["appearance"].get("color_hist") is not None:
                            app_dist = appearance_distance(hist_tr["appearance"]["color_hist"], det_appearance["color_hist"])
                            if app_dist < 0.4:
                                hist_score = 1.0 - app_dist
                                if hist_score > best_hist_score:
                                    best_hist_score = hist_score
                                    best_hist = hist_tr
                    if best_hist is not None:
                        track_id = int(best_hist["id"])
                        track_by_id[track_id] = {
                            "id": track_id,
                            "bbox": det_box,
                            "measurement_bbox": det_box,
                            "last_seen": now,
                            "created_at": best_hist.get("created_at", now),
                            "appearance": det_appearance,
                            "score": score,
                            "status": "confirmed",
                            "hits": max(TRACK_MIN_HITS, int(best_hist.get("hits", TRACK_MIN_HITS))),
                            "misses": 0,
                            "vx": 0.0,
                            "vy": 0.0,
                            "vw": 0.0,
                            "vh": 0.0,
                        }
                        logger.debug(f"[{camera_id}] Re-matched track ID={track_id} from history")

                if track_id is None:
                    if score < NEW_TRACK_MIN_CONFIDENCE:
                        continue

                    if has_duplicate_track_overlap(
                        det_box=det_box,
                        track_by_id=track_by_id,
                        now_ts=now,
                        overlap_iou=TRACK_DUPLICATE_IOU,
                        recent_only_sec=max(TRACK_OUTPUT_HOLD_TIME, 0.5),
                    ):
                        continue

                    track_id = next_id
                    next_id += 1
                    initial_hits = 1
                    track_by_id[track_id] = {
                        "id": track_id,
                        "bbox": det_box,
                        "measurement_bbox": det_box,
                        "last_seen": now,
                        "created_at": now,
                        "appearance": det_appearance,
                        "score": score,
                        "status": "confirmed" if initial_hits >= TRACK_MIN_HITS else "tentative",
                        "hits": initial_hits,
                        "misses": 0,
                        "vx": 0.0,
                        "vy": 0.0,
                        "vw": 0.0,
                        "vh": 0.0,
                    }

                matched_ids.add(track_id)
            except Exception as e:
                logger.warning(f"[{camera_id}] Error processing unmatched detection: {type(e).__name__}: {e}")
                continue

        # Output selection: confirmed tracks by default; optional tentative output via config.
        for tr in track_by_id.values():
            if tr.get("status") == "deleted":
                continue

            is_confirmed = tr.get("status") == "confirmed"
            is_tentative = tr.get("status") == "tentative"

            if tr["id"] in matched_ids:
                if is_confirmed or (OUTPUT_TENTATIVE_TRACKS and is_tentative):
                    output_track_ids.add(tr["id"])
                continue

            if not is_confirmed:
                continue
            if int(tr.get("misses", 0)) > TRACK_MAX_MISSES:
                continue
            if now - tr.get("last_seen", 0.0) > TRACK_OUTPUT_HOLD_TIME:
                continue
            if suppress_overlapping_holds(tr, output_track_ids, track_by_id, HOLD_SUPPRESS_IOU):
                continue
            output_track_ids.add(tr["id"])

        # Build final detections from stable track state (one bbox per track_id).
        detections = []
        for track_id in sorted(output_track_ids):
            tr = track_by_id.get(track_id)
            if not tr:
                continue
            is_confirmed = tr.get("status") == "confirmed"
            # P1-1: render the smoothed track box — see select_output_bbox()
            # docstring. measurement_bbox remains the source of truth fed to
            # fall detection below, which needs true measurements, not the
            # smoothed/lagged box.
            bbox_out = select_output_bbox(tr)
            x1, y1, x2, y2 = bbox_out
            w_box = max(0.0, x2 - x1)
            h_box = max(0.0, y2 - y1)
            if w_box <= 1.0 or h_box <= 1.0:
                continue
            detections.append(Detection(
                cls="person",
                score=float(tr.get("score", 0.0)),
                x=float(x1),
                y=float(y1),
                w=float(w_box),
                h=float(h_box),
                track_id=int(track_id),
                stable=bool(is_confirmed),
                degraded=False,
            ))

        detections = dedupe_output_tracks(detections, OUTPUT_DEDUPE_IOU)

        if raw_dets and not detections:
            best_score = max(det["score"] for det in raw_dets)
            logger.warning(
                f"[{camera_id}] YOLO raw persons={len(raw_dets)} but output detections=0 "
                f"(best_score={best_score:.2f}, track_min_hits={TRACK_MIN_HITS}, "
                f"output_tentative={OUTPUT_TENTATIVE_TRACKS}, "
                f"new_track_min_confidence={NEW_TRACK_MIN_CONFIDENCE:.2f})"
            )

            # Degraded mode: if stable service-side tracking filters
            # everything out but YOLO still sees a person, fall back to raw
            # YOLO boxes so Nx can still render a bbox. The plugin must
            # treat these as render-only detections, not lifecycle events.
            fallback_detections: List[Any] = []
            used_track_ids: set[int] = set()
            reuse_window_sec = max(TRACK_OUTPUT_HOLD_TIME, 0.5)

            for det in sorted(raw_dets, key=lambda item: item["score"], reverse=True):
                det_box = det["bbox"]
                best_track_id = None
                best_iou_score = 0.0

                for tr in track_by_id.values():
                    if tr["id"] in used_track_ids:
                        continue
                    if now - tr.get("last_seen", 0.0) > reuse_window_sec:
                        continue

                    overlap = iou(det_box, tr["bbox"])
                    if overlap > best_iou_score:
                        best_iou_score = overlap
                        best_track_id = int(tr["id"])

                if best_track_id is None or best_iou_score < 0.10:
                    best_track_id = next_id
                    next_id += 1
                    track_by_id[best_track_id] = {
                        "id": best_track_id,
                        "bbox": det_box,
                        "measurement_bbox": det_box,
                        "last_seen": now,
                        "created_at": now,
                        "appearance": None,
                        "score": det["score"],
                        "status": "tentative",
                        "hits": 1,
                        "misses": 0,
                        "vx": 0.0,
                        "vy": 0.0,
                        "vw": 0.0,
                        "vh": 0.0,
                    }
                else:
                    tr = track_by_id[best_track_id]
                    tr["bbox"] = det_box
                    tr["measurement_bbox"] = det_box
                    tr["last_seen"] = now
                    tr["score"] = det["score"]
                    tr["misses"] = 0
                    if tr.get("status") == "lost":
                        tr["status"] = "tentative"
                    track_by_id[best_track_id] = tr

                used_track_ids.add(best_track_id)

                x1, y1, x2, y2 = det_box
                fallback_detections.append(Detection(
                    cls="person",
                    score=float(det["score"]),
                    x=float(x1),
                    y=float(y1),
                    w=float(max(0.0, x2 - x1)),
                    h=float(max(0.0, y2 - y1)),
                    track_id=int(best_track_id),
                    stable=False,
                    degraded=True,
                ))

            if fallback_detections:
                detections = fallback_detections
                logger.warning(
                    f"[{camera_id}] Fallback to raw YOLO detections for plugin continuity: "
                    f"{len(detections)} person(s)"
                )

        ctx.now = now
        ctx.camera_lock = camera_lock
        ctx.camera_frame_idx = camera_frame_idx
        ctx.tracks = tracks
        ctx.track_history_snapshot = track_history_snapshot
        ctx.next_id = next_id
        ctx.fall_detector = fall_detector
        ctx.track_by_id = track_by_id
        ctx.raw_dets = raw_dets
        ctx.matched_ids = matched_ids
        ctx.output_track_ids = output_track_ids
        ctx.detections = detections
