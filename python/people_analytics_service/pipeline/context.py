"""P2-1 step 2: InferContext — the explicit, shared data container that
replaces the ~50 loose local variables `_run_person_detection_pipeline`
used to thread through its ~900 lines by hand.

Pure data container, no behaviour: every field here previously existed as a
bare local variable in api.py. Stages read/write attributes on the same
InferContext instance instead of sharing a Python closure's locals.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np

if TYPE_CHECKING:
    # Only for type-checkers: importing api.Detection at runtime here would
    # be a circular import (api.py imports the pipeline package).
    from ..api import Detection


@dataclass
class InferContext:
    # ---- Inputs (set once, at construction, from the already-decoded frame
    # handed off by the route handler's transport-specific decode step) ----
    frame: np.ndarray
    camera_id: str
    state: Dict[str, Any]
    cam_seq: int
    request_started_at: float
    transport_name: str
    transport_encoded_bytes: int
    transport_raw_bytes: int
    decode_time_ms: float
    route_label: str = "/infer"

    # ---- Control flow (early-return support, mirrors the original
    # function's early `return []` on a decode/YOLO failure) ----
    stopped: bool = False
    result: Optional[List[Any]] = None

    # ---- Populated by PreprocessStage ----
    camera_request_count: int = 0
    preprocess_time_ms: float = 0.0
    H: int = 0
    W: int = 0
    snapshot_source_frame: Optional[np.ndarray] = None
    undistort_crop_offset: Tuple[int, int] = (0, 0)
    pre_roi_shape: Optional[Tuple[int, ...]] = None
    cam_cfg: Optional[Dict[str, Any]] = None
    roi_box: Optional[Tuple[float, float, float, float]] = None
    roi_type: Optional[str] = None
    conf_threshold: float = 0.0
    iou_threshold: float = 0.0
    enable_face_recognition: bool = False

    # ---- Populated by DetectStage ----
    yolo_time_ms: float = 0.0
    yolo_model: Any = None
    yolo_frame: Optional[np.ndarray] = None
    yolo_inv_scale: float = 1.0
    yolo_result: Any = None  # ultralytics Results object

    # ---- Populated by TrackStage ----
    now: float = 0.0
    camera_lock: Optional[threading.Lock] = None
    camera_frame_idx: int = 0
    tracks: List[Dict[str, Any]] = field(default_factory=list)
    track_history_snapshot: List[Dict[str, Any]] = field(default_factory=list)
    next_id: int = 0
    fall_detector: Any = None
    track_by_id: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    raw_dets: List[Dict[str, Any]] = field(default_factory=list)
    matched_ids: Set[int] = field(default_factory=set)
    output_track_ids: Set[int] = field(default_factory=set)
    detections: List["Detection"] = field(default_factory=list)

    # ---- Populated by FallStage ----
    fall_stats: Optional[Dict[str, Any]] = None

    # ---- Populated by FaceStage (also runs the coordinate remap that used
    # to live between Fall and Face — see FaceStage docstring) ----
    # (no extra fields beyond mutating ctx.detections in place)

    # ---- Populated by ZoneStage ----
    zone_violations_this_frame: List[Any] = field(default_factory=list)

    # ---- Populated by PersistStage ----
    tracks_count: int = 0
    unique_count: int = 0
    total_infer_ms: float = 0.0
    metrics_snapshot: Optional[Dict[str, Any]] = None
    new_outbox_ids: Set[int] = field(default_factory=set)
