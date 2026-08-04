"""Shared face identity cache helpers (sync + async worker paths)."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from . import face_engine


def apply_cached_identity_to_det(det: Any, cached: Optional[Dict[str, Any]]) -> None:
    if cached and cached.get("recognized"):
        det.recognized = True
        det.person_id = cached.get("person_id")
        det.person_name = cached.get("name")
        det.person_gender = cached.get("gender")
        det.person_age = cached.get("age")
        det.person_no = cached.get("no")
        return
    det.recognized = False
    det.person_id = None
    det.person_name = None
    det.person_gender = None
    det.person_age = None
    det.person_no = None


def commit_face_match(
    state: Dict[str, Any],
    camera_lock: Any,
    track_id: int,
    match: face_engine.FaceMatch,
    camera_frame_idx: int,
) -> Dict[str, Any]:
    now_mono = time.monotonic()
    with camera_lock:
        identity: Dict[int, Dict[str, Any]] = state.setdefault("identity", {})
        person_no_map: Dict[str, int] = state.setdefault("person_no_map", {})
        no = person_no_map.get(match.person_id)
        if no is None:
            no = int(state.get("next_person_no", 1))
            person_no_map[match.person_id] = no
            state["next_person_no"] = no + 1
        cached = {
            "recognized": True,
            "person_id": match.person_id,
            "name": match.name,
            "gender": match.gender,
            "age": match.age,
            "no": no,
            "score": match.score,
            "last_attempt": camera_frame_idx,
            "last_attempt_ts": now_mono,
            "pending": False,
        }
        identity[track_id] = cached
    return cached


def commit_face_miss(
    state: Dict[str, Any],
    camera_lock: Any,
    track_id: int,
    camera_frame_idx: int,
) -> Dict[str, Any]:
    now_mono = time.monotonic()
    cached = {
        "recognized": False,
        "person_id": None,
        "name": None,
        "gender": None,
        "age": None,
        "no": None,
        "last_attempt": camera_frame_idx,
        "last_attempt_ts": now_mono,
        "pending": False,
    }
    with camera_lock:
        identity: Dict[int, Dict[str, Any]] = state.setdefault("identity", {})
        identity[track_id] = cached
    return cached
