"""Unit tests for async face worker."""

from __future__ import annotations

import threading
import time

import numpy as np


def test_face_worker_processes_job(monkeypatch):
    from people_analytics_service import face_worker
    from people_analytics_service.face_engine import FaceMatch

    state = {"identity": {}, "person_no_map": {}, "next_person_no": 1}
    lock = threading.Lock()
    called = {"n": 0}

    def fake_recognize(_crop):
        called["n"] += 1
        return FaceMatch(
            person_id="p1",
            name="Alice",
            gender="F",
            age=70,
            score=0.91,
        )

    monkeypatch.setattr(face_worker.face_engine, "recognize_crop", fake_recognize)

    pool = face_worker.FaceWorkerPool(workers=1, max_queue=4)
    pool.start()
    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    ok = pool.submit(
        face_worker.FaceJob(
            camera_id="cam1",
            track_id=7,
            crop=crop,
            state=state,
            camera_lock=lock,
            frame_idx=10,
        )
    )
    assert ok is True
    deadline = time.time() + 3.0
    while time.time() < deadline:
        with lock:
            cached = state["identity"].get(7)
        if cached and cached.get("recognized"):
            break
        time.sleep(0.05)
    with lock:
        cached = state["identity"][7]
    assert cached["recognized"] is True
    assert cached["name"] == "Alice"
    assert called["n"] == 1
