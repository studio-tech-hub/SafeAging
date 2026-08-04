"""Background face recognition worker — keeps /infer YOLO path off the hot thread."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from . import face_engine
from .config import logger
from .face_identity import commit_face_match, commit_face_miss

_pool: Optional["FaceWorkerPool"] = None
_pool_lock = threading.Lock()


@dataclass
class FaceJob:
    camera_id: str
    track_id: int
    crop: np.ndarray
    state: Dict[str, Any]
    camera_lock: threading.Lock
    frame_idx: int


class FaceWorkerPool:
    def __init__(self, workers: int, max_queue: int) -> None:
        self.workers = max(1, workers)
        self.max_queue = max(4, max_queue)
        self._queue: queue.Queue[FaceJob | None] = queue.Queue(maxsize=self.max_queue)
        self._threads: list[threading.Thread] = []
        self._started = False
        self._dropped = 0
        self._completed = 0

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        for idx in range(self.workers):
            t = threading.Thread(
                target=self._loop,
                name=f"face-worker-{idx}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)
        logger.info(
            "Face worker pool started workers=%d queue_max=%d",
            self.workers,
            self.max_queue,
        )

    def submit(self, job: FaceJob) -> bool:
        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            self._dropped += 1
            return False

    def stats(self) -> Dict[str, int]:
        return {
            "queue_depth": self._queue.qsize(),
            "dropped": self._dropped,
            "completed": self._completed,
        }

    def _loop(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                if job.crop is None or job.crop.size == 0:
                    commit_face_miss(job.state, job.camera_lock, job.track_id, job.frame_idx)
                    continue
                match = face_engine.recognize_crop(job.crop)
                if match is not None:
                    commit_face_match(
                        job.state,
                        job.camera_lock,
                        job.track_id,
                        match,
                        job.frame_idx,
                    )
                else:
                    commit_face_miss(job.state, job.camera_lock, job.track_id, job.frame_idx)
                self._completed += 1
            except Exception as exc:
                logger.warning(
                    "[face-worker][%s] track=%s error: %s",
                    job.camera_id,
                    job.track_id,
                    exc,
                )
                commit_face_miss(job.state, job.camera_lock, job.track_id, job.frame_idx)
            finally:
                self._queue.task_done()


def get_pool() -> Optional[FaceWorkerPool]:
    return _pool


def ensure_started(workers: int, max_queue: int) -> FaceWorkerPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = FaceWorkerPool(workers=workers, max_queue=max_queue)
            _pool.start()
        return _pool


def submit_recognition(
    *,
    camera_id: str,
    track_id: int,
    crop: np.ndarray,
    state: Dict[str, Any],
    camera_lock: threading.Lock,
    frame_idx: int,
) -> bool:
    pool = get_pool()
    if pool is None:
        return False
    return pool.submit(
        FaceJob(
            camera_id=camera_id,
            track_id=track_id,
            crop=crop.copy(),
            state=state,
            camera_lock=camera_lock,
            frame_idx=frame_idx,
        )
    )


def mark_attempt_pending(
    state: Dict[str, Any],
    camera_lock: threading.Lock,
    track_id: int,
    frame_idx: int,
) -> None:
    """Record that a face job was queued so we do not re-queue every frame."""
    now_mono = time.monotonic()
    with camera_lock:
        identity: Dict[int, Dict[str, Any]] = state.setdefault("identity", {})
        prev = identity.get(track_id) or {}
        identity[track_id] = {
            **prev,
            "recognized": bool(prev.get("recognized")),
            "pending": True,
            "last_attempt": frame_idx,
            "last_attempt_ts": now_mono,
        }
