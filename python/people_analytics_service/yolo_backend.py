"""YOLO inference backend selection for edge accelerators (QCS6490).

Backends
--------
- ``cpu``      — Ultralytics + ONNX Runtime CPU (default)
- ``cpu_lean`` — Direct ORT session pool (lower overhead; needs .onnx at >=640px)
- ``qnn_htp``  — ONNX Runtime QNN plugin → Hexagon NPU (needs oe-linux QNN libs)
- ``qnn_gpu``  — ONNX Runtime QNN plugin → Adreno GPU
- ``qnn``      — alias for ``qnn_htp``

Not NVIDIA CUDA. On QCS6490 OE Linux, PyPI onnxruntime-qnn wheels ship Android QNN
libs — HTP/GPU may fail until QAIRT oe-linux build is installed. ONNX Runtime 1.24.4
CPU is still ~4x faster than 1.27 on this board (~140 ms vs ~550 ms/frame).
"""

from __future__ import annotations

import contextlib
import os
import queue
import threading
from typing import Any, Iterator, List, Optional

import cv2
import numpy as np
import torch

from .config import CONFIG, DEVICE, IOU_THRESHOLD, MIN_INPUT_RESOLUTION, MODEL_PATH, USE_HALF, logger
from .qnn_ep_registry import ensure_qnn_registered
from .tracking import greedy_nms_indices

_ORT_PATCH_ACTIVE = False
_ACTIVE_BACKEND = "cpu"
_EP_NAME = "QNNExecutionProvider"

# Lean ORT pool (set by load_yolo_model when backend=cpu_lean).
_LEAN_POOL: Optional["_LeanPool"] = None


def active_backend() -> str:
    return _ACTIVE_BACKEND


def _register_qnn_plugin() -> bool:
    return ensure_qnn_registered()


def _qnn_backend_lib(backend: str) -> str:
    if CONFIG.qnn_backend_path:
        return CONFIG.qnn_backend_path
    try:
        import onnxruntime_qnn as qnn_ep

        if backend == "qnn_gpu" and hasattr(qnn_ep, "get_qnn_gpu_path"):
            return qnn_ep.get_qnn_gpu_path()
        if hasattr(qnn_ep, "get_qnn_htp_path"):
            return qnn_ep.get_qnn_htp_path()
    except Exception:
        pass
    return "libQnnHtp.so" if backend != "qnn_gpu" else "libQnnGpu.so"


def _qnn_ep_devices() -> list[Any]:
    import onnxruntime as ort

    if not _register_qnn_plugin():
        return []
    return [d for d in ort.get_ep_devices() if d.ep_name == _EP_NAME]


def qnn_runtime_status() -> dict[str, Any]:
    info: dict[str, Any] = {
        "requested_backend": CONFIG.yolo_backend,
        "active_backend": _ACTIVE_BACKEND,
        "qnn_backend_path": CONFIG.qnn_backend_path or None,
        "qnn_htp_arch": CONFIG.qnn_htp_arch or None,
        "qnn_soc_model": CONFIG.qnn_soc_model or None,
        "qnn_context_cache": bool(CONFIG.qnn_context_cache),
        "qnn_context_cache_dir": CONFIG.qnn_context_cache_dir,
        "qnn_execution_provider": False,
        "qnn_ep_devices": 0,
        "available_providers": [],
        "qnn_ready": False,
        "note": "Qualcomm QNN plugin — not NVIDIA CUDA",
    }
    try:
        import onnxruntime as ort

        if not (CONFIG.yolo_backend or "").strip().lower().startswith("qnn"):
            info["available_providers"] = list(ort.get_available_providers())
            return info

        _register_qnn_plugin()
        info["available_providers"] = list(ort.get_available_providers())
        devices = _qnn_ep_devices()
        info["qnn_ep_devices"] = len(devices)
        info["qnn_ready"] = len(devices) > 0
        info["qnn_execution_provider"] = info["qnn_ready"]
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


# ──────────────────────────────────────────────────────────────────────────
# Lean ONNX Runtime path
# ──────────────────────────────────────────────────────────────────────────
# Ultralytics shares one ORT session across all camera requests; under 3
# concurrent streams that one session's intra-op pool (+ torch OMP threads)
# oversubscribes the 6×A55 cores → ~270  ms/frame. A small pool of dedicated
# sessions, each pinned to a few intra-op threads (pool_size × threads ≈ cores),
# removes the contention and runs the same model ~3-4× faster, while producing
# Ultralytics-compatible result objects (verified box-for-box on real frames).

class _Cell:
    __slots__ = ("_v",)

    def __init__(self, v: Any) -> None:
        self._v = v

    def item(self) -> Any:
        return self._v

    def tolist(self) -> list:
        return list(self._v)


class _Indexable:
    """Mimics a length-1 tensor row so ``box.xyxy[0].tolist()`` etc. work."""

    __slots__ = ("_v",)

    def __init__(self, v: Any) -> None:
        self._v = v

    def __getitem__(self, idx: int) -> _Cell:
        if idx != 0:
            raise IndexError(idx)
        return _Cell(self._v)


class _Box:
    __slots__ = ("xyxy", "conf", "cls")

    def __init__(self, xyxy: list, conf: float, cls: float) -> None:
        self.xyxy = _Indexable(xyxy)
        self.conf = _Indexable(conf)
        self.cls = _Indexable(cls)


class _Result:
    """Minimal stand-in for an Ultralytics Results object (only .boxes is used)."""

    __slots__ = ("boxes",)

    def __init__(self, boxes: Optional[List[_Box]]) -> None:
        self.boxes = boxes


def _nms_indices(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Greedy NMS on xyxy boxes (P1-3).

    Delegates to tracking.greedy_nms_indices(), which operates directly on
    xyxy boxes via tracking.iou(). This used to call cv2.dnn.NMSBoxes with
    boxes_xyxy passed as-is, but that API expects [x, y, w, h] -- feeding it
    xyxy silently inflated every box (interpreting x2/y2 as absolute width/
    height instead of x1/y1-relative width/height), which made suppression
    decisions depend on a box's absolute position in the frame rather than
    its true overlap with other boxes. Reusing the tracker's own IoU
    implementation removes that redundant, format-fragile NMS entirely.
    """
    if boxes_xyxy.size == 0:
        return []
    return greedy_nms_indices(boxes_xyxy.tolist(), scores.tolist(), float(iou_thresh))


def _row_is_canvas_space(
    rx0: float,
    ry0: float,
    rx2: float,
    ry2: float,
    frame_w: int,
    frame_h: int,
    coord_space: str,
) -> bool:
    """Decide whether one end2end output row is in letterbox-canvas space (P1-3).

    ``coord_space`` is normally "auto": already-exported models carry no
    recorded coordinate-space contract today, so this falls back to a
    heuristic that flags a row as canvas-space if any coordinate overflows
    the frame on *either* axis. The original heuristic only checked vertical
    overflow (``ry2 > frame_h``), which misclassifies canvas-space rows in a
    portrait frame with horizontal letterbox padding: the model's square
    canvas pads out the shorter axis, so a portrait frame overflows
    horizontally (rx) while a landscape frame overflows vertically (ry) —
    checking only ry silently skipped the rescale for the portrait case.

    Checking both axes has no downside for already-correct rows: genuine
    original-frame boxes are clamped to [0, frame_w] x [0, frame_h] by
    construction (see the ``else`` branch in _parse_end2end and its
    downstream clamping), so they can never trigger a false positive here.

    Set END2END_COORD_SPACE to "canvas" or "frame" once a given export's
    contract is confirmed (e.g. documented at export time in
    tools/qnn/export_yolo26_qdq.py), to skip this heuristic entirely.
    """
    if coord_space == "canvas":
        return True
    if coord_space == "frame":
        return False
    return (
        rx2 > float(frame_w) + 0.5
        or rx0 > float(frame_w) + 0.5
        or ry2 > float(frame_h) + 0.5
        or ry0 > float(frame_h) + 0.5
    )


def _create_qnn_session(model_path: str, qnn_backend: str):
    """Create an ORT session on QNN HTP; try plugin EP API then legacy providers."""
    import onnxruntime as ort

    sess_options = _build_qnn_session_options(qnn_backend)
    qnn_opts: dict[str, Any] = {"backend_path": _qnn_backend_lib(qnn_backend)}
    if qnn_backend == "qnn_htp":
        qnn_opts["htp_performance_mode"] = CONFIG.qnn_htp_performance_mode
        if CONFIG.qnn_htp_arch:
            qnn_opts["htp_arch"] = CONFIG.qnn_htp_arch
            qnn_opts["device_id"] = "0"
            if CONFIG.qnn_soc_model:
                qnn_opts["soc_model"] = CONFIG.qnn_soc_model
        if CONFIG.qnn_vtcm_mb:
            qnn_opts["vtcm_mb"] = CONFIG.qnn_vtcm_mb
        if CONFIG.qnn_htp_fp16:
            qnn_opts["enable_htp_fp16_precision"] = "1"

    try:
        return ort.InferenceSession(model_path, sess_options=sess_options)
    except Exception as primary_exc:
        logger.warning("QNN add_provider_for_devices failed (%s), trying legacy providers API", primary_exc)
        legacy_opts = ort.SessionOptions()
        if CONFIG.qnn_disable_cpu_fallback:
            legacy_opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        sess = ort.InferenceSession(
            model_path,
            sess_options=legacy_opts,
            providers=[_EP_NAME, "CPUExecutionProvider"],
            provider_options=[qnn_opts, {}],
        )
        if _EP_NAME not in sess.get_providers():
            raise RuntimeError(f"QNN legacy session missing HTP provider: {sess.get_providers()}") from primary_exc
        return sess


class _LeanPool:
    """Thread-safe pool of ORT sessions for concurrent multi-camera inference."""

    def __init__(
        self,
        model_path: str,
        pool_size: int,
        intra_threads: int,
        *,
        qnn_backend: Optional[str] = None,
    ) -> None:
        import onnxruntime as ort

        self.model_path = model_path
        self.pool_size = max(1, pool_size)
        self.backend = qnn_backend or "cpu"
        self._free: "queue.Queue[Any]" = queue.Queue()
        self._lock = threading.Lock()
        self.imgsz = 0
        self.input_name = ""
        self._nhwc_input = False
        self._end2end = True
        for _ in range(self.pool_size):
            if qnn_backend:
                if not _register_qnn_plugin():
                    raise RuntimeError("onnxruntime-qnn plugin is not available")
                sess = _create_qnn_session(model_path, qnn_backend)
            else:
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = max(1, intra_threads)
                sess_options.inter_op_num_threads = 1
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess = ort.InferenceSession(
                    model_path,
                    sess_options=sess_options,
                    providers=["CPUExecutionProvider"],
                )
            if not self.input_name:
                inp = sess.get_inputs()[0]
                self.input_name = inp.name
                out = sess.get_outputs()[0]
                out_shape = [int(d) if d not in (None, "") else 0 for d in out.shape]
                self._end2end = len(out_shape) == 3 and out_shape[-1] == 6
                shape = inp.shape  # [1, 3, H, W] or [1, H, W, 3]
                try:
                    dims = [int(d) if d not in (None, "") else 0 for d in shape]
                    if len(dims) >= 4 and dims[1] == 3:
                        self.imgsz = dims[2] or dims[3] or int(CONFIG.yolo_imgsz)
                        self._nhwc_input = False
                    elif len(dims) >= 4 and dims[3] == 3:
                        self.imgsz = dims[1] or dims[2] or int(CONFIG.yolo_imgsz)
                        self._nhwc_input = True
                    else:
                        self.imgsz = int(dims[-1] or CONFIG.yolo_imgsz)
                except Exception:
                    self.imgsz = int(CONFIG.yolo_imgsz)
                if self.imgsz < MIN_INPUT_RESOLUTION:
                    raise ValueError(
                        f"ONNX model input {self.imgsz}px < minimum {MIN_INPUT_RESOLUTION}px "
                        f"({model_path}); export with tools/export_yolo26_onnx.py --imgsz 640"
                    )
                if qnn_backend:
                    providers = sess.get_providers()
                    if "QNNExecutionProvider" not in providers:
                        raise RuntimeError(f"QNN session missing HTP provider: {providers}")
            self._free.put(sess)
        logger.info(
            "Lean ORT pool ready: backend=%s model=%s size=%d threads=%d imgsz=%d end2end=%s",
            self.backend,
            model_path,
            self.pool_size,
            intra_threads,
            self.imgsz,
            self._end2end,
        )

    @contextlib.contextmanager
    def lease(self) -> Iterator[Any]:
        sess = self._free.get()
        try:
            yield sess
        finally:
            self._free.put(sess)

    def _letterbox(self, frame: np.ndarray):
        new = self.imgsz
        h, w = frame.shape[:2]
        r = min(new / h, new / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((new, new, 3), 114, dtype=np.uint8)
        top = (new - nh) // 2
        left = (new - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, r, top, left

    def predict(
        self,
        frame: np.ndarray,
        conf: float,
        classes: Optional[set] = None,
    ) -> _Result:
        canvas, r, top, left = self._letterbox(frame)
        if self._nhwc_input:
            x = np.ascontiguousarray(canvas[:, :, ::-1].astype(np.float32)[None] / 255.0)
        else:
            x = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            x = np.ascontiguousarray(x[None])
        with self.lease() as sess:
            out = sess.run(None, {self.input_name: x})[0]
        if self._end2end:
            fh, fw = frame.shape[:2]
            return self._parse_end2end(out, conf, classes, fh, fw, r, top, left)
        return self._parse_raw_head(out, conf, classes, r, top, left)

    def _parse_end2end(
        self,
        out: np.ndarray,
        conf: float,
        classes: Optional[set],
        frame_h: int,
        frame_w: int,
        r: float,
        top: int,
        left: int,
    ) -> _Result:
        rows = out[0]
        boxes: List[_Box] = []
        inv_r = 1.0 / r if r else 1.0
        for row in rows:
            score = float(row[4])
            if score < conf:
                continue
            cls = float(row[5])
            cls_id = int(round(cls))
            if classes is not None and cls_id not in classes:
                # YOLO26 end2end ONNX embeds NMS rank/slot in the last column (often ~3),
                # not the COCO class id. Person-only inference passes classes={0}.
                if not (classes == {0} or classes == {0.0}):
                    continue
                cls_id = 0
            rx0, ry0, rx2, ry2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            # QNN hybrid end2end may emit either original-frame xyxy or letterbox-canvas xyxy.
            if _row_is_canvas_space(rx0, ry0, rx2, ry2, frame_w, frame_h, CONFIG.end2end_coord_space):
                x1 = (rx0 - left) * inv_r
                y1 = (ry0 - top) * inv_r
                x2 = (rx2 - left) * inv_r
                y2 = (ry2 - top) * inv_r
            else:
                x1, y1, x2, y2 = rx0, ry0, rx2, ry2
            x1 = max(0.0, min(x1, float(frame_w - 1)))
            y1 = max(0.0, min(y1, float(frame_h - 1)))
            x2 = max(0.0, min(x2, float(frame_w)))
            y2 = max(0.0, min(y2, float(frame_h)))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(_Box([x1, y1, x2, y2], score, float(cls_id)))
        return _Result(boxes if boxes else None)

    def _parse_raw_head(
        self,
        out: np.ndarray,
        conf: float,
        classes: Optional[set],
        r: float,
        top: int,
        left: int,
    ) -> _Result:
        pred = np.asarray(out)
        if pred.ndim == 3:
            pred = pred[0]
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T
        if pred.shape[1] < 5:
            raise ValueError(f"Unexpected YOLO raw head shape {pred.shape}")

        boxes_xywh = pred[:, :4].astype(np.float32)
        cls_scores = pred[:, 4:].astype(np.float32)
        if cls_scores.shape[1] == 1:
            scores = cls_scores[:, 0]
            cls_ids = np.zeros(len(scores), dtype=np.int32)
        else:
            cls_ids = cls_scores.argmax(axis=1).astype(np.int32)
            scores = cls_scores[np.arange(len(cls_ids)), cls_ids]

        keep = scores >= conf
        if classes is not None:
            keep &= np.isin(cls_ids, list(classes))
        boxes_xywh = boxes_xywh[keep]
        scores = scores[keep]
        cls_ids = cls_ids[keep]
        if len(scores) == 0:
            return _Result(None)

        cx, cy, w, h = boxes_xywh.T
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        picked = _nms_indices(boxes_xyxy, scores, IOU_THRESHOLD)
        if not picked:
            return _Result(None)

        inv_r = 1.0 / r if r else 1.0
        boxes: List[_Box] = []
        for idx in picked:
            bx1 = (float(boxes_xyxy[idx, 0]) - left) * inv_r
            by1 = (float(boxes_xyxy[idx, 1]) - top) * inv_r
            bx2 = (float(boxes_xyxy[idx, 2]) - left) * inv_r
            by2 = (float(boxes_xyxy[idx, 3]) - top) * inv_r
            boxes.append(_Box([bx1, by1, bx2, by2], float(scores[idx]), float(cls_ids[idx])))
        return _Result(boxes if boxes else None)


def _resolve_backend_name() -> str:
    return _resolve_backend_name_impl()


def _resolve_backend_name_impl() -> str:
    raw = (CONFIG.yolo_backend or "cpu").strip().lower()
    if raw == "qnn":
        return "qnn_htp"
    if raw in {"cpu", "cpu_lean", "qnn_htp", "qnn_gpu"}:
        return raw
    logger.warning("Invalid YOLO_BACKEND='%s', using cpu", raw)
    return "cpu"


def _build_qnn_session_options(backend: str):
    import onnxruntime as ort

    devices = _qnn_ep_devices()
    if not devices:
        raise RuntimeError("No QNN EP devices — install onnxruntime-qnn (aarch64)")

    backend_path = _qnn_backend_lib(backend)
    qnn_opts: dict[str, Any] = {"backend_path": backend_path}
    if backend == "qnn_htp":
        qnn_opts["htp_performance_mode"] = CONFIG.qnn_htp_performance_mode
        if CONFIG.qnn_htp_arch:
            qnn_opts["htp_arch"] = CONFIG.qnn_htp_arch
            qnn_opts["device_id"] = "0"
            if CONFIG.qnn_soc_model:
                qnn_opts["soc_model"] = CONFIG.qnn_soc_model
        if CONFIG.qnn_vtcm_mb:
            qnn_opts["vtcm_mb"] = CONFIG.qnn_vtcm_mb
        if CONFIG.qnn_htp_fp16:
            qnn_opts["enable_htp_fp16_precision"] = "1"

    logger.info(
        "QNN EP: backend=%s backend_path=%s options=%s devices=%d",
        backend,
        backend_path,
        {k: v for k, v in qnn_opts.items() if k != "backend_path"},
        len(devices),
    )
    sess_options = ort.SessionOptions()
    if CONFIG.qnn_disable_cpu_fallback:
        sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if CONFIG.qnn_context_cache:
        os.makedirs(CONFIG.qnn_context_cache_dir, exist_ok=True)
        model_stem = os.path.basename(MODEL_PATH).rsplit(".", 1)[0]
        context_path = os.path.join(CONFIG.qnn_context_cache_dir, f"{model_stem}_{backend}_ctx.onnx")
        sess_options.add_session_config_entry("ep.context_enable", "1")
        sess_options.add_session_config_entry("ep.context_file_path", context_path)
        logger.info("QNN context cache enabled: %s", context_path)
    sess_options.add_provider_for_devices(devices, qnn_opts)
    return sess_options


@contextlib.contextmanager
def _patched_ort_inference_session(backend: str) -> Iterator[None]:
    """Patch InferenceSession so Ultralytics ONNX uses QNN plugin EP."""
    global _ORT_PATCH_ACTIVE
    import onnxruntime as ort

    qnn_sess_options = _build_qnn_session_options(backend)
    original_init = ort.InferenceSession.__init__

    def _init_with_qnn(
        self,
        path_or_bytes,
        sess_options=None,
        providers=None,
        provider_options=None,
        **kwargs,
    ):
        try:
            original_init(self, path_or_bytes, sess_options=qnn_sess_options, **kwargs)
        except Exception as exc:
            logger.warning("QNN InferenceSession failed (%s), using CPU", exc)
            if sess_options is None:
                sess_options = ort.SessionOptions()
            original_init(
                self,
                path_or_bytes,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
                **kwargs,
            )

    ort.InferenceSession.__init__ = _init_with_qnn  # type: ignore[method-assign]
    _ORT_PATCH_ACTIVE = True
    try:
        yield
    finally:
        ort.InferenceSession.__init__ = original_init  # type: ignore[method-assign]
        _ORT_PATCH_ACTIVE = False


def _session_uses_qnn(model: Any) -> bool:
    try:
        predictor = getattr(model, "predictor", None) or model
        session = getattr(predictor, "session", None)
        if session is not None:
            providers = session.get_providers()
            return _EP_NAME in providers and providers[0] == _EP_NAME
    except Exception:
        pass
    return False


def _load_ultralytics(model_path: str, *, use_qnn: bool, qnn_backend: str) -> Any:
    from ultralytics import YOLO

    original_torch_load = torch.load

    def patched_torch_load(f, *args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_torch_load(f, *args, **kwargs)

    torch.load = patched_torch_load
    try:
        if use_qnn:
            with _patched_ort_inference_session(qnn_backend):
                model = YOLO(model_path)
        else:
            model = YOLO(model_path)
        try:
            model.fuse()
        except Exception:
            pass
        if not use_qnn:
            try:
                model.to(DEVICE)
            except Exception as exc:
                logger.warning("Failed to move YOLO model to %s: %s", DEVICE, exc)
        return model
    finally:
        torch.load = original_torch_load


class _LeanModel:
    """Sentinel returned for the cpu_lean backend (inference goes through the pool)."""

    __slots__ = ()


def load_yolo_model(model_path: Optional[str] = None) -> Any:
    global _ACTIVE_BACKEND, _LEAN_POOL

    path = model_path or MODEL_PATH
    suffix = path.rsplit(".", 1)[-1].lower() if "." in path else "pt"
    backend = _resolve_backend_name_impl()

    if backend in {"cpu_lean", "qnn_htp"} and suffix != "onnx":
        logger.warning("YOLO_BACKEND=%s needs .onnx; using cpu for %s", backend, path)
        backend = "cpu"

    if backend in {"cpu_lean", "qnn_htp"}:
        try:
            pool_size = max(1, int(os.getenv("YOLO_LEAN_POOL_SIZE", "3")))
            threads = max(1, int(os.getenv("YOLO_LEAN_THREADS", "2")))
            qnn_backend = backend if backend == "qnn_htp" else None
            _LEAN_POOL = _LeanPool(
                path,
                pool_size=pool_size,
                intra_threads=threads,
                qnn_backend=qnn_backend,
            )
            _ACTIVE_BACKEND = backend
            logger.info("YOLO loaded with lean ORT pool backend (%s)", backend)
            return _LeanModel()
        except Exception as exc:
            logger.error(
                "Lean ORT pool init failed for %s (%s)",
                backend,
                exc,
            )
            if backend == "qnn_htp":
                logger.warning("QNN HTP unavailable — falling back to cpu_lean")
                backend = "cpu_lean"
                try:
                    _LEAN_POOL = _LeanPool(
                        path,
                        pool_size=pool_size,
                        intra_threads=threads,
                        qnn_backend=None,
                    )
                    _ACTIVE_BACKEND = "cpu_lean"
                    logger.info("YOLO loaded with lean ORT pool backend (cpu_lean fallback)")
                    return _LeanModel()
                except Exception as cpu_exc:
                    logger.error("cpu_lean fallback also failed (%s)", cpu_exc)
            backend = "cpu"

    if backend.startswith("qnn") and suffix != "onnx":
        logger.warning("YOLO_BACKEND=%s needs .onnx; using cpu for %s", backend, path)
        backend = "cpu"

    logger.info("Loading YOLO from %s (backend=%s)", path, backend)

    if backend.startswith("qnn") and _qnn_ep_devices():
        try:
            model = _load_ultralytics(path, use_qnn=True, qnn_backend=backend)
            if _session_uses_qnn(model):
                _ACTIVE_BACKEND = backend
                logger.info("YOLO loaded with QNN (%s)", backend)
            else:
                _ACTIVE_BACKEND = "cpu"
                logger.warning(
                    "QNN session fell back to CPU (oe-linux QAIRT libs required for HTP/GPU). "
                    "Using fast ONNX Runtime CPU path."
                )
            return model
        except Exception as exc:
            logger.error("QNN YOLO load failed (%s), falling back to CPU", exc)

    model = _load_ultralytics(path, use_qnn=False, qnn_backend="cpu")
    _ACTIVE_BACKEND = "cpu"
    logger.info("YOLO loaded with CPU backend (onnxruntime CPU)")
    return model


def predict_yolo(model: Any, frame: Any, **kwargs: Any) -> Any:
    if _ACTIVE_BACKEND in {"cpu_lean", "qnn_htp"} and _LEAN_POOL is not None:
        conf = float(kwargs.get("conf", CONFIG.confidence_threshold))
        cls_arg = kwargs.get("classes")
        cls_set = set(int(c) for c in cls_arg) if cls_arg else None
        return _LEAN_POOL.predict(frame, conf, cls_set)
    predict_kwargs = dict(kwargs)
    if _ACTIVE_BACKEND == "cpu":
        predict_kwargs.setdefault("device", DEVICE)
        predict_kwargs.setdefault("half", USE_HALF)
    else:
        predict_kwargs.pop("device", None)
        predict_kwargs.pop("half", None)
    return model.predict(frame, **predict_kwargs)[0]
