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

from .config import CONFIG, DEVICE, MIN_INPUT_RESOLUTION, MODEL_PATH, USE_HALF, logger

_ORT_PATCH_ACTIVE = False
_ACTIVE_BACKEND = "cpu"
_QNN_REGISTERED = False
_EP_NAME = "QNNExecutionProvider"

# Lean ORT pool (set by load_yolo_model when backend=cpu_lean).
_LEAN_POOL: Optional["_LeanPool"] = None


def active_backend() -> str:
    return _ACTIVE_BACKEND


def _register_qnn_plugin() -> bool:
    global _QNN_REGISTERED
    if _QNN_REGISTERED:
        return True
    try:
        import onnxruntime as ort
        import onnxruntime_qnn as qnn_ep

        ort.register_execution_provider_library(_EP_NAME, qnn_ep.get_library_path())
        # Skel libs bundled in the wheel — do not point ADSP at host SNPE paths.
        os.environ.pop("ADSP_LIBRARY_PATH", None)
        qnn_dir = os.path.dirname(qnn_ep.__file__)
        os.environ["LD_LIBRARY_PATH"] = (
            qnn_dir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        )
        _QNN_REGISTERED = True
        return True
    except ImportError:
        return False
    except Exception as exc:
        logger.warning("QNN plugin registration failed: %s", exc)
        return False


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


class _LeanPool:
    """Thread-safe pool of ORT sessions for concurrent multi-camera inference."""

    def __init__(self, model_path: str, pool_size: int, intra_threads: int) -> None:
        import onnxruntime as ort

        self.model_path = model_path
        self.pool_size = max(1, pool_size)
        self._free: "queue.Queue[Any]" = queue.Queue()
        self._lock = threading.Lock()
        self.imgsz = 0
        self.input_name = ""
        for _ in range(self.pool_size):
            so = ort.SessionOptions()
            so.intra_op_num_threads = max(1, intra_threads)
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(
                model_path, sess_options=so, providers=["CPUExecutionProvider"]
            )
            if not self.input_name:
                inp = sess.get_inputs()[0]
                self.input_name = inp.name
                shape = inp.shape  # [1, 3, H, W]
                try:
                    self.imgsz = int(shape[-1])
                except Exception:
                    self.imgsz = int(CONFIG.yolo_imgsz)
                if self.imgsz < MIN_INPUT_RESOLUTION:
                    raise ValueError(
                        f"ONNX model input {self.imgsz}px < minimum {MIN_INPUT_RESOLUTION}px "
                        f"({model_path}); export with tools/export_yolo26_onnx.py --imgsz 640"
                    )
            self._free.put(sess)
        logger.info(
            "Lean ORT pool ready: model=%s size=%d threads=%d imgsz=%d",
            model_path,
            self.pool_size,
            intra_threads,
            self.imgsz,
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
        x = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        x = np.ascontiguousarray(x[None])
        with self.lease() as sess:
            out = sess.run(None, {self.input_name: x})[0]
        # End-to-end NMS export: (1, N, 6) = [x1, y1, x2, y2, score, cls] in input space.
        rows = out[0]
        boxes: List[_Box] = []
        inv_r = 1.0 / r if r else 1.0
        for row in rows:
            score = float(row[4])
            if score < conf:
                continue
            cls = float(row[5])
            if classes is not None and int(cls) not in classes:
                continue
            x1 = (float(row[0]) - left) * inv_r
            y1 = (float(row[1]) - top) * inv_r
            x2 = (float(row[2]) - left) * inv_r
            y2 = (float(row[3]) - top) * inv_r
            boxes.append(_Box([x1, y1, x2, y2], score, cls))
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

    if backend in {"cpu_lean"} and suffix != "onnx":
        logger.warning("YOLO_BACKEND=cpu_lean needs .onnx; using cpu for %s", path)
        backend = "cpu"

    if backend == "cpu_lean":
        try:
            pool_size = max(1, int(os.getenv("YOLO_LEAN_POOL_SIZE", "3")))
            threads = max(1, int(os.getenv("YOLO_LEAN_THREADS", "2")))
            _LEAN_POOL = _LeanPool(path, pool_size=pool_size, intra_threads=threads)
            _ACTIVE_BACKEND = "cpu_lean"
            logger.info("YOLO loaded with lean ORT pool backend")
            return _LeanModel()
        except Exception as exc:
            logger.error("Lean ORT pool init failed (%s), falling back to ultralytics CPU", exc)
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
    if _ACTIVE_BACKEND == "cpu_lean" and _LEAN_POOL is not None:
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
