"""Build ORT provider list for insightface (CPU or QNN GPU on QCS6490)."""

from __future__ import annotations

from typing import Any


def build_face_providers() -> tuple[list[str], list[dict[str, Any]] | None, str]:
    """Return (providers, provider_options, backend_label)."""
    from .config import FACE_BACKEND
    from .qnn_ep_registry import ensure_qnn_registered

    backend = (FACE_BACKEND or "cpu").strip().lower()
    if backend != "qnn_gpu":
        return ["CPUExecutionProvider"], None, "cpu"

    try:
        import onnxruntime as ort
        import onnxruntime_qnn as qnn_ep

        if not ensure_qnn_registered():
            from .config import logger

            logger.warning("[face] QNN GPU setup failed (plugin unavailable); using CPU")
            return ["CPUExecutionProvider"], None, "cpu"

        gpu_path = qnn_ep.get_qnn_gpu_path()
        qnn_opts: dict[str, Any] = {"backend_path": gpu_path}
        return (
            ["QNNExecutionProvider", "CPUExecutionProvider"],
            [qnn_opts, {}],
            "qnn_gpu",
        )
    except Exception as exc:
        from .config import logger

        logger.warning("[face] QNN GPU setup failed (%s); using CPU", exc)
        return ["CPUExecutionProvider"], None, "cpu"
