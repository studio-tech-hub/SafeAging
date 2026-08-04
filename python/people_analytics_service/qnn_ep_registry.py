"""One-time ONNX Runtime QNN execution provider registration.

YOLO (HTP) and face (GPU) both use QNNExecutionProvider with different
``backend_path`` values per session. Register the plugin once and treat an
already-registered provider as success.
"""

from __future__ import annotations

import os
from typing import Final

_EP_NAME: Final = "QNNExecutionProvider"
_REGISTERED = False


def ensure_qnn_registered() -> bool:
    """Register onnxruntime-qnn if needed. Safe to call from multiple modules."""
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        import onnxruntime as ort

        if _EP_NAME in ort.get_available_providers():
            _REGISTERED = True
            return True

        import onnxruntime_qnn as qnn_ep

        try:
            ort.register_execution_provider_library(_EP_NAME, qnn_ep.get_library_path())
        except Exception as reg_exc:
            if _EP_NAME in ort.get_available_providers():
                _REGISTERED = True
                return True
            raise reg_exc
        qnn_dir = os.path.dirname(qnn_ep.__file__)
        os.environ["LD_LIBRARY_PATH"] = (
            qnn_dir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        )
        _REGISTERED = True
        return True
    except ImportError:
        return False
    except Exception as exc:
        from .config import logger

        logger.warning("QNN plugin registration failed: %s", exc)
        return False
