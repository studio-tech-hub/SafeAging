#!/usr/bin/env python3
"""Patch face/QNN service code into the running container and probe dual QNN sessions."""
import sys

from _ssh_common import REPO_ROOT, connect, get_credentials, run, upload_files

FILES = [
    "face_engine.py", "face_providers.py", "face_identity.py",
    "face_worker.py", "api.py", "config.py", "yolo_backend.py",
    "qnn_ep_registry.py",
]

REL_PATHS = [f"python/people_analytics_service/{f}" for f in FILES]

CMD = r"""
cd /root/SafeAging
for f in face_engine.py face_providers.py face_identity.py face_worker.py api.py config.py yolo_backend.py qnn_ep_registry.py; do
  docker cp python/people_analytics_service/$f safeaging-analytics:/app/python/people_analytics_service/$f
done
docker restart safeaging-analytics
sleep 40
echo '=== env ==='
docker exec safeaging-analytics printenv MODEL_PATH YOLO_BACKEND FACE_BACKEND ENABLE_FACE_ASYNC
echo '=== logs ==='
docker logs safeaging-analytics 2>&1 | grep -E 'Face backend|Face worker|insightface|Lean ORT pool|end2end|QNN plugin' | tail -16
echo '=== dual QNN probe ==='
docker exec safeaging-analytics python3 - <<'PY'
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import onnxruntime as ort
import onnxruntime_qnn as qnn
from people_analytics_service.qnn_ep_registry import ensure_qnn_registered
from people_analytics_service.face_providers import build_face_providers

ensure_qnn_registered()
print("providers", ort.get_available_providers())
model = os.environ.get("MODEL_PATH", "/app/models/yolo26n_backbone_qdq.onnx")
htp = qnn.get_qnn_htp_path()
gpu = qnn.get_qnn_gpu_path()
so = ort.SessionOptions()
so.intra_op_num_threads = 1
htp_sess = ort.InferenceSession(
    model, sess_options=so,
    providers=["QNNExecutionProvider", "CPUExecutionProvider"],
    provider_options=[{"backend_path": htp, "htp_performance_mode": "burst", "htp_arch": "68"}, {}],
)
print("htp", htp_sess.get_providers())
fp, fpo, fl = build_face_providers()
print("face", fl, fp)
from insightface.app import FaceAnalysis
kw = {"name": "buffalo_s", "providers": fp}
if fpo:
    kw["provider_options"] = fpo
app = FaceAnalysis(**kw)
app.prepare(ctx_id=-1, det_size=(640, 640))
print("face_ok", fl)
PY
"""


def main() -> int:
    creds = get_credentials()
    c = connect(creds, timeout=20)
    upload_files(c, REL_PATHS, creds=creds, repo_root=REPO_ROOT)
    code, _, _ = run(c, CMD, timeout=180)
    c.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
