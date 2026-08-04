#!/usr/bin/env python3
"""Deploy heterogeneous pipeline (NPU backbone QDQ + GPU face async) on AI Box."""
from __future__ import annotations

import sys

from _ssh_common import REPO_ROOT, connect, get_credentials, run, upload_files

UPLOAD = [
    "python/people_analytics_service/face_engine.py",
    "python/people_analytics_service/face_providers.py",
    "python/people_analytics_service/face_identity.py",
    "python/people_analytics_service/face_worker.py",
    "python/people_analytics_service/api.py",
    "python/people_analytics_service/config.py",
    "python/people_analytics_service/yolo_backend.py",
    "python/people_analytics_service/qnn_ep_registry.py",
    "tools/export_yolo26_onnx.py",
    "tools/qnn/export_yolo26_qdq.py",
    "tools/qnn/validate_qdq_detect.py",
    "docker-compose.aibox-hetero.yml",
]

DEPLOY = r"""
set -e
cd /root/SafeAging
mkdir -p models calib_frames

echo "=== [1] export backbone ONNX (no NMS) ==="
docker cp tools/export_yolo26_onnx.py safeaging-analytics:/tmp/export_yolo26_onnx.py
docker exec safeaging-analytics sh -c '
  pip install -q "ultralytics>=8.4.0" 2>/dev/null || true
  if [ -f /app/models/yolo26n.pt ]; then PT=/app/models/yolo26n.pt
  elif [ -f models/yolo26n.pt ]; then PT=models/yolo26n.pt
  else PT=; fi
  if [ -n "$PT" ]; then
    python3 /tmp/export_yolo26_onnx.py --model "$PT" --out /app/models/yolo26n_backbone.onnx --no-nms --imgsz 640
  elif [ ! -f /app/models/yolo26n_backbone.onnx ]; then
    echo "WARN: no yolo26n.pt — skip backbone export; need manual yolo26n_backbone.onnx"
  fi
'
docker cp safeaging-analytics:/app/models/yolo26n_backbone.onnx models/yolo26n_backbone.onnx 2>/dev/null || true
test -f models/yolo26n_backbone.onnx && ls -la models/yolo26n_backbone.onnx

echo "=== [2] QDQ backbone from calib ==="
docker cp calib_frames/. safeaging-analytics:/app/calib_frames/ 2>/dev/null || true
docker cp models/yolo26n_backbone.onnx safeaging-analytics:/app/models/yolo26n_backbone.onnx 2>/dev/null || true
docker cp tools/qnn/export_yolo26_qdq.py safeaging-analytics:/tmp/export_yolo26_qdq.py
docker cp tools/qnn/validate_qdq_detect.py safeaging-analytics:/tmp/validate_qdq_detect.py
docker exec safeaging-analytics python3 /tmp/export_yolo26_qdq.py \
  --input /app/models/yolo26n_backbone.onnx \
  --calib-dir /app/calib_frames \
  --output /app/models/yolo26n_backbone_qdq.onnx \
  --imgsz 640 --max-images 96 --activation-type uint16 --weight-type uint8
docker cp safeaging-analytics:/app/models/yolo26n_backbone_qdq.onnx models/yolo26n_backbone_qdq.onnx
ls -la models/yolo26n_backbone_qdq.onnx

echo "=== [3] validate QDQ backbone (raw head) ==="
docker exec safeaging-analytics python3 /tmp/validate_qdq_detect.py \
  --float-model /app/models/yolo26n_backbone.onnx \
  --qdq-model /app/models/yolo26n_backbone_qdq.onnx \
  --calib-dir /app/calib_frames --min-max-conf 0.005

echo "=== [4] patch service code ==="
for f in face_engine.py face_providers.py face_identity.py face_worker.py api.py config.py yolo_backend.py qnn_ep_registry.py; do
  docker cp python/people_analytics_service/$f safeaging-analytics:/app/python/people_analytics_service/$f
done

echo "=== [5] deploy heterogeneous compose ==="
docker compose -f docker-compose.yml -f docker-compose.aibox-hostdb.yml \
  -f docker-compose.aibox-qnn.yml -f docker-compose.aibox-hetero.yml up -d --force-recreate analytics
sleep 40

echo "=== [5b] patch code into running container ==="
for f in face_engine.py face_providers.py face_identity.py face_worker.py api.py config.py yolo_backend.py qnn_ep_registry.py; do
  docker cp python/people_analytics_service/$f safeaging-analytics:/app/python/people_analytics_service/$f
done
docker restart safeaging-analytics
sleep 40

echo "=== [6] health ==="
curl -sf http://127.0.0.1:18000/health | python3 -c "
import json,sys
d=json.load(sys.stdin)
p=d.get('pipeline',{})
print('status',d.get('status'),'backend',p.get('yolo_backend'))
for k,v in sorted((p.get('cameras') or {}).items()):
    print(k[:24], 'infer_ms', v.get('avg_infer_ms'))
"

echo "=== [7] face backend log ==="
docker logs safeaging-analytics 2>&1 | grep -E 'Face backend|Face worker|insightface|qnn_gpu|Lean ORT pool' | tail -8

systemctl restart networkoptix-mediaserver
sleep 20
echo DONE
"""


def main() -> int:
    creds = get_credentials()
    c = connect(creds, timeout=30)
    upload_files(c, UPLOAD, creds=creds, repo_root=REPO_ROOT)
    code, _, _ = run(c, DEPLOY, timeout=1200)
    c.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
