#!/usr/bin/env bash
# Full QNN bring-up on AI Box: export backbone → ORT-QNN smoke → deploy overlay.
# Run ON the box:  cd /root/SafeAging && bash tools/qnn/deploy_qnn_htp.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

echo "=== [1/4] export yolo26n_backbone.onnx (nms=False) ==="
docker cp tools/export_yolo26_onnx.py safeaging-analytics:/tmp/export_yolo26_onnx.py
docker exec safeaging-analytics python3 /tmp/export_yolo26_onnx.py \
  --model /app/models/yolo26n.pt \
  --out /app/models/yolo26n_backbone.onnx \
  --imgsz 640 --no-nms
docker cp safeaging-analytics:/app/models/yolo26n_backbone.onnx models/yolo26n_backbone.onnx
ls -la models/yolo26n_backbone.onnx

echo "=== [2/4] ORT-QNN smoke (--entrypoint bash, no full service) ==="
docker stop safeaging-analytics || true
sleep 2
cat > /tmp/qnn_backbone_smoke.py <<'PY'
import time, numpy as np, onnxruntime as ort, onnxruntime_qnn as qnn
ort.register_execution_provider_library("QNNExecutionProvider", qnn.get_library_path())
model="/app/models/yolo26n_backbone.onnx"
opts={"backend_path":qnn.get_qnn_htp_path(),"soc_model":"35","htp_arch":"68","device_id":"0","htp_performance_mode":"burst"}
devices=[d for d in ort.get_ep_devices() if d.ep_name=="QNNExecutionProvider"]
so=ort.SessionOptions(); so.add_provider_for_devices(devices, opts)
t0=time.perf_counter()
sess=ort.InferenceSession(model, sess_options=so)
print("providers", sess.get_providers(), "init_s", round(time.perf_counter()-t0,1))
inp=sess.get_inputs()[0]; out=sess.get_outputs()[0]
print("in", inp.shape, "out", out.shape)
x=np.random.rand(1,640,640,3).astype("float32") if int(inp.shape[-1] or 0)==3 else np.random.rand(1,3,640,640).astype("float32")
t0=time.perf_counter()
for _ in range(10): sess.run(None, {inp.name: x})
print("infer_ms", round((time.perf_counter()-t0)/10*1000,1))
PY

docker run --rm --entrypoint bash \
  --device /dev/adsprpc-smd --device /dev/adsprpc-smd-secure --device /dev/ion --device /dev/kgsl-3d0 \
  -v "$PWD/models:/app/models:ro" -v /root/wheels:/wheels:ro \
  -v /dsp:/dsp:ro -v /usr/lib/rfsa/adsp:/usr/lib/rfsa/adsp:ro \
  -v /usr/lib/libcdsprpc.so:/qnn-host-libs/libcdsprpc.so:ro \
  -v /opt/qti/usr/lib/liblog.so.0.0.0.liblog:/qnn-host-libs/liblog.so.0:ro \
  -v /usr/lib/libion.so.0.0.0:/qnn-host-libs/libion.so.0:ro \
  -v /tmp/qnn_backbone_smoke.py:/tmp/qnn_backbone_smoke.py:ro \
  -e PRODUCT_SOC=6490 -e DSP_ARCH=68 -e LD_LIBRARY_PATH=/qnn-host-libs:/usr/lib \
  safeaging-analytics:latest -lc '
pip install -q --no-deps --force-reinstall /wheels/onnxruntime-1.24.4*.whl /wheels/onnxruntime_qnn-2.2.0*.whl
python3 /tmp/qnn_backbone_smoke.py
' | tee /tmp/qnn_backbone_smoke.log

grep -q infer_ms /tmp/qnn_backbone_smoke.log
grep -q QNNExecutionProvider /tmp/qnn_backbone_smoke.log

echo "=== [3/4] optional native QAIRT smoke ==="
if [[ -x tools/qnn/native_htp_smoke.sh ]]; then
  bash tools/qnn/native_htp_smoke.sh "$PWD/models/yolo26n_backbone.onnx" || echo "WARN: native smoke failed (ORT path may still work)"
fi

echo "=== [4/4] deploy QNN overlay ==="
docker compose -f docker-compose.yml -f docker-compose.aibox-hostdb.yml -f docker-compose.aibox-qnn.yml build analytics
docker compose -f docker-compose.yml -f docker-compose.aibox-hostdb.yml -f docker-compose.aibox-qnn.yml up -d --force-recreate analytics

for i in $(seq 1 40); do
  st=$(docker inspect -f '{{.State.Health.Status}}' safeaging-analytics 2>/dev/null || echo missing)
  line=$(docker logs safeaging-analytics 2>&1 | grep 'Lean ORT pool ready' | tail -1)
  echo "t=$((i*5))s health=$st $line"
  echo "$line" | grep -q 'backend=qnn_htp' && [[ "$st" == "healthy" ]] && break
  sleep 5
done

curl -sf http://127.0.0.1:18000/health | python3 -m json.tool | head -30
echo "Done. Check logs for: Lean ORT pool ready: backend=qnn_htp end2end=False"
