# QNN / NPU Roadmap — QCS6490

## Goal

Offload `yolo26n` @ 640 INT8/QDQ to Hexagon HTP via ONNX Runtime QNN EP (QAIRT oe-linux libs).

## Why not CPU yolo26s

Benchmark on box: `yolo26s` ~3× slower than `yolo26n` on 6×A55. Commercial path is **NPU for detect**, CPU for face/tracking.

## Steps

1. **QAIRT oe-linux** on box (`/opt/qairt`) — `tools/qnn/install_qairt_aibox.sh`
2. **Wheels pinned**: `onnxruntime==1.24.4` + `onnxruntime-qnn==2.2.0` (in `docker/wheels/`)
3. **Export INT8 ONNX** with QDQ from `yolo26n.pt` @ 640 (calibration set from box cameras)
4. **Compose overlay**: `docker-compose.aibox-qnn.yml` — `YOLO_BACKEND=qnn_htp`
5. **Context binary cache** — QNN EP session options for faster restart (HTP compiled graph)
6. **A/B**: same plugin @ 2fps, compare `avg_infer_ms` CPU lean vs QNN HTP

## Reference

- Qualcomm AI Hub: filter QCS6490, real-time 5–60 pred/s models as baseline
- Previous blocker: `QNN_DEVICE_ERROR_INVALID_CONFIG` — retry after QAIRT oe-linux + ADSP mount

## Deploy command (when ready)

```bash
docker compose -f docker-compose.yml \
  -f docker-compose.aibox-hostdb.yml \
  -f docker-compose.aibox-qnn.yml up -d --build analytics
```

Keep `docker-compose.aibox-hostdb.yml` CPU profile as rollback.
