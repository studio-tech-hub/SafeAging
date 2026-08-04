# QNN / NPU Roadmap — QCS6490

## Goal

Offload `yolo26n` @ 640 INT8/QDQ to Hexagon HTP via ONNX Runtime QNN EP (QAIRT oe-linux libs).

## Why not CPU yolo26s

Benchmark on box: `yolo26s` ~3× slower than `yolo26n` on 6×A55. Commercial path is **NPU for detect**, CPU for face/tracking.

## Steps

1. **QAIRT oe-linux** on box (`/opt/qairt`) — `tools/qnn/install_qairt_aibox.sh`
2. **Wheels pinned**: `onnxruntime==1.24.4` + `onnxruntime-qnn==2.2.0` (in `docker/wheels/`)
3. **Export INT8/QDQ ONNX** with `tools/qnn/export_yolo26_qdq.py` from `yolo26n.onnx` @ 640
   using calibration frames from the deployed cameras.
4. **Compose overlay**: `docker-compose.aibox-qnn.yml` — `YOLO_BACKEND=qnn_htp`,
   `MODEL_PATH=/app/models/yolo26n_qdq_synth.onnx`
5. **Context binary cache** — QNN EP session options for faster restart (HTP compiled graph)
6. **A/B**: same plugin @ 2fps, compare `avg_infer_ms` CPU lean vs QNN HTP

## Current status (2026-06-26)

Smoke tests on the QCS6490 box:

- QNN EP plugin registers and exposes one `QNNExecutionProvider` EP-device.
- **HTP inference works at ~27–34ms/frame** with **`yolo26n_qdq_synth.onnx`** (QDQ INT8) when using
  the plugin EP API (`SessionOptions.add_provider_for_devices`) with wheel-bundled QAIRT **2.46.0**
  libs and FastRPC deps mounted (`libcdsprpc`, `liblog`, `libion`).
- **Production deployed 2026-06-26**: 3 cameras avg **~65 ms/cam** all-in (`yolo_ms` ~50–80 ms), down from ~580 ms on CPU.
- **`yolo26n.onnx` (float end2end) fails** on HTP with NHWC layout bug (`Conv_token_1` not assigned to QNN EP) — do not use for QNN.
- **`yolo26n_qdq_synth.onnx`**: runs on HTP (~28 ms) but **outputs all-zero confidences** — do not use in production until re-exported with real camera calibration frames.
- **Only one QNN ORT session per process** — `YOLO_LEAN_POOL_SIZE=1` for `qnn_htp` (pool>1 triggers NHWC layout EP bugs).
- Model on disk must be **NCHW** `[1,3,640,640]` — an accidental NHWC export broke lean pool + warmup.
- Do **not** override `ADSP_LIBRARY_PATH` to `/opt/qairt/...` when using the
  PyPI `onnxruntime-qnn` wheel — ORT must use the skels bundled in the wheel.
- Do **not** set `QNN_DISABLE_CPU_FALLBACK=true` for YOLO end2end exports yet:
  `TopK`/post-NMS nodes still run on CPU while the backbone runs on HTP.
- Production path: `YOLO_BACKEND=qnn_htp` + lean ORT pool (`size=1`) + **`yolo26n_qdq_synth.onnx`** @ 640.

## Previous blocker (resolved)

The earlier `QNN_DEVICE_ERROR_INVALID_CONFIG` was caused by a combination of:

1. Using legacy `providers=["QNNExecutionProvider"]` instead of the plugin EP API.
2. Forcing `session.disable_cpu_ep_fallback=1` while TopK nodes require CPU.
3. Overriding `ADSP_LIBRARY_PATH` / `LD_LIBRARY_PATH` to QAIRT 2.40 libs that
   conflict with wheel QAIRT 2.46.

## Remaining work

1. **ORT-QNN layout bug** on end2end YOLO (`add_provider_for_devices` → NHWC `Conv_token_0` not assigned to QNN EP). Reboot does not fix. Legacy `providers=[QNN, CPU]` silently falls back to CPU-only.
2. Export **backbone-only** ONNX (`nms=False`) + optional QDQ; run TopK/NMS on CPU in lean pool.
3. Fallback: **native `qnn-net-run` sidecar** with QAIRT 2.40 oe-linux libs (validator already passes).
4. Re-benchmark after HTP path works; target ~100ms/cam all-in (YOLO ~30ms proven once in smoke).

## Correctness fixes (P1-3)

As of 2026-08, the deployed model is `yolo26n_backbone_qdq.onnx` running with
`end2end=True` (confirmed via the `Lean ORT pool ready` startup log line) —
i.e. despite the filename, this build actually emits NMS-embedded output and
runs through `_parse_end2end`, **not** `_parse_raw_head`/`_nms_indices`. That
means the `_parse_end2end` letterbox coordinate-space fix (widened
overflow heuristic + `END2END_COORD_SPACE` override — see
`working_pipeline/CPU_PRODUCTION_PROFILE.md` → "Lean/QNN NMS and letterbox
coordinate fixes (P1-3)") is the one that's live and matters here today, not
just for a future re-enablement. Always re-check the actual `end2end=` value
in the startup log before assuming which code path is active — it's derived
from the model's real output shape, not from the model filename.

The `_nms_indices()` xyxy/xywh format fix (also P1-3) still matters for any
box that goes through `_parse_raw_head` — i.e. the `cpu_lean` backend, or a
future backbone-only (`nms=False`) QNN export — and is applied automatically
regardless of which path is active; no config change needed.

## Deploy (on AI Box)

```bash
cd /root/SafeAging
# Sync repo from dev PC first (scp/rsync), then:
bash tools/qnn/deploy_qnn_htp.sh
```

Rollback to CPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.aibox-hostdb.yml up -d --force-recreate analytics
```

## Reference

- Qualcomm AI Hub: filter QCS6490, real-time 5–60 pred/s models as baseline
- QCS6490 uses HTP arch `68` / `hexagon-v68`.
- HTP requires QDQ/quantized ONNX; QNN GPU can run FP32/FP16 but is not working on this box yet.

## Deploy command (when ready)

```bash
docker compose -f docker-compose.yml \
  -f docker-compose.aibox-hostdb.yml \
  -f docker-compose.aibox-qnn.yml up -d --build analytics
```

Keep `docker-compose.aibox-hostdb.yml` CPU profile as rollback.
