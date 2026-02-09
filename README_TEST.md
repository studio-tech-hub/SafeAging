# YOLOv8 FLOW2 Nx Analytics Plugin - Build & Test

## 1) Prerequisites
- Linux host with Nx Meta / Nx Witness Media Server installed.
- Nx Metadata SDK extracted locally.
- OpenCV development packages installed (`core`, `imgproc`, `imgcodecs`).
- Python AI Service running with endpoint `POST /infer`.

## 2) Build Plugin (Linux)
```bash
cd /path/to/SafeAging
mkdir -p build_flow2
cd build_flow2

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DNX_METADATA_SDK_DIR=/opt/nx/metadata_sdk

cmake --build . -j"$(nproc)"
```

Expected output plugin library:
- `build_flow2/libyolov8_flow2_plugin.so`

## 3) Install Plugin into Nx Media Server
Adjust paths for your environment:

```bash
sudo mkdir -p /opt/networkoptix-metavms/mediaserver/bin/plugins/yolov8_flow2_plugin
sudo cp build_flow2/libyolov8_flow2_plugin.so \
  /opt/networkoptix-metavms/mediaserver/bin/plugins/yolov8_flow2_plugin/
```

Optional (copy manifest reference):
```bash
sudo cp src/manifest.json \
  /opt/networkoptix-metavms/mediaserver/bin/plugins/yolov8_flow2_plugin/
```

Restart Media Server:
```bash
sudo systemctl restart networkoptix-mediaserver
```

## 4) Runtime Configuration
Set environment variables for Media Server service (example with systemd override):

```bash
sudo systemctl edit networkoptix-mediaserver
```

Add:
```ini
[Service]
Environment=NX_AI_SERVICE_URL=http://127.0.0.1:18000
Environment=NX_AI_TIMEOUT_CONNECT_MS=250
Environment=NX_AI_TIMEOUT_READ_MS=400
Environment=NX_AI_TIMEOUT_WRITE_MS=250
Environment=NX_AI_SAMPLE_FPS=5.0
Environment=NX_AI_QUEUE_SIZE=4
Environment=NX_AI_SEND_WIDTH=640
Environment=NX_AI_JPEG_QUALITY=80
Environment=NX_AI_CIRCUIT_FAILS=3
Environment=NX_AI_CIRCUIT_OPEN_MS=3000
Environment=NX_AI_FALL_FINISH_MS=3000
Environment=NX_AI_SYNTH_TRACK_TTL_MS=2000
Environment=NX_AI_TRACK_MAP_TTL_MS=60000
Environment=NX_AI_LOG_THROTTLE_MS=5000
```

Apply:
```bash
sudo systemctl daemon-reload
sudo systemctl restart networkoptix-mediaserver
```

## 5) Nx Validation Plan
1. Open Nx Desktop Client and connect to server.
2. Add camera (if not already added).
3. Open camera settings -> Analytics -> enable plugin `YOLOv8 FLOW2 Analytics`.
4. Verify live view:
   - Bounding boxes appear on people.
5. Verify event/rule:
   - Create rule on event type `Fall detected`.
   - Simulate fall scenario so AI service returns `fall_detected=true`.
   - Confirm action triggers (popup/email/webhook/etc.).
6. Verify timeline/archive:
   - Confirm `Fall detected` event appears on timeline.
   - Replay archive at event timestamp and verify overlay + event transition.

## 6) AI Service Local Test
Health check:
```bash
curl -s http://127.0.0.1:18000/health
```

Infer test with base64 image:
```bash
IMG_B64=$(base64 -w0 /path/to/test.jpg)
curl -s -X POST http://127.0.0.1:18000/infer \
  -H "Content-Type: application/json" \
  -d "{\"camera_id\":\"test_cam\",\"image\":\"${IMG_B64}\"}" | jq .
```

Expected response: JSON array of detections with fields like:
- `cls`, `score`, `x`, `y`, `w`, `h`, `track_id`, `fall_detected`

## 7) Troubleshooting

### Nx server logs
```bash
journalctl -u networkoptix-mediaserver -f
```

### Plugin not loaded
- Check plugin `.so` path and permissions.
- Ensure plugin filename is copied under Media Server `plugins/` directory.
- Restart Media Server after deploy.

### No detections in Nx
- Validate AI service URL and port from environment.
- Validate `/infer` works via curl.
- Reduce `NX_AI_SAMPLE_FPS` to lower load (e.g. `2.0`).
- Increase `NX_AI_TIMEOUT_READ_MS` slightly if inference is slow.

### Too many dropped frames
- Increase `NX_AI_QUEUE_SIZE` (e.g. from `4` to `8`).
- Lower `NX_AI_SAMPLE_FPS`.
- Reduce `NX_AI_SEND_WIDTH` (e.g. `640` -> `480`).

### JSON parsing errors
- Compare AI response shape against expected array of detection objects.
- Ensure `Content-Type: application/json` and valid UTF-8 JSON.
- Use `jq .` to validate response:
```bash
curl -s -X POST http://127.0.0.1:18000/infer ... | jq .
```

## Design Notes
- Frame callback path (`pushUncompressedVideoFrame`) does **not** do network IO.
- Callback only samples, converts frame, and enqueues bounded frame jobs.
- Worker thread per `DeviceAgent` performs `/infer`, metadata creation, and event transitions.
- Queue is bounded with **drop-oldest/keep-newest** policy to prevent memory growth.
- AI failures use short timeouts + circuit breaker + log throttling to keep plugin responsive.
- Metadata packets use original Nx `frame.timestampUs`, ensuring archive/timeline alignment.

