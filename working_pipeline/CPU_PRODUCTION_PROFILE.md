# CPU Production Profile — AI Box QCS6490

## Active configuration

| Item | Value |
|------|-------|
| YOLO | `yolo26n.onnx` @ 640 |
| Backend | `cpu_lean` pool **2×2 threads** (~4 cores, headroom for Nx OS) |
| Face | `buffalo_s`, retry Unknown every **8s** on stable tracks |
| Fall | Bbox heuristic; pose **off** by default (tier-2 crop when enabled) |
| ORT | **1.24.4** pinned (do not float to 1.27+ on this board) |

## Per-camera ROI (admin API)

Set via `PUT /admin/camera-configs/{camera_id}`:

```json
{
  "extra": {
    "roi": {
      "type": "rect",
      "rect": { "x_min": 0.05, "y_min": 0.25, "x_max": 0.95, "y_max": 1.0 }
    }
  }
}
```

Polygon:

```json
{
  "extra": {
    "roi": {
      "type": "polygon",
      "polygon_json": "[[0.1,0.3],[0.9,0.3],[0.9,1.0],[0.1,1.0]]"
    }
  }
}
```

## Plugin settings (per camera)

| Setting | Value |
|---------|-------|
| Target Enqueue FPS | **2** |
| Detection period | **2** |
| Frame queue max | **1** |
| service_api_key | same as `API_KEY` in compose |

## Health / degraded

`/health` returns `degraded` when:

- `pipeline_latency_high` — avg ≥ 650ms or p95 ≥ 800ms
- `host_cpu_overloaded` — load1/cores ≥ 85%

Nx plugin polls `/health` and emits diagnostic events.

## Do NOT on CPU

- `yolo26s` @ 640 (tested ~1400ms/cam)
- Full-frame pose every N frames
- `YOLO_LEAN_POOL_SIZE=3` with 3+ cameras unless load verified < 80%
