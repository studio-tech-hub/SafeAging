# CPU Production Profile — AI Box QCS6490

## First boot / secrets (P0-2, P0-3)

`docker compose up` refuses to start until every required secret is set in
`.env` — there are no shipped default passwords. Generate all of them in one
command:

```bash
python tools/generate_secrets.py
```

This writes `API_KEY`, `POSTGRES_PASSWORD`, `MINIO_ROOT_PASSWORD`, and
`GRAFANA_ADMIN_PASSWORD` into `.env` (and keeps `S3_ACCESS_KEY`/`S3_SECRET_KEY`
in sync with the MinIO root credentials). It is safe to re-run — it never
overwrites a secret that already looks real; use `--force` to rotate
everything unconditionally.

Then, before starting the stack:

1. Run `docker compose config` (no `up`) and confirm it prints the resolved
   config with no `required variable ... is missing a value` errors.
2. For an AI Box / Nx plugin deployment: set the same value as `API_KEY` in
   `.env` as the **Service API Key** setting on every camera's YOLO26 plugin
   in Nx Client — otherwise the plugin gets HTTP 401 and detections stop.
3. `docker compose up -d`.

Rotating a secret on an **existing** installation: `POSTGRES_PASSWORD` /
`MINIO_ROOT_PASSWORD` only take effect on first initialization of their data
volume — see the note printed by `generate_secrets.py --force` if you need
to rotate a live box (requires recreating the volume, or updating the
password inside the running Postgres/MinIO container to match).

Pre-flight checks (no `.env` required, safe to run anytime, ready to wire
into CI — see P1-7):

```bash
python tools/check_compose_security.py    # static lint: no compose file may ship weak defaults
python tools/verify_compose_config.py     # dynamic: docker compose config fails empty, succeeds populated
```

## Secrets hygiene (P0-4)

`.env` now holds real credentials, not dev placeholders — treat it like any
other secret material:

- **File permissions.** `tools/generate_secrets.py` chmods `.env` to `600`
  (owner read/write only) every time it writes, on any POSIX host (AI Box,
  Linux dev, WSL). This is a no-op on native Windows — chmod bits don't map
  onto NTFS ACLs there, so rely on normal filesystem/user-account isolation,
  or do secret generation from WSL, for anything production-facing. If you
  ever hand-edit `.env` instead of using the generator, re-run
  `chmod 600 .env` yourself afterward.
- **Never commit or attach `.env`.** It's gitignored; when filing a support
  ticket or attaching logs, double-check `.env` itself is never included, and
  strip any pasted `docker compose config` output (which prints *resolved*
  values, including secrets) before sharing it externally.
- **`*_FILE` alternative (Docker secrets convention).** The analytics service
  additionally accepts `API_KEY_FILE`, `DATABASE_URL_FILE`, `S3_ACCESS_KEY_FILE`,
  `S3_SECRET_KEY_FILE`, and `SMTP_PASSWORD_FILE` — each points at a file whose
  contents are used as the secret, taking precedence over the inline env var
  of the same name if both are set. This is purely additive: nothing changes
  for existing `.env`-only deployments. It exists for customers who want to
  mount secrets from an orchestrator (Docker/Kubernetes secrets, HashiCorp
  Vault, etc.) instead of plain environment variables. The `postgres`,
  `minio`, and `grafana` containers already support the equivalent convention
  natively via their official images — `POSTGRES_PASSWORD_FILE`,
  `MINIO_ROOT_USER_FILE`/`MINIO_ROOT_PASSWORD_FILE` (single underscore), and
  `GF_SECURITY_ADMIN_PASSWORD__FILE` (**double** underscore — Grafana's own
  convention; setting both the plain var and the `__FILE` var on Grafana is
  an error, not a fallback). None of this is wired into the shipped compose
  files by default (that would require operators to have secret files
  already staged before first boot); it's documented here for anyone
  layering their own `secrets:` overlay on top.
- **If you ever build a diagnostic/support-bundle tool** (planned as part of
  the P1 observability work), route any dumped configuration through
  `people_analytics_service.config.redact_env_dict()` (dict of env-var-style
  key/value pairs) or `redact_url_credentials()` (for a `DATABASE_URL`-shaped
  string) first — don't invent a new redaction pattern per feature. The same
  convention should be reused for future secret-bearing features (licensing
  keys, remote update tokens in P4).

## Active configuration

| Item | Value |
|------|-------|
| YOLO | `yolo26n.onnx` @ 640 |
| Backend | `cpu_lean` pool **2×2 threads** (~4 cores, headroom for Nx OS) |
| Face | `buffalo_s`, retry Unknown every **8s** on stable tracks |
| Fall | Bbox heuristic; pose **off** by default (tier-2 crop when enabled) |
| ORT | **1.24.4** pinned (do not float to 1.27+ on this board) |

## Bounding-box smoothing (P1-1)

**Visual behavior change:** rendered/reported boxes are now visibly smoother
than before this fix — this is intended, not a regression. `BBOX_SMOOTHING`
(default **0.85**) was already being computed every frame via `smooth_bbox()`
in `tracking.py`, but the output-selection step in `api.py` was discarding
that result and rendering the raw, unsmoothed YOLO measurement instead. That
was the single largest contributor to reported "jittery boxes" — the fix
(`select_output_bbox()` in `tracking.py`) simply renders the value that was
already being computed.

- **Fall detection is unaffected.** It's fed from `measurement_bbox` (the raw
  YOLO measurement) explicitly, not from the rendered/smoothed box, so
  velocity/aspect-ratio fall math still sees true, undamped motion.
- **To opt out** and go back to today's exact razor-sharp/no-lag boxes, set
  `BBOX_SMOOTHING=0` — `smooth_bbox()` is a no-op at `alpha<=0`, so the
  rendered box is bit-for-bit the raw measurement again. No other flag is
  needed; this is a bug fix, not a new feature.
- Large/fast movements still "snap" quickly rather than lagging noticeably —
  `smooth_bbox()`'s adaptive alpha already increases blending speed in
  proportion to how far the box moved between frames.

## Detection recall tuning (P1-2)

**Recommended-default change, not a forced one:** any deployment with an
explicit value already set for `CONFIDENCE_THRESHOLD`, `PERSON_MIN_HW_RATIO`,
or `ROI_Y_MIN`/`ROI_Y_MAX` in its `.env` keeps that value unchanged — these
remain plain env-tunable settings. Only the code-level *default* (used when
the var is unset) and the aspect-ratio filter's *logic* changed. Check
`Config schema version` in the startup logs (`log_config_summary()`) to see
which generation of defaults a running deployment was built against —
version `1` = pre-P1-2 defaults, `2` = P1-2.

**Why this changed — recall vs. precision tradeoff:** the old defaults were
tuned purely to suppress false positives (furniture misclassified as people),
at a direct, uncompensated cost to the product's core value proposition
(catching people, catching falls):

- **`CONFIDENCE_THRESHOLD`: 0.75 → 0.55.** A nano model at 640px with a 0.75
  floor was silently dropping small/far people below that score. Downstream
  tracking (`TRACK_MIN_HITS`, post-NMS, dedupe) already suppresses transient
  low-confidence noise before it becomes a "stable" output, so lowering the
  floor trades a small amount of extra low-confidence noise for materially
  better recall on small/far people. Rotate this back up per-site if a
  specific camera's furniture/lighting produces persistent false positives
  that survive tracking.
- **`PERSON_MIN_HW_RATIO` (default unchanged, **0.80**) is now track-state-
  aware instead of a blanket filter.** Previously it ran on every raw YOLO box
  before any track association, rejecting anything wider than the ratio
  regardless of context — which is exactly the shape a *fallen* person
  produces, directly conflicting with fall detection. It now only blocks
  **brand-new track candidates** (protecting against furniture false-positives
  on track creation); a box that overlaps an **already-confirmed** track
  bypasses the filter (`overlaps_confirmed_track()` in `tracking.py`, using
  the same `MATCH_IOU_THRESHOLD` gating the tracker itself uses for "is this
  plausibly the same person"). A real person transitioning to a fallen/wide
  posture mid-track is therefore never dropped by this filter again.
- **`ROI_Y_MIN`/`ROI_Y_MAX` invalid-range fallback: `[0.3, 1.0]` → `[0.0,
  1.0]`.** This only applies when `ROI_Y_MIN >= ROI_Y_MAX` (an invalid
  config) — it used to silently fall back to a crop tuned for one specific
  camera framing, which can hide a standing person's head on any other
  camera. An invalid config now degrades to "no crop" (full frame), never to
  a guessed one.

Validate any tuning change against recorded footage before rolling it to a
pilot site — this is an empirical tuning change, not just a code fix: run a
before/after pass on at least one small/far-person clip and one fall-
simulation clip and compare missed-detection / false-positive counts.

## Lean/QNN NMS and letterbox coordinate fixes (P1-3)

**Scope:** only affects the `cpu_lean` and `qnn_htp`/`qnn_gpu`/`qnn` backends
(`YOLO_BACKEND` other than the default `cpu`). The default Ultralytics CPU
backend uses Ultralytics' own internal NMS and letterbox handling and was
never affected by either bug below.

- **NMS box-format bug, fixed.** `yolo_backend._nms_indices()` (used by the
  `cpu_lean` raw-head path, `_parse_raw_head`) passed `xyxy` boxes straight
  into `cv2.dnn.NMSBoxes()`, which expects `[x, y, w, h]`. That silently
  reinterpreted every box's `x2`/`y2` as an absolute width/height instead of
  `x1`/`y1`-relative, inflating boxes by an amount that grows with distance
  from the frame origin — so suppression decisions ended up depending on a
  box's *absolute position in the frame*, not just its true overlap with
  other boxes (translation-variant NMS, which is never correct). Fixed by
  replacing it with `tracking.greedy_nms_indices()`, a plain greedy NMS built
  on the tracker's own already-tested `iou()`, removing the redundant
  format-fragile OpenCV NMS call entirely. `post_nms_dedupe()` in
  `tracking.py` now shares the same implementation. See
  `TestGreedyNmsIndices.test_translation_invariant_near_and_far_from_origin`
  and `TestNmsIndices.test_translation_invariant_regression` for the exact
  before/after case (same relative box geometry, true IoU ~0.39 below a 0.45
  threshold, used to flip outcome purely based on frame position).
- **End2end letterbox coordinate-space heuristic, widened.** The QNN
  NMS-embedded (`end2end`) output path (`_parse_end2end`) has to guess
  whether the model emitted letterbox-canvas coordinates or already-rescaled
  original-frame coordinates, since already-exported models carry no
  recorded contract. The old heuristic only checked vertical overflow
  (`ry2 > frame_h`), which misses the mirror case on a **portrait** source
  frame: the model's square canvas pads out the *shorter* axis, so a portrait
  frame overflows horizontally instead of vertically, and the old check
  silently skipped the rescale for that case. Now checks both axes
  (`_row_is_canvas_space()` in `yolo_backend.py`) — has no downside for
  already-correct rows, since genuine frame-space boxes are clamped to
  `[0, frame_w] x [0, frame_h]` by construction and can never trigger a false
  positive on either axis.
- **New: `END2END_COORD_SPACE` explicit override** (`auto` default / `canvas`
  / `frame`, in `.env.example`). The overflow heuristic is fundamentally only
  reliable when the canvas coordinate range can actually exceed the frame's
  own pixel dimensions — for any camera resolution larger than the model's
  square canvas on **both** axes (e.g. a typical 1080p/720p landscape camera
  vs. a 640x640 canvas), overflow will never be observed on either axis
  regardless of the true coordinate space, and `auto` will default to
  treating output as already-frame-space. **Before relying on `qnn_htp` /
  `qnn_gpu` in production, confirm the real contract** with
  `tools/qnn/benchmark_yolo_backend.py` (now prints a `box_coord_range` line
  and a warning if every detected box stays under 640px on a much larger
  frame — a sign of un-rescaled canvas-space output) and set
  `END2END_COORD_SPACE=canvas` or `=frame` explicitly once confirmed, rather
  than depending on the heuristic long-term.

Neither change alters output on the default `cpu` backend. Re-run
`tools/qnn/validate_qdq_detect.py` and `tools/qnn/benchmark_yolo_backend.py`
on real QNN hardware before promoting any `cpu_lean`/`qnn_*` deployment —
detection counts may shift slightly as duplicate/incorrectly-suppressed boxes
from the NMS fix resolve.

## Additive binary/multipart transport for /infer (P1-4)

**Status: additive, opt-in, not yet field-validated.** Default behavior is
unchanged; do not enable in production until benchmarked on real hardware.

- The analytics service now also accepts `POST /infer/binary`
  (`multipart/form-data`: `camera_id` form field + raw JPEG bytes as the
  `image` file part), alongside the original `POST /infer`
  (JSON body with a base64-encoded frame). Both routes share the exact same
  detection/tracking pipeline (`_run_person_detection_pipeline()` in
  `api.py`) after their respective decode step, so there is no behavioral
  difference between them beyond the wire format.
- The plugin's transport is now a small Strategy interface
  (`ITransportClient` in `transport_client.h`) with two implementations:
  `JsonBase64Transport` (current default, thin wrapper around the existing,
  proven `ObjectDetector::run()`/`callPythonService()` path) and
  `MultipartBinaryTransport` (new, additive). Selected once per
  `settingsReceived()` call via the new `transport_mode` plugin setting
  (`json_base64` default, or `binary`); an unrecognized value falls back to
  `json_base64`. No plugin rebuild is required to keep using the old
  behavior — the setting simply defaults to it.
- Base64 inflates the JPEG payload by ~33% and costs encode/decode CPU on
  both sides, on every frame, on every camera — pure overhead on an
  already CPU-constrained edge box. The binary transport removes both.
- `debug_dump_output` (annotated output-frame dump) is not yet implemented
  for `transport_mode=binary`; use `json_base64` if you need that debug
  feature. `debug_dump_input` works for both (binary transport writes the
  already-JPEG-encoded bytes to disk directly, with no decode round-trip).
- **Before setting `transport_mode=binary` on any production camera**,
  side-by-side benchmark CPU/latency per frame against `json_base64` on the
  same camera feed on real AI Box hardware, and confirm detections are
  identical between the two transports for the same input frames (see
  `TestInferBinary.test_infer_binary_first_frame_matches_json_transport` in
  `python/tests/integration/test_api.py` for the equivalent Python-side
  parity check). Only flip the default in a later release once a full
  field-validation cycle has passed.

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

## Network hardening

On the AI Box overlay, analytics binds to `127.0.0.1:18000` only. Nx mediaserver/plugin
can call the service locally, but LAN clients cannot call `/health`, `/status`, `/metrics`,
or `/infer` directly.

## Ops readiness check

Run this before/after deploy and after camera/plugin changes:

```bash
python3 tools/ops_aibox_check.py --api-key "$API_KEY"
```

From a dev machine:

```bash
python tools/ops_aibox_check.py --host 192.168.1.210 --user root --api-key "$API_KEY"
```

It checks localhost binding, API key enforcement, active camera traffic, per-camera latency,
Docker service state, disk headroom, MinIO storage placement, and alert email configuration.

## Do NOT on CPU

- `yolo26s` @ 640 (tested ~1400ms/cam)
- Full-frame pose every N frames
- `YOLO_LEAN_POOL_SIZE=3` with 3+ cameras unless load verified < 80%
