# Service Transport Backlog

Date: 2026-04-26
Status: Backlog / technical debt
Scope: Python analytics service transport for `/infer`

## Current compatibility contract

The current `/infer` contract is intentionally kept unchanged for the running C++ plugin:

- Request content type: JSON
- Fields:
  - `camera_id`
  - `image` as base64

The `image` field currently supports two payload styles inside the same JSON + base64 envelope:

1. Legacy JPEG/PNG bytes encoded as base64
2. Current plugin raw BGR payload encoded as base64
   - header: `"BGR"` + width + height
   - body: raw BGR bytes

This path must remain supported until the plugin and service migrate together.

## Why this is technical debt

JSON + base64 is simple and compatible, but it adds avoidable overhead:

- Payload expansion from base64
- CPU cost for encode/decode on both plugin and service
- Extra memory copies before inference
- Higher end-to-end latency, especially at higher frame rates or larger frames

The current custom raw-BGR-inside-base64 path avoids JPEG compression artifacts, but still pays the base64 and JSON transport cost.

## Recommended migration target

Recommended primary target: `multipart/form-data`

Why multipart first:

- Keeps request metadata (`camera_id`, optional format hints) explicit
- Sends image bytes as a real binary part instead of base64 text
- Easier compatibility story than switching directly to a fully custom raw binary protocol
- Can support both encoded image bytes and raw BGR bytes during migration

Secondary option: raw binary upload (`application/octet-stream`)

This may be useful later for the lowest overhead path, but it usually needs extra metadata in headers or query params and is less self-describing than multipart.

## Proposed migration shape

### Service side

Add a new transport path without removing the current one:

Option A:
- Keep `/infer` for JSON + base64
- Add `/infer_multipart` for binary payloads

Option B:
- Keep `/infer`
- Branch by `Content-Type`:
  - `application/json` -> current compatibility path
  - `multipart/form-data` -> new binary path

Service work items:

- Parse multipart file uploads
- Accept metadata fields such as `camera_id`
- Optionally accept transport hints such as:
  - `image_format=jpeg|png|bgr_raw`
  - `width`
  - `height`
  - `pixel_format=bgr8`
- Decode directly from binary bytes without base64
- Preserve the current response contract

### Plugin side

Plugin work items:

- Stop wrapping frame payloads in JSON base64
- Send image bytes as multipart file data
- Keep `camera_id` as a normal form field
- If raw BGR is kept, send explicit metadata for width/height/pixel format
- Roll out behind a config flag first

## Backward-compatible rollout

Recommended rollout plan:

1. Land multipart support in the Python service while keeping JSON + base64 unchanged
2. Add plugin support for multipart behind a feature flag
3. Test side-by-side in staging with the same cameras
4. Measure latency, CPU, and memory improvements
5. Switch production plugin instances gradually
6. Deprecate JSON + base64 only after the old plugin path is no longer needed

## Success criteria for the future migration

- No change to inference response contract
- Lower transport overhead than JSON + base64
- Lower decode CPU and fewer memory copies
- Lower p50/p95 end-to-end infer latency
- Clean fallback path during rollout

## Code note

As of this backlog entry, the current JSON + base64 decode logic has been isolated in the service so the future transport migration can replace that layer without rewriting the rest of the inference pipeline.
