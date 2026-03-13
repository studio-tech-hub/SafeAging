# UUID Track Cache Fix Guide

## 1. Problem Statement

In the previous implementation, `uuidFromTrackId(int trackId)` used a single global map:

- Key: `track_id`
- Value: generated UUID
- Lifetime: unbounded

This caused two production issues:

1. **Unbounded memory growth**
   - Every new `track_id` was kept forever.
   - Long-running systems accumulated stale entries with no cleanup.

2. **Cross-camera ID collision risk**
   - Same `track_id` value from different cameras mapped to the same UUID.
   - This could merge entities from different camera streams incorrectly.

---

## 2. Business Logic Impact

### Why this matters

- `track_id` from AI inference is only unique **within a camera/session scope**, not globally.
- UUID is consumed by NX metadata pipeline as object identity.
- Wrong identity mapping impacts:
  - person counting integrity
  - event correlation accuracy
  - stateful event lifecycle (`started`/`finished`, `fallDetected`) correctness

### Business Risk if not fixed

- False correlation across cameras (Camera A person appears as Camera B person).
- Event timelines become unreliable for incident review.
- Memory pressure increases over runtime, causing potential instability.

---

## 3. Requirements (BR)

### Functional BR

- BR-F1: Mapping key must include `camera_id` + `track_id`.
- BR-F2: Existing mapping should be reused while track is still active/recent.
- BR-F3: Stale mappings must expire by TTL.
- BR-F4: Cache size must be bounded by `max_entries` with eviction.
- BR-F5: Invalid track IDs (`<= 0`) must not pollute cache.

### Non-functional BR

- BR-N1: Thread-safe access.
- BR-N2: Ongoing operation without memory growing unbounded.
- BR-N3: Backward-compatible behavior for detector output schema.

### Done Criteria

- RAM no longer grows unbounded because stale entries are cleaned.
- Same `track_id` in different cameras no longer shares UUID.

---

## 4. Code Changes Applied

### Main file changed

- `src/sample_company/vms_server_plugins/opencv_object_detection/object_detector.cpp`

### API-level change inside file

- Old: `uuidFromTrackId(int trackId)`
- New: `uuidFromTrackId(const std::string& cameraId, int trackId)`

### Internal cache redesign

- Added cache model with lifecycle metadata:
  - `TrackUuidEntry { uuid, lastSeen }`
  - `TrackUuidCache { entriesByCamera, totalEntries, lastCleanup, mutex }`
- `entriesByCamera` structure:
  - `unordered_map<string cameraId, unordered_map<int trackId, TrackUuidEntry>>`

### Cleanup policy

- Periodic cleanup interval: `30s`
- TTL: `5 minutes`
- Max entries: `20000`
- Eviction strategy:
  1. Remove TTL-expired entries.
  2. If still above max, evict oldest `lastSeen` entries.

### Call site updates

- Legacy path: `uuidFromTrackId(requestCameraId, trackId)`
- FLOW2/multipart path: `uuidFromTrackId(cameraId, trackId)`

---

## 5. Updated Runtime Behavior

### Identity scope

- UUID is now scoped by `(camera_id, track_id)`.
- Example:
  - Camera A, track 7 -> UUID_X
  - Camera B, track 7 -> UUID_Y (different)

### TTL semantics

- If the same `(camera_id, track_id)` is seen before TTL expiry, UUID is reused.
- After TTL expiry, a new UUID may be generated on next sighting.

### Invalid IDs

- `track_id <= 0` returns random UUID and is not cached.
- Prevents map pollution from malformed/unknown IDs.

---

## 6. Use Cases Covered

### UC-1: Same person track continues on same camera

- Input: same `camera_id`, same `track_id`, frequent frames
- Expected: stable UUID reused

### UC-2: Same numeric track_id appears on different cameras

- Input: camera A track=10, camera B track=10
- Expected: two different UUIDs

### UC-3: Long-running system with many unique tracks

- Input: continuous new tracks over hours/days
- Expected: cache bounded by TTL and `max_entries`, no unbounded growth

### UC-4: Track ID recycled after long inactivity

- Input: `(camera_id, track_id)` not seen > TTL, appears again
- Expected: new UUID assigned (old mapping considered stale)

---

## 7. Edge Cases

- EC-1: Empty camera ID
  - Normalized to `__default_camera__` to avoid empty-key ambiguity.

- EC-2: Negative/zero track ID
  - Not cached; random UUID returned.

- EC-3: High cardinality burst
  - If cache exceeds `max_entries`, oldest entries are evicted.

- EC-4: Multi-thread access
  - Guarded by single mutex in cache object.

- EC-5: Camera churn (many temporary camera IDs)
  - Empty per-camera buckets are removed during cleanup.

---

## 8. Verification Checklist (QA)

1. **Cross-camera isolation test**
   - Send detections with `track_id=1` for `camera_A` and `camera_B`.
   - Verify resulting NX object metadata track UUID differs.

2. **Stability test same camera**
   - Repeated detections with same `(camera_id, track_id)`.
   - Verify UUID remains stable while within TTL.

3. **TTL expiry test**
   - Stop emitting a `(camera_id, track_id)` for > TTL.
   - Emit again and verify UUID changes.

4. **Memory soak test**
   - Simulate high churn of track IDs over long runtime.
   - Observe process RSS/heap trend plateaus instead of monotonic growth.

5. **Invalid input test**
   - Use `track_id=0/-1`.
   - Verify no cache growth from those IDs.

---

## 9. Tuning Knobs

Current constants in `object_detector.cpp`:

- `kUuidCacheTtl = 5 minutes`
- `kCleanupInterval = 30 seconds`
- `kUuidCacheMaxEntries = 20000`

Tuning guidance:

- Increase TTL if track continuity must survive longer temporary dropouts.
- Decrease TTL if track IDs are frequently recycled and strict freshness is needed.
- Increase max_entries for very large multi-camera deployments.

---

## 10. English Prompt for Cursor/Copilot/GitHub (Ready to Paste)

```text
You are working in a C++ NX VMS plugin codebase.

Problem:
The current UUID mapping helper uses a global unbounded cache keyed only by track_id:
- Memory grows unbounded over long runtime.
- Same track_id from different cameras can map to the same UUID.

Target file:
- src/sample_company/vms_server_plugins/opencv_object_detection/object_detector.cpp

Please implement the following:

1) Replace:
   uuidFromTrackId(int trackId)
with:
   uuidFromTrackId(const std::string& cameraId, int trackId)

2) Redesign cache to be camera-scoped and lifecycle-aware:
   - Key scope must be (camera_id, track_id)
   - Store uuid + last_seen timestamp
   - Thread-safe (mutex)

3) Add cleanup policy:
   - TTL-based expiration (remove stale entries)
   - Max-size bound with eviction of oldest entries if cache is still above limit
   - Periodic cleanup trigger (e.g. every 30 seconds) and on overflow

4) Handle invalid track IDs:
   - track_id <= 0 should not be cached
   - return a random UUID for such cases

5) Update all call sites in object_detector.cpp:
   - legacy flow should pass a camera id string (existing "nx_camera" is acceptable there)
   - multipart flow must pass real cameraId variable

6) Keep behavior backward-compatible for detection output except identity correctness:
   - same (camera_id, track_id) should keep UUID while active
   - same track_id across different cameras must produce different UUIDs

7) Add concise comments describing why this prevents memory leak and cross-camera collisions.

Acceptance criteria:
- Cache memory no longer grows unbounded in long-running high-churn scenarios.
- No UUID collision between camera A track_id=X and camera B track_id=X.
- Build remains clean (no API change outside this source file required).

Also provide a short markdown note under docs/ explaining:
- root cause
- implemented cache model
- TTL + max-size policy
- test checklist
```

---

## 11. Summary

This fix transforms UUID mapping from an unbounded global integer map into a bounded, camera-scoped, lifecycle-managed cache. It addresses both memory growth risk and cross-camera identity contamination while preserving expected tracking continuity.
