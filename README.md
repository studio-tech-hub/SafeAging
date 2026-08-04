# SafeAging

**Human-centric AI video analytics for elder care, healthcare, and safety-critical
environments — built on YOLO26, running at the edge, integrated directly into
Nx Witness VMS.**

SafeAging turns cameras you already own into a system that detects and tracks
people, recognizes faces, detects falls, monitors restricted zones, and raises
alerts — without sending video off-site and without replacing your existing
VMS. It's designed first for nursing homes and healthcare facilities, but the
same core (detection + tracking + zones + face ID + alerting) applies to
factories, schools, hotels, and other environments that need to know *who* is
*where* from camera feeds.

> **Status:** pre-commercial, under active hardening. See
> [`working_pipeline/`](working_pipeline/) for the engineering roadmap
> currently in progress (P0 = critical blockers before commercial release,
> P1–P4 = performance/architecture/cleanup work).

---

## Table of contents

- [What it does](#what-it-does)
- [Architecture at a glance](#architecture-at-a-glance)
- [Repository layout](#repository-layout)
- [Quickstart (Docker Compose)](#quickstart-docker-compose)
- [Building the Nx Witness plugin](#building-the-nx-witness-plugin)
- [Configuration](#configuration)
- [Testing](#testing)
- [Documentation map](#documentation-map)
- [Security](#security)
- [Third-party licenses](#third-party-licenses)
- [License](#license)

## What it does

- **Person detection & tracking** — YOLO26 (ONNX Runtime), stable per-camera
  track IDs, unique-person counting, configurable regions of interest.
- **Face recognition & identity** — local (no cloud) InsightFace
  (SCRFD + ArcFace, 512-d embeddings), enroll from a photo, a video, or
  directly from a live track ("Unknown" on screen → enroll → named).
- **Fall detection** — multi-signal heuristic (fall velocity, angle change,
  aspect-ratio change) with multi-frame confirmation, plus an optional
  YOLO-Pose tier for higher accuracy.
- **Zone monitoring** — forbidden / entry / exit / ROI polygon zones per
  camera, stateful (no alert spam while someone stands in a zone).
- **Event logging & evidence** — every event persists to PostgreSQL with a
  snapshot image in MinIO/S3-compatible storage, queryable by camera, type,
  time, person, or zone.
- **Alerting** — deduplicated, rate-limited email alerts (SMTP) with retry,
  for falls and zone violations.
- **Edge resilience** — inference keeps running even if the central
  PostgreSQL is unreachable; events queue in a local SQLite outbox and sync
  once connectivity returns.
- **Ops-ready** — Prometheus metrics, JSON structured logs, a standardized
  `healthy` / `degraded` / `not_ready` health model the Nx plugin polls
  directly.

For a full, sales-oriented feature walkthrough (Vietnamese), see
[`docs/GIOI_THIEU_SAFEAGING.md`](docs/GIOI_THIEU_SAFEAGING.md).

## Architecture at a glance

```text
 IP Camera (RTSP)
        │
        ▼
 Nx Witness Server  ──loads──▶  SafeAging Plugin (C++, src/)
        │                              │  HTTP + X-API-Key
        │ video archive/playback       ▼
        │                       Analytics Service (Python/FastAPI, python/)
        │                       YOLO26 → tracking → face ID → fall → zones
        │                              │
        │                 ┌────────────┼────────────────┐
        │                 ▼            ▼                ▼
        │           PostgreSQL     MinIO / S3      SMTP (alerts)
        │           (metadata)     (snapshots)      + Prometheus/Grafana
        ▼
 Operator views live video + AI overlays in the same Nx Witness client
```

- **Nx never loses video ownership** — the plugin only reads frames and
  writes back detections/events as Nx metadata; SafeAging does not store or
  serve video.
- **The plugin never talks to the database directly** — only to the
  analytics service's HTTP API. All persistence is the service's
  responsibility.
- **The analytics service degrades, not crashes** — if PostgreSQL is down,
  inference and Nx-side detection keep working; only DB-backed features
  (event history query, cross-restart person/zone config) degrade until
  connectivity returns.

See [`working_pipeline/ARCHITECTURE_AND_ROADMAP.md`](working_pipeline/ARCHITECTURE_AND_ROADMAP.md)
for the full architecture writeup, ownership boundaries, and API contracts.

## Repository layout

| Path | What's there |
|---|---|
| `python/people_analytics_service/` | The analytics service — FastAPI app, YOLO/ONNX inference, tracking, face recognition, fall/zone logic, DB + object-storage access, admin API. |
| `python/tests/` | Unit, integration, and load tests for the analytics service. |
| `src/sample_company/vms_server_plugins/opencv_object_detection/` | The Nx Witness plugin (C++) — frame capture, transport to the analytics service, Nx metadata/event generation. (Directory name is a legacy identifier — see [P2-6 in the roadmap](working_pipeline/) for the tracked rebrand plan.) |
| `config/` | Nx plugin manifest (`manifest.json`) and CMake build config for the plugin. |
| `docker-compose.yml` + `docker-compose.aibox-*.yml` | Full-stack deployment: analytics service, PostgreSQL, MinIO, Prometheus, Grafana, plus AI Box (Qualcomm QCS6490 / QNN) overlays. |
| `tools/` | Deployment/ops scripts: secret generation, security linting, AI Box SSH tooling, QNN export/calibration utilities. |
| `working_pipeline/` | Engineering-internal docs: architecture, deployment/operations guide, and the active audit → implementation roadmap (P0–P4). |
| `docs/` | Customer/business-facing material (product overview, sales collateral). |

## Quickstart (Docker Compose)

Requirements: Docker + Docker Compose v2. (For the AI Box / ARM64 target, see
[`working_pipeline/CPU_PRODUCTION_PROFILE.md`](working_pipeline/CPU_PRODUCTION_PROFILE.md).)

```bash
# 1. Generate every required secret (API key, DB/MinIO/Grafana passwords) into .env.
#    Compose refuses to start with empty/placeholder secrets by design — see P0-2/P0-3
#    in the roadmap for why.
python tools/generate_secrets.py

# 2. (Optional but recommended) sanity-check the resolved config before starting anything.
docker compose config --quiet && echo OK

# 3. Build and start the full stack (analytics service + Postgres + MinIO + Prometheus + Grafana).
docker compose up --build -d

# 4. Confirm it's alive.
curl http://localhost:18000/health
```

A healthy response looks like `{"status": "healthy", ...}`. If you see
`"degraded"` with `"reason_codes": ["auth_disabled"]`, you skipped step 1 or
set `ALLOW_INSECURE_NO_AUTH=true` — see [Security](#security) below before
going any further.

Everyday commands (see [`Makefile`](Makefile) for the full list):

```bash
make logs            # tail the analytics service logs
make test-unit        # fast unit tests, no running service required
make test-integration  # requires: make up
docker compose down    # stop everything
docker compose down -v # stop and wipe all data volumes
```

## Building the Nx Witness plugin

The plugin (`src/sample_company/vms_server_plugins/opencv_object_detection/`)
is built against the Nx Metadata SDK and deployed as a DLL/`.so` into an Nx
Witness Server's plugin directory. Use the provided build scripts:

```powershell
# Windows
tools/build_plugin_windows.ps1 -NxMetadataSdkDir <path-to-metadata_sdk> -Package
```

```bash
# Linux (AI Box target)
tools/build_plugin_linux.sh --sdk-dir /path/to/metadata_sdk --package
```

See [`working_pipeline/CPU_PRODUCTION_PROFILE.md`](working_pipeline/CPU_PRODUCTION_PROFILE.md)
and [`working_pipeline/DEPLOYMENT_AND_OPERATIONS.md`](working_pipeline/DEPLOYMENT_AND_OPERATIONS.md)
for full build/deploy/plugin-configuration steps, including wiring the
plugin's **Service API Key** setting to match `API_KEY` in `.env`.

## Configuration

All runtime configuration is environment-variable driven — see
[`.env.example`](.env.example) for the full, documented list (detection
thresholds, tracking parameters, fall/zone tuning, database/object storage,
alerting, monitoring, and secrets). Highlights:

- Every secret (`API_KEY`, `POSTGRES_PASSWORD`, `MINIO_ROOT_PASSWORD`,
  `GRAFANA_ADMIN_PASSWORD`, ...) is **required** — there are no shipped weak
  defaults; `python tools/generate_secrets.py` generates them all in one
  step.
- Every secret also accepts a `*_FILE` variant (e.g. `API_KEY_FILE=/run/secrets/api_key`)
  for customers who want to mount secrets from an orchestrator instead of
  plain environment variables (standard Docker secrets convention).
- `ALLOW_INSECURE_NO_AUTH=true` is an explicit, opt-in-only escape hatch for
  isolated lab/dev use — never set it on a network-reachable box.

## Testing

```bash
make test-unit          # unit tests — no running service needed
make test-integration    # integration tests — requires: make up
make test-cpp            # C++ circuit-breaker unit tests (requires g++)
python tools/check_compose_security.py   # static lint: no compose file may ship a weak default
python tools/verify_compose_config.py    # dynamic: docker compose config fails empty, succeeds populated
```

## Documentation map

| Doc | Audience | Notes |
|---|---|---|
| [`working_pipeline/ARCHITECTURE_AND_ROADMAP.md`](working_pipeline/ARCHITECTURE_AND_ROADMAP.md) | Engineers, architects | Current architecture baseline, ownership boundaries, API contracts. Most up to date architecture doc in the repo. |
| [`working_pipeline/CPU_PRODUCTION_PROFILE.md`](working_pipeline/CPU_PRODUCTION_PROFILE.md) | DevOps, deployers | AI Box production profile, first-boot secrets, active tuned config. |
| [`working_pipeline/DEPLOYMENT_AND_OPERATIONS.md`](working_pipeline/DEPLOYMENT_AND_OPERATIONS.md) | DevOps | Docker, monitoring, backup/restore, troubleshooting. |
| [`working_pipeline/QNN_ROADMAP.md`](working_pipeline/QNN_ROADMAP.md) | Engineers | Qualcomm QNN/NPU acceleration status and plan. |
| [`docs/GIOI_THIEU_SAFEAGING.md`](docs/GIOI_THIEU_SAFEAGING.md) | Sales, customers (Vietnamese) | Full product/feature overview and use cases. |
| [`SECURITY.md`](SECURITY.md) | Everyone | Responsible disclosure process and current security posture. |

> **Note:** `working_pipeline/PROJECT_SUMMARY.md`, `IMPLEMENTATION_TEMPLATES.md`,
> `VISUAL_ARCHITECTURE_GUIDE.md`, and `DOCUMENTATION_INDEX.md` are early
> planning documents from the project's initial design phase (January 2026)
> and describe some things (e.g. an `/detect` endpoint, Azure Face API) that
> do not match the current implementation. They're kept for historical
> context; prefer `ARCHITECTURE_AND_ROADMAP.md` and the actual source for
> anything current.

## Security

See [`SECURITY.md`](SECURITY.md) for the responsible-disclosure process and
a summary of security hardening already in place (fail-closed authentication,
no shipped default credentials, secrets-file support, automated compose
security linting). Please report vulnerabilities privately — do not open a
public issue.

## Third-party licenses

This project depends on third-party libraries, each under their own license
(see [`requirements.txt`](requirements.txt) for the full pinned list).
Notably: **`ultralytics`** (used for YOLO26 model export/training) ships
under **AGPL-3.0** by default; confirm license compliance — or acquire an
Ultralytics Enterprise License — before commercial distribution. See
[`LICENSE`](LICENSE) for details on what still needs legal review before
release.

## License

**Proprietary — draft, pending legal review.** See [`LICENSE`](LICENSE).
This is not yet a finalized license; do not treat its current text as
final commercial terms.
