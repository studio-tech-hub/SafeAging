# 🏗️ SafeAging - Architecture & Roadmap

**Date:** May 10, 2026  
**Status:** Updated Architecture Baseline  
**Scope:** Windows-first development, production deployment later on AI Box as edge inference node

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Current State Analysis](#current-state-analysis)
3. [System Architecture](#system-architecture)
4. [Ownership & Responsibilities](#ownership--responsibilities)
5. [Communication Pipeline](#communication-pipeline)
6. [Data Storage & API Strategy](#data-storage--api-strategy)
7. [Implementation Roadmap](#implementation-roadmap)
8. [Development Workflow](#development-workflow)
9. [Success Metrics](#success-metrics)
10. [Conclusion](#conclusion)

---

## 🎯 Project Overview

### Requirements

- Real-time person detection and tracking from NX-integrated cameras
- Person-related metadata management
- Zone / forbidden area monitoring
- Fall detection and alerting
- Event persistence for later investigation
- Snapshot / crop storage for review and recognition workflows
- Resilience when central infrastructure is temporarily unreachable

### Production scope

The system will run in the following logical shape:

```text
Camera
→ Nx Server + Nx Plugin
→ AI Service (edge, AI Box)
→ PostgreSQL (central metadata DB)
→ MinIO / S3-compatible object storage

Nx Archive / NAS
→ video archive / playback / timeline
```

### Core architecture principles

1. Nx owns video
2. PostgreSQL owns metadata
3. Object storage owns images
4. Plugin never accesses DB
5. AI keeps running if DB is down
6. Events retry asynchronously
7. Local cache is mandatory
8. Health states are standardized

---

## 📊 Current State Analysis

### ✅ What the current repo already has

**Plugin (C++)**
- NX plugin integration
- Frame capture and preprocessing
- HTTP client to Python service
- Queueing and backpressure logic
- Circuit breaker and retry logic
- Metadata/event generation to NX

**Service (Python / FastAPI)**
- `GET /health`
- `POST /infer`
- `GET /status`
- in-memory tracking state
- fall detection
- ROI / undistort preprocessing
- model warmup and readiness state

**Build / tooling**
- CMake-based plugin build
- PowerShell build helper on Windows
- service startup entrypoint
- manual test scripts

### ❌ What is still missing for production

**Persistence**
- PostgreSQL-backed metadata store
- migration strategy
- central config storage

**Edge resilience**
- durable local outbox
- durable config cache
- replay after DB recovery

**Business APIs**
- persons API
- zones API
- events API
- reset/admin API with production semantics

**Observability**
- Prometheus metrics endpoint
- structured JSON logs
- alert thresholds for backlog and stale config

**Testing**
- automated unit tests
- automated integration tests
- load tests for multi-camera scenarios

**Packaging / release hygiene**
- normalized manifest / versioning
- cleaned legacy naming
- release artifacts suitable for deployment handoff

---

## 🏛️ System Architecture

### High-level overview

```text
┌──────────────────────────────┐
│ NX VMS / Nx Server           │
│ - camera stream integration  │
│ - video archive / playback   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Nx Plugin (C++)             │
│ - frame sampling            │
│ - queue / backpressure      │
│ - health polling            │
│ - metadata to NX            │
└──────────────┬───────────────┘
               │ REST
               ▼
┌──────────────────────────────────────────┐
│ AI Service (Python / FastAPI)           │
│ - inference                             │
│ - tracking / fall / zone logic          │
│ - local SQLite outbox + config cache    │
│ - background sync workers               │
│ - metrics / health / status             │
└───────┬─────────────────────┬───────────┘
        │                     │
        ▼                     ▼
┌──────────────────┐   ┌────────────────────┐
│ PostgreSQL 16+   │   │ MinIO / S3 storage │
│ central metadata │   │ snapshots / crops  │
└──────────────────┘   └────────────────────┘
```

### Execution environments

#### Development environment

- Primary development machine: **Windows laptop**
- Goal:
  - run service locally
  - run plugin build locally
  - simulate central infra via Docker

#### Production environment

- AI Box runs:
  - AI Service
  - optional Nx plugin runtime side if needed in deployment topology
- Central server / NAS runs:
  - PostgreSQL
  - MinIO
  - Prometheus / Grafana
  - Nx archive if deployed centrally

### Why edge inference + central metadata

This split is chosen because it:

- reduces central GPU dependency
- keeps inference close to the camera processing path
- allows central backup and reporting for metadata
- avoids storing business-critical state only on the edge
- scales better across multiple AI boxes later

---

## 👤 Ownership & Responsibilities

This project is now planned for **one engineer**, not two separate teams.

### Single-owner responsibilities

The same owner is responsible for:

- plugin reliability
- service backend
- schema design
- observability
- testing
- release packaging
- deployment readiness

### Workstreams

#### 1. Plugin workstream

- service communication
- health polling
- NX diagnostic events
- queue/backpressure behavior
- packaging/build hygiene

#### 2. Service workstream

- inference pipeline
- business logic
- metadata persistence
- local outbox/cache
- retry workers
- metrics/logging

#### 3. Infrastructure workstream

- PostgreSQL
- MinIO
- Prometheus / Grafana
- Windows local Docker stack
- deployment handoff toward AI Box

---

## 🔄 Communication Pipeline

### Phase P1 decision

**Plugin ↔ AI Service remains REST/HTTP.**

Rationale:

- current repo already speaks REST
- avoids scope explosion from gRPC migration
- allows health, persistence, testing, and observability to be finished first

### API contract: Plugin ↔ Service

#### Primary endpoints

```text
GET  /health
GET  /status
GET  /metrics
POST /infer
```

#### Current inference transport

For Phase P1, keep compatibility with the current plugin:

- content type: `application/json`
- fields:
  - `camera_id`
  - `image` as base64

This is not the long-term ideal transport, but it preserves compatibility while production-hardening the rest of the system.

#### Deferred transport improvement

Future optimization can move to:

- `multipart/form-data`

without changing:

- detection response schema
- track semantics
- health semantics

### API contract: Business / Admin side

The service should expose production APIs for:

```text
GET    /persons
GET    /persons/{person_id}
POST   /persons
PUT    /persons/{person_id}

GET    /zones
POST   /zones
PUT    /zones/{zone_id}

GET    /events
GET    /events/{event_id}

POST   /reset/camera/{camera_id}
POST   /reset/all
```

### Health model

#### Top-level statuses

- `healthy`
- `degraded`
- `not_ready`

#### Condition / reason codes

- `db_unreachable`
- `service_unreachable`
- `config_stale`
- `outbox_backlog_high`

#### Response example

```json
{
  "status": "degraded",
  "ready": true,
  "reason_codes": ["db_unreachable", "outbox_backlog_high"],
  "dependencies": {
    "postgres": "down",
    "object_storage": "up",
    "config_cache": "fresh"
  },
  "service_uptime_seconds": 1240.5
}
```

### Plugin behavior against health states

#### `healthy`

- normal infer calls
- normal diagnostics

#### `degraded`

- continue inference when possible
- emit clear NX diagnostic warning
- include reason codes in diagnostic payload

#### `not_ready`

- plugin should not assume the service is broken
- poll again with shorter interval
- avoid flooding service during warmup/startup

### Error handling and retry strategy

#### Rule 1: inference must not depend on central DB availability

If central PostgreSQL is down:

- inference still runs
- health becomes `degraded`
- event persistence moves to local outbox

#### Rule 2: outbox pattern is mandatory

Service flow:

1. inference creates event candidate
2. event is written to local SQLite outbox
3. request finishes without waiting for PostgreSQL
4. background worker syncs to PostgreSQL and MinIO later

#### Rule 3: retry uses exponential backoff

Suggested sequence:

```text
1s → 2s → 5s → 10s → 30s → 60s → 300s
```

#### Rule 4: idempotency is mandatory

Every event must have a unique `event_id`:

- prefer `UUIDv7` or `ULID`
- PostgreSQL enforces `UNIQUE(event_id)`
- object storage key naming also derives from `event_id`

This ensures safe retries without duplicated records.

---

## 💾 Data Storage & API Strategy

### Data ownership model

#### Nx / NAS owns video

- raw video
- playback
- archive
- timeline

#### PostgreSQL owns metadata

- persons
- person embeddings
- zones
- events
- alerts
- camera configs

#### Object storage owns images

- snapshots
- crops
- person reference images

#### SQLite on edge owns temporary durable state

- pending outbox jobs
- cached config data
- sync checkpoints

### Central PostgreSQL schema

#### Required tables for Phase P1

```sql
persons
person_embeddings
zones
events
alerts
camera_configs
```

#### Suggested roles

**persons**
- identity/profile metadata
- name, age, gender, notes, room, status

**person_embeddings**
- face embedding or person reference vectors
- linked to person records

**zones**
- zone geometry and zone type
- per-camera scoping

**events**
- canonical event log
- fall, zone violation, detection lifecycle, etc.

**alerts**
- delivery state of email / notification side effects

**camera_configs**
- per-camera thresholds and runtime rules

### Edge SQLite schema

#### Required local tables

```sql
outbox_jobs
cached_camera_configs
cached_zones
cached_watchlists
cached_person_embeddings
sync_state
```

#### Purpose

**outbox_jobs**
- event persistence jobs
- alert jobs
- image upload jobs

**cached_* tables**
- allow continued operation when central DB is unavailable

**sync_state**
- version and checkpoint tracking

### Object storage layout

Recommended object key layout:

```text
snapshots/{camera_id}/{yyyy}/{mm}/{dd}/{event_id}.jpg
crops/{camera_id}/{yyyy}/{mm}/{dd}/{event_id}_{track_id}.jpg
persons/{person_id}/reference/{image_id}.jpg
```

### API semantics

#### `/infer`

- optimized for plugin consumption
- no heavy side effects inline
- should remain as lean as possible

#### business/admin APIs

- backed by PostgreSQL
- may read through local cache when appropriate
- should be versionable later if contract grows

---

## 🗂️ Implementation Roadmap

### Timeline overview

This roadmap is designed for **one person** working sequentially.

```text
Phase 0: Local development baseline on Windows
Phase 1: Central PostgreSQL persistence
Phase 2: Edge SQLite outbox + cache
Phase 3: Business APIs + background workers
Phase 4: Observability + health hardening
Phase 5: Plugin reliability + automated tests
Phase 6: Packaging / release cleanup + AI Box readiness
```

### Phase 0: Windows local baseline

Goals:

- keep development fast on Windows
- simulate production dependencies locally

Tasks:

- create `docker-compose.dev.yml`
- run:
  - PostgreSQL
  - MinIO
  - Prometheus
- normalize `.env`
- define service-to-central connection strings

### Phase 1: PostgreSQL persistence

Maps to:

- `Service P1.1`
- partially `Service P1.2`

Tasks:

- define schema
- choose ORM / DAL strategy
- add migrations
- persist:
  - persons
  - zones
  - events
  - alerts
  - camera configs

Definition of done:

- service can read/write central metadata
- no plugin changes required for initial persistence path

### Phase 2: Edge SQLite outbox + cache

Maps to:

- `Service P1.3`
- `Service P1.5`

Tasks:

- create local `edge_state.db`
- enable WAL mode
- implement durable outbox
- implement local config cache
- implement sync state tracking
- add idempotent `event_id` handling

Definition of done:

- DB outage does not stop inference
- pending jobs survive service restart

### Phase 3: Business APIs and background workers

Maps to:

- `Service P1.2`
- `Service P1.3`

Tasks:

- add person APIs
- add zone APIs
- add event APIs
- add production reset/admin APIs
- add worker for:
  - event persistence
  - alert delivery
  - image upload

Definition of done:

- `/infer` path stays lightweight
- side effects happen asynchronously

### Phase 4: Observability and health hardening

Maps to:

- `Service P1.5`
- `Plugin P1.1`

Tasks:

- add `GET /metrics`
- expose Prometheus metrics
- add structured JSON logs
- standardize health contract
- include reason codes and dependency states
- define warning thresholds for:
  - stale config
  - outbox backlog

Definition of done:

- operators can detect degraded states early
- plugin can act on health consistently

### Phase 5: Plugin reliability and automated tests

Maps to:

- `Plugin P1.1`
- `Plugin P1.3`
- `Plugin P1.4`
- `Service P1.4`

Tasks:

- plugin health polling
- NX diagnostic events with clear degraded semantics
- queue metric thresholds
- integration tests for:
  - queue/backpressure
  - circuit breaker
  - reconnect behavior
  - metadata / event consistency
- unit tests for:
  - tracking
  - fall detection
  - ROI
  - undistort
- load tests for multiple cameras

Definition of done:

- resilience logic is proven by tests, not only by logs

### Phase 6: Packaging, release cleanup, and AI Box readiness

Maps to:

- `Plugin P1.2`

Tasks:

- clean manifest/version duplication
- remove legacy naming where possible
- standardize build scripts
- define release artifact contents
- document Linux/AI Box deployment prerequisites

Definition of done:

- release artifacts are predictable
- deployment handoff is clean

---

## 🚀 Development Workflow

### Solo workflow

This project no longer uses a two-team structure.

Recommended workflow:

1. architecture decision
2. minimal implementation
3. local test
4. observability
5. hardening
6. packaging

### Branching strategy

Recommended:

- `main` for stable baseline
- short-lived feature branches:
  - `feature/service-persistence`
  - `feature/plugin-health`
  - `feature/outbox-worker`

### Definition of done

A task is only complete when:

- code compiles / runs
- config is documented
- tests exist where applicable
- logs / metrics are visible
- failure mode is understood

### Testing strategy

#### Unit tests

- tracking logic
- fall detection heuristics
- ROI handling
- undistort path
- outbox state transitions

#### Integration tests

- service API with PostgreSQL
- service API with MinIO
- DB outage and recovery behavior
- plugin ↔ service retry behavior

#### Load tests

- multiple cameras
- service degradation under backlog
- reconnect after dependency outage

### Version management

Use semantic versioning:

```text
MAJOR.MINOR.PATCH
```

Guidelines:

- increment patch for bugfix/hardening
- increment minor for backward-compatible features
- increment major only for contract-breaking changes

### Release artifacts

#### Plugin release artifact

- plugin binary
- manifest
- version metadata
- deployment notes

#### Service release artifact

- Python package / deployment directory
- env template
- migration bundle
- docker compose or deployment instructions

### Deferred items

Not in Phase P1:

- gRPC transport
- Redis-based edge runtime
- advanced dashboard/reporting layer
- full multi-node orchestration

---

## 📊 Success Metrics

### Performance targets

- plugin-side frame handling remains real-time oriented
- inference endpoint remains responsive under normal load
- queue backlog stays bounded under expected camera rate

### Reliability targets

- inference continues when PostgreSQL is temporarily unavailable
- events are replayed after dependency recovery
- no duplicate event records under retry
- plugin produces actionable NX diagnostics

### Observability targets

Minimum required metrics:

- `infer_requests_total`
- `infer_latency_ms`
- `db_up`
- `object_storage_up`
- `outbox_pending_total`
- `outbox_oldest_age_seconds`
- `config_last_sync_age_seconds`
- `events_synced_total`
- `events_sync_failed_total`

### Suggested warning thresholds

#### config stale

- warning if config cannot be refreshed for more than **5 minutes**

#### outbox backlog high

- warning if pending jobs exceed **100**
- or oldest pending job age exceeds **60 seconds**

These should remain configurable, not hardcoded forever.

---

## 🎯 Conclusion

This updated architecture intentionally separates:

- **video responsibilities** into Nx / NAS
- **metadata responsibilities** into PostgreSQL
- **image storage responsibilities** into MinIO
- **runtime resilience responsibilities** into the edge AI service with SQLite outbox/cache

It is designed to:

- work on Windows during development
- migrate cleanly to AI Box deployment later
- avoid coupling the plugin to the database
- keep inference alive during central outages
- provide a clear path for the current P1 tasks

The immediate priority is not protocol redesign or distributed complexity. The immediate priority is:

1. persistence
2. outbox/cache resilience
3. observability
4. plugin health semantics
5. automated tests
6. release hygiene

---

**Document Version:** 2.0
