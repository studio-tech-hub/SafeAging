# 🗺️ Visual Architecture & Decision Guide

Quick reference diagrams and decision trees for the elderly care management system.

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          NX VMS (Video Source)                              │
│                    Provides video streams @ 30fps                            │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │ RTSP/HTTP
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PLUGIN LAYER (C++)                                   │
│                    (Process frames, track objects)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  device_agent.cpp                                                           │
│  ├─ Capture frames from NX VMS                                              │
│  ├─ Send to Service API for inference                                       │
│  ├─ Receive detection results                                               │
│  ├─ Track objects (IoU + appearance-based)                                  │
│  ├─ Check for falls (bounding box patterns)                                 │
│  ├─ Validate zones (geometry)                                               │
│  └─ Generate NX events with metadata                                        │
│                                                                             │
│  Interfaces:                                                                 │
│  Input:  ← NX VMS video frames                                              │
│  Output: → Service API (HTTP POST)                                          │
│           → NX VMS events                                                    │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │ HTTP POST
                                     │ Port 18000
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SERVICE LAYER (Python)                                  │
│                  (AI processing, analytics, integration)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  FastAPI Service (service.py)                                               │
│  ├─ POST /detect                                                             │
│  │  └─ Run YOLO26 inference on frame                                        │
│  │     └─ Detect people in frame                                            │
│  │     └─ Return bounding boxes + confidence                                │
│  │                                                                           │
│  ├─ POST /analytics/fall-detection                                          │
│  │  └─ Analyze bounding box sequence                                        │
│  │     └─ Detect fall patterns                                              │
│  │     └─ Return confidence + reason                                        │
│  │                                                                           │
│  ├─ POST /analytics/zone-check                                              │
│  │  └─ Check if position violates zones                                     │
│  │     └─ Load zone definitions from DB                                     │
│  │     └─ Return zone violations                                            │
│  │                                                                           │
│  ├─ GET /person/{id}                                                        │
│  │  └─ Get person profile from database                                     │
│  │                                                                           │
│  ├─ POST /person                                                             │
│  │  └─ Create/update person profile                                         │
│  │     └─ Extract face embedding (Azure API)                                │
│  │                                                                           │
│  ├─ GET /health                                                              │
│  │  └─ Health check + status                                                │
│  │                                                                           │
│  └─ GET /metrics                                                             │
│     └─ Performance metrics for monitoring                                    │
│                                                                             │
│  Components:                                                                 │
│  ├─ YOLO26 Model (inference engine)                                         │
│  ├─ Fall Detector (algorithm)                                               │
│  ├─ Zone Validator (geometry)                                               │
│  ├─ Face Recognition (Azure integration)                                    │
│  ├─ Email Service (SendGrid integration)                                    │
│  └─ Database Client (SQLAlchemy)                                            │
│                                                                             │
│  Interfaces:                                                                 │
│  Input:  ← Plugin HTTP requests                                             │
│           ← External API callbacks                                           │
│  Output: → Database (PostgreSQL)                                            │
│           → Email (SendGrid)                                                │
│           → Analytics (Prometheus)                                          │
│           → Logs (ELK Stack)                                                │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────────┐
           │                         │                             │
           ▼                         ▼                             ▼
┌──────────────────────┐  ┌──────────────────────┐   ┌──────────────────────┐
│  PostgreSQL Database │  │   External Services  │   │  Monitoring Stack    │
├──────────────────────┤  ├──────────────────────┤   ├──────────────────────┤
│ persons              │  │ Azure Face API       │   │ Prometheus           │
│ face_embeddings      │  │ ├─ Face detection    │   │ ├─ Metrics storage   │
│ events               │  │ ├─ Face matching     │   │ │                    │
│ zones                │  │ └─ Age/gender check  │   │ Grafana              │
│ tracking_data        │  │                      │   │ ├─ Dashboards        │
│ alerts               │  │ SendGrid API         │   │ │                    │
│                      │  │ ├─ Email sending     │   │ ELK Stack            │
│ Replication:         │  │ └─ Delivery tracking │   │ ├─ Log aggregation   │
│ ├─ Master-Slave      │  │                      │   │ ├─ Search & analyze  │
│ ├─ Backup daily      │  │ Twilio (optional)    │   │ └─ Visualization     │
│ └─ 7-day retention   │  │ └─ SMS alerts        │   │                      │
│                      │  │                      │   │ Alert Manager        │
│ HA: Multi-region     │  │ Firebase (optional)  │   │ └─ Alert routing     │
│ Backup: S3/Cold      │  │ └─ Push notifications│   │                      │
└──────────────────────┘  └──────────────────────┘   └──────────────────────┘
```

---

## Data Flow Diagram

```
Frame Processing Pipeline:
────────────────────────────

NX VMS
  │ Video stream
  ▼
[Plugin] Capture Frame (30fps)
  │
  ├─→ Preprocess (resize, normalize)
  │
  ▼
[Plugin] Send to Service
  │ HTTP POST /detect
  │
  ▼
[Service] YOLO26 Inference
  │
  ├─→ Detect people (bounding boxes)
  ├─→ Filter low confidence detections
  ├─→ Apply NMS (non-maximum suppression)
  │
  ▼
[Service] Return Detection Results
  │ HTTP 200 + JSON
  │
  ▼
[Plugin] Receive Detections
  │
  ├─→ Match with existing tracks
  ├─→ Create new tracks
  ├─→ Update track history
  │
  ▼
[Plugin] Analyze Each Track
  │
  ├─→ Check for fall patterns
  │   └─→ [Service] POST /analytics/fall-detection
  │   └─→ [Service] Check bounding box history
  │   └─→ [Service] Return fall confidence
  │   └─→ IF FALL DETECTED: Send alert
  │
  ├─→ Check zone violations
  │   └─→ [Service] POST /analytics/zone-check
  │   └─→ [Service] Load zone definitions from DB
  │   └─→ [Service] Check point-in-polygon
  │   └─→ IF ZONE VIOLATED: Send alert
  │
  └─→ Get person info
      └─→ [Service] GET /person/{person_id}
      └─→ [Service] Query database
      └─→ Attach name/metadata to track
  │
  ▼
[Plugin] Generate NX Events
  │
  ├─→ Create event metadata
  ├─→ Attach detection data
  ├─→ Set event type & severity
  │
  ▼
[NX VMS] Receive Events
  │
  └─→ Display in NX interface
  └─→ Log in NX database
  └─→ Trigger NX rules/actions


Alert Delivery Pipeline:
───────────────────────

[Service] Event Created
  │
  ├─→ Save to PostgreSQL
  │
  ├─→ IF TYPE = FALL or ZONE_VIOLATION
  │   │
  │   ├─→ Get person emergency contact from DB
  │   │
  │   ├─→ Create email (SendGrid)
  │   │   ├─ To: emergency_contact@example.com
  │   │   ├─ Subject: ⚠️ [EVENT_TYPE] Alert
  │   │   └─ Body: Person name, location, timestamp, action
  │   │
  │   ├─→ Queue email (async)
  │   │
  │   ├─→ [SendGrid API] Send email
  │   │   └─ Track delivery status
  │   │
  │   └─→ Update alert status in DB (sent/failed)
  │
  ├─→ Log to ELK Stack
  │   └─ Make searchable & analyzable
  │
  └─→ Update Prometheus metrics
      └─ Track event counts, latencies
```

---

## Team Interaction Diagram

```
Daily Workflow:
───────────────

09:30 Daily Standup (15 min)
    │
    ├─→ TEAM A presents progress
    │   "Completed: Database schema, fall detector logic"
    │   "Today: Integration with service endpoints"
    │   "Blocker: None"
    │
    ├─→ TEAM B presents progress
    │   "Completed: Plugin refactoring, HTTP client"
    │   "Today: Zone validation integration"
    │   "Blocker: Need API documentation"
    │
    └─→ ACTION: TEAM A updates API docs immediately

Throughout Day:
    │
    ├─→ TEAM A develops analytics engine
    │   └─→ Updates shared API documentation
    │   └─→ Commits code to feature/team-a/analytics
    │   └─→ Runs unit tests
    │
    ├─→ TEAM B integrates with service
    │   └─→ Tests HTTP client against TEAM A's endpoints
    │   └─→ Commits code to feature/team-b/integration
    │   └─→ Reports integration issues in Slack
    │
    └─→ Async communication via Slack
        ├─ Quick questions: < 1 min response
        ├─ Technical discussions: Pull request comments
        └─ Blocking issues: Schedule 30-min sync

Friday Weekly Sync (1 hour):
    │
    ├─→ DEMO: What each team built (20 min)
    │   ├─ TEAM A: Fall detection working end-to-end
    │   ├─ TEAM B: Plugin calling service API successfully
    │   └─ Integration test: Both working together
    │
    ├─→ IDENTIFY BLOCKERS (15 min)
    │   ├─ Performance: Inference too slow?
    │   ├─ Interface: API contract mismatch?
    │   └─ Testing: Integration test failing?
    │
    ├─→ PLAN NEXT WEEK (20 min)
    │   ├─ Adjust priorities
    │   ├─ Identify dependencies
    │   └─ Allocate resources
    │
    └─→ DOCUMENT DECISIONS
        └─→ Update PROJECT_SUMMARY.md
```

---

## API Contract Decision Tree

```
Plugin needs to process a frame:
│
└─→ Does frame contain people?
    │
    ├─ NO: Skip inference (save time)
    │
    └─ YES: Send to Service
        │
        ├─→ [Service] Run YOLO26
        │
        ├─→ Get detections back
        │   │
        │   └─→ For each detection:
        │       │
        │       ├─→ Did we see this person before?
        │       │   │
        │       │   ├─ YES: Update track
        │       │   │ └─→ Check for fall?
        │       │   │ └─→ Check zones?
        │       │   │
        │       │   └─ NO: Create new track
        │       │       └─→ Is it a known person?
        │       │           ├─ YES: Attach ID
        │       │           └─ NO: Unknown person
        │       │
        │       └─→ Generate event for NX
        │
        └─→ Continue tracking...


Fall Detection Decision Tree:
────────────────────────────

Track history has 3+ detections?
│
├─ NO: Not enough data
│
└─ YES: Send to fall detection
    │
    └─→ [Service] Analyze patterns
        │
        ├─→ Calculate aspect ratio change
        │   (Is person becoming horizontal?)
        │
        ├─→ Calculate height change
        │   (Is person getting shorter?)
        │
        ├─→ Calculate Y-position drop
        │   (Is person moving down?)
        │
        └─→ Combine scores
            │
            └─→ Score >= 0.6 (threshold)?
                │
                ├─ YES: FALL DETECTED
                │   ├─→ Generate critical event
                │   ├─→ Send email alert
                │   └─→ Log in database
                │
                └─ NO: Normal movement
                    └─→ Continue tracking


Zone Violation Decision Tree:
────────────────────────────

Person position known?
│
├─ NO: Skip zone check
│
└─ YES: Check against all zones
    │
    └─→ [Service] Load zones for camera
        │
        └─→ For each zone:
            │
            ├─→ Is zone active?
            │   ├─ NO: Skip
            │   └─ YES: Continue
            │
            └─→ Is person position inside zone?
                │
                ├─ NO: Continue to next zone
                │
                └─ YES: VIOLATION DETECTED
                    │
                    ├─→ Is it forbidden zone?
                    │   ├─ YES: High severity
                    │   └─ NO: Medium severity
                    │
                    ├─→ Was alert already sent?
                    │   ├─ YES: Don't spam
                    │   └─ NO: Send alert
                    │
                    ├─→ Generate event
                    ├─→ Send email to staff
                    └─→ Log in database
```

---

## Technology Stack Decision Matrix

```
Component               TEAM A          TEAM B
────────────────────────────────────────────────
Language                Python          C++
Framework               FastAPI         NX SDK + CMake
Database                PostgreSQL      N/A (Service's concern)
ORM                     SQLAlchemy      N/A
Cache                   Redis           (Service's concern)
HTTP                    httpx/aiohttp   libcurl
Face API                Azure Face      (Service's concern)
Email                   SendGrid        (Service's concern)
Testing                 pytest          gtest
Build                   pip             CMake
Containerization        Docker          (via Python service)
Logging                 structlog       fprintf/cerr
Metrics                 prometheus_client  (via HTTP)
```

---

## Performance Targets Visualization

```
Service Inference Timeline (Target: < 50ms):
────────────────────────────────────────────
0ms ┌─ Frame received
    │
2ms ├─ Preprocessing (resize, normalize)
    │
35ms├─ YOLO26 Inference
    │
42ms├─ NMS & post-processing
    │
48ms└─ Return results
    ✓ Within 50ms target

Plugin HTTP Call Timeline (Target: < 2.5s):
──────────────────────────────────────────
0ms ┌─ Build HTTP request
    │
10ms├─ TCP connection (connection pool)
    │
20ms├─ Send HTTP POST
    │
100ms├─ Service processes request
    │
110ms├─ Receive response
    │
115ms└─ Parse response
    ✓ Within 2.5s timeout (with large margin)


Database Query Timeline (Target: < 50ms):
────────────────────────────────────────
0ms ┌─ Query prepared
    │
2ms ├─ Network latency to DB
    │
5ms ├─ Query execution
    │
7ms ├─ Network latency back
    │
10ms└─ Result available
    ✓ Well within 50ms target


Fall Detection End-to-End (Target: < 5 seconds):
────────────────────────────────────────────────
0ms ┌─ Person starts falling
    │
100ms├─ Frame captured
    │
135ms├─ Detection + inference (50ms latency)
    │
250ms├─ Plugin sends fall detection request
    │
350ms├─ Service analyzes pattern (100ms)
    │
360ms├─ Fall confirmed
    │
370ms├─ Generate alert event
    │
400ms├─ Send email to SendGrid (via async queue)
    │
2500ms├─ Email sent and tracked
    │
5000ms└─ Notification displayed (varies by email client)
    ✓ Alert initiated within 2.5 seconds
```

---

## Database Schema Relationship Diagram

```
persons
├─ id (PK)
├─ name
├─ age
├─ gender
├─ room_number
├─ emergency_contact
├─ face_image
├─ face_embedding
├─ status (active/inactive/visiting)
├─ created_at
└─ updated_at
    │
    ├─ ← face_embeddings.person_id (FK)
    ├─ ← events.person_id (FK)
    ├─ ← tracking_data.person_id (FK)
    └─ ← alerts.person_id (FK)

face_embeddings (one-to-many)
├─ id (PK)
├─ person_id (FK)
├─ embedding (512-dim vector)
├─ confidence
├─ camera_id
└─ capture_date

events (one-to-many)
├─ id (PK)
├─ event_type (detection/fall/zone_violation)
├─ person_id (FK) [optional, null if unknown]
├─ track_id
├─ camera_id
├─ severity
├─ description
├─ metadata
└─ created_at
    │
    └─ → alerts.event_id (FK)

alerts (one-to-many)
├─ id (PK)
├─ event_id (FK)
├─ person_id (FK)
├─ alert_type (email/sms/push)
├─ recipient
├─ status (pending/sent/failed)
├─ sent_at
└─ error_message

zones (per-camera config)
├─ id (PK)
├─ camera_id
├─ zone_name
├─ zone_type (forbidden/danger/normal)
├─ polygon (coordinates)
├─ is_active
├─ created_at
├─ updated_at
└─ metadata

tracking_data (temporary/recent)
├─ id (PK)
├─ person_id (FK) [optional]
├─ track_id
├─ camera_id
├─ position (x, y)
├─ velocity (dx, dy)
├─ bbox (x1, y1, x2, y2)
└─ timestamp
```

---

## Deployment Architecture Diagram

```
Development (Laptop)
════════════════════
┌──────────────────────────────────────┐
│ Service (Python)                     │
│ localhost:18000                      │
├──────────────────────────────────────┤
│ PostgreSQL                           │
│ localhost:5432                       │
├──────────────────────────────────────┤
│ Plugin (compiled locally)            │
│ NX VMS localhost:7001                │
└──────────────────────────────────────┘


Staging (Virtual Machine)
══════════════════════════
┌──────────────────────────────────────┐
│ Docker Network                       │
├──────────────────────────────────────┤
│ Service Container   Service Container│
│ :18000              :18001           │
│                                      │
│ PostgreSQL Container                 │
│ :5432                                │
│                                      │
│ Prometheus / Grafana                 │
│ :9090 / :3000                        │
└──────────────────────────────────────┘


Production (Multi-Node)
═══════════════════════
┌──────────────────────────────────────┐
│ Nginx Load Balancer                  │
│ :80, :443                            │
└────────────┬─────────────────────────┘
             │
   ┌─────────┼─────────┐
   │         │         │
   ▼         ▼         ▼
┌─────┐  ┌─────┐  ┌─────┐
│ App │  │ App │  │ App │   (Service Replicas)
│ 1   │  │ 2   │  │ 3   │
└─────┘  └─────┘  └─────┘
   │         │         │
   └─────────┼─────────┘
             │
             ▼
       ┌──────────────┐
       │ PostgreSQL   │
       │ (HA + slave) │
       └──────────────┘
             │
   ┌─────────┼─────────┐
   │         │         │
   ▼         ▼         ▼
┌──────┐ ┌──────┐ ┌──────┐
│ Prom │ │Grafana│ │ ELK  │  (Monitoring)
└──────┘ └──────┘ └──────┘
```

---

## Deployment Checklist Timeline

```
Week 1: Environment Setup
────────────────────────
Day 1:
  ☐ Git repo created & branched
  ☐ Development environments validated
  ☐ First daily standup completed

Day 2-3:
  ☐ TEAM A: Database designed
  ☐ TEAM B: Plugin refactored
  ☐ Testing frameworks setup

Day 4-5:
  ☐ Basic integration test works
  ☐ API documented (v0.1)
  ☐ Weekly sync: adjust plan


Week 3-4: Integration Phase
────────────────────────────
  ☐ Fall detection end-to-end working
  ☐ Zone validation integrated
  ☐ Email alerts functional
  ☐ Face API integration tested
  ☐ Integration tests all passing


Week 5-6: Optimization Phase
────────────────────────────
  ☐ Performance benchmarks met
  ☐ Load testing completed
  ☐ Memory optimization done
  ☐ Docker image created
  ☐ Monitoring stack setup


Week 7-8: Production Phase
──────────────────────────
  ☐ Docker-compose fully functional
  ☐ Monitoring dashboards live
  ☐ Alerting rules configured
  ☐ Backup procedures tested
  ☐ Documentation complete
  ☐ Team trained on operations
  ☐ Production deployment successful
```

---

**Document Version:** 1.0  
**Created:** January 18, 2026  
**Status:** Ready for Use
