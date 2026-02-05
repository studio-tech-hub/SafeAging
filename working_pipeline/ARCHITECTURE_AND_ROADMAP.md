# 🏗️ Hệ Thống Quản Lý Con Người - Architecture & Roadmap

**Date:** January 18, 2026  
**Status:** Architecture Planning  
**Project:** Elderly Care Management System (YOLOv8 People Analytics)

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Current State Analysis](#current-state-analysis)
3. [System Architecture](#system-architecture)
4. [Team Division & Responsibilities](#team-division--responsibilities)
5. [Communication Pipeline](#communication-pipeline)
6. [Data Storage & API Strategy](#data-storage--api-strategy)
7. [Implementation Roadmap](#implementation-roadmap)
8. [Development Workflow](#development-workflow)

---

## 🎯 Project Overview

### Requirements
- **Person Recognition:** Detect and identify individuals via face recognition
- **People Counting:** Real-time count of people in frame
- **Face-Based Profile Storage:** Name, age, gender, and personal information
- **Danger Zones & Forbidden Areas:** Alert notifications when entering restricted zones
- **Fall Detection:** Immediate email notification on fall events

### System Scope
- **Video Source:** Network cameras (NX VMS integration)
- **Processing:** YOLOv8 model-based detection + tracking
- **Analysis:** Real-time events (counting, fall detection, zone violation)
- **Notification:** Email alerts for critical events
- **Data Persistence:** Face embeddings, person profiles, event logs

---

## 📊 Current State Analysis

### ✅ What We Have
```
Plugin (C++)
├── YOLOv8 Integration ✓
├── Object Detection ✓
├── Object Tracking ✓
└── NX VMS Communication ✓

Service (Python)
├── FastAPI REST API ✓
├── Model Loading ✓
├── Frame Processing ✓
├── HTTP Client to Plugin ✓
└── Basic Inference ✓

Infrastructure
├── CMake Build System ✓
├── Requirements Management ✓
├── Startup Scripts ✓
└── Configuration via ENV ✓
```

### ❌ What's Missing
```
Core Features
├── Face Recognition & Embedding
├── Person Re-identification (ReID)
├── Fall Detection Model/Algorithm
├── Zone Management System
├── Multi-camera Coordination
└── Alert/Email Service

Data Layer
├── Database Design
├── Face Feature Store
├── Event Log Management
├── Person Profile Management
└── Zone Configuration Storage

Integration
├── Third-party Face API (Microsoft, AWS, etc.)
├── Email Service Integration
├── Data Export/Analytics APIs
└── Frontend Dashboard

DevOps
├── Docker Containerization
├── Health Monitoring
├── Error Recovery
└── Multi-instance Coordination
```

---

## 🏛️ System Architecture

### High-Level Overview
```
┌─────────────────────────────────────────────────────────────┐
│                    NX VMS (Video Manager)                   │
│                   (Provides Video Streams)                   │
└───────────────────────┬─────────────────────────────────────┘
                        │ RTSP/HTTP
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         PLUGIN LAYER (C++) - Process Video                  │
├─────────────────────────────────────────────────────────────┤
│ Device Agent (device_agent.cpp)                             │
│ ├─ Frame Capture & Preprocessing                            │
│ ├─ Object Detection (calls Service)                         │
│ ├─ Object Tracking                                          │
│ └─ Metadata Generation                                      │
│                                                             │
│ Object Detector (object_detector.cpp)                       │
│ ├─ HTTP Client to Service                                   │
│ ├─ Detection Result Parsing                                 │
│ └─ Error Handling & Retry Logic                             │
│                                                             │
│ Object Tracker (object_tracker.cpp)                         │
│ ├─ Track Management                                         │
│ ├─ ID Association                                           │
│ └─ Flicker Suppression                                      │
└────────────┬────────────────────────────┬────────────────────┘
             │ HTTP Port 18000             │ Event Output
             │ Inference Requests          │ to NX VMS
             ▼                             │
┌─────────────────────────────────────────┼───────────────────┐
│       SERVICE LAYER (Python)             │                   │
├─────────────────────────────────────────┼───────────────────┤
│ FastAPI Server (service.py)              │                   │
│ ├─ GET /health - Status Check            │                   │
│ ├─ POST /detect - Frame Inference        │                   │
│ ├─ POST /track - Track Update            │                   │
│ ├─ GET /metrics - Performance Stats      │                   │
│ └─ POST /reset - Reset State             │                   │
│                                          │                   │
│ YOLOv8 Model                             │                   │
│ ├─ Person Detection                      │                   │
│ ├─ Frame Preprocessing                   │                   │
│ ├─ Inference Engine                      │                   │
│ └─ Post-processing & NMS                 │                   │
│                                          │                   │
│ Analytics Engine (NEW)                   │                   │
│ ├─ Fall Detection Algorithm               │                   │
│ ├─ Zone Validation                       │                   │
│ ├─ Person Counting Logic                 │                   │
│ └─ Event Generation                      │                   │
└────────┬──────────────────────────────────┴───────────────────┘
         │ REST API
         ├──────────────────────────────────────┐
         │                                      │
         ▼                                      ▼
┌──────────────────────────┐      ┌─────────────────────────┐
│   DATA LAYER (Database)  │      │  EXTERNAL SERVICES      │
├──────────────────────────┤      ├─────────────────────────┤
│ PostgreSQL / MongoDB     │      │ Face Recognition API    │
│ ├─ Person Profiles       │      │ (Azure/AWS/OpenAI)      │
│ ├─ Face Embeddings       │      │                         │
│ ├─ Event Logs            │      │ Email Service           │
│ ├─ Zone Definitions      │      │ (SendGrid/AWS SES)      │
│ └─ Tracking Data         │      │                         │
│                          │      │ Analytics Service       │
│                          │      │ (ELK/DataDog)           │
└──────────────────────────┘      └─────────────────────────┘
```

### Component Breakdown

#### Plugin Layer (C++)
```
Responsibilities:
- Capture frames from NX VMS
- Call Service API for inference
- Manage object tracking
- Generate NX metadata events
- Handle plugin lifecycle

Interfaces:
- ← NX VMS Frame Stream
- → Service HTTP (Port 18000)
- → NX VMS Event Metadata
```

#### Service Layer (Python)
```
Responsibilities:
- Load & run YOLOv8 model
- Process frames (preprocessing, inference)
- Run analytics algorithms
- Coordinate with databases
- Manage service health

Interfaces:
- ← Plugin HTTP Requests (Port 18000)
- ← Database Connections
- → External APIs (Face, Email, Analytics)
```

#### Data Layer
```
Responsibilities:
- Persist person profiles
- Store face embeddings
- Log all events
- Manage zone definitions
- Track person history

Interfaces:
- ← Service writes
- → Service reads
```

---

## 👥 Team Division & Responsibilities

### Team Structure
```
┌─────────────────────────────────────────┐
│    Project Manager / Architect (You)    │
│  - System Design                        │
│  - Integration Coordination              │
│  - Deployment & DevOps                   │
└──────────┬──────────────────────┬────────┘
           │                      │
           ▼                      ▼
    ┌─────────────┐        ┌──────────────┐
    │  TEAM A     │        │   TEAM B     │
    │  Service    │        │   Plugin     │
    │  Engineer   │        │   Engineer   │
    └─────────────┘        └──────────────┘
```

### TEAM A: Service Engineer
**Focus:** Backend Logic, Data Processing, External Integrations

#### Phase 1: Foundation (Week 1-2)
- [ ] Extend `service.py` with analytics engine
  - [ ] Fall detection algorithm
  - [ ] Person counting logic
  - [ ] Zone validation checks
  - [ ] Event generation system
- [ ] Design & implement database schema
  - [ ] Person profiles table
  - [ ] Face embeddings table
  - [ ] Event logs table
  - [ ] Zone definitions table
- [ ] Create database abstraction layer
  - [ ] Connection pooling
  - [ ] CRUD operations
  - [ ] Transaction handling
- [ ] Add new service endpoints
  - [ ] POST /analytics/fall-detection
  - [ ] POST /analytics/zone-check
  - [ ] GET /person/{id}
  - [ ] POST /person (create/update)
  - [ ] GET /events

#### Phase 2: Integration (Week 3-4)
- [ ] Face recognition service integration
  - [ ] Azure Face API client
  - [ ] Embedding extraction
  - [ ] Face matching algorithm
  - [ ] Person re-identification
- [ ] Email notification service
  - [ ] SendGrid/AWS SES integration
  - [ ] Template system
  - [ ] Delivery tracking
- [ ] Analytics & monitoring
  - [ ] Event aggregation
  - [ ] Performance metrics
  - [ ] Error tracking
- [ ] Testing & validation
  - [ ] Unit tests for analytics
  - [ ] Integration tests with DB
  - [ ] Load testing
  - [ ] Error scenario testing

#### Phase 3: Optimization (Week 5+)
- [ ] Performance tuning
  - [ ] Query optimization
  - [ ] Caching strategy
  - [ ] Batch processing
- [ ] Advanced features
  - [ ] Multi-camera coordination
  - [ ] Person trajectory tracking
  - [ ] Heat map generation
  - [ ] Behavior analysis

#### Deliverables (TEAM A)
```
Service Files:
├── service.py (extended)
│   ├── Analytics Engine
│   ├── Fall Detection
│   ├── Zone Management
│   └── Event System
├── analytics/
│   ├── fall_detector.py
│   ├── zone_validator.py
│   ├── person_counter.py
│   └── event_generator.py
├── database/
│   ├── db_client.py
│   ├── models.py
│   ├── migrations/
│   └── queries.py
├── integrations/
│   ├── face_recognition.py
│   ├── email_service.py
│   ├── analytics_service.py
│   └── config.py
└── tests/
    ├── test_analytics.py
    ├── test_database.py
    ├── test_integrations.py
    └── test_service.py

Configuration:
├── docker-compose.yml
├── .env.example
├── requirements.txt (updated)
└── API_DOCUMENTATION.md
```

---

### TEAM B: Plugin Engineer
**Focus:** NX VMS Integration, Video Processing, Reliability

#### Phase 1: Foundation (Week 1-2)
- [ ] Refactor object_detector.cpp
  - [ ] Add health monitoring endpoints
  - [ ] Implement retry logic with exponential backoff
  - [ ] Add detailed logging
  - [ ] Connection pooling
  - [ ] Timeout configurations
- [ ] Enhance device_agent.cpp
  - [ ] Metadata enrichment
  - [ ] Event filtering
  - [ ] State management
  - [ ] Error recovery
- [ ] Create configuration system
  - [ ] Plugin settings (detection_interval, confidence, etc.)
  - [ ] Feature toggles
  - [ ] Dynamic reloading
- [ ] Add comprehensive logging
  - [ ] Structured logging
  - [ ] Performance profiling
  - [ ] Diagnostic endpoints

#### Phase 2: Feature Enhancement (Week 3-4)
- [ ] Implement fall detection trigger
  - [ ] Detect fall patterns from bounding box changes
  - [ ] Cross-reference with service fall detection
  - [ ] Generate high-priority events
  - [ ] Track fall confirmation
- [ ] Zone management integration
  - [ ] Load zone definitions from service
  - [ ] Validate object positions against zones
  - [ ] Generate zone violation events
  - [ ] Cache zone data locally
- [ ] Person profile enrichment
  - [ ] Attach person IDs to detections
  - [ ] Query service for known person info
  - [ ] Include metadata in NX events
  - [ ] Handle new/unknown persons
- [ ] Real-time event generation
  - [ ] Create NX event objects
  - [ ] Set event metadata
  - [ ] Handle event lifecycle
  - [ ] Error handling

#### Phase 3: Stability & Performance (Week 5+)
- [ ] Multi-camera support
  - [ ] Camera identification
  - [ ] Per-camera configuration
  - [ ] Load balancing
  - [ ] Service discovery
- [ ] Performance optimization
  - [ ] Frame buffering
  - [ ] Adaptive frame skipping
  - [ ] Memory management
  - [ ] CPU profiling
- [ ] Resilience patterns
  - [ ] Circuit breaker pattern
  - [ ] Graceful degradation
  - [ ] Auto-recovery
  - [ ] Health checks

#### Deliverables (TEAM B)
```
Plugin Files (C++):
├── src/sample_company/vms_server_plugins/opencv_object_detection/
│   ├── device_agent.cpp (enhanced)
│   ├── device_agent.h
│   ├── object_detector.cpp (enhanced)
│   ├── object_detector.h
│   ├── object_tracker.cpp (enhanced)
│   ├── object_tracker.h
│   ├── analytics_processor.cpp (new)
│   ├── analytics_processor.h (new)
│   ├── zone_validator.cpp (new)
│   ├── zone_validator.h (new)
│   ├── http_client.cpp (new)
│   ├── http_client.h (new)
│   └── config_manager.cpp (new)
│       config_manager.h (new)
├── CMakeLists.txt (updated)
└── tests/
    ├── test_device_agent.cpp
    ├── test_object_detector.cpp
    ├── test_analytics_processor.cpp
    └── test_http_client.cpp

Configuration & Build:
├── build.bat (updated)
├── build.sh (updated)
├── CMakeSettings.json (updated)
└── PLUGIN_DEVELOPMENT.md
```

---

## 🔄 Communication Pipeline

### API Contract (Plugin ↔ Service)

#### 1. **Inference Request**
```http
POST /detect HTTP/1.1
Host: localhost:18000
Content-Type: application/json

{
  "frame_id": "camera_1_2026_01_18_120000_001",
  "camera_id": "camera_1",
  "timestamp": 1705574400000,
  "frame_data": "base64_encoded_image",
  "frame_height": 1080,
  "frame_width": 1920,
  "metadata": {
    "location": "hallway_floor_2",
    "frame_quality": "high"
  }
}

Response 200 OK:
{
  "detections": [
    {
      "track_id": 1,
      "class": "person",
      "confidence": 0.92,
      "bbox": {
        "x1": 100, "y1": 200, "x2": 300, "y2": 600
      },
      "center": { "x": 200, "y": 400 },
      "velocity": { "dx": 5.2, "dy": -1.1 },
      "appearance": {
        "color_dominant": [100, 150, 200],
        "embedding": [...]
      }
    }
  ],
  "frame_count": 12850,
  "inference_time_ms": 35.5
}
```

#### 2. **Analytics Query (Fall Detection)**
```http
POST /analytics/fall-detection HTTP/1.1
Host: localhost:18000
Content-Type: application/json

{
  "track_id": 1,
  "person_id": "person_123",
  "bboxes": [
    { "x1": 100, "y1": 100, "x2": 300, "y2": 600 },
    { "x1": 102, "y1": 150, "x2": 310, "y2": 550 },
    { "x1": 105, "y1": 200, "x2": 305, "y2": 500 }
  ],
  "timestamps": [1000, 1033, 1066],
  "frame_rate": 30
}

Response 200 OK:
{
  "is_falling": true,
  "confidence": 0.87,
  "frame_index": 2,
  "fall_type": "sudden_drop",
  "recommendation": "ALERT"
}
```

#### 3. **Zone Validation**
```http
POST /analytics/zone-check HTTP/1.1
Host: localhost:18000
Content-Type: application/json

{
  "camera_id": "camera_1",
  "person_id": "person_123",
  "track_id": 1,
  "position": { "x": 200, "y": 400 },
  "timestamp": 1705574400000
}

Response 200 OK:
{
  "zones": [
    {
      "zone_id": "zone_restricted_1",
      "zone_name": "Medical Storage",
      "violation": true,
      "severity": "high",
      "action": "SEND_ALERT"
    }
  ]
}
```

#### 4. **Person Query/Update**
```http
GET /person/person_123 HTTP/1.1
Host: localhost:18000

Response 200 OK:
{
  "id": "person_123",
  "name": "Nguyễn Văn A",
  "age": 75,
  "gender": "M",
  "face_embedding": [...],
  "metadata": {
    "room": "301",
    "emergency_contact": "0909123456"
  },
  "created_at": "2025-12-01T10:30:00Z",
  "updated_at": "2026-01-18T12:00:00Z"
}
```

#### 5. **Health & Status**
```http
GET /health HTTP/1.1
Host: localhost:18000

Response 200 OK:
{
  "status": "healthy",
  "model_loaded": true,
  "database": "connected",
  "gpu_memory_mb": 2048,
  "avg_inference_ms": 32.5,
  "uptime_seconds": 86400
}
```

### Data Flow Diagram
```
Plugin (C++)                    Service (Python)             Database
    │                              │                            │
    ├─ Capture Frame ─────────────>│                            │
    │                              ├─ Preprocess Frame          │
    │                              ├─ Run YOLOv8 Inference      │
    │                              ├─ Post-processing           │
    │ <─ Detection Results ────────┤                            │
    │                              │                            │
    ├─ Analyze Movement ───────────>│ Run Fall Detection         │
    │                              ├─ Query Person Profile ────>│
    │ <─ Fall Alert ───────────────┤                   Profile <┤
    │                              │                            │
    ├─ Check Zones ───────────────>│ Validate Zone <──────────┤
    │                              │                  Zone Def  │
    │ <─ Zone Violation ───────────┤                            │
    │                              │                            │
    ├─ Generate NX Event ─ Done    ├─ Log Event ──────────────>│
    │                              ├─ Send Email              │
    │                              ├─ Update Person History ──>│
    │                              │                            │
```

### Error Handling & Retry Strategy

```python
# Plugin Side (object_detector.cpp)
1. Request → Service
2. Timeout (2.5s) → Retry with backoff
3. 5xx Error → Exponential backoff: 100ms, 200ms, 400ms, ...
4. Connection Failed → Circuit breaker + fallback
5. After 3 retries → Log error, skip frame, continue

# Service Side (service.py)
1. Database error → Log, return 503 Service Unavailable
2. Model inference error → Return partial results or cache
3. External API timeout → Graceful degradation
4. Out of memory → Clear cache, restart model
```

---

## 💾 Data Storage & API Strategy

### Database Schema

#### PostgreSQL (Recommended for structured data)
```sql
-- People/Person Profiles
CREATE TABLE persons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    age INT,
    gender CHAR(1),
    embedding BYTEA,  -- Face embedding vector
    face_image BYTEA,  -- Reference face image
    room_number VARCHAR(50),
    emergency_contact VARCHAR(20),
    status ENUM('active', 'inactive', 'visiting'),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    INDEX idx_name (name),
    INDEX idx_status (status)
);

-- Face Embeddings (optimized for search)
CREATE TABLE face_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    person_id UUID REFERENCES persons(id),
    embedding VECTOR(512),  -- Using pgvector for similarity search
    confidence FLOAT,
    capture_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    camera_id VARCHAR(50),
    INDEX idx_person (person_id)
);

-- Events Log
CREATE TABLE events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type ENUM('detection', 'fall', 'zone_violation', 'counting'),
    person_id UUID REFERENCES persons(id),
    track_id INT,
    camera_id VARCHAR(50),
    severity ENUM('low', 'medium', 'high', 'critical'),
    description TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed BOOLEAN DEFAULT FALSE,
    INDEX idx_person (person_id),
    INDEX idx_camera (camera_id),
    INDEX idx_created (created_at),
    INDEX idx_severity (severity)
);

-- Zones (restricted areas)
CREATE TABLE zones (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id VARCHAR(50) NOT NULL,
    zone_name VARCHAR(100) NOT NULL,
    zone_type ENUM('forbidden', 'danger', 'normal'),
    polygon POLYGON,  -- Coordinates of zone boundary
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    INDEX idx_camera (camera_id)
);

-- Tracking Data (recent/temporary)
CREATE TABLE tracking_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    person_id UUID REFERENCES persons(id),
    track_id INT,
    camera_id VARCHAR(50),
    position POINT,
    timestamp TIMESTAMP,
    velocity POINT,
    bbox RECT,
    INDEX idx_person (person_id),
    INDEX idx_camera (camera_id),
    INDEX idx_timestamp (timestamp)
);

-- Alerts/Notifications Sent
CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id UUID REFERENCES events(id),
    person_id UUID REFERENCES persons(id),
    alert_type ENUM('email', 'sms', 'push'),
    recipient VARCHAR(255),
    status ENUM('pending', 'sent', 'failed'),
    sent_at TIMESTAMP,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_person (person_id),
    INDEX idx_status (status)
);
```

#### MongoDB (Alternative for flexible events)
```javascript
// persons collection
{
  _id: ObjectId(),
  name: "Nguyễn Văn A",
  age: 75,
  gender: "M",
  room: "301",
  face_embedding: [0.1, 0.2, ...], // 512-dim vector
  face_image: Binary(),
  status: "active",
  emergency_contact: "0909123456",
  created_at: ISODate(),
  updated_at: ISODate(),
  metadata: {}
}

// events collection
{
  _id: ObjectId(),
  event_type: "fall",
  person_id: ObjectId(),
  camera_id: "camera_1",
  timestamp: ISODate(),
  severity: "critical",
  description: "Fall detected at hallway floor 2",
  metadata: {
    track_id: 1,
    bbox: { x1: 100, y1: 200, x2: 300, y2: 600 },
    confidence: 0.87
  }
}
```

### Third-Party Services & APIs

#### 1. **Face Recognition**
```
Option A: Microsoft Azure Face API
├─ Pros: Accurate, GDPR-compliant, integrates with Azure infrastructure
├─ Cons: $1-10 per 1000 requests
├─ Use: Person identification, age/gender estimation
└─ Endpoint: https://[region].face.cognitive.microsoft.com/

Option B: AWS Rekognition
├─ Pros: Good accuracy, integrated with AWS, supports video analysis
├─ Cons: $0.0001 per image for face detection
├─ Use: Person detection, face matching
└─ Endpoint: AWS SDK only

Option C: Google Cloud Vision
├─ Pros: Excellent accuracy, good for batch processing
├─ Cons: Similar pricing to Azure
├─ Use: General computer vision tasks
└─ Endpoint: https://vision.googleapis.com/

RECOMMENDATION: Azure Face API
- Better for elderly care (better age/gender accuracy)
- GDPR compliance important for medical data
- Can detect face mask, glasses, hair (useful for tracking)
```

**Integration Code Pattern:**
```python
# azure_face_service.py
from azure.cognitiveservices.vision.face import FaceClient

class AzureFaceService:
    def __init__(self, endpoint: str, api_key: str):
        self.client = FaceClient(endpoint, CognitiveServicesCredentials(api_key))
    
    def get_face_id(self, image_data: bytes) -> str:
        """Detect face and get unique face_id"""
        result = self.client.face.detect_in_stream(
            image_stream=image_data,
            return_face_id=True,
            return_face_attributes=['age', 'gender']
        )
        return result[0]['faceId']
    
    def verify_faces(self, face_id_1: str, face_id_2: str) -> float:
        """Compare two faces, return confidence score (0-1)"""
        result = self.client.face.verify_face_to_face(face_id_1, face_id_2)
        return result.confidence
    
    def find_similar(self, face_id: str, face_list_id: str) -> List[dict]:
        """Find similar faces in a group"""
        results = self.client.face.find_similar(
            face_id, large_face_list_id=face_list_id
        )
        return results
```

#### 2. **Email Notification Service**
```
Option A: SendGrid
├─ Pros: Reliable, good templates, good API
├─ Cons: $19.95/month minimum
├─ Features: HTML templates, scheduling, bounce handling
└─ Rate: up to 100 emails/second

Option B: AWS SES (Simple Email Service)
├─ Pros: Cheap ($0.10 per 1000 emails), AWS integration
├─ Cons: Lower rate limits initially
├─ Features: Bulk sending, custom headers
└─ Rate: 14 emails/second initially

Option C: Mailgun
├─ Pros: Good balance, free tier available
├─ Cons: Some limitations on free tier
├─ Features: Email validation, webhooks
└─ Rate: 600 requests/minute

RECOMMENDATION: AWS SES (cost-effective) or SendGrid (ease-of-use)
```

**Integration Code Pattern:**
```python
# email_service.py
import sendgrid
from sendgrid.helpers.mail import Mail, Email, Content

class EmailAlertService:
    def __init__(self, api_key: str):
        self.sg = sendgrid.SendGridAPIClient(api_key)
    
    def send_fall_alert(
        self,
        recipient: str,
        person_name: str,
        camera: str,
        timestamp: str
    ):
        """Send fall detection alert"""
        message = Mail(
            from_email='alerts@elderly-care.com',
            to_emails=recipient,
            subject='⚠️ Fall Detection Alert',
            html_content=f"""
            <h2>Fall Detected!</h2>
            <p><strong>Person:</strong> {person_name}</p>
            <p><strong>Location:</strong> {camera}</p>
            <p><strong>Time:</strong> {timestamp}</p>
            <p><strong>Action Required:</strong> Check immediately</p>
            """
        )
        response = self.sg.send(message)
        return response.status_code == 202
```

#### 3. **Data Analytics & Monitoring**
```
Option A: ELK Stack (Elasticsearch, Logstash, Kibana)
├─ Pros: Open source, powerful analytics, beautiful dashboards
├─ Cons: Requires infrastructure management
├─ Use: Event analytics, trend analysis, real-time dashboards
└─ Cost: Free (self-hosted)

Option B: DataDog
├─ Pros: Cloud-based, excellent dashboards, APM monitoring
├─ Cons: Expensive ($15+/host/month)
├─ Use: System monitoring, performance analysis, alerting
└─ Cost: Pay-as-you-go

Option C: Grafana + Prometheus
├─ Pros: Popular, open source, good for metrics
├─ Cons: Different focus (metrics vs logs)
├─ Use: System metrics, performance monitoring
└─ Cost: Free (self-hosted)

RECOMMENDATION: Grafana + Prometheus for metrics + ELK for events
```

#### 4. **SMS/Push Notifications (Optional)**
```
Option A: Twilio
├─ Pros: Reliable, supports SMS and push
├─ Cons: More expensive
├─ Use: Critical alerts via SMS
└─ Rate: $0.0075 per SMS in Vietnam

Option B: Firebase Cloud Messaging (FCM)
├─ Pros: Free for basic usage, good integration
├─ Cons: Requires mobile app
├─ Use: Push notifications to staff app
└─ Cost: Free

RECOMMENDATION: Twilio for SMS, FCM for mobile push
```

### Data Flow to External Services
```
Service (Python)
    │
    ├─ Event Generated
    │   │
    │   ├─ Save to DB ──────────> PostgreSQL
    │   │
    │   ├─ Extract Face ─────────> Azure Face API
    │   │                         (Get person ID)
    │   │
    │   ├─ Check Zone ────────────> PostgreSQL
    │   │                         (Validate)
    │   │
    │   ├─ Generate Alert
    │   │   │
    │   │   ├─ Email ────────────> SendGrid/AWS SES
    │   │   │
    │   │   ├─ SMS (Critical) ────> Twilio
    │   │   │
    │   │   └─ Push (App) ────────> Firebase FCM
    │   │
    │   └─ Log Metrics ──────────> ELK Stack / Datadog
    │
    └─ Every 5min: Analytics
        └─ Generate Reports ─────> Elasticsearch
```

---

## 🗂️ Implementation Roadmap

### Timeline Overview
```
Week 1-2: Foundation Setup
├─ TEAM A: Database design & analytics engine core
├─ TEAM B: Plugin refactoring & health monitoring
└─ Both: Testing infrastructure, CI/CD setup

Week 3-4: Integration
├─ TEAM A: Face API integration, email service
├─ TEAM B: Fall detection & zone management in plugin
└─ Both: Cross-team testing & API refinement

Week 5-6: Enhancement
├─ TEAM A: Advanced analytics, performance optimization
├─ TEAM B: Multi-camera support, stability improvements
└─ Both: Load testing, stress testing, production hardening

Week 7-8: Deployment & Iteration
├─ TEAM A: Dashboard/reporting, monitoring
├─ TEAM B: Docker containerization, deployment automation
└─ Both: Production deployment, monitoring, iterative improvements
```

### Phase 1: Foundation (Week 1-2)

**TEAM A Milestones:**
- [ ] Database schema finalized
- [ ] Database abstraction layer (ORM/client)
- [ ] Basic CRUD endpoints
- [ ] Fall detection algorithm (MVP)
- [ ] Unit tests for analytics

**TEAM B Milestones:**
- [ ] Refactored object_detector with error handling
- [ ] Health monitoring endpoints
- [ ] Configuration system implemented
- [ ] Comprehensive logging added
- [ ] Unit tests for detector

**Integration Points:**
- [ ] Define API contract
- [ ] Test basic /detect endpoint
- [ ] Error codes standardization

### Phase 2: Integration (Week 3-4)

**TEAM A Milestones:**
- [ ] Azure Face API integrated
- [ ] Email service configured
- [ ] Person identification working
- [ ] Zone validation logic
- [ ] Integration tests passing

**TEAM B Milestones:**
- [ ] Fall detection in plugin complete
- [ ] Zone check integration
- [ ] Person profile enrichment
- [ ] Event metadata generation
- [ ] Integration tests passing

**Integration Points:**
- [ ] End-to-end fall detection test
- [ ] Zone violation alert test
- [ ] Plugin ↔ Service communication verified

### Phase 3: Enhancement (Week 5-6)

**TEAM A Milestones:**
- [ ] Multi-camera coordination
- [ ] Performance optimization
- [ ] Caching strategy implemented
- [ ] Analytics dashboard started
- [ ] Load testing completed

**TEAM B Milestones:**
- [ ] Resilience patterns implemented
- [ ] Performance profiling done
- [ ] Memory optimization complete
- [ ] Docker image created
- [ ] Deployment automation setup

**Integration Points:**
- [ ] Performance benchmarks met
- [ ] Stress testing passed
- [ ] Production readiness checklist

### Phase 4: Deployment (Week 7-8)

**TEAM A Milestones:**
- [ ] Analytics dashboard complete
- [ ] Monitoring/alerting setup
- [ ] Database backups configured
- [ ] Documentation complete
- [ ] Handoff to ops team

**TEAM B Milestones:**
- [ ] Plugin production-ready
- [ ] Deployment guide complete
- [ ] Monitoring integration done
- [ ] Logging aggregation setup
- [ ] Incident response procedures

**Integration Points:**
- [ ] Production deployment complete
- [ ] Monitoring dashboards live
- [ ] Alerting system operational
- [ ] Team handoff successful

---

## 🚀 Development Workflow

### Git Workflow

```
Main Repo Structure:
├── main (stable, production)
├── develop (integration, staging)
├── feature/team-a/* (service features)
├── feature/team-b/* (plugin features)
└── hotfix/* (critical fixes)

Branching Strategy:
1. Create feature branch from develop
   git checkout -b feature/team-a/fall-detection develop

2. Regular commits with meaningful messages
   git commit -m "feat(fall-detection): Add pose estimation algorithm"

3. Push and create Pull Request
   git push origin feature/team-a/fall-detection

4. Code Review (other team + project lead)
   - Automated tests must pass
   - Code review approval required
   - Integration verified

5. Merge to develop
   - Squash commits for clarity
   - Delete feature branch

6. After team coordination, merge develop → main
```

### Code Review Checklist

**For TEAM A (Service) PRs:**
```
□ Unit tests added/passing
□ Integration tests added/passing
□ Database migrations provided
□ API documentation updated
□ Error handling implemented
□ Logging added at appropriate levels
□ No security issues (SQL injection, auth, etc.)
□ Performance considerations addressed
□ Backward compatible or migration provided
```

**For TEAM B (Plugin) PRs:**
```
□ Unit tests added/passing
□ Memory leak checks done
□ Crash scenarios handled
□ HTTP client retry logic tested
□ Timeout handling verified
□ Error messages are diagnostic
□ Plugin stability maintained
□ No blocking operations
□ Resource cleanup verified
```

### Daily Communication

**Daily Standup (15 min)**
```
Format: Sync Meeting (Video/Slack)
Time: 09:30 AM
Attendees: Both teams + Project Lead

Each Person:
1. What did I accomplish yesterday?
2. What am I working on today?
3. Any blockers?

TEAM A Focus: API stability, DB performance
TEAM B Focus: Plugin reliability, HTTP communication
```

**Weekly Sync (1 hour)**
```
1. Review completed work (20 min)
   - Demo new features
   - Show test results
   - Integration status

2. Identify blockers (15 min)
   - API contract issues
   - Performance bottlenecks
   - Testing coverage gaps

3. Plan next week (25 min)
   - Adjust priorities
   - Dependency management
   - Resource allocation
```

### Testing Strategy

**TEAM A Testing**
```python
# unit_tests/ - Test individual functions
tests/
├── test_fall_detector.py
├── test_zone_validator.py
├── test_person_counter.py
├── test_database.py
└── test_external_apis.py

# integration_tests/ - Test Service endpoints
tests/
├── test_detect_endpoint.py
├── test_fall_detection_flow.py
├── test_zone_check_flow.py
├── test_person_management.py
└── test_email_alerts.py

# load_tests/ - Performance testing
tests/
└── test_load_service.py
    └── Simulate 10+ cameras @ 30fps
```

**TEAM B Testing**
```cpp
// unit_tests/ - Test components
tests/
├── test_object_detector.cpp
├── test_http_client.cpp
├── test_zone_validator.cpp
└── test_config_manager.cpp

// integration_tests/ - Test plugin lifecycle
tests/
├── test_plugin_startup.cpp
├── test_frame_processing.cpp
├── test_service_communication.cpp
└── test_event_generation.cpp

// stress_tests/ - Long-running tests
tests/
├── test_memory_leak.cpp
├── test_high_frame_rate.cpp
└── test_error_recovery.cpp
```

### Version Management

```
Semantic Versioning: MAJOR.MINOR.PATCH

Service Version: service_version.txt
Plugin Version: plugin_version.txt

Example Release:
v0.1.0 (Initial MVP)
├─ Person detection
├─ Basic counting
└─ Fall detection alert

v0.2.0 (Face Recognition)
├─ Azure Face API integration
├─ Person profile management
└─ Face matching

v1.0.0 (Production Ready)
├─ Zone management
├─ Multi-camera support
├─ Full analytics dashboard
└─ SLA requirements met
```

---

## 📊 Success Metrics

### Performance Targets

**Service (Python)**
```
Inference Time: < 50ms per frame
  - YOLOv8n: ~35ms
  - Preprocessing: ~10ms
  - Post-processing: ~5ms

API Response Time: < 100ms @ 30fps
  - P95: < 80ms
  - P99: < 150ms

Database Queries: < 50ms
  - Simple lookups: < 10ms
  - Complex queries: < 50ms

Memory Usage: < 4GB
  - Model: ~2.5GB
  - Cache: ~1GB
  - Headroom: ~0.5GB
```

**Plugin (C++)**
```
HTTP Request Latency: < 2.5s timeout
  - Healthy: 100-500ms
  - Degraded: 500-2000ms

Plugin Processing: < 50ms per frame
  - Event generation: < 10ms
  - NX communication: < 30ms

Memory Usage: < 500MB per camera
  - Tracking state: ~200MB
  - Buffer: ~200MB
  - Misc: ~100MB

CPU Usage: < 30% per camera @ 30fps
```

### Reliability Targets

```
Service Uptime: > 99.5% (43.2 minutes/month downtime)
Plugin Stability: > 99% (no crash/restart)
Detection Accuracy: > 95% for people detection
False Positive Rate: < 5% for events
Alert Delivery: > 99% within 5 seconds
```

### User Experience Targets

```
Fall Detection Response: < 5 seconds from event to alert
Zone Violation Alert: < 2 seconds from entry to notification
Person Recognition: > 90% accuracy for known persons
Dashboard Load Time: < 2 seconds
Report Generation: < 30 seconds for monthly reports
```

---

## 📚 Documentation Plan

**TEAM A to Create:**
- [ ] API Documentation (OpenAPI/Swagger)
- [ ] Database Schema Guide
- [ ] Analytics Algorithm Documentation
- [ ] Integration Guide (Face API, Email, etc.)
- [ ] Troubleshooting Guide
- [ ] Performance Tuning Guide

**TEAM B to Create:**
- [ ] Plugin Development Guide
- [ ] Build Instructions (Windows, Linux)
- [ ] Configuration Reference
- [ ] NX VMS Integration Guide
- [ ] Troubleshooting Guide
- [ ] Performance Profiling Guide

**Shared Documentation:**
- [ ] System Architecture Overview (this document)
- [ ] API Contract & Examples
- [ ] Deployment Guide
- [ ] Operational Handbook
- [ ] Release Notes
- [ ] Contributing Guidelines

---

## 🎯 Conclusion

This architecture provides:

✅ **Clear separation of concerns** - Service team handles logic, Plugin team handles stability
✅ **Well-defined interfaces** - HTTP API contract is the single source of truth
✅ **Scalability** - Can add more cameras, services, or processing nodes
✅ **Reliability** - Error handling, retry logic, and graceful degradation
✅ **Professional workflow** - Git flow, code review, testing, and documentation
✅ **Data security** - Database design supports GDPR, audit logs, access control
✅ **Monitoring** - Built-in health checks, metrics, and alerting

The two-team structure enables parallel development while the HTTP API contract ensures seamless integration. Regular syncs and clear communication channels will keep both teams aligned toward the project goals.

---

**Document Version:** 1.0  
**Last Updated:** January 18, 2026  
**Status:** Ready for Implementation  
