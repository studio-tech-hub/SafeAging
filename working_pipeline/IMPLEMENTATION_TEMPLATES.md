# 🛠️ Implementation Templates & Code Examples

This document provides practical code templates that TEAM A and TEAM B can use as starting points for their development.

---

## TEAM A: Service Layer Templates

### 1. Database Models (models.py)

```python
# models.py - SQLAlchemy models for database
from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, Enum, JSON, LargeBinary
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import UUID, BYTEA, VECTOR
from datetime import datetime
import uuid
import enum

Base = declarative_base()

class PersonStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    VISITING = "visiting"

class EventType(str, enum.Enum):
    DETECTION = "detection"
    FALL = "fall"
    ZONE_VIOLATION = "zone_violation"
    COUNTING = "counting"

class Severity(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Person(Base):
    __tablename__ = "persons"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    age = Column(Integer)
    gender = Column(String(1))
    room_number = Column(String(50))
    emergency_contact = Column(String(20))
    status = Column(Enum(PersonStatus), default=PersonStatus.ACTIVE, index=True)
    
    # Face data
    face_image = Column(BYTEA)  # PNG/JPG image
    face_embedding = Column(BYTEA)  # Serialized numpy array
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON, default={})
    
    def __repr__(self):
        return f"<Person(id={self.id}, name={self.name})>"

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    person_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    embedding = Column(VECTOR(512))  # 512-dimensional face embedding
    confidence = Column(Float)
    camera_id = Column(String(50), index=True)
    capture_date = Column(DateTime, default=datetime.utcnow, index=True)

class Event(Base):
    __tablename__ = "events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type = Column(Enum(EventType), index=True)
    person_id = Column(UUID(as_uuid=True), index=True)
    track_id = Column(Integer)
    camera_id = Column(String(50), index=True)
    severity = Column(Enum(Severity), index=True)
    description = Column(String(500))
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    processed = Column(Boolean, default=False)

class Zone(Base):
    __tablename__ = "zones"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    camera_id = Column(String(50), nullable=False, index=True)
    zone_name = Column(String(100))
    zone_type = Column(String(20))  # 'forbidden', 'danger', 'normal'
    polygon = Column(JSON)  # List of (x, y) points
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)

class TrackingData(Base):
    __tablename__ = "tracking_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    person_id = Column(UUID(as_uuid=True), index=True)
    track_id = Column(Integer, index=True)
    camera_id = Column(String(50), index=True)
    position = Column(JSON)  # {"x": float, "y": float}
    velocity = Column(JSON)  # {"dx": float, "dy": float}
    bbox = Column(JSON)  # {"x1": int, "y1": int, "x2": int, "y2": int}
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_id = Column(UUID(as_uuid=True))
    person_id = Column(UUID(as_uuid=True), index=True)
    alert_type = Column(String(20))  # 'email', 'sms', 'push'
    recipient = Column(String(255))
    status = Column(String(20))  # 'pending', 'sent', 'failed'
    sent_at = Column(DateTime)
    error_message = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
```

### 2. Analytics Engine Core (analytics/fall_detector.py)

```python
# analytics/fall_detector.py - Fall detection algorithm
import numpy as np
from typing import List, Dict, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)

class FallDetector:
    """
    Detects falls based on bounding box movement patterns.
    
    Algorithm:
    1. Track vertical position change (drop in center Y coordinate)
    2. Track horizontal spread change (width vs height ratio)
    3. Detect sudden changes in movement velocity
    4. Confirm with time window analysis
    """
    
    def __init__(
        self,
        history_frames: int = 10,
        fall_threshold: float = 0.6,
        velocity_threshold: float = 50.0
    ):
        self.history_frames = history_frames
        self.fall_threshold = fall_threshold
        self.velocity_threshold = velocity_threshold
        self.bbox_history = deque(maxlen=history_frames)
        self.velocity_history = deque(maxlen=history_frames)
    
    def add_frame(self, bbox: Dict[str, int], timestamp: float) -> None:
        """
        Add a new frame's bounding box to history.
        
        Args:
            bbox: {"x1": int, "y1": int, "x2": int, "y2": int}
            timestamp: Frame timestamp in ms
        """
        self.bbox_history.append((bbox, timestamp))
        
        # Calculate velocity if we have 2 frames
        if len(self.bbox_history) >= 2:
            prev_bbox, prev_time = self.bbox_history[-2]
            curr_bbox, curr_time = self.bbox_history[-1]
            
            # Center of bounding boxes
            prev_center = self._get_center(prev_bbox)
            curr_center = self._get_center(curr_bbox)
            
            # Time delta in seconds
            dt = (curr_time - prev_time) / 1000.0 if curr_time > prev_time else 1.0
            
            # Velocity in pixels/second
            dx = (curr_center[0] - prev_center[0]) / dt if dt > 0 else 0
            dy = (curr_center[1] - prev_center[1]) / dt if dt > 0 else 0
            
            velocity = np.sqrt(dx**2 + dy**2)
            self.velocity_history.append(velocity)
    
    def detect_fall(self) -> Tuple[bool, float, str]:
        """
        Detect if a fall has occurred.
        
        Returns:
            (is_falling, confidence, reason)
        """
        if len(self.bbox_history) < 3:
            return False, 0.0, "Not enough frames"
        
        # Calculate metrics
        aspect_ratio_change = self._calculate_aspect_ratio_change()
        height_change = self._calculate_height_change()
        velocity_spike = self._detect_velocity_spike()
        y_drop = self._calculate_y_drop()
        
        # Scoring
        score = 0.0
        reasons = []
        
        # Check aspect ratio change (person becomes more horizontal)
        if aspect_ratio_change > 2.0:  # Height/Width ratio decreased significantly
            score += 0.4
            reasons.append(f"aspect_ratio_change={aspect_ratio_change:.2f}")
        
        # Check sudden drop in position (downward movement)
        if y_drop > 100:  # Dropped more than 100 pixels
            score += 0.3
            reasons.append(f"y_drop={y_drop:.0f}px")
        
        # Check velocity spike (sudden movement)
        if velocity_spike > self.velocity_threshold:
            score += 0.2
            reasons.append(f"velocity_spike={velocity_spike:.0f}px/s")
        
        # Check height reduction
        if height_change < 0.5:  # Height reduced to 50% or less
            score += 0.1
            reasons.append(f"height_change={height_change:.2f}x")
        
        is_fall = score >= self.fall_threshold
        reason = ", ".join(reasons) if reasons else "No fall indicators"
        
        return is_fall, score, reason
    
    def reset(self) -> None:
        """Clear history for new person/track."""
        self.bbox_history.clear()
        self.velocity_history.clear()
    
    # Helper methods
    
    def _get_center(self, bbox: Dict[str, int]) -> Tuple[float, float]:
        """Get center point of bbox."""
        x_center = (bbox['x1'] + bbox['x2']) / 2.0
        y_center = (bbox['y1'] + bbox['y2']) / 2.0
        return x_center, y_center
    
    def _get_width(self, bbox: Dict[str, int]) -> float:
        """Get width of bbox."""
        return bbox['x2'] - bbox['x1']
    
    def _get_height(self, bbox: Dict[str, int]) -> float:
        """Get height of bbox."""
        return bbox['y2'] - bbox['y1']
    
    def _calculate_aspect_ratio_change(self) -> float:
        """
        Calculate how much the aspect ratio has changed.
        Returns: ratio of height to width changes
        """
        if len(self.bbox_history) < 2:
            return 1.0
        
        first_bbox = self.bbox_history[0][0]
        last_bbox = self.bbox_history[-1][0]
        
        first_h = self._get_height(first_bbox)
        first_w = self._get_width(first_bbox)
        last_h = self._get_height(last_bbox)
        last_w = self._get_width(last_bbox)
        
        first_ratio = first_h / first_w if first_w > 0 else 1.0
        last_ratio = last_h / last_w if last_w > 0 else 1.0
        
        return first_ratio / last_ratio if last_ratio > 0 else 1.0
    
    def _calculate_height_change(self) -> float:
        """Calculate how much height has changed (ratio)."""
        if len(self.bbox_history) < 2:
            return 1.0
        
        first_bbox = self.bbox_history[0][0]
        last_bbox = self.bbox_history[-1][0]
        
        first_h = self._get_height(first_bbox)
        last_h = self._get_height(last_bbox)
        
        return last_h / first_h if first_h > 0 else 1.0
    
    def _calculate_y_drop(self) -> float:
        """Calculate how much the person moved down (in pixels)."""
        if len(self.bbox_history) < 2:
            return 0.0
        
        first_bbox = self.bbox_history[0][0]
        last_bbox = self.bbox_history[-1][0]
        
        first_center = self._get_center(first_bbox)
        last_center = self._get_center(last_bbox)
        
        # Positive = moved down, Negative = moved up
        return last_center[1] - first_center[1]
    
    def _detect_velocity_spike(self) -> float:
        """Detect if there's a sudden velocity increase."""
        if len(self.velocity_history) < 3:
            return 0.0
        
        velocities = list(self.velocity_history)
        
        # Compare last velocity to average of previous velocities
        if len(velocities) > 1:
            avg_prev = np.mean(velocities[:-1])
            last_vel = velocities[-1]
            
            # Spike if last velocity is significantly higher
            spike = max(0, last_vel - avg_prev)
            return spike
        
        return 0.0
```

### 3. Zone Validator (analytics/zone_validator.py)

```python
# analytics/zone_validator.py - Zone validation logic
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.
    
    Args:
        point: (x, y) coordinate
        polygon: List of (x, y) vertices
    
    Returns:
        True if point is inside polygon
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

class ZoneValidator:
    """Validates if a position is in forbidden/danger zones."""
    
    def __init__(self):
        self.zones = {}  # zone_id -> zone_data
    
    def add_zone(
        self,
        zone_id: str,
        camera_id: str,
        zone_name: str,
        zone_type: str,  # 'forbidden', 'danger'
        polygon: List[Tuple[float, float]]
    ) -> None:
        """Add a zone definition."""
        self.zones[zone_id] = {
            'camera_id': camera_id,
            'zone_name': zone_name,
            'zone_type': zone_type,
            'polygon': polygon
        }
    
    def check_position(
        self,
        camera_id: str,
        position: Dict[str, float],
        person_id: str = None
    ) -> List[Dict]:
        """
        Check if a position violates any zones.
        
        Args:
            camera_id: Camera identifier
            position: {"x": float, "y": float}
            person_id: Optional person ID for logging
        
        Returns:
            List of zone violations
        """
        violations = []
        x, y = position['x'], position['y']
        point = (x, y)
        
        for zone_id, zone_data in self.zones.items():
            if zone_data['camera_id'] != camera_id:
                continue  # Zone is for different camera
            
            if point_in_polygon(point, zone_data['polygon']):
                violations.append({
                    'zone_id': zone_id,
                    'zone_name': zone_data['zone_name'],
                    'zone_type': zone_data['zone_type'],
                    'violation': True,
                    'severity': 'high' if zone_data['zone_type'] == 'forbidden' else 'medium',
                    'action': 'SEND_ALERT'
                })
                
                if person_id:
                    logger.warning(
                        f"Zone violation: person={person_id}, "
                        f"zone={zone_data['zone_name']}, "
                        f"position=({x}, {y})"
                    )
        
        return violations
```

### 4. Enhanced Service Endpoints (service_v2.py - additions)

```python
# Add these to service.py for analytics endpoints

@app.post("/analytics/fall-detection")
async def detect_fall(request_data: dict) -> dict:
    """
    Detect fall from bounding box sequence.
    
    Request:
    {
        "track_id": int,
        "person_id": str,
        "bboxes": [{"x1": int, "y1": int, "x2": int, "y2": int}, ...],
        "timestamps": [float, ...],
        "frame_rate": int
    }
    """
    try:
        track_id = request_data['track_id']
        bboxes = request_data['bboxes']
        timestamps = request_data['timestamps']
        
        detector = FallDetector()
        
        for bbox, timestamp in zip(bboxes, timestamps):
            detector.add_frame(bbox, timestamp)
        
        is_falling, confidence, reason = detector.detect_fall()
        
        return {
            "is_falling": is_falling,
            "confidence": confidence,
            "frame_index": len(bboxes) - 1,
            "fall_type": "sudden_drop" if is_falling else None,
            "reason": reason,
            "recommendation": "ALERT" if is_falling else "NORMAL"
        }
    except Exception as e:
        logger.error(f"Fall detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analytics/zone-check")
async def check_zone(request_data: dict) -> dict:
    """
    Check if person is in any restricted zones.
    
    Request:
    {
        "camera_id": str,
        "person_id": str,
        "position": {"x": float, "y": float},
        "timestamp": float
    }
    """
    try:
        camera_id = request_data['camera_id']
        position = request_data['position']
        person_id = request_data.get('person_id')
        
        # Load zones from database
        zones = db.get_zones_for_camera(camera_id)
        validator = ZoneValidator()
        for zone in zones:
            validator.add_zone(
                zone['id'],
                zone['camera_id'],
                zone['zone_name'],
                zone['zone_type'],
                zone['polygon']
            )
        
        violations = validator.check_position(camera_id, position, person_id)
        
        return {"zones": violations}
    except Exception as e:
        logger.error(f"Zone check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/person/{person_id}")
async def get_person(person_id: str) -> dict:
    """Get person profile."""
    try:
        person = db.get_person(person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        
        return {
            "id": person.id,
            "name": person.name,
            "age": person.age,
            "gender": person.gender,
            "room": person.room_number,
            "emergency_contact": person.emergency_contact,
            "status": person.status,
            "face_embedding": None,  # Don't send embedding over HTTP
            "created_at": person.created_at.isoformat(),
            "updated_at": person.updated_at.isoformat()
        }
    except Exception as e:
        logger.error(f"Get person error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/person")
async def create_person(request_data: dict) -> dict:
    """Create or update person profile."""
    try:
        person = db.create_or_update_person(
            name=request_data['name'],
            age=request_data.get('age'),
            gender=request_data.get('gender'),
            room_number=request_data.get('room'),
            emergency_contact=request_data.get('emergency_contact'),
            metadata=request_data.get('metadata', {})
        )
        
        return {"id": str(person.id), "status": "created"}
    except Exception as e:
        logger.error(f"Create person error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/events")
async def get_events(
    camera_id: str = None,
    person_id: str = None,
    event_type: str = None,
    limit: int = 100
) -> dict:
    """Get recent events with optional filtering."""
    try:
        events = db.get_events(
            camera_id=camera_id,
            person_id=person_id,
            event_type=event_type,
            limit=limit
        )
        
        return {
            "count": len(events),
            "events": [
                {
                    "id": str(e.id),
                    "type": e.event_type,
                    "person_id": str(e.person_id),
                    "camera_id": e.camera_id,
                    "severity": e.severity,
                    "description": e.description,
                    "created_at": e.created_at.isoformat()
                }
                for e in events
            ]
        }
    except Exception as e:
        logger.error(f"Get events error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

---

## TEAM B: Plugin Layer Templates

### 1. Enhanced HTTP Client (http_client.h)

```cpp
// http_client.h - Robust HTTP client with retries and timeouts
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <curl/curl.h>
#include <chrono>

class HttpClient {
public:
    struct Response {
        int status_code;
        std::string body;
        bool success;
        std::string error_message;
    };
    
    HttpClient(const std::string& service_url, int timeout_ms = 2500);
    ~HttpClient();
    
    // Main methods
    Response post_detect(const std::string& json_payload);
    Response post_fall_detection(const std::string& json_payload);
    Response post_zone_check(const std::string& json_payload);
    Response get_person(const std::string& person_id);
    Response get_health();
    
    // Configuration
    void set_timeout(int ms) { timeout_ms_ = ms; }
    void set_max_retries(int retries) { max_retries_ = retries; }
    void set_retry_backoff(int initial_ms) { retry_backoff_ms_ = initial_ms; }
    
private:
    std::string service_url_;
    int timeout_ms_;
    int max_retries_;
    int retry_backoff_ms_;
    CURL* curl_;
    
    Response post_with_retry(const std::string& endpoint, const std::string& payload);
    Response post_internal(const std::string& endpoint, const std::string& payload);
    
    static size_t write_callback(void* contents, size_t size, size_t nmemb, std::string* s);
};
```

### 2. Analytics Processor (analytics_processor.h)

```cpp
// analytics_processor.h - Process analytics on plugin side
#pragma once

#include <vector>
#include <deque>
#include <memory>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct BBox {
    float x1, y1, x2, y2;
    
    float get_width() const { return x2 - x1; }
    float get_height() const { return y2 - y1; }
    float get_center_x() const { return (x1 + x2) / 2.0f; }
    float get_center_y() const { return (y1 + y2) / 2.0f; }
};

class AnalyticsProcessor {
public:
    AnalyticsProcessor();
    
    // Fall detection
    void add_detection(int track_id, const BBox& bbox, uint64_t timestamp_ms);
    json detect_fall(int track_id);
    
    // Zone checking
    json check_zones(
        const std::string& camera_id,
        int track_id,
        const BBox& bbox,
        const std::string& person_id,
        uint64_t timestamp_ms
    );
    
    // Clear history for new person
    void reset_track(int track_id);
    
private:
    struct TrackHistory {
        std::deque<BBox> bboxes;
        std::deque<uint64_t> timestamps;
        static const size_t MAX_HISTORY = 10;
    };
    
    std::map<int, TrackHistory> track_histories_;
    
    // Helper methods for fall detection
    float calculate_aspect_ratio_change(const TrackHistory& history);
    float calculate_height_change(const TrackHistory& history);
    float calculate_y_drop(const TrackHistory& history);
    float detect_velocity_spike(const TrackHistory& history);
};
```

### 3. Configuration Manager (config_manager.h)

```cpp
// config_manager.h - Centralized configuration
#pragma once

#include <string>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class ConfigManager {
public:
    static ConfigManager& instance() {
        static ConfigManager instance;
        return instance;
    }
    
    // Load configuration from environment or file
    void load_from_env();
    void load_from_file(const std::string& config_path);
    
    // Getters
    std::string get_service_url() const;
    int get_service_timeout_ms() const;
    int get_detection_interval_frames() const;
    float get_confidence_threshold() const;
    int get_max_http_retries() const;
    bool is_fall_detection_enabled() const;
    bool is_zone_check_enabled() const;
    
    // Setters
    void set_service_url(const std::string& url);
    void set_service_timeout_ms(int timeout);
    void set_detection_interval_frames(int frames);
    
private:
    ConfigManager() = default;
    
    std::map<std::string, std::string> config_;
    
    std::string get_env_or_default(
        const std::string& key,
        const std::string& default_value
    );
};
```

### 4. Enhanced Device Agent (device_agent_v2.cpp - key additions)

```cpp
// device_agent.cpp - Key additions for analytics
// Add to existing DeviceAgent class

#include "config_manager.h"
#include "analytics_processor.h"
#include "http_client.h"

class DeviceAgent : public ConsumingDeviceAgent {
private:
    std::unique_ptr<HttpClient> http_client_;
    std::unique_ptr<AnalyticsProcessor> analytics_;
    ConfigManager& config_;
    
    // Tracking per detection
    struct TrackingInfo {
        std::string person_id;
        std::string person_name;
        uint64_t last_seen_ms;
        std::vector<BBox> recent_bboxes;
    };
    std::map<int, TrackingInfo> active_tracks_;
    
    // Fall detection state
    std::map<int, bool> fall_alert_sent_;
    
    // Zone violations (debounce)
    struct ZoneViolationState {
        std::string zone_id;
        uint64_t first_violation_ms;
        bool alert_sent;
    };
    std::map<std::string, ZoneViolationState> zone_violations_;

public:
    DeviceAgent(...) {
        http_client_ = std::make_unique<HttpClient>(
            config_.get_service_url(),
            config_.get_service_timeout_ms()
        );
        analytics_ = std::make_unique<AnalyticsProcessor>();
    }

    bool processDetections(
        const std::vector<Detection>& detections,
        uint64_t timestamp_ms
    ) {
        for (const auto& detection : detections) {
            int track_id = detection.track_id;
            
            // 1. Get person profile (if known)
            std::string person_id;
            std::string person_name;
            if (!detection.embedding.empty()) {
                auto response = http_client_->get_person(detection.person_id);
                if (response.success) {
                    json person_data = json::parse(response.body);
                    person_id = person_data["id"];
                    person_name = person_data["name"];
                }
            }
            
            // 2. Add to tracking
            TrackingInfo& track_info = active_tracks_[track_id];
            track_info.person_id = person_id;
            track_info.person_name = person_name;
            track_info.last_seen_ms = timestamp_ms;
            
            BBox bbox{
                detection.bbox.x1,
                detection.bbox.y1,
                detection.bbox.x2,
                detection.bbox.y2
            };
            track_info.recent_bboxes.push_back(bbox);
            
            // Keep only last 10 detections
            if (track_info.recent_bboxes.size() > 10) {
                track_info.recent_bboxes.erase(
                    track_info.recent_bboxes.begin()
                );
            }
            
            // 3. Check for fall (every 5 frames)
            if (config_.is_fall_detection_enabled() &&
                m_frameIndex % 5 == 0 &&
                track_info.recent_bboxes.size() >= 3) {
                
                analytics_->add_detection(track_id, bbox, timestamp_ms);
                json fall_result = analytics_->detect_fall(track_id);
                
                if (fall_result["is_falling"] && !fall_alert_sent_[track_id]) {
                    sendFallAlert(person_id, person_name, detection.bbox);
                    fall_alert_sent_[track_id] = true;
                }
            }
            
            // 4. Check zones
            if (config_.is_zone_check_enabled()) {
                auto zone_result = analytics_->check_zones(
                    m_cameraId,
                    track_id,
                    bbox,
                    person_id,
                    timestamp_ms
                );
                
                json zones = zone_result["zones"];
                if (!zones.empty()) {
                    sendZoneViolationAlert(
                        person_id,
                        person_name,
                        zones[0],
                        detection.bbox
                    );
                }
            }
        }
        
        return true;
    }

private:
    void sendFallAlert(
        const std::string& person_id,
        const std::string& person_name,
        const BBox& bbox
    ) {
        // Create NX event for fall
        auto event = generateEvent(
            "fall_detection",
            person_name.empty() ? "Unknown" : person_name,
            "critical"
        );
        
        // Log details
        std::cerr << "[FALL ALERT] Person: " << person_name
                  << " (" << person_id << ")"
                  << " Position: (" << bbox.x1 << "," << bbox.y1 << ")"
                  << std::endl;
    }
    
    void sendZoneViolationAlert(
        const std::string& person_id,
        const std::string& person_name,
        const json& zone,
        const BBox& bbox
    ) {
        // Create NX event for zone violation
        std::string zone_name = zone["zone_name"];
        auto event = generateEvent(
            "zone_violation",
            person_name.empty() ? "Unknown" : person_name +
            " in " + zone_name,
            "high"
        );
        
        std::cerr << "[ZONE VIOLATION] Person: " << person_name
                  << " in zone: " << zone_name
                  << std::endl;
    }
};
```

---

## Communication Examples

### Example 1: Fall Detection Flow

**Plugin → Service:**
```json
POST /detect
{
  "frame_id": "cam1_001234",
  "camera_id": "camera_1",
  "timestamp": 1705574400000,
  "frame_data": "iVBORw0KGgo...",
  "metadata": {"location": "hallway_floor2"}
}

Response:
{
  "detections": [
    {
      "track_id": 1,
      "class": "person",
      "bbox": {"x1": 100, "y1": 100, "x2": 250, "y2": 400}
    }
  ]
}
```

**Plugin → Service (Fall Check):**
```json
POST /analytics/fall-detection
{
  "track_id": 1,
  "bboxes": [
    {"x1": 100, "y1": 100, "x2": 250, "y2": 400},
    {"x1": 102, "y1": 110, "x2": 248, "y2": 390},
    {"x1": 105, "y1": 200, "x2": 245, "y2": 350}
  ],
  "timestamps": [1000, 1033, 1066]
}

Response:
{
  "is_falling": true,
  "confidence": 0.87,
  "recommendation": "ALERT"
}
```

**Service → Database:**
```python
db.create_event(
    event_type='fall',
    person_id=person.id,
    camera_id='camera_1',
    severity='critical',
    description='Fall detected at hallway floor 2'
)
```

**Service → Email:**
```python
email_service.send_fall_alert(
    recipient='nurse@facility.com',
    person_name='Nguyễn Văn A',
    camera='Hallway Floor 2',
    timestamp='2026-01-18 12:34:56'
)
```

---

This provides teams A and B with concrete starting points for their implementation while maintaining the agreed API contracts.

---

**Document Version:** 1.0  
**Created:** January 18, 2026
