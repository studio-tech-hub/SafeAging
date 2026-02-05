# 📋 TÓM TẮT - Hệ Thống Quản Lý Con Người Tại Viện Dưỡng Lão

**Ngày tạo:** 18/01/2026  
**Trạng thái:** Sẵn sàng triển khai  
**Thời gian thực hiện:** 8 tuần  

---

## 🎯 Mục Tiêu Dự Án

Phát triển hệ thống quản lý con người toàn diện cho viện dưỡng lão với:
- ✅ Nhận diện & đếm người trong khung hình
- ✅ Lưu thông tin người (Tên, tuổi, giới tính) via face recognition
- ✅ Định nghĩa vùng nguy hiểm/vùng cấm
- ✅ Cảnh báo khi người vào vùng cấm (qua email)
- ✅ Phát hiện hành động té ngã ngay lập tức (email alert trong < 5 giây)

---

## 📊 Phân Chia Công Việc: 2 Người 2 Nhóm

### **TEAM A: Kỹ Sư Service (Backend)**
**Trách nhiệm:** Logic xử lý, database, tích hợp AI

**Công nghệ:**
- Python + FastAPI (REST API)
- PostgreSQL (cơ sở dữ liệu)
- Azure Face API (nhận diện khuôn mặt)
- SendGrid (gửi email)

**Sản phẩm (8 tuần):**
| Tuần | Công việc | Kết quả |
|------|----------|--------|
| 1-2 | Schema DB + fall detection | Database + algorithm |
| 3-4 | Face API + Email service | Tích hợp ngoài |
| 5-6 | Tối ưu hiệu năng | Performance target |
| 7-8 | Dashboard + monitoring | Production ready |

---

### **TEAM B: Kỹ Sư Plugin (NX VMS)**
**Trách nhiệm:** Xử lý video, tích hợp NX VMS, tính ổn định

**Công nghệ:**
- C++ + NX VMS SDK
- CMake + OpenCV
- libcurl (HTTP client)

**Sản phẩm (8 tuần):**
| Tuần | Công việc | Kết quả |
|------|----------|--------|
| 1-2 | Cải thiện HTTP client | Robust communication |
| 3-4 | Fall detection + Zone check | Phát hiện sự kiện |
| 5-6 | Tối ưu + Docker | Production ready |
| 7-8 | Multi-camera support | Scale up |

---

## 🔄 Pipeline Giao Tiếp

### **API Chính (Plugin → Service, Port 18000)**

```
1. POST /detect
   Input:  Frame dạng base64
   Output: Bounding box + confidence
   Latency: < 50ms

2. POST /analytics/fall-detection
   Input:  Lịch sử bounding boxes (3-10 frames)
   Output: {is_falling: true/false, confidence: 0-1}
   Latency: < 100ms

3. POST /analytics/zone-check
   Input:  {camera_id, person_id, position}
   Output: {zones: [...violations...]}
   Latency: < 50ms

4. GET /person/{person_id}
   Input:  Person ID
   Output: {name, age, gender, room, contact}
   Latency: < 30ms

5. GET /health
   Input:  (none)
   Output: Status + stats
   Latency: < 10ms
```

### **Dòng Chảy Dữ Liệu**

```
Plugin (C++)
    ↓ HTTP POST frame
Service (Python)
    ↓ Run YOLOv8
    ↓ Detect people
    ↓ Track motion
    ↓ Analyze patterns
    ├─→ Fall detector → DB (event)
    ├─→ Zone check → DB (violation)
    ├─→ Get person info → Azure Face API
    ├─→ Send email → SendGrid
    └─→ Log metrics → Prometheus
    ↓
Database (PostgreSQL)
    ├─ Persons (tên, tuổi, giới tính)
    ├─ Events (fall, zone_violation)
    ├─ Face embeddings
    └─ Zones (vùng cấm)
```

---

## 💾 Cơ Sở Dữ Liệu

### **PostgreSQL Schema (6 bảng)**

```sql
persons
├─ id, name, age, gender, room_number
├─ emergency_contact
├─ face_image, face_embedding
└─ status, created_at, updated_at

events
├─ id, event_type (fall/zone_violation)
├─ person_id, camera_id, track_id
├─ severity (low/medium/high/critical)
└─ description, metadata, created_at

face_embeddings
├─ id, person_id, embedding (512-dim)
├─ camera_id, capture_date
└─ confidence

zones
├─ id, camera_id, zone_name
├─ zone_type (forbidden/danger)
├─ polygon (coordinates)
└─ is_active, created_at

alerts
├─ id, event_id, person_id
├─ alert_type (email/sms/push)
├─ recipient, status
└─ sent_at, error_message

tracking_data (tạm thời)
├─ person_id, track_id, camera_id
├─ position (x, y), velocity
└─ bbox, timestamp
```

---

## 🌐 Dịch Vụ Bên Thứ 3 Đề Xuất

| Dịch Vụ | Chức Năng | Giá | Khuyến Cáo |
|---------|----------|-----|-----------|
| **Azure Face API** | Nhận diện khuôn mặt | $1-10/1k | ✅ Best choice (GDPR) |
| **SendGrid** | Gửi email | $20/tháng | ✅ Reliable + templates |
| **AWS SES** | Email alternative | $0.10/1k | Cost-effective |
| **Twilio** | SMS alerts | $0.0075/SMS | Cho alert critical |
| **Firebase FCM** | Push notifications | Free | Mobile app |
| **Prometheus** | Metrics | Free | Open-source |
| **Grafana** | Dashboards | Free | Visualization |
| **ELK Stack** | Log management | Free | Elasticsearch + Kibana |

---

## 🚀 Roadmap 8 Tuần

### **Tuần 1-2: Xây Dựng Nền Tảng**

**TEAM A:**
- [ ] Design database schema
- [ ] Build ORM layer (SQLAlchemy)
- [ ] Implement fall detection algorithm
- [ ] Create service endpoints (/detect, /analytics/*)
- [ ] Unit tests

**TEAM B:**
- [ ] Refactor object_detector.cpp
- [ ] Implement HTTP client with retries
- [ ] Add health monitoring
- [ ] Comprehensive logging
- [ ] Unit tests

**Checkpoint:** Basic integration test works

---

### **Tuần 3-4: Tích Hợp**

**TEAM A:**
- [ ] Azure Face API integration
- [ ] SendGrid email service
- [ ] Person profile management
- [ ] Zone validation logic
- [ ] Integration tests

**TEAM B:**
- [ ] Fall detection trigger in plugin
- [ ] Zone check integration
- [ ] Person profile lookup
- [ ] NX event generation
- [ ] Integration tests

**Checkpoint:** End-to-end fall detection working

---

### **Tuần 5-6: Tối Ưu Hóa**

**TEAM A:**
- [ ] Performance optimization
- [ ] Caching strategy (Redis)
- [ ] Multi-camera coordination
- [ ] Load testing

**TEAM B:**
- [ ] Memory optimization
- [ ] Resilience patterns
- [ ] Docker containerization
- [ ] Stress testing

**Checkpoint:** All performance targets met

---

### **Tuần 7-8: Triển Khai Production**

**TEAM A:**
- [ ] Analytics dashboard
- [ ] Monitoring setup
- [ ] Documentation
- [ ] Production deployment

**TEAM B:**
- [ ] Deployment automation
- [ ] Logging aggregation
- [ ] Incident response procedures
- [ ] Final testing

**Checkpoint:** Production ready & live

---

## 📈 Mục Tiêu Hiệu Năng

**Service (Python):**
```
Inference: < 50ms per frame
API response: < 100ms @ 30fps
DB queries: < 50ms
Memory: < 4GB
```

**Plugin (C++):**
```
HTTP latency: 100-500ms (healthy)
Processing: < 50ms per frame
Memory: < 500MB per camera
CPU: < 30% per camera
```

**Business:**
```
Fall detection latency: < 5 seconds
Zone violation alert: < 2 seconds
Alert delivery: > 99% within 5 seconds
Detection accuracy: > 95%
Service uptime: > 99.5%
```

---

## 🔐 An Toàn & Bảo Mật

```
Data Protection:
├─ Face embeddings encrypted at rest (AES-256)
├─ PostgreSQL with SSL/TLS
├─ JWT authentication for API
└─ GDPR compliance for face data

Network Security:
├─ HTTPS/TLS for all external APIs
├─ API rate limiting
├─ Database access control
└─ Secret management via .env

Privacy:
├─ Face data deletion policies
├─ Audit logs for sensitive operations
├─ Per-camera access control
└─ Data retention policies
```

---

## 📦 Triển Khai Docker

```yaml
Services:
├─ Service (Python) × 2 instances
├─ PostgreSQL (database)
├─ Redis (cache)
├─ Nginx (load balancer)
├─ Prometheus (metrics)
├─ Grafana (dashboards)
├─ Elasticsearch (logs)
└─ Kibana (log analysis)

docker-compose up -d   # Start all
docker-compose down -v # Stop & cleanup
docker-compose logs -f service_1 # View logs
```

---

## 📞 Giao Tiếp Hằng Ngày

### **Daily Standup (15 phút)**
```
09:30 AM
├─ TEAM A: Completed + Today + Blockers
├─ TEAM B: Completed + Today + Blockers
└─ Action: Xử lý blocker ngay
```

### **Weekly Sync (1 giờ)**
```
Friday 10:00 AM
├─ Demo features (20 min)
├─ Identify blockers (15 min)
├─ Plan next week (20 min)
└─ Update documentation (5 min)
```

### **Async Communication:**
```
- Slack: Quick questions (< 1 min response)
- PR comments: Technical discussion
- Schedule sync: Blocking issues
```

---

## ✅ Checklist Tuần 1

**Chuẩn Bị:**
- [ ] Clone repo + setup Git branching
- [ ] Schedule daily standup (09:30 AM)
- [ ] Create shared Slack channel
- [ ] Assign TEAM A & B members
- [ ] Setup development environments

**Ngày 1:**
- [ ] Team kickoff meeting (1 hour)
- [ ] Architecture walkthrough
- [ ] Review API contracts
- [ ] Setup testing frameworks

**Ngày 2-5:**
- [ ] TEAM A: Start database design
- [ ] TEAM B: Start plugin refactoring
- [ ] Daily 15-min standups
- [ ] Reference docs for questions

**Ngày 5:**
- [ ] Weekly sync meeting
- [ ] Demo progress
- [ ] Plan Week 2

---

## 📚 Tài Liệu Chi Tiết

| File | Kích thước | Nội dung |
|------|-----------|---------|
| **ARCHITECTURE_AND_ROADMAP.md** | 8000 words | System design + roadmap |
| **IMPLEMENTATION_TEMPLATES.md** | 3000 words | Code examples |
| **DEPLOYMENT_AND_OPERATIONS.md** | 4000 words | Production setup |
| **PROJECT_SUMMARY.md** | 2000 words | Executive summary |
| **VISUAL_ARCHITECTURE_GUIDE.md** | 2000 words | Diagrams & quick reference |
| **DOCUMENTATION_INDEX.md** | 2000 words | Navigation guide |

**Total:** 20,000+ words, production-ready architecture

---

## 🎯 Bắt Đầu Ngay Hôm Nay

### **Step 1: Đọc tài liệu (2 giờ)**
```
1. Đọc file này (15 phút)
2. ARCHITECTURE_AND_ROADMAP.md (90 phút)
3. VISUAL_ARCHITECTURE_GUIDE.md (30 phút)
```

### **Step 2: Quy Hoạch Tuần 1 (1 giờ)**
```
1. Review team responsibilities
2. Discuss architecture approach
3. Setup development environments
4. Schedule daily standups
```

### **Step 3: Bắt Đầu Coding (Tuần 1)**
```
TEAM A: Copy models từ IMPLEMENTATION_TEMPLATES.md
TEAM B: Copy HTTP client từ IMPLEMENTATION_TEMPLATES.md
Cả 2: Setup testing frameworks
```

---

## 🎓 Lợi Ích Của Architecture Này

✅ **Separation of Concerns** - Mỗi team làm việc độc lập  
✅ **Clear API Contract** - Dễ tích hợp, dễ debug  
✅ **Scalable** - Từ 5 camera → 100+ camera  
✅ **Professional** - Monitoring, alerting, backup built-in  
✅ **Secure** - Encryption, GDPR compliance  
✅ **Cost-effective** - Dùng open-source + strategic SaaS  

---

## 📞 Liên Hệ & Hỗ Trợ

**Nếu có câu hỏi:**
1. Tìm kiếm trong 6 documents
2. Check VISUAL_ARCHITECTURE_GUIDE.md (diagrams)
3. Check PROJECT_SUMMARY.md (FAQ)
4. Schedule 30-min sync với team

**Trách Nhiệm:**
- Project Lead: ARCHITECTURE_AND_ROADMAP.md
- TEAM A: IMPLEMENTATION_TEMPLATES.md (sections 1-3)
- TEAM B: IMPLEMENTATION_TEMPLATES.md (sections 4-7)
- DevOps: DEPLOYMENT_AND_OPERATIONS.md

---

## 🚀 Tóm Tắt Nhanh

| Yếu Tố | Chi Tiết |
|--------|---------|
| **Thời gian** | 8 tuần |
| **Nhân sự** | 2 người (TEAM A + TEAM B) |
| **Công nghệ chính** | Python FastAPI + C++ NX Plugin |
| **Database** | PostgreSQL |
| **Nhân diện khuôn mặt** | Azure Face API |
| **Email** | SendGrid |
| **Monitoring** | Prometheus + Grafana + ELK |
| **Deployment** | Docker + Load Balancer |
| **API** | HTTP REST (JSON) |
| **Performance** | Inference < 50ms, Alert < 5sec |
| **Uptime** | > 99.5% |

---

**Bạn đã sẵn sàng triển khai! 🎯**

---

**Version:** 1.0  
**Ngày tạo:** 18/01/2026  
**Trạng thái:** Production Ready
