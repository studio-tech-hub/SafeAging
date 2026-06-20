# 📋 Tóm Tắt Kiến Trúc - SafeAging

**Ngày cập nhật:** 10/05/2026  
**Trạng thái:** Kiến trúc chốt cho Phase P1  
**Mục tiêu triển khai:** Làm trên Windows trước, triển khai production sau trên AI Box (edge inference) + hạ tầng trung tâm

---

## 🎯 Mục Tiêu Dự Án

Xây dựng hệ thống phân tích người cho môi trường viện dưỡng lão với các mục tiêu chính:

- Phát hiện và theo dõi người trong khung hình từ camera NX
- Ghi nhận sự kiện như đếm người, ngã, vào vùng cấm
- Quản lý dữ liệu nghiệp vụ như người, vùng, cảnh báo, cấu hình camera
- Lưu ảnh snapshot/crop để phục vụ điều tra hoặc nhận diện
- Đảm bảo hệ thống vẫn chạy khi DB trung tâm tạm thời mất kết nối

---

## ✅ Quyết Định Kiến Trúc Đã Chốt

### Mô hình production

```text
Camera
→ Nx Server + Nx Plugin
→ AI Service (edge, chạy trên AI Box)
→ PostgreSQL (metadata DB trung tâm)
→ MinIO / S3-compatible Object Storage (ảnh/snapshot/crop)

Nx Archive / NAS
→ video / playback / timeline
```

### Boundary rules

- Plugin chỉ gọi AI Service
- Plugin không truy cập PostgreSQL trực tiếp
- AI Service là lớp duy nhất làm việc với DB trung tâm
- Nx chịu trách nhiệm video/archive/playback
- AI Service chịu trách nhiệm inference, business logic, event, sync và retry

### Giao tiếp kỹ thuật

- **Plugin ↔ AI Service:** REST/HTTP
- **Không dùng gRPC trong Phase P1**
- **Transport inference hiện tại:** giữ `POST /infer` theo JSON + base64 để tương thích với plugin hiện có
- **Nâng cấp transport sau này:** có thể chuyển sang `multipart/form-data` mà không đổi response contract

### Edge local persistence

- **Chốt dùng SQLite**
- Không đưa Redis vào scope P1
- SQLite trên edge sẽ dùng cho:
  - local outbox
  - local config cache
  - sync state

---

## 🧱 Phân Tách Trách Nhiệm Dữ Liệu

### 1. Video / Archive / Playback

- Thuộc về **Nx Archive / NAS**
- Không lưu video thô trong PostgreSQL

### 2. Metadata nghiệp vụ

- Thuộc về **PostgreSQL**
- Bao gồm:
  - `persons`
  - `person_embeddings`
  - `zones`
  - `events`
  - `alerts`
  - `camera_configs`

### 3. Snapshot / Crop Images

- Thuộc về **MinIO / S3-compatible object storage**
- Có thể dùng NAS/file share ở môi trường tạm thời, nhưng production khuyến nghị MinIO

### 4. Database chỉ lưu metadata

- `image_path`
- `image_url`
- `object_key`
- metadata của event/person/zone/alert

---

## 🔄 Chính Sách Khi Mất Kết Nối Mạng / DB

### Khi AI Service mất kết nối PostgreSQL trung tâm

- Inference **vẫn tiếp tục chạy**
- Event **không được drop ngay**
- Event phải được ghi vào **SQLite outbox**
- Worker nền retry đồng bộ lại với PostgreSQL và object storage
- Retry dùng **exponential backoff**

### Local cache bắt buộc phải có

- `zones`
- `person_embeddings`
- `watchlists`
- `camera_configs`
- `AI thresholds/rules`

### Khi DB phục hồi

- Worker nền tự sync lại event pending
- Sync config nếu version trung tâm mới hơn cache local

---

## 🩺 Chuẩn Trạng Thái Health

### Top-level status

- `healthy`
- `degraded`
- `not_ready`

### Reason / condition codes

- `db_unreachable`
- `service_unreachable`
- `config_stale`
- `outbox_backlog_high`

### Ý nghĩa

- `healthy`: mọi dependency chính đang hoạt động bình thường
- `degraded`: hệ thống vẫn chạy nhưng có suy giảm chức năng hoặc rủi ro vận hành
- `not_ready`: service đang startup, warmup model, hoặc đang sync config ban đầu

### Ví dụ

```json
{
  "status": "degraded",
  "ready": true,
  "reason_codes": ["db_unreachable", "outbox_backlog_high"],
  "dependencies": {
    "postgres": "down",
    "object_storage": "up",
    "config_cache": "fresh"
  }
}
```

---

## 🛠️ Công Nghệ Chốt Cho Phase P1

| Thành phần | Lựa chọn |
|-----------|----------|
| Video / playback | Nx Archive / NAS |
| Plugin ↔ Service | REST/HTTP |
| Metadata DB trung tâm | PostgreSQL 16+ |
| Object storage | MinIO |
| Edge outbox/cache | SQLite |
| Monitoring | Prometheus + Grafana |
| Logging | Structured JSON logging |
| Email alert | Worker nền + provider ngoài (ví dụ SendGrid/AWS SES) |

---

## 📦 Trách Nhiệm Của Từng Thành Phần

### Nx Server + Plugin

- Nhận frame từ camera
- Gọi `POST /infer`
- Poll `GET /health`
- Đẩy metadata/event sang NX
- Hiển thị diagnostic event rõ ràng khi service degraded hoặc unavailable

### AI Service

- Chạy model inference
- Theo dõi người, fall detection, zone logic
- Ghi event vào local outbox trước
- Sync event, alert, snapshot metadata về PostgreSQL và MinIO
- Cache config local để tiếp tục chạy khi DB down

### PostgreSQL trung tâm

- Nơi lưu metadata chuẩn của toàn hệ thống
- Không chứa video thô
- Không bị plugin truy cập trực tiếp

### MinIO / Object Storage

- Lưu snapshot, crop image, ảnh nhận diện
- DB chỉ lưu path/key/url

---

## 🚀 Roadmap P1 Cho Một Người Làm

### Tuần 1: Dựng nền local dev trên Windows

- Dựng `docker-compose.dev.yml`
- Chạy local:
  - PostgreSQL
  - MinIO
  - Prometheus
- Chuẩn hóa env/config cho service
- Chốt schema metadata trung tâm

### Tuần 2: Service persistence nền tảng

- Thêm PostgreSQL cho:
  - persons
  - events
  - zones
  - alerts
  - camera_configs
- Thêm migration và DAL/ORM

### Tuần 3: SQLite local outbox + cache

- Tạo local SQLite database trên edge
- Thêm:
  - outbox jobs
  - config cache
  - sync state
- Bổ sung idempotent `event_id`

### Tuần 4: API nghiệp vụ thật

- Thêm API:
  - person
  - zone
  - event
  - reset
- Tách rõ API infer và API business/admin

### Tuần 5: Background worker + retry

- Worker nền cho:
  - event persistence
  - alert/email
  - object storage upload
- Exponential backoff
- Không làm nặng luồng `/infer`

### Tuần 6: Observability + health chuẩn

- Thêm `GET /metrics`
- Prometheus metrics
- JSON structured logs
- Health states + reason codes
- Ngưỡng cảnh báo cho backlog/config stale

### Tuần 7: Plugin reliability

- Health polling từ plugin sang service
- Diagnostic event rõ ràng cho:
  - service down
  - service degraded
  - queue pressure
  - circuit breaker state
- Review queue policy và threshold cảnh báo

### Tuần 8: Test + packaging + release cleanup

- Unit test và integration test
- Load test nhiều camera
- Chuẩn hóa build script, manifest, versioning, release artifact
- Loại bỏ dấu vết legacy khỏi packaging

---

## 🎯 Mapping Trực Tiếp Sang Các Task P1

### Plugin

- `Plugin P1.1`
  - Health polling từ plugin sang service
  - Diagnostic event rõ ràng theo `healthy/degraded/not_ready`

- `Plugin P1.2`
  - Chuẩn hóa packaging/build/versioning/release artifact
  - Dọn legacy naming và hardcoded version

- `Plugin P1.3`
  - Integration test cho queue/backpressure, circuit breaker, reconnect, metadata consistency

- `Plugin P1.4`
  - Giữ queue policy realtime nhưng thêm metric và threshold cảnh báo rõ

### Service

- `Service P1.1`
  - PostgreSQL cho metadata trung tâm

- `Service P1.2`
  - API thật cho person, zone, event, reset

- `Service P1.3`
  - Background worker cho alert/email, event persistence, object upload

- `Service P1.4`
  - Unit test + integration test + load test

- `Service P1.5`
  - Prometheus metrics, JSON logging, metrics endpoint sử dụng được

---

## ✅ Checklist Bắt Đầu Trên Windows

- [ ] Cài Docker Desktop
- [ ] Dựng PostgreSQL local bằng Docker
- [ ] Dựng MinIO local bằng Docker
- [ ] Dựng Prometheus local bằng Docker
- [ ] Chuẩn hóa `.env` cho service
- [ ] Thiết kế schema PostgreSQL trung tâm
- [ ] Thiết kế schema SQLite edge
- [ ] Chốt JSON contract cho `/health`
- [ ] Chốt danh sách metric cho `/metrics`

---

## 📈 Mục Tiêu Vận Hành

- Inference vẫn chạy khi PostgreSQL down
- Event không mất khi DB hoặc object storage lỗi tạm thời
- Plugin không cần biết DB trung tâm
- Snapshot/crop không nằm trong PostgreSQL
- Health state và diagnostic event có ngôn ngữ thống nhất
- Có thể triển khai từ Windows dev sang AI Box sau này mà không đổi kiến trúc lõi

---

## 🚀 Tóm Tắt Nhanh

| Hạng mục | Quyết định chốt |
|---------|------------------|
| Nhân sự | 1 người làm, roadmap tuần tự |
| Kiến trúc deploy | AI Service chạy edge trên AI Box |
| DB trung tâm | PostgreSQL 16+ |
| Ảnh/snapshot | MinIO |
| Video | Nx Archive / NAS |
| Plugin ↔ Service | REST |
| Cache/Outbox local | SQLite |
| Monitoring | Prometheus + Grafana |
| Logging | Structured JSON |
| Nguyên tắc cốt lõi | AI vẫn chạy khi DB down |

---

**Version:** 2.0
