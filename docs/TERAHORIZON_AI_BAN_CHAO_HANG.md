# HỆ THỐNG CAMERA AI GIÁM SÁT THÔNG MINH

## Triển khai bởi **TeraHorizon AI**

---

**Phiên bản tài liệu:** 1.0  
**Cập nhật:** Tháng 7/2026  
**Đối tượng:** Ban lãnh đạo, phòng vận hành, phòng IT, phòng mua sắm, đối tác tích hợp  
**Liên hệ:** TeraHorizon AI — *(điền hotline / email / website)*

---

## Mục lục

1. [Lời mở đầu — Tại sao cần hệ thống này?](#1-lời-mở-đầu--tại-sao-cần-hệ-thống-này)
2. [TeraHorizon AI là ai?](#2-terahorizon-ai-là-ai)
3. [Hệ thống Camera AI Giám Sát là gì?](#3-hệ-thống-camera-ai-giám-sát-là-gì)
4. [Hệ thống làm được những gì? — Danh sách đầy đủ](#4-hệ-thống-làm-được-những-gì--danh-sách-đầy-đủ)
5. [Cách hoạt động — Giải thích cho người không chuyên kỹ thuật](#5-cách-hoạt-động--giải-thích-cho-người-không-chuyên-kỹ-thuật)
6. [Kiến trúc hệ thống & mô hình triển khai](#6-kiến-trúc-hệ-thống--mô-hình-triển-khai)
7. [Mức độ Production — Đã sẵn sàng thực tế chưa?](#7-mức-độ-production--đã-sẵn-sàng-thực-tế-chưa)
8. [Giao diện vận hành — Người dùng dùng gì hàng ngày?](#8-giao-diện-vận-hành--người-dùng-dùng-gì-hàng-ngày)
9. [Ứng dụng theo ngành & kịch bản thực tế](#9-ứng-dụng-theo-ngành--kịch-bản-thực-tế)
10. [Lợi ích kinh doanh & ROI](#10-lợi-ích-kinh-doanh--roi)
11. [So sánh với giải pháp truyền thống](#11-so-sánh-với-giải-pháp-truyền-thống)
12. [Điểm mạnh cạnh tranh của TeraHorizon AI](#12-điểm-mạnh-cạnh-tranh-của-terahorizon-ai)
13. [Khả năng mở rộng trong tương lai](#13-khả-năng-mở-rộng-trong-tương-lai)
14. [Gói triển khai & quy trình làm việc với TeraHorizon AI](#14-gói-triển-khai--quy-trình-làm-việc-với-terahorizon-ai)
15. [Câu hỏi thường gặp (FAQ)](#15-câu-hỏi-thường-gặp-faq)
16. [Bước tiếp theo — Demo, Pilot & Báo giá](#16-bước-tiếp-theo--demo-pilot--báo-giá)

---

## 1. Lời mở đầu — Tại sao cần hệ thống này?

Camera giám sát ngày nay **đã thấy** — nhưng **chưa hiểu** và **chưa hành động** thay con người.

Trong viện dưỡng lão, bệnh viện, nhà máy, trường học hay tòa nhà văn phòng, hàng trăm camera ghi hình 24/7. Nhưng:

- **Không ai** có thể ngồi xem màn hình liên tục cả ngày lẫn đêm.
- Khi có sự cố (té ngã, đi lạc, vào vùng cấm), nhân viên thường **phát hiện muộn** — phải tua lại video hàng giờ mới tìm được thời điểm.
- Nhiều giải pháp AI trên thị trường là **ứng dụng rời**, không gắn với hệ thống camera (VMS) đang dùng — IT phải vận hành song song, nhân viên phải học thêm phần mềm mới.
- Giải pháp cloud phụ thuộc internet — **mất mạng là mất cảnh báo**, không phù hợp môi trường chăm sóc sức khỏe.

**Hệ thống Camera AI Giám Sát của TeraHorizon AI** giải quyết trọn bộ vấn đề trên:

> *Biến camera hiện có thành "mắt thông minh" — tự phát hiện người, nhận diện danh tính, cảnh báo té ngã, giám sát vùng cấm, ghi nhận bằng chứng và báo cáo — ngay tại cơ sở, không cần gửi video ra ngoài.*

---

## 2. TeraHorizon AI là ai?

**TeraHorizon AI** là đơn vị triển khai và vận hành **hệ thống Camera AI Giám Sát thông minh** — nền tảng phân tích video tập trung vào **con người**, được thiết kế và tối ưu cho môi trường **edge** (xử lý tại chỗ), tích hợp sâu với hệ thống quản lý video (VMS) phổ biến.

**Cam kết của TeraHorizon AI:**

| Cam kết | Ý nghĩa với khách hàng |
|---------|------------------------|
| **Không thay camera** | Tận dụng camera IP / VMS đang có — giảm chi phí đầu tư |
| **AI chạy tại chỗ** | Video không bắt buộc ra internet — bảo mật, tuân thủ quy định |
| **Tích hợp VMS** | Nhân viên vẫn xem màn hình quen thuộc (NX Witness) |
| **Triển khai trọn gói** | Khảo sát → lắp đặt → cấu hình → đào tạo → hỗ trợ vận hành |
| **Mở rộng theo nhu cầu** | Từ 2 camera pilot đến hàng chục camera, nhiều cơ sở |

---

## 3. Hệ thống Camera AI Giám Sát là gì?

Đây là **bộ giải pháp phần mềm end-to-end** gồm ba thành phần chính, do TeraHorizon AI triển khai và cấu hình tại cơ sở khách hàng:


┌─────────────┐     ┌──────────────────────┐     ┌─────────────────────────┐
│   Camera    │ ──► │  VMS + Plugin AI       │ ──► │  AI Box (xử lý tại chỗ) │
│  (RTSP/IP)  │     │  (NX Witness + Plugin) │     │  Phân tích & logic      │
└─────────────┘     └──────────────────────┘     └────────────┬────────────┘
                                                              │
              ┌───────────────────────────────────────────────┼────────────────────────┐
              ▼                               ▼               ▼                        ▼
       ┌─────────────┐               ┌─────────────┐   ┌─────────────┐         ┌─────────────┐
       │ PostgreSQL  │               │ MinIO / S3  │   │ Email cảnh  │         │ Prometheus  │
       │ (metadata)  │               │ (ảnh sự kiện)│   │ báo / API   │         │ / Grafana   │
       └─────────────┘               └─────────────┘   └─────────────┘         └─────────────┘


### Thành phần 1 — Plugin tích hợp VMS (C++)

- Gắn trực tiếp vào **NX Witness** (Network Optix) — nền tảng VMS được nhiều doanh nghiệp và cơ sở y tế tại Việt Nam sử dụng.
- Tự động lấy khung hình từ camera, gửi sang bộ xử lý AI.
- **Vẽ khung bao quanh người**, hiển thị **tên**, **tuổi**, **sự kiện** ngay trên giao diện VMS mà nhân viên đang xem.
- Phát sinh sự kiện VMS chuẩn: *Phát hiện người*, *Té ngã*, *Vi phạm vùng* — tích hợp timeline, playback, báo động VMS.

### Thành phần 2 — AI Service (bộ não xử lý)

- Chạy trên **AI Box** đặt tại cơ sở — xử lý inference, tracking, nhận diện mặt, phát hiện té ngã, kiểm tra vùng.
- Công nghệ: Python/FastAPI, mô hình YOLO thế hệ mới, nhận diện khuôn mặt ArcFace, ONNX Runtime.
- **API quản trị đầy đủ** + **giao diện web vận hành** (Operations Console).

### Thành phần 3 — Lưu trữ & cảnh báo

- **PostgreSQL:** hồ sơ người, vùng giám sát, nhật ký sự kiện, cấu hình camera.
- **MinIO (S3):** ảnh snapshot tại thời điểm sự kiện — bằng chứng nhẹ, dễ sao lưu.
- **Email SMTP:** gửi cảnh báo té ngã / vi phạm vùng tới danh sách nhân viên trực.
- **Prometheus + Grafana:** giám sát sức khỏe hệ thống cho phòng IT.

> **Một câu để nhớ:** *"Camera của bạn đã thấy — TeraHorizon AI giúp camera hiểu và hành động."*

---

## 4. Hệ thống làm được những gì? — Danh sách đầy đủ

Dưới đây là **toàn bộ khả năng** hệ thống Camera AI Giám Sát của TeraHorizon AI ở thời điểm hiện tại — đã triển khai production, không phải demo trên giấy.

---

### 4.1. Phát hiện & theo dõi người (Person Detection & Tracking)

| Khả năng | Mô tả chi tiết |
|----------|----------------|
| Phát hiện người | Nhận diện người trong khung hình với độ tin cậy cao, mô hình YOLO thế hệ mới |
| ID theo dõi ổn định | Cùng một người giữ cùng một số track qua nhiều khung hình — không nhảy số liên tục |
| Đếm người | Đếm số người unique xuất hiện trong khu vực camera |
| Làm mượt khung bao | Khung hiển thị trên VMS mượt, không giật |
| Nhiều camera đồng thời | Mỗi camera có trạng thái tracking riêng, xử lý song song |
| Vùng quan tâm (ROI) | Chỉ phân tích phần hình ảnh cần thiết — bỏ qua cửa sổ, tường, vùng nhiễu |
| Tùy chỉnh theo camera | Ngưỡng confidence, tần suất phân tích, ROI riêng từng camera |

**Ứng dụng thực tế:** Đếm người trong phòng chờ, hành lang, sân chơi; theo dõi ai đang ở đâu trong khu vực giám sát.

---

### 4.2. Nhận diện khuôn mặt & quản lý danh tính (Face Recognition)

| Khả năng | Mô tả chi tiết |
|----------|----------------|
| Nhận diện real-time | Hiển thị *(Tên, Giới tính, Số thứ tự)* trên khung bao người trên màn VMS |
| Đăng ký khuôn mặt | Từ ảnh, video, hoặc **trực tiếp từ khung hình live** (enroll from track) |
| Hồ sơ người | Họ tên, ngày sinh (tuổi tự tính), giới tính, phòng, ghi chú |
| Gallery embedding | Vector 512 chiều (ArcFace), so khớp cosine — nhanh, chính xác |
| Tự động thử lại Unknown | Người chưa biết được thử nhận diện lại theo lịch — không bỏ sót |
| ReID (tái nhận diện) | So khớp ảnh sự kiện cũ với hồ sơ đã đăng ký — tự gắn tên cho sự kiện trước đó |
| Xử lý local | Không gửi ảnh khuôn mặt lên cloud — dữ liệu nằm tại cơ sở |

**Ứng dụng thực tế:** Biết **ai** đang ở hành lang lúc 2h sáng; onboard cư dân mới trong vài phút; audit trail theo từng người.

**Enroll from live track — tính năng đặc biệt:**

1. Nhân viên thấy "Unknown" trên màn hình VMS.
2. Mở Operations Console → tab Live → chọn track → bấm **Enroll**.
3. Nhập tên, phòng, ngày sinh → xong.
4. Từ giây sau, camera hiển thị tên người đó trên VMS.

→ Không cần chụp ảnh riêng, không cần phiên đăng ký phức tạp.

---

### 4.3. Phát hiện té ngã (Fall Detection)

| Khả năng | Mô tả chi tiết |
|----------|----------------|
| Phân tích chuyển động | Tốc độ rơi, thay đổi góc, tỷ lệ chiều cao/rộng khung bao |
| Xác nhận đa khung hình | Yêu cầu N khung liên tiếp (mặc định 3) — giảm báo nhầm |
| YOLO-Pose (tùy chọn) | Phân tích tư thế cơ thể — tăng độ chính xác khi cần |
| Cảnh báo đa kênh | Sự kiện VMS + lưu DB + gửi email |
| Reset trạng thái | Theo camera hoặc toàn hệ thống — sau khi xử lý xong sự cố |
| Tune theo môi trường | ROI loại trừ giường, tăng ngưỡng xác nhận, bật pose tier-2 |

**Tín hiệu phát hiện té ngã:**

| Tín hiệu | Mô tả |
|----------|-------|
| Vận tốc rơi | Người di chuyển xuống nhanh bất thường |
| Thay đổi góc | Tư thế từ đứng → nằm |
| Tỷ lệ H/W | Khung bao chuyển từ cao → rộng (nằm ngang) |
| Pose (tùy chọn) | Góc thân so với phương đứng ≥ ngưỡng |

**Ứng dụng thực tế:** Viện dưỡng lão, bệnh viện khoa lão, nhà ở người cao tuổi sống một mình, nhà máy khu vực nguy hiểm.

---

### 4.4. Giám sát vùng (Zone Monitoring)

Hỗ trợ **bốn loại vùng**, vẽ bằng **đa giác (polygon)** trên khung hình camera:

| Loại vùng | Ý nghĩa | Hành vi |
|-----------|---------|---------|
| **Forbidden** (Cấm) | Khu vực không được vào | Cảnh báo khi người bước vào |
| **Entry** (Vào) | Ranh giới vào khu | Cảnh báo khi người đi vào vùng |
| **Exit** (Ra) | Ranh giới ra khỏi khu an toàn | Cảnh báo khi người rời vùng an toàn |
| **ROI** (Quan tâm) | Vùng phân tích | Chỉ giới hạn vùng AI xử lý — không cảnh báo |

**Đặc điểm nổi bật:**

- Mỗi camera có **nhiều vùng** độc lập.
- Logic **có trạng thái** — không spam cảnh báo liên tục khi người đứng trong vùng.
- Sự kiện lưu kèm ID vùng, tên vùng, track ID người vi phạm.

**Ứng dụng thực tế:**

- Cửa ra ngoài viện dưỡng lão (forbidden).
- Cầu thang ban đêm (forbidden).
- Khu bếp, kho thuốc (forbidden).
- Cổng ra sân chơi trường mầm non (exit).
- Vùng máy cắt đang chạy trong nhà máy (forbidden).

---

### 4.5. Ghi nhận sự kiện & bằng chứng hình ảnh

Mọi sự kiện quan trọng được ghi vào cơ sở dữ liệu với:

- Thời gian chính xác (timestamp).
- Camera nguồn.
- Loại sự kiện: detection, fall, zone_violation, v.v.
- Khung bao (bbox) tại thời điểm xảy ra.
- Liên kết người (nếu đã nhận diện).

**Ảnh snapshot** tự động upload lên object storage (MinIO/S3).

**Tra cứu:** theo camera, loại sự kiện, thời gian, người, vùng — xem ảnh trực tiếp từ giao diện quản trị.

→ Khi gia đình hoặc cơ quan quản lý hỏi "chuyện gì xảy ra lúc 2h15 sáng?" — trả lời trong **30 giây**, có ảnh, có thời gian, có tên người (nếu đã enroll).

---

### 4.6. Cảnh báo & thông báo (Alerting)

| Khả năng | Chi tiết |
|----------|----------|
| Email SMTP | Gửi khi té ngã, vi phạm vùng cấm / vùng an toàn |
| Chống trùng lặp (dedupe) | Không gửi 10 email cho cùng một sự cố |
| Giới hạn tần suất (rate limit) | Tránh flood email khi camera báo liên tục |
| Retry tự động | Gửi lại khi SMTP tạm thời lỗi |
| Nội dung email | Tên người (nếu có), camera, thời gian, loại sự kiện |

**Ứng dụng:** Điều dưỡng trực đêm nhận email trên điện thoại → chạy tới ngay. Bảo vệ nhận cảnh báo khi có người vào khu kỹ thuật.

---

### 4.7. Phân tích & báo cáo (Analytics)

| Báo cáo | Nội dung |
|---------|----------|
| Dashboard tổng quan | Số sự kiện, người, vùng, camera, trạng thái hệ thống |
| Timeseries | Xu hướng sự kiện theo giờ / ngày / tuần (24h / 7 ngày / 30 ngày) |
| Theo camera | Số té ngã, vi phạm vùng, track duy nhất |
| Theo vùng | Số lần vi phạm, lần cuối cùng |
| Cảnh báo | Thống kê gửi email: thành công / thất bại / bị chặn |

→ Ban quản lý review hàng tháng: khu vực nào té ngã nhiều? Vùng nào vi phạm thường xuyên? Cần bổ sung nhân sự ca nào?

---

### 4.8. Vận hành & giám sát hệ thống

| Khả năng | Chi tiết |
|----------|----------|
| Health check | Trạng thái `healthy` / `degraded` / `not_ready` — chuẩn hóa |
| Plugin tự poll health | VMS hiển thị cảnh báo khi AI Service suy giảm |
| Prometheus metrics | Tích hợp Grafana — latency, CPU, số inference |
| Log JSON có cấu trúc | Dễ tích hợp ELK / SIEM |
| Retention policy | Tự dọn dữ liệu cũ (mặc định 30 ngày sự kiện) |
| API Key auth | Bảo vệ endpoint quản trị |
| Rate limiting | Chống abuse API |

---

### 4.9. Khả năng chịu lỗi (Resilience) — Điểm khác biệt quan trọng

| Tình huống | Hành vi hệ thống |
|------------|------------------|
| PostgreSQL trung tâm mất kết nối | AI **vẫn chạy inference** và cảnh báo local |
| Mất mạng WAN | Sự kiện ghi vào **SQLite outbox** tại edge, sync lại khi DB phục hồi |
| AI Service quá tải | Plugin có **circuit breaker, retry, backpressure** — không treo VMS |
| Cache local | Vùng, embedding khuôn mặt, cấu hình camera — vẫn dùng được khi DB chậm |

→ Trong môi trường chăm sóc sức khỏe, **mất vài phút cảnh báo có thể là mất mạng sống**. Hệ thống TeraHorizon AI được thiết kế để **không dừng** khi hạ tầng trung tâm gặp sự cố.

---

## 5. Cách hoạt động — Giải thích cho người không chuyên kỹ thuật

Hãy tưởng tượng hệ thống Camera AI Giám Sát như một **y tá số** ngồi cạnh màn hình camera:

```
 Bước 1          Bước 2              Bước 3                    Bước 4
┌────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ Camera │───►│ VMS + Plugin │───►│ AI Box (bộ não) │───►│ Kết quả trả về   │
│  quay  │    │ lấy khung    │    │ phân tích       │    │ vài trăm ms      │
└────────┘    └──────────────┘    └─────────────────┘    └──────────────────┘
```

**Luồng chi tiết:**

1. **Camera quay** → hình ảnh đi vào hệ thống quản lý video (NX Witness).
2. **Plugin TeraHorizon AI** lấy từng khung hình (vài khung/giây) gửi sang AI Box.
3. **AI Box** trả lời trong vài trăm mili giây:
   - *"Có 2 người — người số 1 là Bà Nguyễn Thị A, phòng 201"*
   - *"Người số 2 chưa biết tên (Unknown)"*
   - *"Không có té ngã, không vi phạm vùng"*
4. **Plugin vẽ khung + tên** lên màn hình VMS mà nhân viên đang xem.
5. Nếu **có té ngã** hoặc **bước vào vùng cấm**:
   - VMS hiện cảnh báo.
   - Hệ thống chụp ảnh, lưu sự kiện.
   - Gửi email cho điều dưỡng / bảo vệ.
6. **Ban đêm**, khi không ai xem màn hình — AI vẫn làm việc. Sáng hôm sau, quản lý mở dashboard xem báo cáo.

**Điểm quan trọng:** Video gốc vẫn do NX Witness lưu trữ (NAS/archive). Hệ thống TeraHorizon AI chỉ lưu **metadata + ảnh chụp sự kiện** — nhẹ, dễ sao lưu, dễ tuân thủ quy định bảo mật dữ liệu cá nhân.

---

## 6. Kiến trúc hệ thống & mô hình triển khai

### 6.1. Kiến trúc Edge-First (khuyến nghị)

```
Cơ sở khách hàng (viện / nhà máy / trường)
├── Camera IP (RTSP)
├── NX Witness Server
├── AI Box (TeraHorizon AI chạy tại đây)     ← xử lý AI local
└── Switch mạng nội bộ

Trung tâm (DC / văn phòng / cloud nội bộ)
├── PostgreSQL (metadata)
├── MinIO (ảnh snapshot)
└── SMTP / monitoring (Grafana)
```

**Lợi ích edge:**

| Lợi ích | Giá trị |
|---------|---------|
| Độ trễ thấp | < 1 giây từ sự cố → cảnh báo |
| Bảo mật | Video không ra internet |
| Chịu lỗi | Vẫn hoạt động khi mất kết nối WAN |
| Chi phí băng thông | Không upload video liên tục |

### 6.2. Thiết bị edge đã kiểm chứng

TeraHorizon AI triển khai trên **AI Box QCS6490** (Qualcomm, ARM aarch64):

| Thông số | Giá trị production |
|----------|-------------------|
| Mô hình YOLO | yolo26n @ 640px |
| Backend | CPU lean pool 2×2 threads |
| Face | buffalo_s, retry Unknown mỗi 8s |
| Fall | Bbox heuristic; pose tắt mặc định |
| Tần suất | ~2 FPS inference / camera |
| Quy mô | 2–4 camera / AI Box (profile CPU) |
| Triển khai | Docker Compose — một lệnh |
| Tăng tốc NPU | QNN HTP trên roadmap — đã benchmark ~65ms/camera (3 cam) |

### 6.3. Yêu cầu hạ tầng tối thiểu

| Thành phần | Yêu cầu |
|------------|---------|
| AI Box / Server edge | 4+ CPU cores, 6+ GB RAM (tùy số camera) |
| Camera | RTSP/IP, độ phân giải ≥ 720p khuyến nghị |
| VMS | NX Witness (hoặc tương thích plugin) |
| DB trung tâm | PostgreSQL 16+ |
| Object storage | MinIO hoặc S3-compatible |
| Mạng | LAN 1Gbps nội bộ; WAN cho sync metadata (không bắt buộc realtime) |

### 6.4. Tích hợp VMS

- **NX Witness** (Network Optix) — plugin chính thức, đã production.
- Plugin settings: FPS enqueue, timeout, retry, health poll — cấu hình per camera.
- Sự kiện VMS native: detection, prolonged detection, fall, zone violation.

---

## 7. Mức độ Production — Đã sẵn sàng thực tế chưa?

**Câu trả lời ngắn: Có.** Hệ thống Camera AI Giám Sát của TeraHorizon AI **không phải prototype** — đã qua giai đoạn pilot trên phần cứng edge thật.

### Bằng chứng production

| Hạng mục | Trạng thái |
|----------|------------|
| Deploy trên AI Box QCS6490 | ✅ Đã chạy production |
| Profile CPU tune cho 2–4 camera | ✅ Đã document & verify |
| Plugin NX + circuit breaker | ✅ Production |
| Face recognition local | ✅ Production |
| Fall detection multi-signal | ✅ Production |
| Zone monitoring 4 loại | ✅ Production |
| Email alert dedupe + rate limit | ✅ Production |
| Edge outbox khi mất DB | ✅ Production |
| Operations Console web | ✅ Production |
| Prometheus + Grafana + alert rules | ✅ Có |
| Script kiểm tra ops tự động | ✅ `ops_aibox_check.py` |
| Docker Compose đa overlay | ✅ Có |
| Unit + integration tests | ✅ 150+ test cases |
| Tài liệu kỹ thuật & vận hành | ✅ Đầy đủ |

### Độ trễ thực tế (tham khảo)

| Cấu hình | Latency all-in / camera |
|----------|-------------------------|
| CPU lean (2–4 cam) | ~500–650 ms (degraded nếu > 650ms avg) |
| QNN HTP (3 cam, đã benchmark) | ~65 ms/camera |

### Health monitoring

Hệ thống tự báo `degraded` khi:

- Latency pipeline cao (avg ≥ 650ms hoặc p95 ≥ 800ms).
- CPU host quá tải (load1/cores ≥ 85%).

→ Plugin VMS tự poll và hiển thị cảnh báo — IT biết trước khi nhân viên phàn nàn.

---

## 8. Giao diện vận hành — Người dùng dùng gì hàng ngày?

Khách hàng sử dụng **hai giao diện chính** — không cần cài thêm phần mềm phức tạp:

### 8.1. NX Witness (giao diện quen thuộc)

- Nhân viên an ninh, điều dưỡng **vẫn xem camera như cũ**.
- Thêm: khung bao người, tên, cảnh báo té ngã/vùng ngay trên live view.
- Timeline sự kiện VMS tích hợp.

### 8.2. Operations Console (giao diện web TeraHorizon AI)

Truy cập qua trình duyệt — không cần cài app:

| Tab | Chức năng | Ai dùng |
|-----|-----------|---------|
| **Dashboard** | Tổng quan số liệu, trạng thái hệ thống | Quản lý, IT |
| **Live** | Xem track đang live, Enroll Unknown ngay | Điều dưỡng, hành chính |
| **Persons** | Quản lý hồ sơ người, upload ảnh, lịch sử sự kiện | Hành chính, y tá trưởng |
| **Events** | Nhật ký sự kiện, lọc, xem snapshot | Quản lý, điều tra sự cố |
| **Zones** | Tạo/sửa/vô hiệu hóa vùng giám sát | IT, kỹ thuật viên |

### 8.3. REST API (`/admin/*`)

Toàn bộ chức năng có API — tích hợp với:

- Phần mềm quản lý nội bộ (HIS, ERP).
- App di động tùy chỉnh.
- Dashboard báo cáo riêng của khách hàng.

---

## 9. Ứng dụng theo ngành & kịch bản thực tế

Hệ thống được thiết kế ban đầu cho **chăm sóc người cao tuổi**, nhưng lõi công nghệ (phát hiện người + tracking + vùng + nhận diện + cảnh báo) **áp dụng rộng** cho mọi môi trường cần giám sát hành vi con người qua camera.

---

### 🏥 9.1. Y tế & Chăm sóc sức khỏe

#### Viện dưỡng lão / Nhà an dưỡng

| Nhu cầu | Giải pháp TeraHorizon AI |
|---------|--------------------------|
| Té ngã ban đêm | Fall detection + email điều dưỡng trực |
| Cư dân đi lạc | Zone forbidden tại cửa, cầu thang |
| Không biết ai ở đâu | Face recognition + hiển thị tên trên VMS |
| Onboard cư dân mới | Enroll from live track — vài phút |
| Báo cáo gia đình | Event log + snapshot có timestamp |

#### Bệnh viện / Phòng khám

- Giám sát bệnh nhân nguy cơ té ngã (khoa lão, chỉnh hình).
- Cảnh báo bệnh nhân rời giường / rời khu điều trị.
- Kiểm soát vùng cách ly, khu hạn chế.
- Đếm người phòng chờ.

#### Home care — Chăm sóc tại gia

- 2 camera: phòng khách + cầu thang.
- Con cái nhận email té ngã — yên tâm khi bố mẹ ở xa.

---

### 🏫 9.2. Giáo dục

#### Trường mầm non / Tiểu học

- Cảnh báo khi trẻ ra khỏi vùng sân an toàn (zone exit).
- Đếm số trẻ trong sân chơi.
- Giám sát khu vực cô lập (cầu thang, bãi đậu xe).

#### Trường THPT / Đại học

- Giám sát phòng thí nghiệm, kho thiết bị (zone forbidden).
- Đếm lưu lượng cổng phụ.
- An ninh campus sau giờ học.

---

### 🏭 9.3. Sản xuất & Công nghiệp

#### Nhà máy sản xuất

| Nhu cầu | Giải pháp |
|---------|-----------|
| Worker vào vùng máy đang chạy | Zone forbidden + email tổ trưởng ca |
| Khu hóa chất, điện cao áp | Zone forbidden + audit trail |
| Đếm người khu sản xuất | Person count vs ca trực |
| Té ngã khu làm việc | Fall detection |
| ISO 45001 audit | Event log + snapshot + timestamp |

#### Kho bãi / Logistics

- Cảnh báo người đi bộ trong vùng xe nâng.
- Kiểm soát kho lạnh, kho giá trị cao.

#### GMP — Thực phẩm / Dược phẩm

- Kiểm soát vùng sạch — ai vào, lúc nào.

---

### 🏨 9.4. Khách sạn & Lưu trú

- Giám sát kho, phòng server, tầng hầm.
- Nhận diện VIP / nhân viên.
- An ninh hành lang ban đêm.

---

### 🛒 9.5. Bán lẻ & Thương mại

- Đếm lưu lượng khách theo khu vực.
- Giám sát kho hàng (staff only).
- Té ngã tại food court, khu thang cuốn.

---

### 🏛️ 9.6. Công cộng & An ninh

- Chung cư: té ngã sảnh, hầm xe.
- Cảnh báo người vào khu kỹ thuật (tầng mái, phòng điện).
- Đếm người tại lối thoát hiểm — hỗ trợ PCCC.

---

### 🏠 9.7. Gia đình (Premium)

- Người cao tuổi sống một mình.
- Bệnh nhân Alzheimer — cảnh báo ra cửa, vùng cấm bếp/cầu thang.

---

## 10. Kịch bản triển khai mẫu — Chi tiết từng bước

### Kịch bản A — Viện dưỡng lão 80 giường, 24 camera

**Bối cảnh:** 80 cư dân, 24 camera IP, đang dùng NX Witness. Ban đêm 2 điều dưỡng trực.

**Triển khai TeraHorizon AI:**

1. Lắp **2 AI Box** tại phòng server (12 cam/box).
2. Cài plugin lên NX Witness.
3. Enroll khuôn mặt 80 cư dân (ảnh + enroll from live, 1–2 tuần).
4. Vẽ vùng: cửa ra ngoài, cầu thang ban đêm, khu bếp.
5. Cấu hình email → điện thoại điều dưỡng trực + quản y.

**Kết quả:**

> Đêm khuya, cư dân bà Lan (85 tuổi) té ngã hành lang tầng 2.  
> AI phát hiện ~2 giây → email + sự kiện VMS.  
> Điều dưỡng chạy tới trong 1 phút.  
> Sự kiện lưu ảnh + timestamp → quản y báo cáo gia đình có bằng chứng.

---

### Kịch bản B — Nhà máy may 500 công nhân, 8 camera

**Bối cảnh:** Giám sát khu máy cắt + lối thoát hiểm. Không cần face recognition.

**Triển khai:**

1. 1 AI Box, 8 camera.
2. Zone forbidden quanh máy đang chạy.
3. Email/SMS tới tổ trưởng ca.

**Kết quả:** Công nhân bước vào vùng máy → cảnh báo ngay. Log audit cho ISO 45001.

---

### Kịch bản C — Trường mầm non, 2 camera sân chơi

**Triển khai:** Zone exit tại cổng sân. Cảnh báo khi trẻ ra khỏi vùng an toàn.

---

### Kịch bản D — Gia đình, bố 78 tuổi sống một mình

**Triển khai:** 2 camera + enroll bố + email 3 con. Không cần VMS phức tạp — RTSP trực tiếp.

---

## 11. Lợi ích kinh doanh & ROI

### 11.1. Giảm rủi ro pháp lý & uy tín

- Bằng chứng timestamp + ảnh khi có khiếu nại.
- Chứng minh đã có biện pháp giám sát chủ động.
- Audit trail đầy đủ theo từng người.

### 11.2. Tối ưu nhân sự

- 1 hệ thống AI thay **hàng chục giờ** xem camera thủ công/tháng.
- Nhân viên tập trung chăm sóc trực tiếp.
- Cảnh báo có lọc — không alarm fatigue.

### 11.3. Cải thiện thời gian phản ứng

| Sự cố | Trước | Sau TeraHorizon AI |
|-------|-------|-------------------|
| Té ngã | Phút → giờ (tua video) | **Giây** (email + VMS) |
| Ra khỏi khu an toàn | Phát hiện khi gặp | **Ngay tại cửa** |
| Tra cứu sự cố | Hàng giờ tua video | **30 giây** (Events tab) |

### 11.4. Tận dụng đầu tư sẵn có

- Không thay toàn bộ camera.
- Không cần hệ thống AI rời không tích hợp VMS.
- Plugin gắn vào NX Witness đang dùng.

### 11.5. Dữ liệu phục vụ quản trị

- Báo cáo té ngã theo khu vực → cải thiện thiết kế (tay vịn, thảm chống trượt).
- Thống kê vi phạm vùng → điều chỉnh ca trực.
- Heatmap hoạt động (roadmap) → tối ưu bố trí nhân sự.

---

## 12. So sánh với giải pháp truyền thống

| Tiêu chí | Camera + xem thủ công | Cloud AI generic | **TeraHorizon AI** |
|----------|----------------------|------------------|-------------------|
| Phát hiện té ngã tự động | ❌ | ✅ (một số) | ✅ |
| Nhận diện "ai" | ❌ | ⚠️ Cloud, phí | ✅ Local |
| Tích hợp VMS NX Witness | ❌ | ❌ | ✅ |
| Chạy offline / edge | ✅ (chỉ ghi hình) | ❌ | ✅ |
| Quản lý vùng linh hoạt | ❌ | ⚠️ | ✅ |
| Enroll từ live camera | ❌ | ❌ | ✅ |
| Audit trail + snapshot | ⚠️ Tua video | ⚠️ | ✅ |
| Dữ liệu tại Việt Nam / on-prem | — | ⚠️ Rủi ro | ✅ |
| Tùy biến theo ngành | — | ⚠️ | ✅ |
| Mất DB vẫn cảnh báo | — | ❌ | ✅ |

---

## 13. Điểm mạnh cạnh tranh của TeraHorizon AI

1. **Tích hợp sâu VMS** — nhân viên không đổi workflow, vẫn xem NX Witness.
2. **Edge-first** — AI tại chỗ, bảo mật, không phụ thuộc internet.
3. **Resilience** — mất DB vẫn chạy, sync sau; không mất sự kiện.
4. **Enroll from live track** — onboarding cư dân/nhân viên cực nhanh.
5. **All-in-one** — detection + tracking + face + fall + zone + alert + analytics trong một nền tảng.
6. **Mở API** — tích hợp ERP, app mobile, dashboard tùy chỉnh.
7. **Tối ưu phần cứng edge ARM** — chi phí AI Box thấp hơn server GPU.
8. **Triển khai linh hoạt** — từ viện 100 giường đến gia đình 2 camera.
9. **Đội ngũ triển khai local** — khảo sát, cấu hình, đào tạo, hỗ trợ tiếng Việt.

---

## 14. Khả năng mở rộng trong tương lai

TeraHorizon AI cam kết lộ trình phát triển liên tục. Khách hàng mua hôm nay **không bị kẹt** — hệ thống mở rộng theo nhu cầu:

### 14.1. Lộ trình kỹ thuật

| Giai đoạn | Tính năng | Lợi ích cho khách hàng |
|-----------|-----------|------------------------|
| **QNN / NPU** | Tăng tốc YOLO trên Hexagon HTP | Nhiều camera hơn / AI Box, latency thấp hơn |
| **Pose nâng cao** | Phát hiện té ngã chính xác hơn | Giảm false positive viện dưỡng lão |
| **SMS / Zalo / Push** | Cảnh báo đa kênh | Điều dưỡng nhận ngay trên điện thoại VN |
| **Rule engine** | Quy tắc tùy chỉnh: "1 người > 5 phút tại vùng X" | Linh hoạt theo quy trình nội bộ |
| **Heatmap** | Bản đồ nhiệt hoạt động | Tối ưu bố trí nhân sự, thiết kế không gian |
| **PPE detection** | Phát hiện không đeo mũ/áo bảo hộ | Vertical nhà máy, GMP |
| **Multi-site** | Quản lý tập trung nhiều cơ sở | Chuỗi viện dưỡng lão, chuỗi nhà máy |
| **Plugin VMS khác** | Milestone, Genetec, v.v. | Mở rộng thị trường |
| **White-label** | OEM cho đối tác tích hợp | Đối tác camera/VMS bán lại |

### 14.2. Mở rộng theo quy mô

```
Giai đoạn 1          Giai đoạn 2           Giai đoạn 3
┌─────────────┐     ┌─────────────┐      ┌─────────────────┐
│ Pilot       │ ──► │ Rollout     │ ──►  │ Multi-site      │
│ 2–4 camera  │     │ 20–50 camera│      │ N cơ sở         │
│ 1 cơ sở     │     │ 1 cơ sở     │      │ Dashboard tập   │
└─────────────┘     └─────────────┘      │ trung           │
                                         └─────────────────┘
```

### 14.3. Mở rộng theo vertical

| Vertical | Tính năng hiện có | Mở rộng thêm |
|----------|-------------------|--------------|
| Viện dưỡng lão | Fall + Face + Zone | SMS gia đình, báo cáo tháng |
| Nhà máy | Zone + Fall | PPE, đếm ca trực |
| Trường học | Zone exit | Đếm trẻ, người lạ (watchlist) |
| Bán lẻ | Person count | Heatmap, dwell time |
| Gia đình | Fall + Zone | App mobile, Zalo |

---

## 15. Gói triển khai & quy trình làm việc với TeraHorizon AI

### 15.1. Các gói dịch vụ

| Gói | Nội dung | Phù hợp |
|-----|----------|---------|
| **Pilot** | 2–4 camera, 30 ngày, tune ngưỡng, báo cáo FP/FN | Khách hàng muốn thử trước khi cam kết |
| **Standard** | Rollout toàn bộ camera, enroll, zone, email, đào tạo | Viện dưỡng lão, nhà máy vừa |
| **Enterprise** | Multi-site, SLA, Grafana, API tích hợp, custom rule | Chuỗi cơ sở, tập đoàn |
| **Gia đình** | 1–2 camera, RTSP, email alert | Home care premium |

### 15.2. Quy trình triển khai 4 giai đoạn

```
Giai đoạn 1 — Khảo sát (1–2 tuần)
├── Đếm camera, vị trí, góc quay
├── Xác định use case ưu tiên
├── Kiểm tra VMS (NX Witness version)
└── Thiết kế sơ đồ vùng giám sát

Giai đoạn 2 — Pilot (2–4 tuần)
├── Triển khai 2–4 camera thí điểm
├── Tune ngưỡng fall / zone / face
├── Enroll nhóm pilot
└── Đào tạo nhân viên vận hành

Giai đoạn 3 — Rollout (4–8 tuần)
├── Mở rộng toàn bộ camera
├── Enroll toàn bộ hồ sơ
├── Cấu hình email theo ca trực
└── Tích hợp monitoring

Giai đoạn 4 — Vận hành & tối ưu (liên tục)
├── Review false positive / false negative hàng tháng
├── Điều chỉnh ROI, vùng, ngưỡng
└── Báo cáo analytics cho ban quản lý
```

### 15.3. TeraHorizon AI cam kết hỗ trợ

| Hạng mục | Cam kết |
|----------|---------|
| Đào tạo vận hành | Buổi training Operations Console + VMS |
| Tài liệu | Hướng dẫn tiếng Việt cho nhân viên |
| Hỗ trợ kỹ thuật | Hotline / ticket trong giờ hành chính (SLA theo gói) |
| Bảo trì | Cập nhật phần mềm, patch bảo mật |
| Tune AI | Hỗ trợ giảm false positive sau pilot |

---

## 16. Câu hỏi thường gặp (FAQ)

### Hệ thống có thay thế camera không?
**Không.** Dùng camera IP hiện có, kết nối qua VMS (NX Witness).

### Video có gửi lên cloud không?
**Không bắt buộc.** AI xử lý tại AI Box local. Chỉ metadata và ảnh snapshot sync về server (có thể đặt nội bộ).

### Nhận diện khuôn mặt có chính xác không?
Phụ thuộc chất lượng ảnh enroll và góc camera. Khuyến nghị ≥ 5 ảnh hoặc enroll from live track. Ngưỡng match có thể tune.

### Báo té ngã có hay báo nhầm không?
Có thể khi ngồi xuống nhanh hoặc nằm giường. Giảm bằng: ROI loại trừ giường, tăng xác nhận khung hình, bật pose. TeraHorizon AI hỗ trợ tune sau pilot.

### Mất internet thì sao?
AI vẫn phân tích và cảnh báo local (sự kiện VMS). Email cần SMTP reachable. Sự kiện queue local, sync DB khi mạng về.

### Cần GPU NVIDIA không?
**Không.** Chạy CPU trên AI Box ARM. Roadmap NPU Qualcomm (QNN) để tăng số camera.

### Tích hợp VMS khác ngoài NX Witness?
Hiện tại plugin chính thức cho NX Witness. Camera RTSP trực tiếp cho pilot hoặc môi trường không VMS.

### Bao nhiêu camera trên một AI Box?
Profile CPU production: **2–4 camera** @ ~2 FPS. Tăng camera → thêm AI Box hoặc bật NPU.

### Dữ liệu khuôn mặt có tuân thủ quy định không?
Embedding lưu local/on-premise. Khách hàng kiểm soát retention. Cần thông báo và đồng ý theo Luật ANM Việt Nam / GDPR.

### Có hỗ trợ tiếng Việt không?
**Có.** Giao diện console, tài liệu đào tạo, hỗ trợ kỹ thuật tiếng Việt.

---

## 17. Thông số kỹ thuật tóm tắt

| Hạng mục | Chi tiết |
|----------|----------|
| Object detection | YOLO26 (ONNX), class: person |
| Tracking | IoU + appearance, Hungarian assignment |
| Face recognition | InsightFace ArcFace, 512-d embedding |
| Fall detection | Multi-signal bbox + optional YOLO-Pose |
| Zone types | forbidden, entry, exit, roi (polygon) |
| VMS integration | NX Witness plugin (C++) |
| AI Service | Python 3.11, FastAPI, ONNX Runtime |
| Database | PostgreSQL 16+ |
| Object storage | MinIO / S3-compatible |
| Edge cache | SQLite (outbox + config) |
| Alerting | SMTP email (dedupe + rate limit + retry) |
| Monitoring | Prometheus + Grafana |
| Security | API Key auth, HTTPS optional, rate limiting |
| Inference input | 640×640 px (configurable) |
| Supported edge | AI Box QCS6490 (aarch64), Docker |

---

## 18. Bước tiếp theo — Demo, Pilot & Báo giá

### TeraHorizon AI mang lại gì cho bạn?

✅ **An tâm hơn** — sự cố được phát hiện sớm, có người phản ứng  
✅ **Minh bạch hơn** — mọi sự kiện có bằng chứng, có thời gian  
✅ **Hiệu quả hơn** — nhân sự tập trung chăm sóc, không ngồi xem camera  
✅ **Linh hoạt hơn** — từ viện dưỡng lão đến nhà máy, trường học, gia đình  

### Đề xuất bước tiếp theo

| Bước | Nội dung | Thời lượng |
|------|----------|------------|
| **1. Demo live** | Xem nhận diện + té ngã + vùng cấm trên camera thật | 30–45 phút |
| **2. Khảo sát sơ bộ** | Đếm camera, use case, VMS hiện tại | 1–2 giờ tại cơ sở |
| **3. Pilot** | 2 camera / 30 ngày — đo false positive rate | 1 tháng |
| **4. Báo giá** | Theo gói: số camera, face on/off, hỗ trợ triển khai | Sau pilot |

### Thông tin liên hệ

| | |
|---|---|
| **Sản phẩm** | Hệ thống Camera AI Giám Sát Thông Minh |
| **Đơn vị triển khai** | **TeraHorizon AI** |
| **Website** | *(điền URL)* |
| **Email** | *(điền email sales)* |
| **Hotline** | *(điền SĐT)* |
| **Demo** | *(điền link đặt lịch)* |

---

> *"Camera của bạn đã thấy — TeraHorizon AI giúp camera hiểu và hành động."*

---

**© 2026 TeraHorizon AI. Bảo lưu mọi quyền.**

*Tài liệu mô tả khả năng hệ thống Camera AI Giám Sát dựa trên phiên bản phần mềm hiện tại. Một số tính năng roadmap có thể thay đổi theo lộ trình phát triển. Vui lòng liên hệ TeraHorizon AI để nhận bản cập nhật mới nhất.*
