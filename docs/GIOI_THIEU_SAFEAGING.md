# SafeAging — Giới Thiệu Giải Pháp Phân Tích Người Thông Minh

**Tài liệu dành cho khách hàng & đội kinh doanh**  
**Phiên bản:** 1.0 · **Cập nhật:** Tháng 6/2026  
**Nhà phát triển:** SafeAging

---

## Mục lục

1. [Tóm tắt điều hành](#1-tóm-tắt-điều-hành)
2. [Vấn đề thị trường mà SafeAging giải quyết](#2-vấn-đề-thị-trường-mà-safeaging-giải-quyết)
3. [SafeAging là gì?](#3-safeaging-là-gì)
4. [Hệ thống làm được những gì?](#4-hệ-thống-làm-được-những-gì)
5. [Cách hoạt động — dễ hiểu cho người không chuyên kỹ thuật](#5-cách-hoạt-động--dễ-hiểu-cho-người-không-chuyên-kỹ-thuật)
6. [Tính năng chi tiết](#6-tính-năng-chi-tiết)
7. [Giao diện vận hành](#7-giao-diện-vận-hành)
8. [Mô hình triển khai & hạ tầng](#8-mô-hình-triển-khai--hạ-tầng)
9. [Lợi ích kinh doanh & ROI](#9-lợi-ích-kinh-doanh--roi)
10. [Ứng dụng theo ngành & lĩnh vực](#10-ứng-dụng-theo-ngành--lĩnh-vực)
11. [Kịch bản sử dụng thực tế](#11-kịch-bản-sử-dụng-thực-tế)
12. [So sánh với giải pháp truyền thống](#12-so-sánh-với-giải-pháp-truyền-thống)
13. [Điểm mạnh cạnh tranh](#13-điểm-mạnh-cạnh-tranh)
14. [Thông số kỹ thuật tóm tắt](#14-thông-số-kỹ-thuật-tóm-tắt)
15. [Quy trình triển khai điển hình](#15-quy-trình-triển-khai-điển-hình)
16. [Câu hỏi thường gặp (FAQ)](#16-câu-hỏi-thường-gặp-faq)
17. [Lộ trình phát triển](#17-lộ-trình-phát-triển)
18. [Bước tiếp theo — liên hệ & demo](#18-bước-tiếp-theo--liên-hệ--demo)

---

## 1. Tóm tắt điều hành

**SafeAging** là nền tảng **phân tích video thông minh tập trung vào con người**, được thiết kế ban đầu cho **chăm sóc người cao tuổi** nhưng có thể mở rộng sang nhiều lĩnh vực khác.

Hệ thống biến **camera giám sát hiện có** thành một lớp trí tuệ nhân tạo có khả năng:

- **Phát hiện và theo dõi người** theo thời gian thực trên từng camera
- **Nhận diện khuôn mặt** và gắn tên, tuổi, phòng cho từng người trên màn hình giám sát
- **Phát hiện té ngã** và gửi cảnh báo tức thì
- **Giám sát vùng cấm / vùng an toàn** (cửa ra ngoài, khu bếp, cầu thang…)
- **Ghi nhận sự kiện, lưu ảnh bằng chứng** và báo cáo thống kê
- **Vận hành ổn định ngay cả khi mạng hoặc máy chủ trung tâm tạm thời gián đoạn**

SafeAging **không thay thế camera** và **không lưu trữ toàn bộ video** — nó bổ sung trí tuệ lên hệ thống VMS (Video Management System) mà khách hàng đã có, tận dụng hạ tầng sẵn có để giảm chi phí đầu tư.

> **Một câu để nhớ:** *"Camera của bạn đã thấy — SafeAging giúp camera hiểu và hành động."*

---

## 2. Vấn đề thị trường mà SafeAging giải quyết

### 2.1. Thiếu nhân lực giám sát liên tục

Trong viện dưỡng lão, bệnh viện, trường học hay nhà máy, không thể có nhân viên nhìn màn hình 24/7. Camera ghi hình nhưng **không tự cảnh báo** khi có sự cố. Hậu quả:

- Té ngã được phát hiện muộn → tăng nguy cơ biến chứng
- Người cao tuổi đi lạc / ra ngoài khu vực an toàn
- Vi phạm an toàn lao động không được xử lý kịp thời

### 2.2. Dữ liệu video "chết" — khó tra cứu

Khi có sự cố, nhân viên phải **tua lại hàng giờ video** để tìm thời điểm, người liên quan. Quy trình này tốn thời gian, dễ sai sót, khó làm báo cáo cho gia đình hoặc cơ quan quản lý.

### 2.3. Hệ thống AI rời rạc, khó tích hợp

Nhiều giải pháp AI trên thị trường là **ứng dụng độc lập**, không gắn với VMS đang dùng, không đồng bộ với quy trình vận hành hiện tại. Đội IT phải vận hành song song nhiều hệ thống.

### 2.4. Phụ thuộc mạng — rủi ro khi mất kết nối

Giải pháp cloud-only dừng hoạt động khi internet chập chờn. Trong môi trường chăm sóc sức khỏe, **mất vài phút cảnh báo có thể là mất mạng sống**.

### 2.5. SafeAging trả lời như thế nào?

| Vấn đề | Giải pháp SafeAging |
|--------|---------------------|
| Không ai xem camera 24/7 | AI phân tích liên tục, cảnh báo tự động |
| Khó tra cứu sự cố | Sự kiện có timestamp, ảnh chụp, liên kết người |
| Không tích hợp VMS | Plugin gắn trực tiếp vào NX Witness / hệ VMS tương thích |
| Mất mạng = mất AI | Xử lý tại edge (AI Box), hàng đợi sự kiện local, sync sau |
| Không biết "ai" trong khung hình | Nhận diện khuôn mặt + hồ sơ người (tên, phòng, tuổi) |

---

## 3. SafeAging là gì?

SafeAging là **bộ giải pháp phần mềm end-to-end** gồm ba thành phần chính:

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   Camera    │ ──► │  NX VMS + Plugin │ ──► │  AI Service (Edge)  │
│  (RTSP/IP)  │     │  (tích hợp VMS)  │     │  Phân tích & logic  │
└─────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                        │
                        ┌───────────────────────────────┼───────────────────────────────┐
                        ▼                               ▼                               ▼
                 ┌─────────────┐               ┌─────────────────┐              ┌──────────────┐
                 │ PostgreSQL  │               │ MinIO / S3      │              │ Email cảnh   │
                 │ (metadata)  │               │ (ảnh snapshot)  │              │ báo / API    │
                 └─────────────┘               └─────────────────┘              └──────────────┘
```

### Thành phần 1 — Plugin VMS (C++)

- Gắn vào **NX Witness** (Network Optix) — một trong những nền tảng VMS phổ biến
- Lấy khung hình từ camera, gửi sang AI Service
- Hiển thị **khung bao quanh người**, **tên**, **sự kiện** trực tiếp trên giao diện VMS
- Phát sinh sự kiện VMS: *Phát hiện người*, *Té ngã*, *Vi phạm vùng*

### Thành phần 2 — AI Service (Python / FastAPI)

- Bộ não xử lý: phát hiện người (YOLO), theo dõi (tracking), nhận diện mặt (ArcFace), phát hiện té ngã, kiểm tra vùng
- Chạy trên **máy edge** (AI Box) ngay tại cơ sở — dữ liệu video không cần ra internet
- API quản trị đầy đủ + giao diện web vận hành

### Thành phần 3 — Lưu trữ & cảnh báo

- **PostgreSQL:** hồ sơ người, vùng giám sát, nhật ký sự kiện
- **MinIO (S3):** ảnh snapshot tại thời điểm sự kiện
- **Email SMTP:** gửi cảnh báo té ngã / vi phạm vùng tới danh sách nhân viên

---

## 4. Hệ thống làm được những gì?

Dưới đây là **danh sách đầy đủ khả năng** của SafeAging ở thời điểm hiện tại:

### 4.1. Phát hiện & theo dõi người (Person Detection & Tracking)

- Phát hiện **người** trong khung hình với độ tin cậy cao (mô hình YOLO thế hệ mới)
- Gán **ID theo dõi ổn định** cho từng người — cùng một người giữ cùng một số track qua nhiều khung hình
- **Đếm số người** xuất hiện trong khu vực camera (unique count)
- Làm mượt khung bao, loại bỏ phát hiện trùng lặp
- Hỗ trợ **nhiều camera đồng thời**, mỗi camera có trạng thái tracking riêng
- Tùy chỉnh **vùng quan tâm (ROI)** theo từng camera — chỉ phân tích phần hình ảnh cần thiết

### 4.2. Nhận diện khuôn mặt & quản lý danh tính (Face Recognition)

- Nhận diện khuôn mặt **real-time** trên khung bao người — hiển thị dạng *(Tên, Giới tính, Số thứ tự)* trên màn VMS
- **Đăng ký khuôn mặt** từ ảnh, video hoặc **trực tiếp từ khung hình live** (enroll from track)
- Quản lý **hồ sơ người:** họ tên, ngày sinh (tuổi tự tính), giới tính, phòng, ghi chú
- Gallery embedding 512 chiều (ArcFace) — so khớp cosine similarity
- Tự động thử lại nhận diện cho người chưa biết (Unknown) theo lịch thời gian
- **ReID (Re-Identification):** so khớp ảnh sự kiện với hồ sơ đã đăng ký, tự động gắn tên cho sự kiện cũ

### 4.3. Phát hiện té ngã (Fall Detection)

- Phân tích **chuỗi chuyển động** của khung bao người: tốc độ rơi, thay đổi góc, tỷ lệ chiều cao/rộng
- **Xác nhận đa khung hình** — tránh báo giả do ngồi xổm, cúi xuống
- Tùy chọn **YOLO-Pose** (phân tích tư thế cơ thể) để tăng độ chính xác khi cần
- Phát sự kiện **Fall detected** lên VMS + lưu DB + gửi email cảnh báo
- Reset trạng thái té ngã theo camera hoặc toàn hệ thống

### 4.4. Giám sát vùng (Zone Monitoring)

Hỗ trợ bốn loại vùng, vẽ bằng **đa giác (polygon)** trên khung hình camera:

| Loại vùng | Ý nghĩa | Hành vi |
|-----------|---------|---------|
| **Forbidden** (Cấm) | Khu vực không được vào | Cảnh báo khi người bước vào |
| **Entry** (Vào) | Ranh giới vào khu | Cảnh báo khi người đi vào vùng |
| **Exit** (Ra) | Ranh giới ra khỏi khu | Cảnh báo khi người rời vùng an toàn |
| **ROI** (Quan tâm) | Vùng phân tích | Chỉ dùng để giới hạn, không cảnh báo |

- Mỗi camera có thể có **nhiều vùng** độc lập
- Phát hiện **có trạng thái** — không spam cảnh báo liên tục khi người đứng trong vùng
- Sự kiện vi phạm vùng được lưu kèm ID vùng, tên vùng, track ID

### 4.5. Ghi nhận sự kiện & bằng chứng hình ảnh

- Mọi sự kiện quan trọng được ghi vào **cơ sở dữ liệu** với:
  - Thời gian chính xác
  - Camera nguồn
  - Loại sự kiện (detection, fall, zone_violation…)
  - Khung bao (bbox) tại thời điểm xảy ra
  - Liên kết người (nếu đã nhận diện)
- **Ảnh snapshot** tự động upload lên object storage (MinIO/S3)
- Tra cứu sự kiện theo camera, loại, thời gian, người, vùng
- Xem ảnh snapshot trực tiếp từ giao diện quản trị

### 4.6. Cảnh báo & thông báo (Alerting)

- Gửi **email cảnh báo** qua SMTP khi:
  - Phát hiện té ngã
  - Vi phạm vùng cấm / vùng an toàn
- **Chống trùng lặp** (dedupe) — không gửi 10 email cho cùng một sự cố
- **Giới hạn tần suất** (rate limit) — tránh flood email khi camera báo liên tục
- **Retry tự động** khi gửi thất bại
- Email kèm thông tin người (nếu đã nhận diện), camera, thời gian

### 4.7. Phân tích & báo cáo (Analytics)

- **Dashboard tổng quan:** số sự kiện, số người, số vùng, số camera
- Thống kê theo **24 giờ / 7 ngày / 30 ngày**
- **Biểu đồ timeseries** — xu hướng sự kiện theo giờ/ngày/tuần
- Thống kê **theo camera:** số té ngã, vi phạm vùng, track duy nhất
- Thống kê **theo vùng:** số lần vi phạm, lần cuối cùng
- Thống kê **gửi cảnh báo:** thành công / thất bại / bị chặn

### 4.8. Vận hành & giám sát hệ thống

- **Health check** chuẩn hóa: `healthy` / `degraded` / `not_ready`
- Plugin VMS tự poll health và hiển thị cảnh báo khi AI Service suy giảm
- **Prometheus metrics** — tích hợp Grafana cho giám sát hạ tầng
- Log JSON có cấu trúc — dễ tích hợp ELK / SIEM
- **Chính sách retention** — tự dọn dữ liệu cũ theo cấu hình (mặc định 30 ngày)

### 4.9. Khả năng chịu lỗi (Resilience)

- AI **vẫn chạy inference** khi PostgreSQL trung tâm tạm mất kết nối
- Sự kiện được ghi vào **SQLite outbox** tại edge, worker nền sync lại khi DB phục hồi
- Cache local: vùng, embedding khuôn mặt, cấu hình camera
- Plugin có **circuit breaker, retry, backpressure** — không làm treo VMS khi AI quá tải

---

## 5. Cách hoạt động — dễ hiểu cho người không chuyên kỹ thuật

Hãy tưởng tượng SafeAging như một **y tá số** ngồi cạnh màn hình camera:

1. **Camera quay** → hình ảnh đi vào hệ thống quản lý video (NX Witness)
2. **Plugin SafeAging** lấy từng khung hình (vài khung/giây) gửi sang "bộ não AI"
3. **AI Service** trả lời trong vài trăm ms:
   - "Có 2 người — người số 1 là **Bà Nguyễn Thị A**, phòng 201"
   - "Người số 2 chưa biết tên (Unknown)"
   - "Không có té ngã, không vi phạm vùng"
4. **Plugin vẽ khung + tên** lên màn hình VMS mà nhân viên đang xem
5. Nếu **có té ngã** hoặc **bước vào vùng cấm**:
   - VMS hiện cảnh báo đỏ
   - Hệ thống chụp ảnh, lưu sự kiện
   - Gửi email cho điều dưỡng / bảo vệ
6. **Ban đêm**, khi không ai xem màn hình — AI vẫn làm việc. Sáng hôm sau, quản lý mở dashboard xem báo cáo.

**Điểm quan trọng:** Video gốc vẫn do NX Witness lưu trữ (NAS/archive). SafeAging chỉ lưu **metadata + ảnh chụp sự kiện** — nhẹ, dễ sao lưu, dễ tuân thủ quy định bảo mật.

---

## 6. Tính năng chi tiết

### 6.1. Phát hiện người — YOLO thế hệ mới

SafeAging sử dụng mô hình **YOLO26** — dòng detector object state-of-the-art, tối ưu cho phát hiện người trong môi trường thực tế:

- Hoạt động tốt với **ánh sáng yếu** (tùy chọn CLAHE enhancement)
- Hỗ trợ **upscale** khung hình nhỏ trước khi inference — phù hợp camera độ phân giải thấp
- Chạy trên **CPU** (AI Box edge) hoặc **NPU/GPU** (Qualcomm QNN trên thiết bị QCS6490)
- ONNX Runtime — triển khai ổn định, không phụ thuộc GPU NVIDIA

### 6.2. Tracking thông minh

Không chỉ "thấy người" — SafeAging **theo dõi liên tục**:

- Thuật toán **Hungarian matching** kết hợp IoU + appearance (histogram màu)
- Phân biệt track **confirmed** vs **tentative** — chỉ hiển thị khi đủ tin cậy
- **Re-match** từ lịch sử khi người tạm bị che khuất rồi xuất hiện lại
- Làm mượt khung bao (bbox smoothing) — hình ảnh trên VMS không giật

### 6.3. Nhận diện khuôn mặt — ArcFace / InsightFace

- Engine **InsightFace** (buffalo_s / buffalo_l) — không gửi ảnh lên cloud
- Embedding **512 chiều**, so khớp cosine — nhanh, chính xác
- **Enroll linh hoạt:**
  - Upload nhiều ảnh (batch)
  - Trích frame từ video
  - **Enroll from live track** — nhân viên thấy "Unknown" trên màn hình → bấm Enroll → nhập tên → xong
- Gallery tự refresh — thay đổi hồ sơ có hiệu lực trong vài chục giây

### 6.4. Phát hiện té ngã — đa tín hiệu

Thuật toán té ngã kết hợp nhiều dấu hiệu:

| Tín hiệu | Mô tả |
|----------|-------|
| Vận tốc rơi | Người di chuyển xuống nhanh bất thường |
| Thay đổi góc | Tư thế từ đứng → nằm |
| Tỷ lệ H/W | Khung bao chuyển từ cao → rộng (nằm ngang) |
| Pose (tùy chọn) | Góc thân so với phương đứng ≥ ngưỡng |

- Yêu cầu **xác nhận N khung liên tiếp** (mặc định 3) trước khi báo té ngã
- Giảm false positive từ: ngồi ghế thấp, cúi nhặt đồ, nằm trên giường (khi ROI cấu hình đúng)

### 6.5. Vùng giám sát — linh hoạt theo từng camera

- Vẽ vùng bằng tọa độ pixel trên khung hình camera
- Quản lý qua API hoặc giao diện web
- Mỗi vùng có tên, loại, trạng thái active/inactive
- Logic **stateful** — chỉ fire event khi **vừa mới** vi phạm, không lặp lại

### 6.6. Cấu hình theo camera

Mỗi camera có thể có cấu hình riêng:

- Ngưỡng confidence / IoU
- Frame period (tần suất phân tích)
- ROI (hình chữ nhật hoặc đa giác)
- Danh sách vùng áp dụng

→ Cho phép **camera hành lang** (cảnh báo ra ngoài) và **camera phòng ngủ** (cảnh báo té ngã) dùng rule khác nhau.

---

## 7. Giao diện vận hành

SafeAging cung cấp **Operations Console** — giao diện web tích hợp sẵn, không cần cài thêm phần mềm:

### Tab Dashboard
- Tổng quan số liệu: sự kiện, người, camera, cảnh báo
- Trạng thái hệ thống nhanh

### Tab Live
- Xem **track đang live** trên từng camera
- Thấy ai đang Unknown → **Enroll ngay** từ khung hình
- Hữu ích cho buổi onboarding cư dân mới

### Tab Persons (Quản lý người)
- Danh sách hồ sơ: tên, tuổi, giới, phòng, trạng thái
- Tạo / sửa / xóa hồ sơ
- Upload ảnh khuôn mặt (batch enrollment)
- Xem lịch sử sự kiện theo từng người

### Tab Events (Nhật ký sự kiện)
- Danh sách sự kiện mới nhất
- Lọc theo loại, camera, thời gian
- Xem **ảnh snapshot** tại thời điểm sự kiện
- Liên kết sự kiện với người (manual hoặc ReID auto-link)

### Tab Zones (Quản lý vùng)
- Tạo / sửa / vô hiệu hóa vùng
- Gán vùng cho camera cụ thể

**Ngoài ra:** toàn bộ chức năng có **REST API** (`/admin/*`) để tích hợp với phần mềm quản lý nội bộ, app di động, hoặc dashboard tùy chỉnh.

---

## 8. Mô hình triển khai & hạ tầng

### 8.1. Kiến trúc Edge-First (khuyến nghị)

```
Cơ sở (viện / nhà máy / trường)
├── Camera IP (RTSP)
├── NX Witness Server
├── AI Box (SafeAging chạy tại đây)     ← xử lý AI local
└── Switch mạng nội bộ

Trung tâm (DC / cloud / văn phòng)
├── PostgreSQL (metadata)
├── MinIO (ảnh)
└── SMTP / monitoring
```

**Lợi ích edge:**
- Độ trễ thấp (< 1 giây từ sự cố → cảnh báo)
- Video không ra internet
- Vẫn hoạt động khi mất kết nối WAN

### 8.2. Thiết bị edge đã kiểm chứng

- **AI Box QCS6490** (Qualcomm) — profile production CPU @ 640px
- Docker Compose — triển khai một lệnh
- Hỗ trợ tăng tốc **QNN HTP** (NPU Hexagon) trên roadmap

### 8.3. Tích hợp VMS

- **NX Witness** (Network Optix) — plugin chính thức
- Plugin settings: FPS enqueue, timeout, retry, health poll
- Sự kiện VMS native: detection, prolonged detection, fall, zone violation

### 8.4. Yêu cầu hạ tầng tối thiểu (tham khảo)

| Thành phần | Yêu cầu |
|------------|---------|
| AI Box / Server edge | 4+ CPU cores, 6+ GB RAM (tùy số camera) |
| Camera | RTSP/IP, độ phân giải ≥ 720p khuyến nghị |
| VMS | NX Witness (hoặc tương thích plugin) |
| DB trung tâm | PostgreSQL 16+ |
| Object storage | MinIO hoặc S3-compatible |
| Mạng | LAN 1Gbps nội bộ; WAN cho sync metadata (không bắt buộc realtime) |

### 8.5. Quy mô camera tham khảo

- **AI Box CPU profile:** 2–4 camera @ ~2 FPS inference mỗi camera (cấu hình production đã tune)
- Tăng số camera: thêm AI Box hoặc bật accelerator NPU/GPU
- Mỗi camera độc lập — scale ngang bằng cách thêm node edge

---

## 9. Lợi ích kinh doanh & ROI

### 9.1. Giảm rủi ro pháp lý & uy tín

- Bằng chứng **timestamp + ảnh** khi có khiếu nại
- Chứng minh đã có biện pháp giám sát chủ động
- Audit trail đầy đủ theo từng người

### 9.2. Tối ưu nhân sự

- 1 AI thay thế **hàng chục giờ** xem camera thủ công mỗi tháng
- Nhân viên tập trung **chăm sóc trực tiếp** thay vì ngồi màn hình
- Cảnh báo có lọc — không alarm fatigue

### 9.3. Cải thiện thời gian phản ứng

- Té ngã: từ **phút → giây** (tùy quy trình phản ứng)
- Người ra khỏi khu an toàn: phát hiện ngay tại cửa
- Giảm chi phí điều trị hậu té ngã (complication cost trong y tế cao gấp nhiều lần chi phí phòng ngừa)

### 9.4. Tận dụng đầu tư camera sẵn có

- Không cần thay toàn bộ camera
- Không cần hệ thống AI riêng biệt không tích hợp VMS
- Plugin gắn vào NX Witness đang dùng

### 9.5. Dữ liệu phục vụ quản trị

- Báo cáo té ngã theo khu vực → cải thiện thiết kế không gian (tay vịn, thảm chống trượt)
- Thống kê vi phạm vùng → điều chỉnh quy trình ca trực
- Heatmap hoạt động (roadmap) → tối ưu bố trí nhân sự

---

## 10. Ứng dụng theo ngành & lĩnh vực

SafeAging được thiết kế cho **chăm sóc người cao tuổi**, nhưng lõi công nghệ (phát hiện người + tracking + vùng + nhận diện + cảnh báo) **áp dụng rộng** cho bất kỳ môi trường nào cần giám sát hành vi con người qua camera.

---

### 🏥 10.1. Y tế & Chăm sóc sức khỏe

#### Viện dưỡng lão / Nhà an dưỡng
- Phát hiện té ngã trong phòng, hành lang, phòng tắm
- Cảnh báo cư dân đi vào khu bếp, kho thuốc (vùng cấm)
- Nhận diện cư dân — biết **ai** đang ở đâu
- Enroll nhanh cư dân mới nhập viện
- Báo cáo cho gia đình / quản lý cơ sở

#### Bệnh viện / Phòng khám
- Giám sát bệnh nhân nguy cơ té ngã (khoa lão, khoa chỉnh hình)
- Cảnh báo bệnh nhân rời giường / rời khu điều trị
- Kiểm soát vùng khu cách ly, khu hạn chế
- Đếm người trong phòng chờ — tối ưu luồng khám

#### Trung tâm phục hồi chức năng
- Giám sát bài tập vật lý trị liệu — phát hiện té ngã
- Theo dõi bệnh nhân trong khu vực tập luyện an toàn

#### Nhà trọ chăm sóc tại gia (Home care agency)
- Lắp camera tại nhà người cao tuổi sống một mình
- Gia đình nhận cảnh báo té ngã qua email
- Không cần người trông 24/7

---

### 🏫 10.2. Giáo dục

#### Trường mầm non / Tiểu học
- Đếm số trẻ trong sân chơi — đối chiếu với danh sách lớp
- Phát hiện trẻ **một mình** ở khu vực cô lập (cầu thang, bãi đậu xe)
- Cảnh báo người lạ trong khuôn viên (kết hợp watchlist)

#### Trường trung học / Đại học
- Giám sát khu vực hạn chế (phòng thí nghiệm, kho thiết bị)
- Đếm lưu lượng ra/vào cổng phụ
- Hỗ trợ an ninh campus sau giờ học

#### Trung tâm đào tạo / Dạy nghề
- Giám sát phòng máy nguy hiểm — cảnh báo khi có người không đeo PPE (roadmap)
- Kiểm soát vùng thực hành hàn, cắt

---

### 🏭 10.3. Sản xuất & Công nghiệp

#### Nhà máy sản xuất
- **An toàn lao động:** cảnh báo worker vào vùng máy đang chạy
- Giám sát khu vực nguy hiểm (kho hóa chất, phòng điện cao áp)
- Đếm người trong khu vực sản xuất — đối chiếu ca trực
- Phát hiện té ngã tại khu vực làm việc

#### Kho bãi / Logistics
- Giám sát vùng xe nâng hoạt động — cảnh báo người đi bộ
- Đếm người tại khu vực xuất/nhập hàng
- Kiểm soát vùng restricted (kho lạnh, kho giá trị cao)

#### Nhà máy thực phẩm / Dược phẩm (GMP)
- Kiểm soát vùng sạch — cảnh báo người không có quyền vào
- Audit trail: ai vào khu sản xuất, lúc nào

#### Mỏ / Công trình xây dựng
- Giám sát khu vực đào sâu, giàn giáo
- Phát hiện té ngã tại công trường
- Đếm worker tại khu vực nguy hiểm

---

### 🏨 10.4. Khách sạn & Dịch vụ lưu trú

#### Khách sạn / Resort
- Giám sát khu vực hạn chế (kho, phòng server, tầng hầm)
- An ninh hành lang — phát hiện người lang thang ban đêm
- Nhận diện VIP / nhân viên (enroll gallery)

#### Ký túc xá / Nhà ở xã hội
- Giám sát khu vực chung — phát hiện sự cố
- Kiểm soát vùng cấm (mái, phòng kỹ thuật)

---

### 🛒 10.5. Bán lẻ & Thương mại

#### Siêu thị / Trung tâm thương mại
- Đếm lưu lượng khách theo khu vực
- Giám sát khu vực kho hàng (staff only)
- Phát hiện té ngã tại khu food court, escalator area

#### Ngân hàng / Chi nhánh tài chính
- Giám sát vùng két sắt, phòng server
- Đếm khách trong phòng giao dịch
- Audit: ai có mặt tại khu vực nhạy cảm

---

### 🏛️ 10.6. Công cộng & An ninh

#### Tòa nhà văn phòng / Chung cư cao tầng
- Giám sát sảnh, hầm xe — phát hiện té ngã (người già, trẻ em)
- Cảnh báo người vào khu vực kỹ thuật (tầng mái, phòng điện)
- Đếm người tại lối thoát hiểm — hỗ trợ PCCC

#### Bệnh viện công / Trạm y tế xã
- Giám sát phòng cấp cứu chờ
- Kiểm soát khu vực thuốc gây nghiện

#### Nhà ga / Sân bay (khu vực staff)
- Giám sát khu vực hạn chế (đường băng side, kho hàng)
- Đếm nhân viên tại checkpoint

#### Nhà tù / Trung tâm giữ giam (phạm vi hạn chế)
- Giám sát khu vực sân tập, phòng ăn
- Phát hiện té ngã, ẩu đả (kết hợp pose roadmap)
- *Lưu ý: cần tuân thủ quy định pháp lý riêng về giám sát*

---

### ⛪ 10.7. Tôn giáo & Cộng đồng

#### Viện dưỡng lão tư nhân / Cơ sở từ thiện
- Tương tự mục y tế — giám sát an toàn người cao tuổi
- Chi phí triển khai thấp hơn giải pháp enterprise

#### Trung tâm cộng đồng / CLB người cao tuổi
- Giám sát hoạt động thể dục — phát hiện té ngã
- Đếm người tham dự sự kiện

---

### 🏠 10.8. Gia đình & Smart Home (Premium)

#### Người cao tuổi sống cùng con cháu
- Camera phòng khách / cầu thang — cảnh báo té ngã
- Con cái nhận email khi có sự cố (đi làm xa vẫn yên tâm)

#### Người khuyết tật / Bệnh nhân Alzheimer
- Cảnh báo khi đi ra cửa / ra khỏi nhà
- Vùng cấm: bếp (nguy cơ cháy), cầu thang

---

### 🎪 10.9. Sự kiện & Giải trí

#### Hội trường / Nhà thi đấu
- Đếm số người trong khu vực — kiểm soát sức chứa (fire code)
- Giám sát khu backstage (staff only)

#### Công viên nước / Khu vui chơi
- Phát hiện té ngã tại khu vực trơn trượt
- Giám sát khu vực kỹ thuật (máy bơm, phòng điện)

---

### 🔬 10.10. Nghiên cứu & Thí nghiệm

#### Phòng lab / Viện nghiên cứu
- Kiểm soát vùng lab động vật / mẫu nguy hiểm
- Audit trail ai vào phòng thí nghiệm

#### Bệnh viện thú y / Trại chăn nuôi công nghiệp
- *Mở rộng:* plugin hiện hỗ trợ detect cat/dog — có thể giám sát vùng vật nuôi

---

## 11. Kịch bản sử dụng thực tế

### Kịch bản A — Viện dưỡng lão 80 giường, 24 camera

**Bối cảnh:** Viện có 80 cư dân, 24 camera IP, đang dùng NX Witness. Ban đêm chỉ còn 2 điều dưỡng trực.

**Triển khai SafeAging:**
1. Lắp AI Box tại phòng server viện
2. Cài plugin SafeAging lên NX Witness
3. Enroll khuôn mặt 80 cư dân (ảnh + enroll from live trong 1–2 tuần đầu)
4. Vẽ vùng: cửa ra ngoài (forbidden), cầu thang (forbidden ban đêm), khu bếp (forbidden)
5. Cấu hình email cảnh báo tới điện thoại điều dưỡng trực + quản y

**Kết quả mong đợi:**
- Đêm khuya, cư dân bà Lan (85 tuổi) té ngã tại hành lang tầng 2
- AI phát hiện trong ~2 giây → email + sự kiện VMS
- Điều dưỡng trực chạy tới trong 1 phút
- Sự kiện lưu kèm ảnh + timestamp → quản y báo cáo gia đình có bằng chứng

---

### Kịch bản B — Nhà máy may 500 công nhân

**Bối cảnh:** Nhà máy 3 tầng, cần giám sát khu vực máy cắt (nguy hiểm) và đếm người tại lối thoát hiểm.

**Triển khai:**
1. 8 camera tại khu máy cắt + lối thoát
2. Vùng forbidden quanh máy đang chạy
3. Không cần face recognition — chỉ detection + zone
4. Cảnh báo SMS/email tới tổ trưởng ca

**Kết quả:**
- Công nhân bước vào vùng máy → cảnh báo ngay
- Giảm vi phạm an toàn lao động, có log audit cho ISO 45001

---

### Kịch bản C — Trường mầm non

**Bối cảnh:** 120 trẻ, sân chơi ngoài trời 2 camera, lo lắng trẻ leo rào hoặc đi một mình.

**Triển khai:**
1. Vùng exit tại cổng ra sân chơi
2. Cảnh báo khi có **1 trẻ** (detection count = 1) ở góc sân cô lập > 30 giây (roadmap rule engine)
3. Hiện tại: cảnh báo zone violation khi trẻ ra khỏi vùng sân an toàn

---

### Kịch bản D — Gia đình có bố mẹ già sống một mình

**Bối cảnh:** Bố 78 tuổi sống một mình, con cái ở tỉnh khác.

**Triển khai:**
1. 2 camera: phòng khách + cầu thang
2. Enroll khuôn mặt bố
3. Email cảnh báo té ngã gửi 3 con
4. Không cần VMS phức tạp — có thể dùng RTSP ingest service trực tiếp

---

## 12. So sánh với giải pháp truyền thống

| Tiêu chí | Camera thường + xem thủ công | Cloud AI generic | **SafeAging** |
|----------|------------------------------|------------------|---------------|
| Phát hiện té ngã tự động | ❌ | ✅ (một số) | ✅ |
| Nhận diện "ai" | ❌ | ⚠️ (cloud, phí) | ✅ (local) |
| Tích hợp VMS NX Witness | ❌ | ❌ | ✅ |
| Chạy offline / edge | ✅ (chỉ ghi hình) | ❌ | ✅ |
| Quản lý vùng linh hoạt | ❌ | ⚠️ | ✅ |
| Enroll từ live camera | ❌ | ❌ | ✅ |
| Audit trail + snapshot | ⚠️ (tua video) | ⚠️ | ✅ |
| Dữ liệu ra ngoài nước | — | ⚠️ rủi ro | ✅ local |
| Tùy biến theo ngành | — | ⚠️ | ✅ |

---

## 13. Điểm mạnh cạnh tranh

1. **Tích hợp sâu VMS** — không phải app riêng, nhân viên vẫn xem NX Witness quen thuộc
2. **Edge-first** — AI tại chỗ, bảo mật dữ liệu, không phụ thuộc internet
3. **Resilience** — mất DB vẫn chạy, sync sau; không mất sự kiện
4. **Enroll from live track** — onboarding cư dân/nhân viên cực nhanh, không cần chụp ảnh riêng
5. **Đa tính năng trong một nền tảng** — detection + tracking + face + fall + zone + alert + analytics
6. **Mở API** — tích hợp ERP, app mobile, dashboard tùy chỉnh
7. **Tối ưu phần cứng edge** — profile production cho AI Box QCS6490, roadmap QNN NPU
8. **Mô hình triển khai linh hoạt** — từ viện 100 giường đến gia đình 2 camera

---

## 14. Thông số kỹ thuật tóm tắt

| Hạng mục | Chi tiết |
|----------|----------|
| **Object detection** | YOLO26 (ONNX), class: person (+ cat/dog) |
| **Tracking** | IoU + appearance, Hungarian assignment |
| **Face recognition** | InsightFace ArcFace, 512-d embedding |
| **Fall detection** | Multi-signal bbox + optional YOLO-Pose |
| **Zone types** | forbidden, entry, exit, roi (polygon) |
| **VMS integration** | NX Witness plugin (C++) |
| **AI Service** | Python 3.11, FastAPI, ONNX Runtime |
| **Database** | PostgreSQL 16+ (metadata) |
| **Object storage** | MinIO / S3-compatible |
| **Edge cache** | SQLite (outbox + config) |
| **Alerting** | SMTP email (dedupe + rate limit + retry) |
| **Monitoring** | Prometheus + Grafana, JSON logs |
| **Security** | API Key auth, HTTPS optional, rate limiting |
| **Inference input** | 640×640 px (minimum, configurable) |
| **Supported edge** | AI Box QCS6490 (aarch64), Docker |

---

## 15. Quy trình triển khai điển hình

### Giai đoạn 1 — Khảo sát (1–2 tuần)
- Đếm camera, vị trí, góc quay
- Xác định use case ưu tiên (té ngã? vùng cấm? nhận diện?)
- Kiểm tra VMS hiện tại (NX Witness version)
- Thiết kế sơ đồ vùng giám sát

### Giai đoạn 2 — Pilot (2–4 tuần)
- Triển khai 2–4 camera thí điểm
- Tune ngưỡng fall / zone / face
- Enroll nhóm người pilot
- Đào tạo nhân viên vận hành console

### Giai đoạn 3 — Rollout (4–8 tuần)
- Mở rộng toàn bộ camera
- Enroll toàn bộ hồ sơ
- Cấu hình email cảnh báo theo ca trực
- Tích hợp monitoring

### Giai đoạn 4 — Vận hành & tối ưu (liên tục)
- Review false positive / false negative hàng tháng
- Điều chỉnh ROI, vùng, ngưỡng
- Báo cáo analytics cho ban quản lý

---

## 16. Câu hỏi thường gặp (FAQ)

### SafeAging có thay thế camera không?
**Không.** SafeAging dùng camera IP hiện có. Yêu cầu camera kết nối được với VMS (NX Witness).

### Video có gửi lên cloud không?
**Không bắt buộc.** AI xử lý tại AI Box local. Chỉ metadata và ảnh snapshot sự kiện sync về server trung tâm (có thể đặt nội bộ).

### Nhận diện khuôn mặt có chính xác không?
Độ chính xác phụ thuộc chất lượng ảnh enroll và góc camera. Khuyến nghị enroll ≥ 5 ảnh hoặc dùng enroll from live track. Ngưỡng match có thể tune.

### Báo té ngã có hay báo nhầm không?
Có thể xảy ra khi người **ngồi xuống nhanh** hoặc **nằm trên giường**. Giảm bằng: cấu hình ROI loại trừ giường, tăng `FALL_CONFIRM_FRAMES`, bật pose tier-2.

### Mất internet thì sao?
AI vẫn phân tích và cảnh báo local (sự kiện VMS). Email cần SMTP reachable. Sự kiện queue trong SQLite, sync DB khi mạng về.

### Cần GPU NVIDIA không?
**Không.** Production profile chạy CPU trên AI Box ARM. Roadmap hỗ trợ Qualcomm NPU (QNN).

### Tích hợp được với VMS khác ngoài NX Witness không?
Hiện tại plugin chính thức cho **NX Witness**. Có thể tích hợp camera RTSP trực tiếp qua RTSP ingest service cho pilot hoặc môi trường không dùng NX.

### Bao nhiêu camera trên một AI Box?
Profile CPU production: **2–4 camera** @ ~2 FPS inference. Tăng camera → thêm AI Box hoặc bật NPU.

### Dữ liệu cá nhân (khuôn mặt) có tuân thủ quy định không?
Embedding lưu local/on-premise. Khách hàng kiểm soát retention (mặc định 30 ngày sự kiện). Cần thông báo và đồng ý theo quy định địa phương (GDPR, Luật ANM Việt Nam…).

### Có hỗ trợ tiếng Việt không?
Giao diện console và API hỗ trợ trường tiếng Việt (ngày sinh, ghi chú). Tài liệu bán hàng và đào tạo có thể cung cấp bản tiếng Việt đầy đủ.

### Giá bao nhiêu?
*(Điền thông tin giá / gói dịch vụ của đội kinh doanh tại đây)*

---

## 17. Lộ trình phát triển

Các tính năng **đã có** được mô tả ở các mục trên. Lộ trình tiếp theo:

| Giai đoạn | Tính năng dự kiến |
|-----------|-------------------|
| **QNN / NPU** | Tăng tốc YOLO trên Hexagon HTP — nhiều camera hơn, latency thấp hơn |
| **Pose nâng cao** | Phát hiện té ngã chính xác hơn, phân tích tư thế ngồi/đứng |
| **SMS / Push** | Cảnh báo qua Zalo, SMS, mobile push |
| **Rule engine** | Quy tắc tùy chỉnh: "1 người đứng > 5 phút tại vùng X" |
| **Heatmap** | Bản đồ nhiệt hoạt động theo thời gian |
| **PPE detection** | Phát hiện không đeo mũ/áo bảo hộ (nhà máy) |
| **Multi-site** | Quản lý tập trung nhiều cơ sở từ một dashboard |

---

## 18. Bước tiếp theo — liên hệ & demo

### SafeAging mang lại gì cho khách hàng?

✅ An tâm hơn — sự cố được phát hiện sớm, có người phản ứng  
✅ Minh bạch hơn — mọi sự kiện có bằng chứng, có thời gian  
✅ Hiệu quả hơn — nhân sự tập trung chăm sóc, không ngồi xem camera  
✅ Linh hoạt hơn — từ viện dưỡng lão đến nhà máy, trường học, gia đình  

### Đề xuất demo

1. **Demo live 30 phút** — xem nhận diện + té ngã + vùng cấm trên camera thật
2. **Pilot 2 camera / 30 ngày** — đo false positive rate tại cơ sở khách hàng
3. **Báo giá** theo gói: số camera, có/không face recognition, hỗ trợ triển khai

---

### Thông tin liên hệ

| | |
|---|---|
| **Sản phẩm** | SafeAging — YOLO26 People Analytics |
| **Website** | *(điền URL)* |
| **Email** | *(điền email sales)* |
| **Hotline** | *(điền SĐT)* |
| **Demo** | *(điền link đặt lịch)* |

---

*Tài liệu này mô tả khả năng hệ thống SafeAging dựa trên phiên bản phần mềm hiện tại. Một số tính năng roadmap có thể thay đổi theo lộ trình phát triển. Vui lòng liên hệ đội SafeAging để nhận bản cập nhật mới nhất.*

**© 2026 SafeAging. All rights reserved.**
