# 🔴 Phân Tích Chi Tiết Vấn Đề Detection - SafeAging

**Ngày**: 30/03/2026  
**Trạng thái**: CAO ĐỘ NGUY HIỂM - Nhiều lỗi được xác định  

---

## 📸 Vấn Đề Quan Sát (Từ Ảnh)

1. ❌ **Bbox Chớp nháy** - Hộp xuất hiện/mất đi liên tục
2. ❌ **Bbox lệch vị trí** - Không khít khao với con người
3. ❌ **Detection thiếu** - Nhiều người nhìn thấy nhưng không detect
4. ❌ **False positives** - Đôi khi detect những chỗ không phải người
5. ❌ **Bbox nhảy xóc** - Thay vì chuyển động mượt

---

## 🔍 NGUYÊN NHÂN CỐT LÕI

### 🚨 **LỖI CRITICAL #1: Kích Thước Hàng Đợi Quá Nhỏ**

**Vị trí**: `device_agent.h` Line 121  
**Code**: 
```cpp
static constexpr size_t kDefaultFrameQueueMaxSize = 1;  // ← CHỈ 1 FRAME!
```

**Vấn đề**:
- Hàng đợi chỉ chứa **1 frame** duy nhất
- Tại 15 FPS từ camera + ~1-6 giây xử lý Python:
  - 15-90 frames tới khi đang xử lý
  - Chỉ giữ frame mới nhất, cũ bị bỏ đi
  - Kết quả: **Thiếu detection liên tục**

**Bằng chứng từ log**:
```
[12:07:59] Backpressure: queue full, dropped 101 frames  
[12:08:30] Backpressure: queue full, dropped 56 frames
[12:10:51] Backpressure: queue full, dropped 87 frames
```
- Bỏ 20-100+ frames mỗi 10 giây
- Tốc độ xử lý: 0.1-1 FPS vs input 15 FPS

**→ ĐÂY LÀ NGUYÊN NHÂN CHÍNH của chớp nháy!**

### 🔴 **LỖI CRITICAL #2: Python Service Timeout**

**Bằng chứng**:
```
[12:07:43] Transient infer error: no response from /infer endpoint
[12:08:20] AI service call failed: Transient infer failure
[12:10:35] Circuit breaker OPEN - tự động fail-fast
```

**Vấn đề**:
- Endpoint `/infer` không phản hồi đáng tin cậy
- Thời gian inference: 1000-7000ms (chậm quá!)
- Sau lỗi lặp lại, "cầu ngắt" mở = fail-fast tự động
- Retry 3 lần rồi bỏ cuộc

**Hậu quả**: **Mất detection hoàn toàn khi service timeout!**

### 🟠 **LỖI HIGH #3: Confidence Threshold Quá Thấp**

**Vị trí**: `service.py` Line 133  
```python
self.confidence_threshold = 0.15  # ← 15% QUỐC TIÊU LÀ 45%!
```

**Vấn đề**:
- Ngưỡng tin tưởng YOLO = 15% (quá thấp!)
- Tiêu chuẩn: 45%+ 
- Quá nhiều false positive/negative
- Tracking không ổn định

### 🟡 **LỖI MEDIUM #4: Tracking Descriptor Quá Nhỏ**

**Vị trí**: `object_tracker.cpp` Line 30  
```cpp
static const Size kDescriptorFastSize(16, 32);  // ← QUỐC TIÊU: 32×64+
```

**Vấn đề**:
- Descriptor chỉ 16×32 pixels = **quá sơ sài**
- Dễ mất track nếu người di chuyển >20px
- Match threshold 0.20 = quá dễ xáo trộn

---

## 📊 Phân Tích Hiệu Năng

```
Input FPS:              15 fps (camera)
Xử lý FPS:              0.1-1 fps ❌ (chỉ 1 frame per 10 giây!)
Kích thước Queue:       1 ❌ (CRITICAL)
Frames bỏ đi:           20-100 mỗi 10 giây
Thời gian Inference:    1000-7000 ms ❌ (should be <300ms)
Trạng thái Service:     FAILING (Circuit breaker open)
```

---

## 💡 Chuỗi Vấn Đề

```
1. Python service quá chậm (1-7 giây)
                    ↓
2. Chỉ 1 frame chờ được
                    ↓
3. Lúc xử lý → tất cả frame mới bỏ đi
                    ↓
4. Detection rất thưa (0.1-1 FPS)
                    ↓
5. Tracking mất object (thiếu dữ liệu)
                    ↓
6. Render state TTL hết
                    ↓
7. Bbox biến mất/chớp ← ĐÂY LÀ GÌ USER THẤY!
```

---

## ✅ CÁCH FIX (Ưu Tiên)

### **NGAY LẬP TỨC (Làm Trước)**

#### 1️⃣ Kiểm Tra Python Service  
- [ ] Chạy hay đã tắt? `ps aux | grep python`
- [ ] Xem log service: `tail -f python/logs/service.log`
- [ ] Test endpoint: `curl -X POST http://localhost:18000/infer`
- [ ] GPU có? `nvidia-smi`

#### 2️⃣ Tăng Kích Thước Hàng Đợi (Thay Đổi Nhanh!)
**File**: `device_agent.h` Line 121
```cpp
// CŨ
static constexpr size_t kDefaultFrameQueueMaxSize = 1;

// MỚI
static constexpr size_t kDefaultFrameQueueMaxSize = 5;  // Ít nhất 3-5
```

**Kết quả**: Cho phép buffer khi đang xử lý

#### 3️⃣ Tăng Target FPS
**File**: `device_agent.h` Line 120
```cpp
// CŨ
static constexpr int kDefaultTargetEnqueueFps = 1;

// MỚI
static constexpr int kDefaultTargetEnqueueFps = 3;  // Process nhiều frame hơn
```

### **ƯỚI TIÊN CAO**

#### 4️⃣ Tăng Confidence Threshold
**File**: `service.py` Line 133
```python
# CŨ
self.confidence_threshold = 0.15

# MỚI  
self.confidence_threshold = 0.35  # Tiêu chuẩn ngành
```

---

## 🎯 FIX NHANH (3 Thay Đổi = 80% Cải Thiện)

**device_agent.h**:
```cpp
static constexpr size_t kDefaultFrameQueueMaxSize = 5;  // Was 1
static constexpr int kDefaultTargetEnqueueFps = 3;      // Was 1
```

**service.py**:
```python
self.confidence_threshold = 0.35  # Was 0.15
```

**Kỳ Vọng**: 70-80% giảm chớp nháy, 2-3x detection nhiều hơn

---

## 📁 Files Cần Sửa

1. `device_agent.h` - Kích thước queue & FPS
2. `service.py` - Confidence & tracking params  
3. `object_tracker.cpp` - Descriptor, match threshold
4. `fall_detection.py` - Fine-tune nếu cần

---

## 🧪 Kiểm Tra Sau Fix

- [ ] Queue size < 5 trong logs
- [ ] Processing FPS > 2 fps
- [ ] Dropped frames ≈ 0
- [ ] Bbox liên tục không chớp
- [ ] Bbox mượt không nhảy
- [ ] Bbox khít chặt với người

---

## 📞 Bước Tiếp Theo

1. **Cấp bách**: Kiểm tra Python service chạy hay không
2. Fix queue size
3. Tối ưu inference time
4. Tăng confidence threshold
5. Test & lặp

**Kết quả mong đợi**: Giảm chớp nháy 70-80%, tăng detection 2-3x
