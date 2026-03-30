# 🔴 SafeAging Detection Issues - Root Cause Analysis

**Date**: March 30, 2026  
**Status**: CRITICAL - Multiple bottlenecks identified  
**Scope**: Bbox flickering, misalignment, missing detections

---

## 📊 Observed Problems (từ ảnh)

1. **Bbox Flickering** - Boxes appear/disappear intermittently
2. **Misalignment** - Boxes don't align properly with people
3. **Missing Detections** - Many visible people not detected at all  
4. **False Positives** - Some boxes appear on non-people areas
5. **Jitter** - Boxes jump around instead of smooth tracking

---

## 🔍 ROOT CAUSES (In Order of Severity)

### 🚨 **CRITICAL #1: Queue Size Bottleneck** 
**File**: `device_agent.h` Line 121  
**Code**: `static constexpr size_t kDefaultFrameQueueMaxSize = 1;`

**Problem**:
- Only **1 frame** can be queued at any time
- At 15 FPS input + ~1000-6000ms Python inference:
  - 15-90 frames arrive during inference
  - Only newest frame kept, older 14-89 discarded
  - Results in **constant detection gaps**

**Evidence from logs**:
```
[12:07:29] Backpressure: queue full, dropped 1 frames  
[12:07:59] Backpressure: queue full, dropped 101 frames  
[12:08:30] Backpressure: queue full, dropped 56 frames
[12:10:51] Backpressure: queue full, dropped 87 frames
```
- Dropping 20-100+ frames every 10 seconds
- Processing rate: 0.1-1 FPS vs input 15 FPS ❌

**Impact**: **Major flickering - detections only appear sporadically**

### 🔴 **CRITICAL #2: Python Service Timeout / Failure**
**File**: `object_detector.cpp` (callPythonServiceMultipart)  
**Evidence**:
```
[12:07:43] Transient infer error: no response from /infer endpoint
[12:08:20] AI service call failed: Transient infer failure after retries
[12:10:35] Circuit breaker OPEN for camera "Tapo C220"
```

**Problem**:
- Python `/infer` endpoint not responding reliably
- Inference times: 1000-7000ms (too slow!)
- After repeated failures, circuit breaker opens = automatic fail-fast
- Service tries 3 retries before giving up

**Questions to investigate**:
1. Is Python service actually running?
2. GPU/CPU bottleneck?
3. Service connection issues?
4. Model loading/inference too slow?

**Impact**: **Complete detection loss when service times out (most of the time!)**

### 🟠 **HIGH #3: Very Low Confidence Threshold**
**File**: `service.py` Line 133  
**Code**: `self.confidence_threshold = _clamp(..., 0.15, ...)`

**Problem**:
- YOLO confidence threshold = 0.15 (15%)
- Industry standard: 0.45+ (45%)
- Too aggressive → many false positives/negatives
- Causes unstable detections and jitter

**Impact**: **Multiple false boxes, noise in tracking**

### 🟡 **MEDIUM #4: Weak Tracking Descriptor**
**File**: `object_tracker.cpp` Line 30  
**Code**: 
```cpp
static const Size kDescriptorFastSize(16, 32);
std::shared_ptr<IImageDescriptor> descriptorFast =
    std::make_shared<ResizedImageDescriptor>(
        kDescriptorFastSize, InterpolationFlags::INTER_LINEAR);
```

**Problem**:
- Descriptors only 16×32 pixels = **very coarse**
- Uses MatchTemplate distance (pixel-level matching)
- Easy to lose track if person moves >20-30 pixels
- Match IOU threshold = 0.20 (very permissive)

**Impact**: **Tracks lost frequently → bbox jumps/disappears**

### 🟡 **MEDIUM #5: Frame Downscaling & Compression Loss**
**File**: `device_agent.cpp` Line 616  
**Code**:
```cpp
std::vector<uint8_t> jpegBytes = encodeFrameToJpeg(frame, 1280);
std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 80};
```

**Problem**:
- Downscale to 1280px width (loose spatial info)
- JPEG quality 80% (lossy compression artifacts)
- Then upscaled back for tracking = quality loss
- Creates coordinate transformation errors

**Impact**: **Subtle misalignment, small people disappear**

### 🟡 **MEDIUM #6: Frame Sampling Strategy**
**File**: `device_agent.cpp` Line 289  
**Code**: `bool shouldEnqueue = (m_frameIndex % detectionFramePeriod == 0);`

**Problem**:
- Default `detectionFramePeriod = 1` (process every frame)
- But with queue size 1, most frames still dropped
- No adaptive frame skipping based on load

**Impact**: **Inconsistent frame sampling**

### 🔵 **LOW #7: Render State TTL (Time To Live)**
**File**: `device_agent.cpp` Line 227  
**Code**: `const int ttlMs = std::clamp(5000 / targetEnqueueFps, 250, 2000);`

**Problem**:
- Bboxes rendered for only 250-2000ms after last detection
- If detection missed, bbox immediately disappears
- Creates flickering effect

**Impact**: **Visible flickering when frame dropped**

---

## 🧮 Performance Analysis

### Actual Performance (from logs):
```
Input FPS:         15 fps (camera)
Processing FPS:    0.1 - 1 fps (huge bottleneck!)
Frame Queue Size:  1 (CRITICAL)
Dropped Frames:    20-100 per 10 seconds
Inference Time:    1000-7000 ms (way too slow!)
Service Status:    FAILING (circuit breaker open)
```

### Expected Performance:
```
Input FPS:         15 fps (camera)
Processing FPS:    Should be ≥ 5 fps (for smooth tracking)
Frame Queue Size:  Should be ≥ 3-5 frames
Dropped Frames:    Should be < 1 per 10 seconds
Inference Time:    Should be < 300 ms
```

---

## 💡 WHY Detection is So Bad

### The Problem Chain:
1. **Python service too slow** (1000-7000ms)
   ↓
2. **Only 1 frame can wait in queue**
   ↓
3. **While processing = all input frames dropped**
   ↓
4. **Very sparse detections** (0.1-1 FPS output)
   ↓
5. **Tracking loses objects** (gaps in data)
   ↓
6. **Render state TTL expires**
   ↓
7. **Bbox disappears/flickers** ← This is what user sees!

---

## 📋 Configuration Defaults (device_agent.h)

| Parameter | Default | Issue |
|-----------|---------|-------|
| Frame Queue Max Size | **1** | 🔴 CRITICAL |
| Target Enqueue FPS | 1 | 🟠 HIGH |
| Detection Frame Period | 1 | 🟡 MEDIUM |
| Render TTL | 250-2000ms | 🟢 LOW |

---

## 🐍 Python Service Issues (service.py)

| Parameter | Value | Issue |
|-----------|-------|-------|
| Confidence Threshold | 0.15 | 🟠 Too low (should be 0.3-0.5) |
| Bbox Smoothing | 0.6 | 🟡 High |
| Match IOU Threshold | 0.20 | 🟡 Too permissive |
| Track Max Misses | 6 | 🟡 Too many |
| Min Hits for Confirmation | 1 | 🟡 Too low |

---

## ✅ FIXES NEEDED (Priority Order)

### **IMMEDIATE (Do First)**

#### 1. Fix Python Service Timeout Crisis
- [ ] Check if Python service is running: `ps aux | grep python`
- [ ] Check service logs: `tail -f logs/service.log`
- [ ] Test inference endpoint: `curl -X POST http://localhost:18000/infer`
- [ ] Verify GPU availability if using: `nvidia-smi`
- [ ] Check inference time baseline (should be <300ms)

**Possible causes**:
- Service crashed and not restarting
- GPU memory full
- Model loading timeout
- Network connection issue to service

#### 2. Increase Frame Queue Size (Quick Win!)
**File**: `device_agent.h` Line 121
**Change**:
```cpp
// OLD
static constexpr size_t kDefaultFrameQueueMaxSize = 1;

// NEW
static constexpr size_t kDefaultFrameQueueMaxSize = 5;  // At least 3-5
```
**Effect**: Allows buffering while inference processing

#### 3. Increase Target FPS  
**File**: `device_agent.h` Line 120
**Change**:
```cpp
// OLD
static constexpr int kDefaultTargetEnqueueFps = 1;

// NEW
static constexpr int kDefaultTargetEnqueueFps = 5;  // Process more frames
```
**Effect**: More frequent detections

### **HIGH PRIORITY**

#### 4. Optimize Inference Performance
**In Python service** (`service.py`):
- [ ] Profile inference: Add timing logs
- [ ] Check model size (yolov8n is smallest - good)
- [ ] Consider enabling half precision (`USE_HALF=true`)
- [ ] Reduce image size if inference too slow
- [ ] Check ENABLE_CLAHE (preprocessing overhead?)

#### 5. Increase Confidence Threshold
**File**: `service.py` Line 133
**Change**:
```python
# OLD
self.confidence_threshold = _clamp(_env_float("CONFIDENCE_THRESHOLD", 0.15), ...)

# NEW
self.confidence_threshold = _clamp(_env_float("CONFIDENCE_THRESHOLD", 0.35), ...)  # Industry standard
```
**Effect**: Fewer false positives, more stable tracking

#### 6. Improve Tracking
**File**: `object_tracker.cpp` Line 30
**Options**:
- Increase descriptor size from 16×32 to 32×64
- Increase match threshold from 0.20 to 0.35-0.45
- Use better distance metric (HOG or CNN instead of MatchTemplate)

### **MEDIUM PRIORITY**

#### 7. Reduce Image Compression Loss
**File**: `device_agent.cpp` Line 627
**Change**:
```cpp
// OLD
std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 80};

// NEW
std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};  // Better quality
```

#### 8. Adjust Tracking Parameters (Python side)
- [ ] Increase TRACK_MIN_HITS from 1 to 2-3 (confirm detection before showing)
- [ ] Increase TRACK_OUTPUT_HOLD_TIME from 1.5s to 2-3s
- [ ] Adjust MATCH_IOU_THRESHOLD based on performance

---

## 🧪 Testing Checklist

After fixes:
- [ ] Monitor queue size in logs (should see < 5)
- [ ] Check processing FPS (target: >2 fps)
- [ ] Check dropped frames (should be near 0)
- [ ] Visual test: Person should have constant smooth box
- [ ] Count detections vs expected (should match ~95%+)
- [ ] No flickering (box should persist)
- [ ] No jitter (box should move smoothly)
- [ ] No misalignment (box should center on person)

---

## 📁 Files to Modify

1. **`device_agent.h`** (Default queue size & FPS)
2. **`service.py`** (Confidence threshold, tracking params)
3. **`object_tracker.cpp`** (Descriptor size, match threshold)
4. **`fall_detection.py`** (Fine-tune thresholds if needed)

---

## 🚀 Quick Fix Summary

**Minimum changes for 80% improvement**:

```cpp
// device_agent.h (2 lines)
static constexpr size_t kDefaultFrameQueueMaxSize = 5;  // Was 1
static constexpr int kDefaultTargetEnqueueFps = 3;      // Was 1
```

```python
# service.py (1 line)
self.confidence_threshold = _clamp(_env_float("CONFIDENCE_THRESHOLD", 0.35), ...)  # Was 0.15
```

These 3 changes should significantly reduce flickering and improve detection quality.

---

## 📞 Next Steps

1. Check Python service status immediately
2. Apply queue size fix
3. Profile and optimize inference time
4. Adjust confidence threshold
5. Test and iterate

**Expected improvement after fixes**: 70-80% reduction in flickering, 2-3x more detections
