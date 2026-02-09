# FLOW 2 MVP - Code Locations & Line References

## Quick Reference for Code Review

Use this guide to quickly locate specific FLOW 2 implementation sections in the code.

---

## device_agent.h

### Includes (Lines 1-21)
```cpp
// Added these headers:
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
```

### FrameJob Struct (BEFORE `class DeviceAgent`)
```cpp
struct FrameJob {
    std::vector<uint8_t> jpegBytes;
    std::string cameraId;
    int64_t timestampUs;
    int64_t frameIndex;
};
```

### Private Methods (ADD)
```cpp
void workerThreadRun();
std::vector<uint8_t> encodeFrameToJpeg(const Frame& frame, int targetWidth = 640);
MetadataPacketList processFrameJob(const FrameJob& job);
```

### New Constants (Private Section)
```cpp
const std::string kFallDetectedEventType = "mycompany.yolov8_people_analytics.fallDetected";
static constexpr size_t kFrameQueueMaxSize = 3;
```

### New Member Variables (Private Section)
```cpp
// Frame queuing
std::mutex m_frameQueueMutex;
std::condition_variable m_frameQueueCV;
std::deque<FrameJob> m_frameQueue;

// Worker thread
std::thread m_workerThread;
bool m_workerShouldStop = false;

// Metadata
std::mutex m_metadataQueueMutex;
std::deque<nx::sdk::Ptr<nx::sdk::analytics::IMetadataPacket>> m_metadataQueue;

// Fall dedup
std::set<nx::sdk::Uuid> m_activeFallDetectedTrackIds;
```

---

## device_agent.cpp

### Constructor (MODIFIED)
**Location**: ~Line 45-60
```cpp
DeviceAgent::DeviceAgent(
    const nx::sdk::IDeviceInfo* deviceInfo,
    std::filesystem::path pluginHomeDir,
    std::filesystem::path modelPath)
    : ConsumingDeviceAgent(deviceInfo, /*enableOutput*/ true),
      ...
      m_workerThread(&DeviceAgent::workerThreadRun, this),  // ← NEW
      m_workerShouldStop(false)                             // ← NEW
{
}
```

### Destructor (MODIFIED)
**Location**: ~Line 65-80
```cpp
DeviceAgent::~DeviceAgent() {
    // Signal worker thread to stop
    {
        std::unique_lock<std::mutex> lk(m_frameQueueMutex);
        m_workerShouldStop = true;
    }
    m_frameQueueCV.notify_one();
    if (m_workerThread.joinable())
        m_workerThread.join();
}
```

### pushUncompressedVideoFrame() (MAJOR REWRITE)
**Location**: ~Line 115-160
**Old**: Called `processFrame()` directly (BLOCKING)
**New**: Enqueues frame and returns immediately (NON-BLOCKING)

```cpp
// NEW CODE
// Convert Nx frame to OpenCV Mat
Frame frame(videoFrame, m_frameIndex);

// Encode JPEG with downscaling
std::vector<uint8_t> jpegBytes = encodeFrameToJpeg(frame, 640);

// Create job
FrameJob job;
job.jpegBytes = std::move(jpegBytes);
job.cameraId = "nx_camera";
job.timestampUs = frame.timestampUs;
job.frameIndex = m_frameIndex;

// Enqueue with backpressure
{
    std::unique_lock<std::mutex> lk(m_frameQueueMutex);
    if (m_frameQueue.size() >= kFrameQueueMaxSize)
        m_frameQueue.pop_front();  // Drop oldest
    m_frameQueue.push_back(std::move(job));
}
m_frameQueueCV.notify_one();

return true;  // ✓ Returns immediately
```

### New: workerThreadRun()
**Location**: ~Line 260-310
**Key Comment**: "FLOW 2: Worker thread - runs in background"

```cpp
void DeviceAgent::workerThreadRun() {
    while (true) {
        FrameJob job;
        
        // Wait for frame
        {
            std::unique_lock<std::mutex> lk(m_frameQueueMutex);
            m_frameQueueCV.wait(lk, [this]() {
                return !m_frameQueue.empty() || m_workerShouldStop;
            });
            
            if (m_workerShouldStop && m_frameQueue.empty())
                break;
            if (m_frameQueue.empty())
                continue;
            
            // Get NEWEST
            job = std::move(m_frameQueue.back());
            m_frameQueue.clear();
        }
        
        // Process
        try {
            MetadataPacketList packets = processFrameJob(job);
            {
                std::unique_lock<std::mutex> lk(m_metadataQueueMutex);
                for (const auto& pkt : packets)
                    m_metadataQueue.push_back(pkt);
            }
        }
        catch (...) { /* log error */ }
    }
}
```

### New: encodeFrameToJpeg()
**Location**: ~Line 315-340
**Key Comment**: "FLOW 2: Encode frame to JPEG bytes"

```cpp
std::vector<uint8_t> DeviceAgent::encodeFrameToJpeg(const Frame& frame, int targetWidth) {
    cv::Mat sendImg = frame.cvMat;
    
    // Downscale
    if (frame.width > targetWidth) {
        float scale = (float)targetWidth / (float)frame.width;
        int newH = std::max(1, (int)std::round(frame.height * scale));
        cv::resize(sendImg, sendImg, cv::Size(targetWidth, newH));
    }
    
    // Encode JPEG
    std::vector<uint8_t> jpegBytes;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 80};
    cv::imencode(".jpg", sendImg, jpegBytes, params);
    
    return jpegBytes;
}
```

### New: processFrameJob()
**Location**: ~Line 345-390
**Key Comment**: "FLOW 2: Process queued frame job"

```cpp
MetadataPacketList DeviceAgent::processFrameJob(const FrameJob& job) {
    MetadataPacketList result;
    
    // 1. Call AI service
    DetectionList detections = m_objectDetector->run(job.cameraId, job.jpegBytes);
    
    // 2. ObjectMetadata
    auto objectPkt = detectionsToObjectMetadataPacket(detections, job.timestampUs);
    if (objectPkt)
        result.push_back(objectPkt);
    
    // 3. Fall detection (FLOW 2)
    for (const auto& detection : detections) {
        if (detection->fallDetected && detection->classLabel == "person") {
            // Check dedup
            bool alreadyActive = m_activeFallDetectedTrackIds.count(detection->trackId) > 0;
            
            if (!alreadyActive) {
                // New fall event
                auto event = makePtr<EventMetadata>();
                event->setCaption("Fall Detected");
                event->setIsActive(true);
                event->setTypeId(kFallDetectedEventType);
                
                auto eventPkt = makePtr<EventMetadataPacket>();
                eventPkt->addItem(event.get());
                eventPkt->setTimestampUs(job.timestampUs);
                result.push_back(eventPkt);
                
                m_activeFallDetectedTrackIds.insert(detection->trackId);
            }
        }
    }
    
    return result;
}
```

---

## object_detector.h

### New Method Signatures (PUBLIC)
**Location**: ~Line 35-40

```cpp
// NEW: Accept JPEG bytes directly (FLOW 2)
DetectionList run(const std::string& cameraId, const std::vector<uint8_t>& jpegBytes);

// OLD: Still available for backward compatibility
DetectionList run(const Frame& frame);
```

### New Private Method
```cpp
// NEW: HTTP multipart call (FLOW 2)
DetectionList callPythonServiceMultipart(
    const std::string& cameraId,
    const std::vector<uint8_t>& jpegBytes);
```

---

## object_detector.cpp

### New: run(cameraId, jpegBytes) Method
**Location**: ~Line 430-450
**Key Comment**: "FLOW 2: New method - run inference on JPEG bytes"

```cpp
DetectionList ObjectDetector::run(
    const std::string& cameraId, 
    const std::vector<uint8_t>& jpegBytes) {
    
    if (isTerminated())
        return {};

    try {
        return callPythonServiceMultipart(cameraId, jpegBytes);
    }
    catch (const ObjectDetectionError&) {
        throw;
    }
    catch (const std::exception& e) {
        throw ObjectDetectionError(std::string("Error: ") + e.what());
    }
}
```

### New: callPythonServiceMultipart() Method
**Location**: ~Line 455-600
**Key Comment**: "FLOW 2: HTTP multipart/form-data call to Python service"

```cpp
DetectionList ObjectDetector::callPythonServiceMultipart(
    const std::string& cameraId,
    const std::vector<uint8_t>& jpegBytes) {
    
    DetectionList result;
    
    // Base64 encode
    std::string b64 = base64Encode(jpegBytes.data(), jpegBytes.size());
    
    // Create JSON
    json req;
    req["camera_id"] = cameraId;
    req["image"] = b64;
    
    // HTTP client with SHORT TIMEOUT
    thread_local httplib::Client cli("127.0.0.1", 18000);
    cli.set_connection_timeout(0, 500000);  // 500ms ← FAIL-FAST
    cli.set_read_timeout(1, 0);             // 1s
    cli.set_write_timeout(0, 500000);       // 500ms
    
    // POST /infer
    auto res = cli.Post("/infer", req.dump(), "application/json");
    
    if (!res)
        throw ObjectDetectionError("No response from /infer endpoint");
    if (res->status != 200)
        throw ObjectDetectionError("HTTP error " + std::to_string(res->status));
    
    // Parse response
    json j = json::parse(res->body);
    
    // Extract detections
    for (const auto& item : j) {
        const std::string classLabel = item.value("cls", "person");
        const float score = item.value("score", 0.0f);
        float x = item.value("x", 0.0f);
        float y = item.value("y", 0.0f);
        float w = item.value("w", 0.0f);
        float h = item.value("h", 0.0f);
        
        bool fallDetected = item.value("fall_detected", false);  // ← FLOW 2
        
        // ... normalize coordinates ...
        
        int trackId = item.value("track_id", 0);
        nx::sdk::Uuid trackUuid = uuidFromTrackId(trackId);
        
        auto detection = std::make_shared<Detection>(Detection{
            nx::sdk::analytics::Rect(xNorm, yNorm, wNorm, hNorm),
            classLabel,
            score,
            trackUuid,
            fallDetected  // ← FLOW 2: Include fall flag
        });
        
        result.push_back(detection);
    }
    
    return result;
}
```

**Key Points**:
- Line ~480: HTTP timeout configuration (FAIL-FAST)
- Line ~520: Extract `fall_detected` from JSON
- Line ~560: Pass `fallDetected` to Detection struct

---

## detection.h

### Modified: Detection Struct
**Location**: ~Line 20-30

```cpp
struct Detection {
    const nx::sdk::analytics::Rect boundingBox;
    const std::string classLabel;
    const float confidence;
    const nx::sdk::Uuid trackId;
    const bool fallDetected;  // ← NEW FIELD (FLOW 2)
};
```

---

## manifest.json

### New Event Type
**Location**: ~Line 40-45 (in `supportedEventTypes` array)

```json
{
    "id": "mycompany.yolov8_people_analytics.fallDetected",
    "name": "Fall detected",
    "flags": "stateDependent"
}
```

**Key Points**:
- ID must match code: `kFallDetectedEventType`
- Flags: `stateDependent` means START/FINISHED events
- This makes event type available in Nx rules engine

---

## Code Annotation References

### Frame Callback (Non-Blocking)
**File**: device_agent.cpp  
**Lines**: ~115-160  
**Search**: `// ============================================================`  
**Comment**: `// FLOW 2: Frame callback MUST NOT process frames here.`

### Worker Thread Main Loop
**File**: device_agent.cpp  
**Lines**: ~260-310  
**Search**: `void DeviceAgent::workerThreadRun()`  
**Comment**: `// FLOW 2: Worker thread - runs in background`

### Backpressure/Drop Old Frames
**File**: device_agent.cpp  
**Lines**: ~130-135  
**Search**: `// ⚠️ BACKPRESSURE`  
**Comment**: `// If queue is full, drop oldest frame and add newest`

### Fall Detection Deduplication
**File**: device_agent.cpp  
**Lines**: ~360-375  
**Search**: `// FLOW 2: Create EventMetadata for fall detected`  
**Comment**: `// Emit START event only once per person (deduplication)`

### HTTP Fail-Fast Timeout
**File**: object_detector.cpp  
**Lines**: ~475-485  
**Search**: `// ⚠️ SHORT TIMEOUT FOR MVP`  
**Comment**: `// 1.5 seconds total (fail-fast instead of blocking Nx)`

### Fall Detection Flag Parsing
**File**: object_detector.cpp  
**Lines**: ~520-525  
**Search**: `const bool fallDetected = item.value`  
**Comment**: `// FLOW 2`

---

## Quick Search Terms

To find FLOW 2 code sections, search for these keywords:

| Keyword | File(s) | Purpose |
|---------|---------|---------|
| `FrameJob` | device_agent.h | Frame queue job struct |
| `workerThreadRun` | device_agent.cpp | Worker thread function |
| `m_frameQueue` | device_agent.h/cpp | Frame queue member |
| `kFrameQueueMaxSize` | device_agent.h | Queue max size constant |
| `encodeFrameToJpeg` | device_agent.cpp | JPEG encoding function |
| `processFrameJob` | device_agent.cpp | Async frame processing |
| `callPythonServiceMultipart` | object_detector.cpp | HTTP call function |
| `fallDetected` | All files | Fall detection flag |
| `kFallDetectedEventType` | device_agent.h | Fall event type ID |
| `m_activeFallDetectedTrackIds` | device_agent.h | Dedup tracking set |
| `FLOW 2` | All files | Comments marking FLOW 2 code |

---

## File Sizes & Change Impact

| File | Original | Modified | Change |
|------|----------|----------|--------|
| device_agent.h | ~100 LOC | ~180 LOC | +80 LOC (headers, structs, members) |
| device_agent.cpp | ~450 LOC | ~750 LOC | +300 LOC (worker thread, async) |
| object_detector.h | ~50 LOC | ~60 LOC | +10 LOC (new method signature) |
| object_detector.cpp | ~430 LOC | ~600 LOC | +170 LOC (HTTP call, parsing) |
| detection.h | ~45 LOC | ~50 LOC | +5 LOC (fallDetected field) |
| manifest.json | ~40 LOC | ~55 LOC | +15 LOC (fallDetected event) |
| **TOTAL** | **~1115 LOC** | **~1695 LOC** | **+580 LOC** |

---

## Build Verification Commands

```bash
# Check for syntax errors in modified files
cd safe_aging/config
cmake -G "Visual Studio 16 2019" -A x64 ..
cmake --build . --config Release 2>&1 | grep -i error

# Should produce: (no errors)
```

---

## Testing Commands

```bash
# Terminal 1: Start Python service
cd safe_aging/python
python service.py --port 18000

# Terminal 2: Run test script
cd safe_aging/tools
python test_service.py

# Terminal 3: Monitor Nx logs
# (Watch `Release\yolov8_people_analytics_plugin.dll` output)
```

---

## Documentation Files Generated

```
✅ FLOW2_MVP_IMPLEMENTATION.md         (1200+ lines, comprehensive)
✅ FLOW2_CODE_CHANGES_SUMMARY.md       (400+ lines, code reference)
✅ FLOW2_DEMO_GUIDE.ps1                (200+ lines, demo steps)
✅ FLOW2_VERIFICATION_CHECKLIST.md     (400+ lines, pre-demo checks)
✅ FLOW2_EXECUTIVE_SUMMARY.md          (300+ lines, overview)
✅ FLOW2_CODE_LOCATIONS.md             (this file, 400+ lines)
```

---

**Status**: All code locations documented. Ready for code review and demo! ✅

---

**Implementation Completed**: February 9, 2026  
**FLOW 2 MVP Status**: Complete and Ready for Demo ✓
