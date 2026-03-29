// device_agent.h
// Copyright 2018-present Network Optix, Inc.
// Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#pragma once

#include <filesystem>
#include <memory>
#include <vector>
#include <set>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <chrono>
#include <cstdint>

#include <nx/sdk/analytics/helpers/event_metadata_packet.h>
#include <nx/sdk/analytics/helpers/object_metadata_packet.h>
#include <nx/sdk/analytics/helpers/consuming_device_agent.h>
#include <nx/sdk/helpers/uuid_helper.h>
#include <nx/sdk/ptr.h>

#include "engine.h"
#include "object_detector.h"
#include "object_tracker.h"

namespace sample_company {
namespace vms_server_plugins {
namespace opencv_object_detection {

// ========================================
// Frame job for async processing
// ========================================
struct FrameJob
{
    std::vector<uint8_t> jpegBytes;  // JPEG encoded frame
    std::string cameraId;
    int64_t timestampUs;
    int64_t frameIndex;
};

class DeviceAgent: public nx::sdk::analytics::ConsumingDeviceAgent
{
public:
    using MetadataPacketList = std::vector<nx::sdk::Ptr<nx::sdk::analytics::IMetadataPacket>>;

public:
    DeviceAgent(
        const nx::sdk::IDeviceInfo* deviceInfo,
        std::filesystem::path pluginHomeDir,
        std::filesystem::path modelPath);

    virtual ~DeviceAgent() override;

protected:
    virtual std::string manifestString() const override;

    virtual bool pushUncompressedVideoFrame(
        const nx::sdk::analytics::IUncompressedVideoFrame* videoFrame) override;
    virtual bool pullMetadataPackets(
        std::vector<nx::sdk::analytics::IMetadataPacket*>* metadataPackets) override;
    virtual void doSetNeededMetadataTypes(
        nx::sdk::Result<void>* outValue,
        const nx::sdk::analytics::IMetadataTypes* neededMetadataTypes) override;
    virtual nx::sdk::Result<const nx::sdk::ISettingsResponse*> settingsReceived() override;

private:
    void reinitializeObjectTrackerOnFrameSizeChanges(const Frame& frame);

    nx::sdk::Ptr<nx::sdk::analytics::ObjectMetadataPacket> detectionsToObjectMetadataPacket(
        const DetectionList& detections,
        int64_t timestampUs);

    MetadataPacketList eventsToEventMetadataPacketList(
        const EventList& events,
        int64_t timestampUs);

    MetadataPacketList processFrame(
        const nx::sdk::analytics::IUncompressedVideoFrame* videoFrame);

    // ============ FLOW 2: Frame queuing & async worker ============
    // Worker thread function that runs in background
    void workerThreadRun();
    
    // Encode frame to JPEG bytes
    std::vector<uint8_t> encodeFrameToJpeg(const Frame& frame, int targetWidth = 0);
    
    // Process queued frame job and return metadata packets
    MetadataPacketList processFrameJob(const FrameJob& job);

private:
    const std::string kPersonObjectType = "nx.base.Person";
    const std::string kCatObjectType = "nx.base.Cat";
    const std::string kDogObjectType = "nx.base.Dog";

    const std::string kDetectionEventType = "sample.opencv_object_detection.detection";
    const std::string kDetectionEventCaptionSuffix = " detected";
    const std::string kDetectionEventDescriptionSuffix = " detected";

    const std::string kProlongedDetectionEventType =
        "sample.opencv_object_detection.prolongedDetection";
    
    // FLOW 2: Fall Detection Event
    const std::string kFallDetectedEventType = "mycompany.yolov8_people_analytics.fallDetected";

    static constexpr int kQueueWarningThrottleSec = 30;
    static constexpr int kMetricsDiagThrottleSec = 30;
    static constexpr int kAiServiceErrorDiagThrottleSec = 30;

    static constexpr int kDefaultDetectionFramePeriod = 2;
    static constexpr int kDefaultTargetEnqueueFps = 8;
    static constexpr size_t kDefaultFrameQueueMaxSize = 3;
    static constexpr int kDefaultMetricsLogPeriodSec = 10;

private:
    bool m_terminated = false;
    bool m_terminatedPrevious = false;

    std::filesystem::path m_pluginHomeDir;
    std::filesystem::path m_modelPath;
    std::string m_cameraName;

    const std::unique_ptr<ObjectDetector> m_objectDetector;
    std::unique_ptr<ObjectTracker> m_objectTracker;
    int m_frameIndex = 0;

    int m_previousFrameWidth = 0;
    int m_previousFrameHeight = 0;

    // ====== ĐẾM NGƯỜI ======
    // Số người trong frame hiện tại (persons đang thấy trên màn hình).
    int m_currentPersons = 0;

    // Tập các trackId person đã từng xuất hiện (đếm không trùng).
    std::set<nx::sdk::Uuid> m_seenPersonIds;
    
    // ============ FLOW 2: Async frame processing ============
    // Mutex + CV for frame queue
    std::mutex m_frameQueueMutex;
    std::condition_variable m_frameQueueCV;
    std::deque<FrameJob> m_frameQueue;
    
    // Worker thread
    std::thread m_workerThread;
    bool m_workerShouldStop = false;
    
    // Outgoing metadata packet queue (non-blocking)
    std::mutex m_metadataQueueMutex;
    std::deque<nx::sdk::Ptr<nx::sdk::analytics::IMetadataPacket>> m_metadataQueue;
    
    // Fall detection deduplication: track which trackIds have active fallDetected events
    std::set<nx::sdk::Uuid> m_activeFallDetectedTrackIds;

    // Track state of person presence to emit start/finish state-dependent events.
    bool m_personDetectionActive = false;

    // ========= Runtime metrics and backpressure diagnostics =========
    std::atomic<uint64_t> m_inFrameCount{0};
    std::atomic<uint64_t> m_enqueuedFrameCount{0};
    std::atomic<uint64_t> m_processedFrameCount{0};
    std::atomic<uint64_t> m_droppedFrameCount{0};
    std::atomic<uint64_t> m_totalInferMs{0};
    std::atomic<uint64_t> m_encodingErrorCount{0};
    std::atomic<uint64_t> m_processingErrorCount{0};
    std::atomic<size_t> m_maxQueueDepth{0};

    std::chrono::steady_clock::time_point m_lastEnqueueTime = std::chrono::steady_clock::time_point::min();
    std::chrono::steady_clock::time_point m_lastQueueWarningTime = std::chrono::steady_clock::time_point::min();
    std::chrono::steady_clock::time_point m_lastMetricsLogTime = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point m_lastMetricsDiagTime = std::chrono::steady_clock::time_point::min();
    std::chrono::steady_clock::time_point m_lastAiServiceErrorDiagTime = std::chrono::steady_clock::time_point::min();

    uint64_t m_droppedSinceLastQueueWarning = 0;
    uint64_t m_lastMetricsInCount = 0;
    uint64_t m_lastMetricsProcessedCount = 0;
    uint64_t m_lastMetricsDroppedCount = 0;
    uint64_t m_lastMetricsInferMs = 0;

    std::atomic<int> m_detectionFramePeriod{kDefaultDetectionFramePeriod};
    std::atomic<int> m_targetEnqueueFps{kDefaultTargetEnqueueFps};
    std::atomic<size_t> m_frameQueueMaxSize{kDefaultFrameQueueMaxSize};
    std::atomic<int> m_metricsLogPeriodSec{kDefaultMetricsLogPeriodSec};
};

} // namespace opencv_object_detection
} // namespace vms_server_plugins
} // namespace sample_company
