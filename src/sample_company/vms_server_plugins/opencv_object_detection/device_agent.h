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

    virtual void doSetNeededMetadataTypes(
        nx::sdk::Result<void>* outValue,
        const nx::sdk::analytics::IMetadataTypes* neededMetadataTypes) override;

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
    std::vector<uint8_t> encodeFrameToJpeg(const Frame& frame, int targetWidth = 640);
    
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

    /** Process every 2nd frame for better detection frequency (reasonable balance). */
    static constexpr int kDetectionFramePeriod = 2;
    
    // ============ FLOW 2: Frame queue config ============
    static constexpr size_t kFrameQueueMaxSize = 3;  // Drop old frames if queue full

private:
    bool m_terminated = false;
    bool m_terminatedPrevious = false;

    std::filesystem::path m_pluginHomeDir;
    std::filesystem::path m_modelPath;

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
};

} // namespace opencv_object_detection
} // namespace vms_server_plugins
} // namespace sample_company
