// device_agent.h
// Copyright 2018-present Network Optix, Inc.
// Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#pragma once

#include <filesystem>
#include <map>
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
#include "frame.h"
#include "object_detector.h"
#include "object_tracker.h"

namespace sample_company {
namespace vms_server_plugins {
namespace opencv_object_detection {

struct FrameJob
{
    std::vector<uint8_t> jpegBytes;
    std::shared_ptr<Frame> frame;
    std::string cameraId;
    int64_t timestampUs;
    int64_t frameIndex;
};

struct RenderedDetectionState
{
    std::shared_ptr<Detection> detection;
    std::chrono::steady_clock::time_point lastSeen =
        std::chrono::steady_clock::time_point::min();
};

class DeviceAgent: public nx::sdk::analytics::ConsumingDeviceAgent
{
public:
    using MetadataPacketList = std::vector<nx::sdk::Ptr<nx::sdk::analytics::IMetadataPacket>>;

public:
    DeviceAgent(
        const nx::sdk::IDeviceInfo* deviceInfo,
        std::filesystem::path pluginHomeDir);

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

    void workerThreadRun();
    std::vector<uint8_t> encodeFrameToJpeg(const Frame& frame, int targetWidth = 0);
    MetadataPacketList processFrameJob(const FrameJob& job);
    MetadataPacketList buildDisabledCleanupPackets(int64_t timestampUs);
    void clearPendingFrameQueue();

    void updateRenderedTrackState(const DetectionList& detections);
    nx::sdk::Ptr<nx::sdk::analytics::ObjectMetadataPacket> renderCurrentObjectMetadataPacket(
        int64_t timestampUs);

private:
    const std::string kPersonObjectType = "nx.base.Person";
    const std::string kCatObjectType = "nx.base.Cat";
    const std::string kDogObjectType = "nx.base.Dog";

    const std::string kDetectionEventType = "sample.opencv_object_detection.detection";
    const std::string kDetectionEventCaptionSuffix = " detected";
    const std::string kDetectionEventDescriptionSuffix = " detected";

    const std::string kProlongedDetectionEventType =
        "sample.opencv_object_detection.prolongedDetection";

    const std::string kFallDetectedEventType = "mycompany.yolov8_people_analytics.fallDetected";

    static constexpr int kQueueWarningThrottleSec = 30;
    static constexpr int kMetricsDiagThrottleSec = 30;
    static constexpr int kAiServiceErrorDiagThrottleSec = 30;

    static constexpr int kDefaultDetectionFramePeriod = 1;
    static constexpr int kDefaultTargetEnqueueFps = 10;
    static constexpr size_t kDefaultFrameQueueMaxSize = 2;
    static constexpr int kDefaultMetricsLogPeriodSec = 10;
    static constexpr int kAcquireBoostEnqueueFps = 10;

private:
    bool m_terminated = false;
    bool m_terminatedPrevious = false;

    std::filesystem::path m_pluginHomeDir;
    std::string m_cameraName;

    const std::unique_ptr<ObjectDetector> m_objectDetector;
    std::unique_ptr<ObjectTracker> m_objectTracker;
    int m_frameIndex = 0;

    int m_previousFrameWidth = 0;
    int m_previousFrameHeight = 0;

    int m_currentPersons = 0;
    std::set<nx::sdk::Uuid> m_seenPersonIds;

    std::mutex m_frameQueueMutex;
    std::condition_variable m_frameQueueCV;
    std::deque<FrameJob> m_frameQueue;

    std::thread m_workerThread;
    bool m_workerShouldStop = false;

    std::mutex m_metadataQueueMutex;
    nx::sdk::Ptr<nx::sdk::analytics::ObjectMetadataPacket> m_latestObjectMetadataPacket;
    std::deque<nx::sdk::Ptr<nx::sdk::analytics::IMetadataPacket>> m_metadataQueue;

    std::mutex m_renderStateMutex;
    std::map<nx::sdk::Uuid, RenderedDetectionState> m_renderedTrackStates;
    std::chrono::steady_clock::time_point m_lastRenderedTrackStateUpdateTime =
        std::chrono::steady_clock::time_point::min();

    std::mutex m_lifecycleStateMutex;
    std::set<nx::sdk::Uuid> m_activeFallDetectedTrackIds;
    bool m_personDetectionActive = false;

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
    std::atomic<int> m_lastEffectiveEnqueueFps{kDefaultTargetEnqueueFps};
    std::atomic<bool> m_detectionEnabled{true};
    std::atomic<bool> m_detectionDisableCleanupPending{false};
};

} // namespace opencv_object_detection
} // namespace vms_server_plugins
} // namespace sample_company
