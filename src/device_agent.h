#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include <nx/sdk/analytics/helpers/consuming_device_agent.h>
#include <nx/sdk/analytics/helpers/event_metadata_packet.h>
#include <nx/sdk/analytics/helpers/object_metadata_packet.h>
#include <nx/sdk/ptr.h>

#include "detection.h"
#include "object_detector.h"

namespace mycompany::yolov8_flow2 {

struct DeviceAgentConfig
{
    DetectorConfig detector;
    double sampleFps = 5.0;
    size_t maxQueueSize = 4;
    int64_t fallFinishGraceUs = 3'000'000;
    int64_t syntheticTrackTtlUs = 2'000'000;
    int64_t trackMapTtlUs = 60'000'000;
    size_t trackMapMaxSize = 50'000; //< max number of track entries to keep
    int logThrottleMs = 5000;
};

class DeviceAgent: public nx::sdk::analytics::ConsumingDeviceAgent
{
public:
    using MetadataPacketList = std::vector<nx::sdk::Ptr<nx::sdk::analytics::IMetadataPacket>>;

    DeviceAgent(
        const nx::sdk::IDeviceInfo* deviceInfo,
        std::filesystem::path pluginHomeDir,
        DeviceAgentConfig config);

    ~DeviceAgent() override;

protected:
    std::string manifestString() const override;
    bool pushUncompressedVideoFrame(const nx::sdk::analytics::IUncompressedVideoFrame* videoFrame) override;
    void doSetNeededMetadataTypes(
        nx::sdk::Result<void>* outValue,
        const nx::sdk::analytics::IMetadataTypes* neededMetadataTypes) override;

private:
    struct FrameJob
    {
        int64_t timestampUs = 0;
        cv::Mat bgrFrame;
    };

    struct SyntheticTrack
    {
        nx::sdk::analytics::Rect bbox;
        int64_t lastSeenUs = 0;
    };

    struct FallTrackState
    {
        int64_t lastSeenUs = 0;
    };

    void workerLoop();
    void processFrameJob(FrameJob job);

    bool shouldSampleFrame(int64_t timestampUs);
    bool convertFrameToBgr(const nx::sdk::analytics::IUncompressedVideoFrame* frame, cv::Mat* outBgr) const;
    void enqueueFrameJob(FrameJob job);

    void resolveTrackIds(DetectionList* detections, int64_t timestampUs);
    int64_t resolveSyntheticTrackId(const nx::sdk::analytics::Rect& bbox, int64_t timestampUs);

    // return uuid for a given camera+track; updates cache last-seen and creates new uuid if needed
    nx::sdk::Uuid uuidFromTrackId(const std::string& cameraId, int64_t trackId, int64_t timestampUs);
    void cleanupTrackState(int64_t timestampUs);

    nx::sdk::Ptr<nx::sdk::analytics::ObjectMetadataPacket> makeObjectPacket(
        const DetectionList& detections,
        int64_t timestampUs);
    MetadataPacketList makeFallEventPackets(const DetectionList& detections, int64_t timestampUs);

    float iou(const nx::sdk::analytics::Rect& a, const nx::sdk::analytics::Rect& b) const;
    void maybeLog(const std::string& message);

private:
    static constexpr const char* kPersonObjectType = "mycompany.yolov8.person";
    static constexpr const char* kGenericObjectType = "mycompany.yolov8.object";
    static constexpr const char* kFallEventType = "mycompany.yolov8.fallDetected";

    const std::filesystem::path m_pluginHomeDir;
    const DeviceAgentConfig m_config;
    const std::string m_cameraId;

    ObjectDetector m_objectDetector;

    std::thread m_worker;
    mutable std::mutex m_queueMutex;
    std::condition_variable m_queueCv;
    std::deque<FrameJob> m_frameQueue;
    bool m_stop = false;

    int64_t m_lastAcceptedTimestampUs = 0;
    int64_t m_minFrameIntervalUs = 0;

    int64_t m_nextSyntheticTrackId = -1;
    std::map<int64_t, SyntheticTrack> m_syntheticTracks;

    // combined cache for track->uuid entries; key includes camera id to avoid cross-device
    struct TrackKey
    {
        std::string cameraId;
        int64_t trackId = 0;

        bool operator<(TrackKey const& other) const
        {
            if (cameraId != other.cameraId)
                return cameraId < other.cameraId;
            return trackId < other.trackId;
        }
    };

    struct UuidEntry
    {
        nx::sdk::Uuid uuid;
        int64_t lastSeenUs = 0;
    };

    std::map<TrackKey, UuidEntry> m_trackUuidByKey;
    std::map<nx::sdk::Uuid, FallTrackState> m_activeFallTracks;

    std::chrono::steady_clock::time_point m_lastLogAt{};
};

} // namespace mycompany::yolov8_flow2
