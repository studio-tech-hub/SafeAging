#include "device_agent.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

#include <nx/sdk/analytics/helpers/event_metadata.h>
#include <nx/sdk/analytics/helpers/event_metadata_packet.h>
#include <nx/sdk/analytics/helpers/object_metadata.h>
#include <nx/sdk/analytics/helpers/object_metadata_packet.h>
#include <nx/sdk/helpers/attribute.h>
#include <nx/sdk/helpers/uuid_helper.h>

namespace mycompany::yolov8_flow2 {

using nx::sdk::Attribute;
using nx::sdk::IAttribute;
using nx::sdk::Ptr;
using namespace nx::sdk::analytics;

namespace {

float clamp01(float value)
{
    return std::max(0.0F, std::min(1.0F, value));
}

} // namespace

DeviceAgent::DeviceAgent(
    const nx::sdk::IDeviceInfo* deviceInfo,
    std::filesystem::path pluginHomeDir,
    DeviceAgentConfig config):
    ConsumingDeviceAgent(deviceInfo, /*enableOutput*/ true),
    m_pluginHomeDir(std::move(pluginHomeDir)),
    m_config(std::move(config)),
    m_cameraId([deviceInfo]() {
        std::ostringstream os;
        os << "nx_cam_" << reinterpret_cast<std::uintptr_t>(deviceInfo);
        return os.str();
    }()),
    m_objectDetector(m_config.detector)
{
    if (m_config.sampleFps > 0.0)
        m_minFrameIntervalUs = static_cast<int64_t>(std::llround(1'000'000.0 / m_config.sampleFps));
    else
        m_minFrameIntervalUs = 0;

    m_worker = std::thread(&DeviceAgent::workerLoop, this);
}

DeviceAgent::~DeviceAgent()
{
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_stop = true;
    }
    m_queueCv.notify_all();

    if (m_worker.joinable())
        m_worker.join();
}

std::string DeviceAgent::manifestString() const
{
    return /*suppress newline*/ 1 + R"json(
{
    "eventTypes": [
        {
            "id": "mycompany.yolov8.fallDetected",
            "name": "Fall detected",
            "flags": "stateDependent"
        }
    ],
    "supportedTypes": [
        {
            "objectTypeId": "mycompany.yolov8.person"
        },
        {
            "objectTypeId": "mycompany.yolov8.object"
        }
    ]
}
)json";
}

void DeviceAgent::doSetNeededMetadataTypes(
    nx::sdk::Result<void>* /*outValue*/,
    const nx::sdk::analytics::IMetadataTypes* /*neededMetadataTypes*/)
{
}

bool DeviceAgent::pushUncompressedVideoFrame(const IUncompressedVideoFrame* videoFrame)
{
    if (!videoFrame)
        return false;

    const int64_t timestampUs = videoFrame->timestampUs();
    if (!shouldSampleFrame(timestampUs))
        return true;

    FrameJob job;
    job.timestampUs = timestampUs;
    if (!convertFrameToBgr(videoFrame, &job.bgrFrame))
        return true;

    enqueueFrameJob(std::move(job));
    return true;
}

void DeviceAgent::workerLoop()
{
    while (true)
    {
        FrameJob job;
        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            m_queueCv.wait(lock, [this]() { return m_stop || !m_frameQueue.empty(); });

            if (m_stop && m_frameQueue.empty())
                break;

            job = std::move(m_frameQueue.front());
            m_frameQueue.pop_front();
        }

        processFrameJob(std::move(job));
    }
}

void DeviceAgent::processFrameJob(FrameJob job)
{
    DetectionList detections = m_objectDetector.run(m_cameraId, job.bgrFrame);
    resolveTrackIds(&detections, job.timestampUs);

    if (const auto objectPacket = makeObjectPacket(detections, job.timestampUs))
    {
        objectPacket->addRef();
        pushMetadataPacket(objectPacket.get());
    }

    auto fallPackets = makeFallEventPackets(detections, job.timestampUs);
    for (auto& packet : fallPackets)
    {
        packet->addRef();
        pushMetadataPacket(packet.get());
    }

    cleanupTrackState(job.timestampUs);
}

bool DeviceAgent::shouldSampleFrame(int64_t timestampUs)
{
    if (m_minFrameIntervalUs <= 0)
    {
        m_lastAcceptedTimestampUs = timestampUs;
        return true;
    }

    if (timestampUs <= 0)
        return true;

    if (m_lastAcceptedTimestampUs > 0 && (timestampUs - m_lastAcceptedTimestampUs) < m_minFrameIntervalUs)
        return false;

    m_lastAcceptedTimestampUs = timestampUs;
    return true;
}

bool DeviceAgent::convertFrameToBgr(const IUncompressedVideoFrame* frame, cv::Mat* outBgr) const
{
    if (!frame || !outBgr)
        return false;

    const int w = frame->width();
    const int h = frame->height();
    if (w <= 0 || h <= 0)
        return false;

    constexpr int PF_RGB24 = 2;
    constexpr int PF_BGR24 = 3;
    constexpr int PF_BGRA32 = 4;
    constexpr int PF_RGBA32 = 5;
    constexpr int PF_YV12 = 6;

    const int pf = static_cast<int>(frame->pixelFormat());

    try
    {
        if (pf == PF_BGR24)
        {
            cv::Mat bgr(h, w, CV_8UC3, const_cast<uint8_t*>(frame->data(0)), frame->lineSize(0));
            *outBgr = bgr.clone();
            return true;
        }

        if (pf == PF_BGRA32)
        {
            cv::Mat bgra(h, w, CV_8UC4, const_cast<uint8_t*>(frame->data(0)), frame->lineSize(0));
            cv::cvtColor(bgra, *outBgr, cv::COLOR_BGRA2BGR);
            return !outBgr->empty();
        }

        if (pf == PF_RGBA32)
        {
            cv::Mat rgba(h, w, CV_8UC4, const_cast<uint8_t*>(frame->data(0)), frame->lineSize(0));
            cv::cvtColor(rgba, *outBgr, cv::COLOR_RGBA2BGR);
            return !outBgr->empty();
        }

        if (pf == PF_RGB24)
        {
            cv::Mat rgb(h, w, CV_8UC3, const_cast<uint8_t*>(frame->data(0)), frame->lineSize(0));
            cv::cvtColor(rgb, *outBgr, cv::COLOR_RGB2BGR);
            return !outBgr->empty();
        }

        if (pf == PF_YV12)
        {
            const auto* data = reinterpret_cast<const uint8_t*>(frame->data(0));
            if (!data)
                return false;

            const int ySize = w * h;
            const int uvSize = ySize / 4;
            std::vector<uint8_t> i420;
            i420.reserve(ySize + 2 * uvSize);

            i420.insert(i420.end(), data, data + ySize); // Y
            i420.insert(i420.end(), data + ySize + uvSize, data + ySize + 2 * uvSize); // U
            i420.insert(i420.end(), data + ySize, data + ySize + uvSize); // V

            cv::Mat i420Mat(h * 3 / 2, w, CV_8UC1, i420.data());
            cv::cvtColor(i420Mat, *outBgr, cv::COLOR_YUV2BGR_I420);
            return !outBgr->empty();
        }
    }
    catch (const std::exception& e)
    {
        maybeLog(std::string("frame conversion failed: ") + e.what());
        return false;
    }

    maybeLog("unsupported frame pixel format: " + std::to_string(pf));
    return false;
}

void DeviceAgent::enqueueFrameJob(FrameJob job)
{
    std::lock_guard<std::mutex> lock(m_queueMutex);

    if (m_frameQueue.size() >= std::max<size_t>(1, m_config.maxQueueSize))
    {
        m_frameQueue.pop_front(); // drop oldest, keep newest
    }

    m_frameQueue.push_back(std::move(job));
    m_queueCv.notify_one();
}

void DeviceAgent::resolveTrackIds(DetectionList* detections, int64_t timestampUs)
{
    if (!detections)
        return;

    for (auto& detection : *detections)
    {
        int64_t key = 0;
        if (detection.aiTrackId.has_value())
        {
            key = *detection.aiTrackId;
        }
        else
        {
            key = resolveSyntheticTrackId(detection.bbox, timestampUs);
        }

        detection.trackId = getOrCreateUuid(key);
        m_trackLastSeenUs[key] = timestampUs;
    }
}

int64_t DeviceAgent::resolveSyntheticTrackId(const Rect& bbox, int64_t timestampUs)
{
    const float iouThreshold = 0.3F;
    int64_t bestTrackId = 0;
    float bestIou = 0.0F;

    for (auto& [trackId, track]: m_syntheticTracks)
    {
        if (timestampUs - track.lastSeenUs > m_config.syntheticTrackTtlUs)
            continue;

        const float overlap = iou(track.bbox, bbox);
        if (overlap > iouThreshold && overlap > bestIou)
        {
            bestIou = overlap;
            bestTrackId = trackId;
        }
    }

    if (bestTrackId == 0)
    {
        const int64_t newId = m_nextSyntheticTrackId--;
        m_syntheticTracks[newId] = SyntheticTrack{bbox, timestampUs};
        return newId;
    }

    m_syntheticTracks[bestTrackId] = SyntheticTrack{bbox, timestampUs};
    return bestTrackId;
}

nx::sdk::Uuid DeviceAgent::getOrCreateUuid(int64_t key)
{
    const auto it = m_trackUuidByKey.find(key);
    if (it != m_trackUuidByKey.end())
        return it->second;

    const nx::sdk::Uuid uuid = nx::sdk::UuidHelper::randomUuid();
    m_trackUuidByKey.emplace(key, uuid);
    return uuid;
}

void DeviceAgent::cleanupTrackState(int64_t timestampUs)
{
    for (auto it = m_syntheticTracks.begin(); it != m_syntheticTracks.end(); )
    {
        if (timestampUs - it->second.lastSeenUs > m_config.syntheticTrackTtlUs)
            it = m_syntheticTracks.erase(it);
        else
            ++it;
    }

    for (auto it = m_trackLastSeenUs.begin(); it != m_trackLastSeenUs.end(); )
    {
        if (timestampUs - it->second > m_config.trackMapTtlUs)
        {
            m_trackUuidByKey.erase(it->first);
            it = m_trackLastSeenUs.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

Ptr<ObjectMetadataPacket> DeviceAgent::makeObjectPacket(const DetectionList& detections, int64_t timestampUs)
{
    if (detections.empty())
        return nullptr;

    auto packet = nx::sdk::makePtr<ObjectMetadataPacket>();
    bool hasItems = false;
    for (const auto& detection : detections)
    {
        auto metadata = nx::sdk::makePtr<ObjectMetadata>();

        const Rect& b = detection.bbox;
        const float x = clamp01(static_cast<float>(b.x));
        const float y = clamp01(static_cast<float>(b.y));
        float w = clamp01(static_cast<float>(b.width));
        float h = clamp01(static_cast<float>(b.height));
        if (x + w > 1.0F)
            w = std::max(0.0F, 1.0F - x);
        if (y + h > 1.0F)
            h = std::max(0.0F, 1.0F - y);
        if (w <= 0.0F || h <= 0.0F)
            continue;

        metadata->setBoundingBox(Rect(x, y, w, h));
        metadata->setConfidence(detection.confidence);
        metadata->setTrackId(detection.trackId);

        if (detection.classLabel == "person")
            metadata->setTypeId(kPersonObjectType);
        else
            metadata->setTypeId(kGenericObjectType);

        metadata->addAttribute(nx::sdk::makePtr<Attribute>(
            IAttribute::Type::string,
            "classLabel",
            detection.classLabel));
        metadata->addAttribute(nx::sdk::makePtr<Attribute>(
            IAttribute::Type::number,
            "confidence",
            std::to_string(detection.confidence)));
        metadata->addAttribute(nx::sdk::makePtr<Attribute>(
            IAttribute::Type::number,
            "fallDetected",
            detection.fallDetected ? "1" : "0"));

        packet->addItem(metadata.releasePtr());
        hasItems = true;
    }

    if (!hasItems)
        return nullptr;

    packet->setTimestampUs(timestampUs);
    return packet;
}

DeviceAgent::MetadataPacketList DeviceAgent::makeFallEventPackets(
    const DetectionList& detections,
    int64_t timestampUs)
{
    MetadataPacketList packets;
    std::set<nx::sdk::Uuid> seenTracks;
    std::set<nx::sdk::Uuid> fallingTracks;

    for (const auto& detection : detections)
    {
        seenTracks.insert(detection.trackId);
        if (detection.fallDetected)
            fallingTracks.insert(detection.trackId);
    }

    for (const auto& trackId : fallingTracks)
    {
        auto [it, inserted] = m_activeFallTracks.emplace(trackId, FallTrackState{timestampUs});
        if (!inserted)
            it->second.lastSeenUs = timestampUs;

        if (!inserted)
            continue;

        auto eventMetadata = nx::sdk::makePtr<EventMetadata>();
        eventMetadata->setTypeId(kFallEventType);
        eventMetadata->setCaption("Fall detected STARTED");
        eventMetadata->setDescription(
            "Track " + nx::sdk::UuidHelper::toStdString(trackId) + " entered fall state");
        eventMetadata->setIsActive(true);

        auto packet = nx::sdk::makePtr<EventMetadataPacket>();
        packet->addItem(eventMetadata.releasePtr());
        packet->setTimestampUs(timestampUs);
        packets.push_back(packet);
    }

    std::vector<nx::sdk::Uuid> toFinish;
    for (const auto& [trackId, state] : m_activeFallTracks)
    {
        if (fallingTracks.find(trackId) != fallingTracks.end())
            continue;

        const bool seenInThisFrame = seenTracks.find(trackId) != seenTracks.end();
        const bool missingTimedOut = (timestampUs - state.lastSeenUs) >= m_config.fallFinishGraceUs;
        if (seenInThisFrame || missingTimedOut)
            toFinish.push_back(trackId);
    }

    for (const auto& trackId : toFinish)
    {
        auto eventMetadata = nx::sdk::makePtr<EventMetadata>();
        eventMetadata->setTypeId(kFallEventType);
        eventMetadata->setCaption("Fall detected FINISHED");
        eventMetadata->setDescription(
            "Track " + nx::sdk::UuidHelper::toStdString(trackId) + " exited fall state");
        eventMetadata->setIsActive(false);

        auto packet = nx::sdk::makePtr<EventMetadataPacket>();
        packet->addItem(eventMetadata.releasePtr());
        packet->setTimestampUs(timestampUs);
        packets.push_back(packet);

        m_activeFallTracks.erase(trackId);
    }

    return packets;
}

float DeviceAgent::iou(const Rect& a, const Rect& b) const
{
    const float ax1 = static_cast<float>(a.x);
    const float ay1 = static_cast<float>(a.y);
    const float ax2 = ax1 + static_cast<float>(a.width);
    const float ay2 = ay1 + static_cast<float>(a.height);

    const float bx1 = static_cast<float>(b.x);
    const float by1 = static_cast<float>(b.y);
    const float bx2 = bx1 + static_cast<float>(b.width);
    const float by2 = by1 + static_cast<float>(b.height);

    const float ix1 = std::max(ax1, bx1);
    const float iy1 = std::max(ay1, by1);
    const float ix2 = std::min(ax2, bx2);
    const float iy2 = std::min(ay2, by2);

    const float iw = std::max(0.0F, ix2 - ix1);
    const float ih = std::max(0.0F, iy2 - iy1);
    const float intersection = iw * ih;

    const float areaA = std::max(0.0F, (ax2 - ax1) * (ay2 - ay1));
    const float areaB = std::max(0.0F, (bx2 - bx1) * (by2 - by1));
    if (areaA <= 0.0F || areaB <= 0.0F)
        return 0.0F;

    return intersection / (areaA + areaB - intersection + 1e-6F);
}

void DeviceAgent::maybeLog(const std::string& message)
{
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastLogAt).count();
    if (m_lastLogAt.time_since_epoch().count() == 0 || elapsed >= m_config.logThrottleMs)
    {
        std::cerr << "[DeviceAgent][" << m_cameraId << "] " << message << std::endl;
        m_lastLogAt = now;
    }
}

} // namespace mycompany::yolov8_flow2
