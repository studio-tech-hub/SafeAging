Tmhung
tmhung3404
Online

Lê Dũng — 3/22/26, 9:53 PM
D:\sdk\metadata_sdk
Tmhung — 3/22/26, 9:54 PM
metavms-metadata_sdk-6.0.6.41837-universal
Lê Dũng — 3/22/26, 9:56 PM
D:\metavms-metadata_sdk-6.0.6.41837-universal\metadata_sdk
Tmhung — 3/22/26, 9:57 PM
cd D:\Part-time\SafeAgingV2\SafeAging
$env:NX_METADATA_SDK_DIR="D:\metavms-metadata_sdk-6.0.6.41837-universal\metadata_sdk"
.\tools\build_plugin_windows.ps1 
-NxMetadataSdkDir "D:\metavms-metadata_sdk-6.0.6.41837-universal\metadata_sdk" -VcvarsVersion "14.29.30133"
Lê Dũng — 3/22/26, 10:23 PM
Viết docs cách build và cài plugin cho nx meta giúp Dũng (viết full luôn nha, đầy đủ từ cách cài conan, tới cách down và set visual installer, và gửi mấy câu lệnh để build + chỉ luôn cái metavms-metadata_sdk-6.0.6.41837-universal)
Check manifest.json, oke thì gửi dũng
goodboy — 3/22/26, 10:57 PM
Build đồ ngon hết chưa
Chạy êm chưa
Tmhung — Yesterday at 12:34 AM
Image
Lê Dũng — Yesterday at 4:39 PM
Attachment file type: unknown
yolov8_people_analytics_plugin.dll
5.41 MB
{
    "id": "mycompany.yolov8_people_analytics",
    "name": "YOLOv8 People Analytics",
    "description": "Analytics plugin using YOLOv8 model for people detection and tracking.",
    "version": "1.0.0",
    "vendor": "HumanCounterV8",

manifest.json
3 KB
Lê Dũng — 4:39 PM
// device_agent.cpp
// Copyright 2018-present Network Optix, Inc.
// Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#include "device_agent.h"
#include <set>

device_agent.cpp
42 KB
// device_agent.h
// Copyright 2018-present Network Optix, Inc.
// Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#pragma once

device_agent.h
8 KB
#include "object_detector.h"
#include "exceptions.h"
#include "frame.h"
#include "logging_utils.h"

#ifdef _MSC_VER

object_detector.cpp
49 KB
﻿
// device_agent.cpp
// Copyright 2018-present Network Optix, Inc.
// Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#include "device_agent.h"
#include <set>
#include <algorithm>
#include <chrono>
#include <exception>
#include <cctype>
#include <type_traits>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>

#include <nx/sdk/i_plugin_diagnostic_event.h>
#include <nx/sdk/analytics/helpers/event_metadata.h>
#include <nx/sdk/analytics/helpers/event_metadata_packet.h>
#include <nx/sdk/helpers/attribute.h>
#include <nx/sdk/helpers/uuid_helper.h>

#include <nx/sdk/analytics/helpers/object_metadata.h>
#include <nx/sdk/analytics/helpers/object_metadata_packet.h>
#include <nx/sdk/helpers/string.h>

#include "detection.h"
#include "exceptions.h"
#include "frame.h"
#include "logging_utils.h"

namespace sample_company
{
    namespace vms_server_plugins
    {
        namespace opencv_object_detection
        {

            using namespace nx::sdk;
            using namespace nx::sdk::analytics;
            using namespace std::string_literals;

            namespace {

                template<typename T, typename = void>
                struct HasNameMethod: std::false_type {};

                template<typename T>
                struct HasNameMethod<T, std::void_t<decltype(std::declval<const T*>()->name())>>:
                    std::true_type {};

                template<typename T, typename = void>
                struct HasIdMethod: std::false_type {};

                template<typename T>
                struct HasIdMethod<T, std::void_t<decltype(std::declval<const T*>()->id())>>:
                    std::true_type {};

                std::string resolveCameraName(const nx::sdk::IDeviceInfo* deviceInfo)
                {
                    if (!deviceInfo)
                        return "unknown_camera";

                    std::string name;

                    if constexpr (HasNameMethod<nx::sdk::IDeviceInfo>::value)
                    {
                        const char* raw = deviceInfo->name();
                        if (raw && *raw)
                            name = raw;
                    }

                    if constexpr (HasIdMethod<nx::sdk::IDeviceInfo>::value)
                    {
                        if (name.empty())
                        {
                            const char* raw = deviceInfo->id();
                            if (raw && *raw)
                                name = raw;
                        }
                    }

                    return name.empty() ? "unknown_camera" : name;
                }

                int parseIntSettingValue(
                    const std::string& raw,
                    int defaultValue,
                    int minValue,
                    int maxValue)
                {
                    if (raw.empty())
                        return defaultValue;

                    try
                    {
                        int value = std::stoi(raw);
                        if (value < minValue)
                            value = minValue;
                        if (value > maxValue)
                            value = maxValue;
                        return value;
                    }
                    catch (...)
                    {
                        return defaultValue;
                    }
                }

                void updateMaxDepth(std::atomic<size_t>& currentMax, size_t depth)
                {
                    size_t observed = currentMax.load();
                    while (depth > observed && !currentMax.compare_exchange_weak(observed, depth))
                    {
                    }
                }
            } // namespace

            DeviceAgent::DeviceAgent(
                const nx::sdk::IDeviceInfo *deviceInfo,
                std::filesystem::path pluginHomeDir,
                std::filesystem::path modelPath)
                : ConsumingDeviceAgent(deviceInfo, /*enableOutput*/ true),
                  m_pluginHomeDir(std::move(pluginHomeDir)),
                  m_modelPath(std::move(modelPath)),
                  m_cameraName(resolveCameraName(deviceInfo)),
                  m_objectDetector(std::make_unique<ObjectDetector>(m_modelPath)),
                  m_objectTracker(std::make_unique<ObjectTracker>()),
                  m_workerThread(&DeviceAgent::workerThreadRun, this), // FLOW 2: Start worker thread
                  m_workerShouldStop(false)
            {
                logutil::log(
                    logutil::Level::info,
                    "DeviceAgent camera_name=\"" + m_cameraName + "\"");
            }

            DeviceAgent::~DeviceAgent()
            {
                // FLOW 2: Signal worker thread to stop and wait for it
                {
                    std::unique_lock<std::mutex> lk(m_frameQueueMutex);
                    m_workerShouldStop = true;
                }
                m_frameQueueCV.notify_one();
                if (m_workerThread.joinable())
                {
                    m_workerThread.join();
                }
            }

            std::string DeviceAgent::manifestString() const
            {
                return /*suppress newline*/ 1 + R"json(
{
    "eventTypes": [
        {
            "id": ")json" +
                       kDetectionEventType + R"json(",
            "name": "Object detected"
        },
        {
            "id": ")json" +
                       kProlongedDetectionEventType + R"json(",
            "name": "Object detected (prolonged)",
            "flags": "stateDependent"
        },
        {
            "id": ")json" +
                       kFallDetectedEventType + R"json(",
            "name": "Fall detected",
            "flags": "stateDependent"
        }
    ],
    "supportedTypes": [
        {
            "objectTypeId": ")json" +
                       kPersonObjectType + R"json("
        },
        {
            "objectTypeId": ")json" +
                       kCatObjectType + R"json("
        },
        {
            "objectTypeId": ")json" +
                       kDogObjectType + R"json("
        }
    ]
}
)json";
            }

            bool DeviceAgent::pushUncompressedVideoFrame(const IUncompressedVideoFrame* videoFrame)
            {
                if (!videoFrame)
                    return false;

                ++m_inFrameCount;
                const auto now = std::chrono::steady_clock::now();

                m_terminated = m_terminated || m_objectDetector->isTerminated();
                if (m_terminated)
                {
                    if (!m_terminatedPrevious)
                    {
                        pushPluginDiagnosticEvent(
                            nx::sdk::IPluginDiagnosticEvent::Level::error,
                            "Plugin is in broken state.",
                            "Disable the plugin.");
                        m_terminatedPrevious = true;
                    }
                    return true;
                }

                const int detectionFramePeriod = std::max(
                    1, m_detectionFramePeriod.load(std::memory_order_relaxed));
                const int targetEnqueueFps = std::max(
                    1, m_targetEnqueueFps.load(std::memory_order_relaxed));
                const size_t frameQueueMaxSize = std::max<size_t>(
                    1, m_frameQueueMaxSize.load(std::memory_order_relaxed));

                bool shouldEnqueue = (m_frameIndex % detectionFramePeriod == 0);
                if (shouldEnqueue && targetEnqueueFps > 0)
                {
                    const auto minIntervalMs = std::chrono::milliseconds(1000 / targetEnqueueFps);
                    if (m_lastEnqueueTime != std::chrono::steady_clock::time_point::min() &&
                        now - m_lastEnqueueTime < minIntervalMs)
                    {
                        shouldEnqueue = false;
                    }
                }

                if (shouldEnqueue)
                {
                    m_lastEnqueueTime = now;
                    try
                    {
                        Frame frame(videoFrame, m_frameIndex);
                        std::vector<uint8_t> jpegBytes = encodeFrameToJpeg(frame, 1280);

                        FrameJob job;
                        job.jpegBytes = std::move(jpegBytes);
                        job.cameraId = m_cameraName;
                        job.timestampUs = frame.timestampUs;
                        job.frameIndex = m_frameIndex;

                        {
                            std::unique_lock<std::mutex> lk(m_frameQueueMutex);
                            if (m_frameQueue.size() >= frameQueueMaxSize)
                            {
                                m_frameQueue.pop_front();
                                ++m_droppedFrameCount;
                                ++m_droppedSinceLastQueueWarning;

                                const auto warnNow = std::chrono::steady_clock::now();
                                if (m_lastQueueWarningTime == std::chrono::steady_clock::time_point::min() ||
                                    warnNow - m_lastQueueWarningTime >=
                                        std::chrono::seconds(kQueueWarningThrottleSec))
                                {
                                    const std::string details =
                                        "Worker thread may be slow; dropped " +
                                        std::to_string(m_droppedSinceLastQueueWarning) +
                                        " frames in last interval. queue_max=" +
                                        std::to_string(frameQueueMaxSize) +
                                        ", target_fps=" + std::to_string(targetEnqueueFps);
                                    pushPluginDiagnosticEvent(
                                        nx::sdk::IPluginDiagnosticEvent::Level::warning,
                                        "Frame queue full - dropping old frames",
                                        details.c_str());

                                    logutil::log(
                                        logutil::Level::warn,
                                        "Backpressure: queue full, dropped " +
                                            std::to_string(m_droppedSinceLastQueueWarning) +
                                            " frames in last interval");

                                    m_droppedSinceLastQueueWarning = 0;
                                    m_lastQueueWarningTime = warnNow;
                                }
                            }

                            m_frameQueue.push_back(std::move(job));
                            updateMaxDepth(m_maxQueueDepth, m_frameQueue.size());
                        }

                        ++m_enqueuedFrameCount;
                        m_frameQueueCV.notify_one();
                    }
                    catch (const std::exception& e)
                    {
                        ++m_encodingErrorCount;
                        logutil::logThrottled(
                            logutil::Level::error,
                            "device_agent.frame_encode_error",
                            std::chrono::seconds(10),
                            std::string("Frame encoding error: ") + e.what());
                        pushPluginDiagnosticEvent(
                            nx::sdk::IPluginDiagnosticEvent::Level::error,
                            "Frame encoding error",
                            e.what());
                    }
                }

                const int metricsLogPeriodSec = std::max(
                    1, m_metricsLogPeriodSec.load(std::memory_order_relaxed));
                if (now - m_lastMetricsLogTime >= std::chrono::seconds(metricsLogPeriodSec))
                {
                    const uint64_t inCount = m_inFrameCount.load();
                    const uint64_t processedCount = m_processedFrameCount.load();
                    const uint64_t droppedCount = m_droppedFrameCount.load();
                    const uint64_t inferMs = m_totalInferMs.load();
                    const size_t maxDepth = m_maxQueueDepth.load();

                    const uint64_t deltaIn = inCount - m_lastMetricsInCount;
                    const uint64_t deltaProcessed = processedCount - m_lastMetricsProcessedCount;
                    const uint64_t deltaDropped = droppedCount - m_lastMetricsDroppedCount;
                    const uint64_t deltaInferMs = inferMs - m_lastMetricsInferMs;

                    const double periodSec =
                        std::chrono::duration<double>(now - m_lastMetricsLogTime).count();
                    const double inFps = (periodSec > 0.0) ? (deltaIn / periodSec) : 0.0;
                    const double procFps = (periodSec > 0.0) ? (deltaProcessed / periodSec) : 0.0;
                    const double avgInferMs =
                        (deltaProcessed > 0) ? (static_cast<double>(deltaInferMs) / deltaProcessed) : 0.0;

                    size_t queueLen = 0;
                    {
                        std::unique_lock<std::mutex> lk(m_frameQueueMutex);
                        queueLen = m_frameQueue.size();
                    }

                    logutil::log(
                        logutil::Level::info,
                        "Pipeline metrics: in_fps=" + std::to_string(inFps) +
                            ", proc_fps=" + std::to_string(procFps) +
                            ", drop=" + std::to_string(deltaDropped) +
                            ", queue=" + std::to_string(queueLen) +
                            ", max_depth=" + std::to_string(maxDepth) +
                            ", avg_infer_ms=" + std::to_string(avgInferMs));

                    if (m_lastMetricsDiagTime == std::chrono::steady_clock::time_point::min() ||
                        now - m_lastMetricsDiagTime >= std::chrono::seconds(kMetricsDiagThrottleSec))
                    {
                        const std::string diag =
                            "in_fps=" + std::to_string(inFps) +
                            ", proc_fps=" + std::to_string(procFps) +
                            ", dropped=" + std::to_string(deltaDropped) +
                            ", queue=" + std::to_string(queueLen) +
                            ", max_depth=" + std::to_string(maxDepth) +
                            ", avg_infer_ms=" + std::to_string(avgInferMs);
                        pushPluginDiagnosticEvent(
                            nx::sdk::IPluginDiagnosticEvent::Level::info,
                            "Pipeline metrics",
                            diag.c_str());
                        m_lastMetricsDiagTime = now;
                    }

                    m_lastMetricsInCount = inCount;
                    m_lastMetricsProcessedCount = processedCount;
                    m_lastMetricsDroppedCount = droppedCount;
                    m_lastMetricsInferMs = inferMs;
                    m_lastMetricsLogTime = now;
                }

                ++m_frameIndex;
                return true;
            }

            bool DeviceAgent::pullMetadataPackets(
                std::vector<nx::sdk::analytics::IMetadataPacket *> *metadataPackets)
            {
                std::unique_lock<std::mutex> lk(m_metadataQueueMutex);
                while (!m_metadataQueue.empty())
                {
                    metadataPackets->push_back(m_metadataQueue.front().releasePtr());
                    m_metadataQueue.pop_front();
                }
                return true;
            }

            nx::sdk::Result<const nx::sdk::ISettingsResponse*> DeviceAgent::settingsReceived()
            {
                const int detectionPeriod = parseIntSettingValue(
                    settingValue("detection_frame_period"),
                    kDefaultDetectionFramePeriod,
                    1,
                    60);
                const int enqueueFps = parseIntSettingValue(
                    settingValue("target_enqueue_fps"),
                    kDefaultTargetEnqueueFps,
                    1,
                    60);
                const int queueMax = parseIntSettingValue(
                    settingValue("frame_queue_max_size"),
                    static_cast<int>(kDefaultFrameQueueMaxSize),
                    1,
                    100);
                const int metricsPeriod = parseIntSettingValue(
                    settingValue("metrics_log_period_sec"),
                    kDefaultMetricsLogPeriodSec,
                    1,
                    300);

                m_detectionFramePeriod.store(detectionPeriod, std::memory_order_relaxed);
                m_targetEnqueueFps.store(enqueueFps, std::memory_order_relaxed);
                m_frameQueueMaxSize.store(static_cast<size_t>(queueMax), std::memory_order_relaxed);
                m_metricsLogPeriodSec.store(metricsPeriod, std::memory_order_relaxed);

                logutil::log(
                    logutil::Level::info,
                    "Applied settings: detection_frame_period=" +
                        std::to_string(m_detectionFramePeriod.load(std::memory_order_relaxed)) +
                        ", target_enqueue_fps=" +
                        std::to_string(m_targetEnqueueFps.load(std::memory_order_relaxed)) +
                        ", frame_queue_max_size=" +
                        std::to_string(m_frameQueueMaxSize.load(std::memory_order_relaxed)) +
                        ", metrics_log_period_sec=" +
                        std::to_string(m_metricsLogPeriodSec.load(std::memory_order_relaxed)));

                return nullptr;
            }

            void DeviceAgent::doSetNeededMetadataTypes(
                nx::sdk::Result<void> *outValue,
                const nx::sdk::analytics::IMetadataTypes * /*neededMetadataTypes*/)
            {
                pushPluginDiagnosticEvent(
                    nx::sdk::IPluginDiagnosticEvent::Level::info,
                    "PLUGIN VERSION",
                    "yolov8_people_analytics_plugin.dll build=2025-12-14 v2");

                if (m_terminated)
                    return;

                try
                {
                    m_objectDetector->ensureInitialized();
                }
                catch (const ObjectDetectorInitializationError &e)
                {
                    *outValue = {ErrorCode::otherError, new String(e.what())};
                    m_terminated = true;
                }
                catch (const ObjectDetectorIsTerminatedError &)
                {
                    m_terminated = true;
                }
            }

            //-------------------------------------------------------------------------------------------------
            // private

            // ============================================================
            // FLOW 2: Worker thread - runs in background
            // Dequeues newest frame, processes it, and pushes metadata
            // ============================================================
            void DeviceAgent::workerThreadRun()
            {
                while (true)
                {
                    FrameJob job;

                    // Wait for frame or shutdown signal
                    {
                        std::unique_lock<std::mutex> lk(m_frameQueueMutex);
                        m_frameQueueCV.wait(lk, [this]()
                                            { return !m_frameQueue.empty() || m_workerShouldStop; });

                        if (m_workerShouldStop && m_frameQueue.empty())
                            break; // Exit thread

                        if (m_frameQueue.empty())
                            continue; // Spurious wakeup, wait again

                        // Dequeue NEWEST frame (drop old ones if multiple in queue)
                        if (m_frameQueue.size() > 1)
                            m_droppedFrameCount.fetch_add(m_frameQueue.size() - 1);
                        job = std::move(m_frameQueue.back());
                        m_frameQueue.clear(); // Drop all other frames
                    }

                    // Process frame job (WITHOUT holding lock)
                    try
                    {
                        const auto started = std::chrono::steady_clock::now();
                        MetadataPacketList metadataPackets = processFrameJob(job);
                        const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - started).count();
                        ++m_processedFrameCount;
                        m_totalInferMs.fetch_add(static_cast<uint64_t>(elapsedMs));

                        // Enqueue metadata packets for Nx to pull
                        {
                            std::unique_lock<std::mutex> lk(m_metadataQueueMutex);
                            for (const auto &pkt : metadataPackets)
                            {
                                m_metadataQueue.push_back(pkt);
                            }
                        }
                    }
                    catch (const std::exception &e)
                    {
                        ++m_processingErrorCount;
                        logutil::logThrottled(
                            logutil::Level::error,
                            "device_agent.worker.process_error",
                            std::chrono::seconds(10),
                            std::string("Worker processing error: ") + e.what());
                        pushPluginDiagnosticEvent(
                            nx::sdk::IPluginDiagnosticEvent::Level::error,
                            "Worker thread: frame processing error",
                            e.what());
                    }
                }
            }

            // ============================================================
            // FLOW 2: Encode frame to JPEG bytes
            // ============================================================
            std::vector<uint8_t> DeviceAgent::encodeFrameToJpeg(const Frame &frame, int targetWidth)
            {
                cv::Mat sendImg = frame.cvMat;

                // Optional downscale for faster HTTP transmission and inference.
                // targetWidth <= 0 means keep full frame size.
                if (targetWidth > 0 && frame.width > targetWidth)
                {
                    float scale = (float)targetWidth / (float)frame.width;
                    int newH = std::max(1, (int)std::round(frame.height * scale));
                    cv::resize(sendImg, sendImg, cv::Size(targetWidth, newH));
                }

                // Encode to JPEG
                std::vector<uint8_t> jpegBytes;
                std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 80}; // 80% quality (balanced)

                if (!cv::imencode(".jpg", sendImg, jpegBytes, params))
                {
                    throw ObjectDetectionError("Failed to encode frame to JPEG");
                }

                return jpegBytes;
            }

            // ============================================================
            // FLOW 2: Process queued frame job
            // ============================================================
            DeviceAgent::MetadataPacketList DeviceAgent::processFrameJob(const FrameJob &job)
            {
                MetadataPacketList result;

                try
                {
                    // Call Python AI service with JPEG bytes
                    DetectionList detections = m_objectDetector->run(job.cameraId, job.jpegBytes);

                    // Create ObjectMetadata for bboxes
                    const auto &objectMetadataPacket =
                        detectionsToObjectMetadataPacket(detections, job.timestampUs);

                    if (objectMetadataPacket)
                        result.push_back(objectMetadataPacket);

                    // Emit state-dependent person presence event (start/finish).
                    bool hasPerson = false;
                    std::set<nx::sdk::Uuid> currentFallDetectedTrackIds;
                    for (const auto &detection : detections)
                    {
                        if (detection->classLabel != "person")
                            continue;

                        hasPerson = true;
                        if (detection->fallDetected)
                            currentFallDetectedTrackIds.insert(detection->trackId);
                    }

                    if (hasPerson != m_personDetectionActive)
                    {
                        EventList personEvents;
                        personEvents.push_back(std::make_shared<Event>(Event{
                            hasPerson ? EventType::detection_started : EventType::detection_finished,
                            job.timestampUs,
                            "person"}));

                        const auto personEventPackets =
                            eventsToEventMetadataPacketList(personEvents, job.timestampUs);
                        result.insert(
                            result.end(),
                            std::make_move_iterator(personEventPackets.begin()),
                            std::make_move_iterator(personEventPackets.end()));

                        m_personDetectionActive = hasPerson;
                    }

                    // Emit state-dependent fall events per track_id.
                    // START: newly fallen tracks.
                    for (const auto &trackId : currentFallDetectedTrackIds)
                    {
                        if (m_activeFallDetectedTrackIds.count(trackId) > 0)
                            continue;

                        auto eventMetadata = nx::sdk::makePtr<nx::sdk::analytics::EventMetadata>();
                        eventMetadata->setCaption("Fall detected");
                        eventMetadata->setDescription(
                            "Person " + nx::sdk::UuidHelper::toStdString(trackId) + " is in fallen state");
                        eventMetadata->setIsActive(true);
                        eventMetadata->setTypeId(kFallDetectedEventType);

                        auto eventPacket = nx::sdk::makePtr<nx::sdk::analytics::EventMetadataPacket>();
                        eventPacket->addItem(eventMetadata.get());
                        eventPacket->setTimestampUs(job.timestampUs);
                        result.push_back(eventPacket);

                        m_activeFallDetectedTrackIds.insert(trackId);
                    }

                    // FINISH: tracks that were fallen before but are no longer fallen now.
                    std::vector<nx::sdk::Uuid> tracksToClear;
                    for (const auto &activeTrackId : m_activeFallDetectedTrackIds)
                    {
                        if (currentFallDetectedTrackIds.count(activeTrackId) > 0)
                            continue;

                        auto eventMetadata = nx::sdk::makePtr<nx::sdk::analytics::EventMetadata>();
                        eventMetadata->setCaption("Fall cleared");
                        eventMetadata->setDescription(
                            "Person " + nx::sdk::UuidHelper::toStdString(activeTrackId) + " is no longer fallen");
                        eventMetadata->setIsActive(false);
                        eventMetadata->setTypeId(kFallDetectedEventType);

                        auto eventPacket = nx::sdk::makePtr<nx::sdk::analytics::EventMetadataPacket>();
                        eventPacket->addItem(eventMetadata.get());
                        eventPacket->setTimestampUs(job.timestampUs);
                        result.push_back(eventPacket);

                        tracksToClear.push_back(activeTrackId);
                    }

                    for (const auto &trackId : tracksToClear)
                        m_activeFallDetectedTrackIds.erase(trackId);
                }
                catch (const ObjectDetectionError &e)
                {
                    logutil::logThrottled(
                        logutil::Level::warn,
                        "device_agent.ai_service_error." + m_cameraName,
                        std::chrono::seconds(kAiServiceErrorDiagThrottleSec),
                        std::string("AI service call failed: ") + e.what());

                    const auto now = std::chrono::steady_clock::now();
                    if (m_lastAiServiceErrorDiagTime == std::chrono::steady_clock::time_point::min() ||
                        now - m_lastAiServiceErrorDiagTime >=
                            std::chrono::seconds(kAiServiceErrorDiagThrottleSec))
                    {
                        pushPluginDiagnosticEvent(
                            nx::sdk::IPluginDiagnosticEvent::Level::warning,
                            "AI service call failed - throttled",
                            e.what());
                        m_lastAiServiceErrorDiagTime = now;
                    }
                }
                catch (const std::exception &e)
                {
                    pushPluginDiagnosticEvent(
                        nx::sdk::IPluginDiagnosticEvent::Level::error,
                        "Unexpected error in processFrameJob",
                        e.what());
                }

                return result;
            }

            DeviceAgent::MetadataPacketList DeviceAgent::eventsToEventMetadataPacketList(
                const EventList &events,
                int64_t timestampUs)
            {
                if (events.empty())
                    return {};

                MetadataPacketList result;

                const auto objectDetectedEventMetadataPacket = makePtr<EventMetadataPacket>();

                for (const std::shared_ptr<Event> &event : events)
                {
                    const auto eventMetadata = makePtr<EventMetadata>();

                    if (event->eventType == EventType::detection_started ||
                        event->eventType == EventType::detection_finished)
                    {
                        static const std::string kStartedSuffix = " STARTED";
                        static const std::string kFinishedSuffix = " FINISHED";

                        const std::string suffix = (event->eventType == EventType::detection_started)
                                                       ? kStartedSuffix
                                                       : kFinishedSuffix;

                        const std::string caption =
                            kClassesToDetectPluralCapitalized.at(event->classLabel) +
                            " detection" + suffix;

                        const std::string description = caption;

                        eventMetadata->setCaption(caption);
                        eventMetadata->setDescription(description);
                        eventMetadata->setIsActive(event->eventType == EventType::detection_started);
                        eventMetadata->setTypeId(kProlongedDetectionEventType);

                        const auto eventMetadataPacket = makePtr<EventMetadataPacket>();
                        eventMetadataPacket->addItem(eventMetadata.get());
                        eventMetadataPacket->setTimestampUs(event->timestampUs);
                        result.push_back(eventMetadataPacket);
                    }
                    else if (event->eventType == EventType::object_detected)
                    {
                        std::string caption = event->classLabel + kDetectionEventCaptionSuffix;
                        caption[0] = (char)toupper(caption[0]);
                        std::string description = event->classLabel + kDetectionEventDescriptionSuffix;
                        description[0] = (char)toupper(description[0]);

                        eventMetadata->setCaption(caption);
                        eventMetadata->setDescription(description);
                        eventMetadata->setIsActive(true);
                        eventMetadata->setTypeId(kDetectionEventType);

                        objectDetectedEventMetadataPacket->addItem(eventMetadata.get());
                    }
                }

                objectDetectedEventMetadataPacket->setTimestampUs(timestampUs);
                result.push_back(objectDetectedEventMetadataPacket);

                return result;
            }

            Ptr<ObjectMetadataPacket> DeviceAgent::detectionsToObjectMetadataPacket(
                const DetectionList &detections,
                int64_t timestampUs)
            {
                using nx::sdk::Attribute;
                using nx::sdk::IAttribute;

                using nx::sdk::analytics::ObjectMetadata;
                using nx::sdk::analytics::ObjectMetadataPacket;

                if (detections.empty())
                    return nullptr;

                const auto objectMetadataPacket = makePtr<ObjectMetadataPacket>();

                // PASS 1: count persons in this frame.
                m_currentPersons = 0;
                std::set<nx::sdk::Uuid> framePersonIds;

                for (const std::shared_ptr<Detection> &detection : detections)
                {
                    if (detection->classLabel == "person")
                    {
                        ++m_currentPersons;
                        framePersonIds.insert(detection->trackId);
                    }
                }

                // Keep unique-person tracking state for future analytics extensions.
                m_seenPersonIds.insert(framePersonIds.begin(), framePersonIds.end());

                // PASS 2: build ObjectMetadata and bbox attributes.
                for (const std::shared_ptr<Detection> &detection : detections)
                {
                    auto objectMetadata = makePtr<ObjectMetadata>();

                    objectMetadata->setBoundingBox(detection->boundingBox);
                    objectMetadata->setConfidence(detection->confidence);
                    objectMetadata->setTrackId(detection->trackId);

                    if (detection->classLabel == "person")
                    {
                        objectMetadata->setTypeId(kPersonObjectType);
                        objectMetadata->addAttribute(makePtr<Attribute>(
                            IAttribute::Type::number,
                            "Count Detect",
                            std::to_string(m_currentPersons)));
                        objectMetadata->addAttribute(makePtr<Attribute>(
                            IAttribute::Type::number,
                            "Fall Detect",
                            "0"));
                    }
                    else if (detection->classLabel == "cat")
                    {
                        objectMetadata->setTypeId(kCatObjectType);
                    }
                    else if (detection->classLabel == "dog")
                    {
                        objectMetadata->setTypeId(kDogObjectType);
                    }

                    objectMetadataPacket->addItem(objectMetadata.releasePtr());
                }

                objectMetadataPacket->setTimestampUs(timestampUs);
                return objectMetadataPacket;
            }
            void DeviceAgent::reinitializeObjectTrackerOnFrameSizeChanges(const Frame &frame)
            {
                const bool frameSizeUnset = m_previousFrameWidth == 0 && m_previousFrameHeight == 0;
                if (frameSizeUnset)
                {
                    m_previousFrameWidth = frame.width;
                    m_previousFrameHeight = frame.height;
                    return;
                }

                const bool frameSizeChanged =
                    frame.width != m_previousFrameWidth ||
                    frame.height != m_previousFrameHeight;

                if (frameSizeChanged)
                {
                    m_objectTracker = std::make_unique<ObjectTracker>();
                    m_previousFrameWidth = frame.width;
                    m_previousFrameHeight = frame.height;
                }
            }

            DeviceAgent::MetadataPacketList DeviceAgent::processFrame(
                const IUncompressedVideoFrame *videoFrame)
            {
                if (m_frameIndex % 200 == 0)
                {
                    logutil::log(
                        logutil::Level::debug,
                        "processFrame pixelFormat=" + std::to_string((int)videoFrame->pixelFormat()) +
                            " w=" + std::to_string(videoFrame->width()) +
                            " h=" + std::to_string(videoFrame->height()) +
                            " lineSize0=" + std::to_string(videoFrame->lineSize(0)));
                }

                try
                {
                    // ⚠️ Frame constructor có thể ném exception (unsupported pixel format, cvtColor fail, etc)
                    Frame frame(videoFrame, m_frameIndex);
                    reinitializeObjectTrackerOnFrameSizeChanges(frame);

                    logutil::logThrottled(
                        logutil::Level::debug,
                        "device_agent.process_frame.call_detector",
                        std::chrono::seconds(10),
                        "Calling detector from legacy processFrame path");

                    // 1) Gọi Python service -> lấy detections đã có track_id
                    std::vector<uint8_t> jpegBytes = encodeFrameToJpeg(frame, 1280);
                    DetectionList detections = m_objectDetector->run(m_cameraName, jpegBytes);

                    // 2) Dùng trực tiếp detections từ Python để tạo ObjectMetadata
                    const auto &objectMetadataPacket =
                        detectionsToObjectMetadataPacket(detections, frame.timestampUs);

                    // 3) Không còn events từ tracking, nên truyền EventList rỗng
                    EventList emptyEvents;
                    const auto &eventMetadataPacketList =
                        eventsToEventMetadataPacketList(emptyEvents, frame.timestampUs);

                    MetadataPacketList result;
                    if (objectMetadataPacket)
                        result.push_back(objectMetadataPacket);

                    result.insert(
                        result.end(),
                        std::make_move_iterator(eventMetadataPacketList.begin()),
                        std::make_move_iterator(eventMetadataPacketList.end()));

                    return result;
                }
                catch (const ObjectDetectionError &e)
                {
                    // Log error nhưng KHÔNG terminate plugin
                    // Plugin sẽ tiếp tục chạy và retry lần sau
                    pushPluginDiagnosticEvent(
                        nx::sdk::IPluginDiagnosticEvent::Level::error,
                        "Object detection failed - will retry next frame",
                        e.what());
                }
                catch (const ObjectTrackingError &e)
                {
                    pushPluginDiagnosticEvent(
                        nx::sdk::IPluginDiagnosticEvent::Level::error,
                        "Object tracking error - will retry next frame",
                        e.what());
                }
                catch (const std::runtime_error &e)
                {
                    // Frame constructor throws std::runtime_error for unsupported pixel format
                    pushPluginDiagnosticEvent(
                        nx::sdk::IPluginDiagnosticEvent::Level::error,
                        "Frame conversion error (unsupported pixel format or OpenCV error) - skipping frame",
                        e.what());
                }
                catch (const std::exception &e)
                {
                    pushPluginDiagnosticEvent(
                        nx::sdk::IPluginDiagnosticEvent::Level::error,
                        "Unexpected error in frame processing",
                        (std::string("Type: ") + typeid(e).name() + " Message: " + e.what()).c_str());
                }
                catch (...)
                {
                    pushPluginDiagnosticEvent(
                        nx::sdk::IPluginDiagnosticEvent::Level::error,
                        "Unknown exception in frame processing",
                        "catch(...) - Unable to determine exception type");
                }

                return {};
            }

        } // namespace opencv_object_detection
    } // namespace vms_server_plugins
} // namespace sample_company



device_agent.cpp
42 KB