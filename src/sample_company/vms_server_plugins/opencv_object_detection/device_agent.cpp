// device_agent.cpp
// Copyright 2018-present Network Optix, Inc.
// Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#include "device_agent.h"
#include <set>
#include <iostream>
#include <chrono>
#include <exception>
#include <cctype>

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
#include "logging.h"

namespace sample_company
{
    namespace vms_server_plugins
    {
        namespace opencv_object_detection
        {

            using namespace nx::sdk;
            using namespace nx::sdk::analytics;
            using namespace std::string_literals;

            DeviceAgent::DeviceAgent(
                const nx::sdk::IDeviceInfo *deviceInfo,
                std::filesystem::path pluginHomeDir,
                std::filesystem::path modelPath)
                : ConsumingDeviceAgent(deviceInfo, /*enableOutput*/ true),
                  m_pluginHomeDir(std::move(pluginHomeDir)),
                  m_modelPath(std::move(modelPath)),
                  m_cameraId(getCameraIdFromDeviceInfo(deviceInfo)),
                  m_objectDetector(std::make_unique<ObjectDetector>(m_modelPath)),
                  m_objectTracker(std::make_unique<ObjectTracker>()),
                  m_workerThread(&DeviceAgent::workerThreadRun, this), // FLOW 2: Start worker thread
                  m_workerShouldStop(false)
            {
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

            std::string DeviceAgent::getCameraIdFromDeviceInfo(const nx::sdk::IDeviceInfo* deviceInfo)
            {
                if (!deviceInfo)
                    return "unknown_camera";

                // Get the device ID from NX SDK
                std::string deviceId = deviceInfo->id();

                // Normalize: replace non-alphanumeric with underscores, lowercase
                std::string normalizedId;
                for (char c : deviceId)
                {
                    if (std::isalnum(c))
                        normalizedId += std::tolower(c);
                    else
                        normalizedId += '_';
                }

                // Ensure not empty
                if (normalizedId.empty())
                    normalizedId = "camera_" + std::to_string(reinterpret_cast<uintptr_t>(deviceInfo));

                return normalizedId;
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

            bool DeviceAgent::pushUncompressedVideoFrame(const IUncompressedVideoFrame *videoFrame)
            {
                if (!videoFrame)
                    return false;

                if (m_frameIndex % 200 == 0)
                {
                    Logger::logThrottled(LogLevel::Debug, "pixel_format",
                        std::chrono::seconds(30),
                        "[DBG] pixelFormat=" + std::to_string((int)videoFrame->pixelFormat()) +
                        " w=" + std::to_string(videoFrame->width()) +
                        " h=" + std::to_string(videoFrame->height()) +
                        " lineSize0=" + std::to_string(videoFrame->lineSize(0)));
                }

                if (m_frameIndex % 200 == 0)
                {
                    Logger::logThrottled(LogLevel::Debug, "frame_arrived",
                        std::chrono::seconds(30),
                        "Frame arrived frame#" + std::to_string(m_frameIndex) +
                        " w=" + std::to_string(videoFrame->width()) +
                        " h=" + std::to_string(videoFrame->height()));
                }

                // Nếu detector đã bị terminate cứng (hiếm), chỉ báo 1 lần rồi bỏ qua frame.
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

                // ============================================================
                // FLOW 2: Frame callback MUST NOT process frames here.
                //         Instead, enqueue frame for async worker thread.
                //         This callback returns immediately (NON-BLOCKING).
                // ============================================================

                // 🔻 Process detection frames regularly:
                const int kPeriod = kDetectionFramePeriod;
                if (m_frameIndex % kPeriod == 0)
                {
                    try
                    {
                        // Convert Nx frame to OpenCV Mat for encoding
                        Frame frame(videoFrame, m_frameIndex);

                        // Encode frame to JPEG with downscale to ~720p width for faster, more stable detection.
                        // 2560x1440 -> 1280x720 (keeps aspect ratio, no crop).
                        std::vector<uint8_t> jpegBytes = encodeFrameToJpeg(frame, 1280);

                        // Create frame job
                        FrameJob job;
                        job.jpegBytes = std::move(jpegBytes);
                        job.cameraId = m_cameraId;
                        job.timestampUs = frame.timestampUs;
                        job.frameIndex = m_frameIndex;

                        // ⚠️ BACKPRESSURE: bounded queue (size 3)
                        // If queue is full, drop oldest frame and add newest
                        {
                            std::unique_lock<std::mutex> lk(m_frameQueueMutex);
                            if (m_frameQueue.size() >= kFrameQueueMaxSize)
                            {
                                // Drop oldest (front) frame to make room
                                m_frameQueue.pop_front();
                                if (!m_frameQueueFullReported)
                                {
                                    Logger::log(LogLevel::Warn,
                                        "Frame queue full - dropping old frames. Worker thread may be slow; increase queue or reduce FPS");
                                    pushPluginDiagnosticEvent(
                                        nx::sdk::IPluginDiagnosticEvent::Level::warning,
                                        "Frame queue full - dropping old frames",
                                        "Worker thread may be slow; increase queue or reduce FPS");
                                    m_frameQueueFullReported = true;
                                }
                            }
                            m_frameQueue.push_back(std::move(job));
                        }
                        m_frameQueueCV.notify_one(); // Wake up worker thread
                    }
                    catch (const std::exception &e)
                    {
                        pushPluginDiagnosticEvent(
                            nx::sdk::IPluginDiagnosticEvent::Level::error,
                            "Frame encoding error",
                            e.what());
                    }
                }

                ++m_frameIndex;
                return true; // ✓ Frame callback returns immediately
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
                        job = std::move(m_frameQueue.back());
                        m_frameQueue.clear(); // Drop all other frames
                    }

                    // Process frame job (WITHOUT holding lock)
                    try
                    {
                        MetadataPacketList metadataPackets = processFrameJob(job);

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
                    pushPluginDiagnosticEvent(
                        nx::sdk::IPluginDiagnosticEvent::Level::error,
                        "AI service call failed - will retry next frame",
                        e.what());
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

                // --- PASS 1: đếm số person trong frame, gom trackId ---
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

                // Cập nhật tập trackId đã từng xuất hiện (đếm không trùng)
                m_seenPersonIds.insert(framePersonIds.begin(), framePersonIds.end());
                const int totalUniquePersons = static_cast<int>(m_seenPersonIds.size());

                // --- PASS 2: tạo ObjectMetadata + gắn attribute + caption ---
                for (const std::shared_ptr<Detection> &detection : detections)
                {
                    auto objectMetadata = makePtr<ObjectMetadata>();

                    objectMetadata->setBoundingBox(detection->boundingBox);
                    objectMetadata->setConfidence(detection->confidence);
                    objectMetadata->setTrackId(detection->trackId);

                    if (detection->classLabel == "person")
                    {
                        objectMetadata->setTypeId(kPersonObjectType);

                        // 1) id từng người (trackId)
                        objectMetadata->addAttribute(makePtr<Attribute>(
                            IAttribute::Type::string,
                            "yolov8_person_id",
                            nx::sdk::UuidHelper::toStdString(detection->trackId)));

                        // 2) số người đang có trong frame hiện tại
                        objectMetadata->addAttribute(makePtr<Attribute>(
                            IAttribute::Type::number,
                            "yolov8_person_count_frame",
                            std::to_string(m_currentPersons)));

                        // 3) tổng số người khác nhau đã đi qua (đếm không trùng)
                        objectMetadata->addAttribute(makePtr<Attribute>(
                            IAttribute::Type::number,
                            "yolov8_person_count_unique",
                            std::to_string(totalUniquePersons)));
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
                    Logger::logThrottled(LogLevel::Debug, "process_frame_pixel_format",
                        std::chrono::seconds(30),
                        "[DBG] pixelFormat=" + std::to_string((int)videoFrame->pixelFormat()) +
                        " w=" + std::to_string(videoFrame->width()) +
                        " h=" + std::to_string(videoFrame->height()) +
                        " lineSize0=" + std::to_string(videoFrame->lineSize(0)));
                }

                try
                {
                    // ⚠️ Frame constructor có thể ném exception (unsupported pixel format, cvtColor fail, etc)
                    Frame frame(videoFrame, m_frameIndex);
                    reinitializeObjectTrackerOnFrameSizeChanges(frame);

                    if (m_frameIndex % 200 == 0)
                    {
                        Logger::logThrottled(LogLevel::Debug, "calling_detector",
                            std::chrono::seconds(30),
                            "Calling detector: about to call Python /infer endpoint");
                    }

                    // 1) Gọi Python service -> lấy detections đã có track_id
                    DetectionList detections = m_objectDetector->run(frame);

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
