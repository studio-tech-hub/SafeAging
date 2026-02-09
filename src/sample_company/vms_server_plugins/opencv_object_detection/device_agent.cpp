// device_agent.cpp
// Copyright 2018-present Network Optix, Inc.
// Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#include "device_agent.h"
#include <set>
#include <iostream>
#include <chrono>
#include <exception>
#include <cctype>
#include <cstdint>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>

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

namespace sample_company {
    namespace vms_server_plugins {
        namespace opencv_object_detection {

            using namespace nx::sdk;
            using namespace nx::sdk::analytics;
            using namespace std::string_literals;

            DeviceAgent::DeviceAgent(
                const nx::sdk::IDeviceInfo* deviceInfo,
                std::filesystem::path pluginHomeDir,
                std::filesystem::path modelPath)
                :
                ConsumingDeviceAgent(deviceInfo, /*enableOutput*/ true),
                m_pluginHomeDir(std::move(pluginHomeDir)),
                m_modelPath(std::move(modelPath)),
                m_objectDetector(std::make_unique<ObjectDetector>(m_modelPath)),
                m_objectTracker(std::make_unique<ObjectTracker>())
            {
                std::ostringstream os;
                os << "nx_cam_" << reinterpret_cast<std::uintptr_t>(deviceInfo);
                m_cameraId = os.str();
            }

            DeviceAgent::~DeviceAgent()
            {
            }

            std::string DeviceAgent::manifestString() const
            {
                return /*suppress newline*/ 1 + R"json(
{
    "eventTypes": [
        {
            "id": ")json" + kDetectionEventType + R"json(",
            "name": "Object detected"
        },
        {
            "id": ")json" + kProlongedDetectionEventType + R"json(",
            "name": "Object detected (prolonged)",
            "flags": "stateDependent"
        },
        {
            "id": ")json" + kFallDetectedEventType + R"json(",
            "name": "Fall detected",
            "flags": "stateDependent"
        }
    ],
    "supportedTypes": [
        {
            "objectTypeId": ")json" + kPersonObjectType + R"json("
        },
        {
            "objectTypeId": ")json" + kCatObjectType + R"json("
        },
        {
            "objectTypeId": ")json" + kDogObjectType + R"json("
        }
    ]
}
)json";
            }

            bool DeviceAgent::pushUncompressedVideoFrame(const IUncompressedVideoFrame* videoFrame)
            {
                if (!videoFrame)
                    return false;

                if (m_frameIndex % 200 == 0)
                {
                    std::cerr << "[DBG] pixelFormat=" << (int)videoFrame->pixelFormat()
                        << " w=" << videoFrame->width()
                        << " h=" << videoFrame->height()
                        << " lineSize0=" << videoFrame->lineSize(0)
                        << std::endl;
                }


                if (m_frameIndex % 200 == 0)
                {
                    pushPluginDiagnosticEvent(
                        nx::sdk::IPluginDiagnosticEvent::Level::info,
                        "Frame arrived",
                        ("frame#" + std::to_string(m_frameIndex) +
                            " w=" + std::to_string(videoFrame->width()) +
                            " h=" + std::to_string(videoFrame->height())).c_str());
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

                // 🔻 Process detection frames regularly:
                // Detect every kDetectionFramePeriod frames for consistent performance
                // Purpose: balance detection accuracy with system load
                const int kPeriod = kDetectionFramePeriod;
                if (m_frameIndex % kPeriod == 0)
                {
                    MetadataPacketList metadataPackets;
                    try
                    {
                        metadataPackets = processFrame(videoFrame);
                    }
                    catch (const std::exception& e)
                    {
                        pushPluginDiagnosticEvent(
                            nx::sdk::IPluginDiagnosticEvent::Level::error,
                            "Frame processing error (processFrame failed)",
                            (std::string("Exception: ") + e.what()).c_str());
                        metadataPackets.clear();
                    }
                    catch (...)
                    {
                        pushPluginDiagnosticEvent(
                            nx::sdk::IPluginDiagnosticEvent::Level::error,
                            "Frame processing error (unknown exception)",
                            "Unknown exception type in processFrame");
                        metadataPackets.clear();
                    }

                    for (const Ptr<IMetadataPacket>& metadataPacket : metadataPackets)
                    {
                        metadataPacket->addRef();
                        pushMetadataPacket(metadataPacket.get());
                    }
                }

                ++m_frameIndex;
                return true;
            }

            void DeviceAgent::doSetNeededMetadataTypes(
                nx::sdk::Result<void>* outValue,
                const nx::sdk::analytics::IMetadataTypes* /*neededMetadataTypes*/)
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
                catch (const ObjectDetectorInitializationError& e)
                {
                    *outValue = { ErrorCode::otherError, new String(e.what()) };
                    m_terminated = true;
                }
                catch (const ObjectDetectorIsTerminatedError&)
                {
                    m_terminated = true;
                }
            }

            //-------------------------------------------------------------------------------------------------
            // private

            DeviceAgent::MetadataPacketList DeviceAgent::eventsToEventMetadataPacketList(
                const EventList& events,
                int64_t timestampUs)
            {
                if (events.empty())
                    return {};

                MetadataPacketList result;

                const auto objectDetectedEventMetadataPacket = makePtr<EventMetadataPacket>();

                for (const std::shared_ptr<Event>& event : events)
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

            DeviceAgent::MetadataPacketList DeviceAgent::fallEventsToEventMetadataPacketList(
                const DetectionList& detections,
                int64_t timestampUs)
            {
                std::set<nx::sdk::Uuid> frameFallingTrackIds;
                for (const std::shared_ptr<Detection>& detection : detections)
                {
                    if (detection->classLabel == "person" && detection->fallDetected)
                        frameFallingTrackIds.insert(detection->trackId);
                }

                const auto eventMetadataPacket = makePtr<EventMetadataPacket>();
                bool hasEvents = false;

                for (const auto& trackId : frameFallingTrackIds)
                {
                    if (m_activeFallTrackIds.find(trackId) != m_activeFallTrackIds.end())
                        continue;

                    const auto eventMetadata = makePtr<EventMetadata>();
                    const std::string trackIdStr = nx::sdk::UuidHelper::toStdString(trackId);
                    const std::string description = "Track ID: " + trackIdStr;
                    eventMetadata->setCaption("Fall detected STARTED");
                    eventMetadata->setDescription(description);
                    eventMetadata->setIsActive(true);
                    eventMetadata->setTypeId(kFallDetectedEventType);
                    eventMetadataPacket->addItem(eventMetadata.get());
                    hasEvents = true;
                }

                for (const auto& trackId : m_activeFallTrackIds)
                {
                    if (frameFallingTrackIds.find(trackId) != frameFallingTrackIds.end())
                        continue;

                    const auto eventMetadata = makePtr<EventMetadata>();
                    const std::string trackIdStr = nx::sdk::UuidHelper::toStdString(trackId);
                    const std::string description = "Track ID: " + trackIdStr;
                    eventMetadata->setCaption("Fall detected FINISHED");
                    eventMetadata->setDescription(description);
                    eventMetadata->setIsActive(false);
                    eventMetadata->setTypeId(kFallDetectedEventType);
                    eventMetadataPacket->addItem(eventMetadata.get());
                    hasEvents = true;
                }

                m_activeFallTrackIds = std::move(frameFallingTrackIds);

                if (!hasEvents)
                    return {};

                eventMetadataPacket->setTimestampUs(timestampUs);
                return {eventMetadataPacket};
            }

            Ptr<ObjectMetadataPacket> DeviceAgent::detectionsToObjectMetadataPacket(
                const DetectionList& detections,
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

                for (const std::shared_ptr<Detection>& detection : detections)
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
                for (const std::shared_ptr<Detection>& detection : detections)
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

            void DeviceAgent::reinitializeObjectTrackerOnFrameSizeChanges(const Frame& frame)
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
                const IUncompressedVideoFrame* videoFrame)
            {
                if (m_frameIndex % 200 == 0)
                {
                    std::cerr << "[DBG] pixelFormat=" << (int)videoFrame->pixelFormat()
                        << " w=" << videoFrame->width()
                        << " h=" << videoFrame->height()
                        << " lineSize0=" << videoFrame->lineSize(0)
                        << std::endl;
                }

                try
                {
                    // ⚠️ Frame constructor có thể ném exception (unsupported pixel format, cvtColor fail, etc)
                    Frame frame(videoFrame, m_frameIndex);
                    reinitializeObjectTrackerOnFrameSizeChanges(frame);

                    if (m_frameIndex % 200 == 0)
                    {
                        pushPluginDiagnosticEvent(
                            nx::sdk::IPluginDiagnosticEvent::Level::info,
                            "Calling detector",
                            "About to call Python /infer endpoint");
                    }

                    // 1) Gọi Python service -> lấy detections đã có track_id
                    DetectionList detections = m_objectDetector->run(frame, m_cameraId);

                    // 2) Dùng trực tiếp detections từ Python để tạo ObjectMetadata
                    const auto& objectMetadataPacket =
                        detectionsToObjectMetadataPacket(detections, frame.timestampUs);

                    // 3) Không còn events từ tracking, nên truyền EventList rỗng
                    EventList emptyEvents;
                    const auto& eventMetadataPacketList =
                        eventsToEventMetadataPacketList(emptyEvents, frame.timestampUs);
                    const auto& fallEventMetadataPacketList =
                        fallEventsToEventMetadataPacketList(detections, frame.timestampUs);

                    MetadataPacketList result;
                    if (objectMetadataPacket)
                        result.push_back(objectMetadataPacket);

                    result.insert(
                        result.end(),
                        std::make_move_iterator(eventMetadataPacketList.begin()),
                        std::make_move_iterator(eventMetadataPacketList.end()));
                    result.insert(
                        result.end(),
                        std::make_move_iterator(fallEventMetadataPacketList.begin()),
                        std::make_move_iterator(fallEventMetadataPacketList.end()));

                    return result;
                }
                catch (const ObjectDetectionError& e)
                {
                    // Log error nhưng KHÔNG terminate plugin
                    // Plugin sẽ tiếp tục chạy và retry lần sau
                    pushPluginDiagnosticEvent(
                        nx::sdk::IPluginDiagnosticEvent::Level::error,
                        "Object detection failed - will retry next frame",
                        e.what());
                }
                catch (const ObjectTrackingError& e)
                {
                    pushPluginDiagnosticEvent(
                        nx::sdk::IPluginDiagnosticEvent::Level::error,
                        "Object tracking error - will retry next frame",
                        e.what());
                }
                catch (const std::runtime_error& e)
                {
                    // Frame constructor throws std::runtime_error for unsupported pixel format
                    pushPluginDiagnosticEvent(
                        nx::sdk::IPluginDiagnosticEvent::Level::error,
                        "Frame conversion error (unsupported pixel format or OpenCV error) - skipping frame",
                        e.what());
                }
                catch (const std::exception& e)
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
