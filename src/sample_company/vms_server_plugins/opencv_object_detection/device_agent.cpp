// device_agent.cpp
// Copyright 2018-present Network Optix, Inc.
// Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#include "device_agent.h"
#include <set>
#include <algorithm>
#include <chrono>
#include <exception>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <opencv2/core.hpp>
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
                int renderTrackHoldMs(int targetEnqueueFps)
                {
                    return std::clamp(1800 / std::max(1, targetEnqueueFps), 220, 420);
                }

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

                // Returns the stable Nx camera UUID used as the service-side camera_id key.
                // Prefers deviceInfo->id() (UUID, stable across renames) and falls back to
                // the display name only when no id is available.
                std::string resolveStableCameraId(const nx::sdk::IDeviceInfo* deviceInfo)
                {
                    if (!deviceInfo)
                        return "unknown_camera";

                    if constexpr (HasIdMethod<nx::sdk::IDeviceInfo>::value)
                    {
                        const char* raw = deviceInfo->id();
                        if (raw && *raw)
                            return raw;
                    }

                    if constexpr (HasNameMethod<nx::sdk::IDeviceInfo>::value)
                    {
                        const char* raw = deviceInfo->name();
                        if (raw && *raw)
                            return raw;
                    }

                    return "unknown_camera";
                }

                std::string trimSettingValue(const std::string& raw)
                {
                    size_t begin = 0;
                    while (begin < raw.size()
                        && std::isspace(static_cast<unsigned char>(raw[begin])))
                    {
                        ++begin;
                    }

                    size_t end = raw.size();
                    while (end > begin
                        && std::isspace(static_cast<unsigned char>(raw[end - 1])))
                    {
                        --end;
                    }

                    return raw.substr(begin, end - begin);
                }

                int parseIntSettingValue(
                    const std::string& raw,
                    int defaultValue,
                    int minValue,
                    int maxValue)
                {
                    const std::string valueText = trimSettingValue(raw);
                    if (valueText.empty())
                        return defaultValue;

                    try
                    {
                        size_t parsedChars = 0;
                        int value = std::stoi(valueText, &parsedChars);
                        if (parsedChars != valueText.size())
                            return defaultValue;
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

                bool parseBoolSettingValue(const std::string& raw, bool defaultValue)
                {
                    std::string value = trimSettingValue(raw);
                    if (value.empty())
                        return defaultValue;

                    std::transform(value.begin(), value.end(), value.begin(),
                        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

                    if (value == "1" || value == "true" || value == "yes" || value == "on")
                        return true;
                    if (value == "0" || value == "false" || value == "no" || value == "off")
                        return false;
                    return defaultValue;
                }

                std::string parseTextSettingValue(
                    const std::string& raw,
                    const std::string& defaultValue)
                {
                    const std::string value = trimSettingValue(raw);
                    return value.empty() ? defaultValue : value;
                }

                std::filesystem::path resolveRelativeDebugDumpRoot(
                    const std::filesystem::path& pluginHomeDir,
                    const std::string& configuredDir,
                    bool* outUsedFallback = nullptr)
                {
                    if (outUsedFallback)
                        *outUsedFallback = false;

                    const auto fallbackRoot =
                        (pluginHomeDir / DebugDumpConfig::kDefaultDumpDir).lexically_normal();
                    const auto fallbackToDefault = [&]() -> std::filesystem::path
                    {
                        if (outUsedFallback)
                            *outUsedFallback = true;
                        return fallbackRoot;
                    };

                    const std::filesystem::path configuredPath(trimSettingValue(configuredDir));
                    if (configuredPath.empty())
                        return fallbackRoot;

                    if (configuredPath.is_absolute()
                        || configuredPath.has_root_name()
                        || configuredPath.has_root_directory())
                    {
                        return fallbackToDefault();
                    }

                    const std::filesystem::path normalized = configuredPath.lexically_normal();
                    if (normalized.empty() || normalized == ".")
                        return fallbackRoot;

                    for (const auto& part: normalized)
                    {
                        if (part == std::filesystem::path(".."))
                            return fallbackToDefault();
                    }

                    return (pluginHomeDir / normalized).lexically_normal();
                }

                std::string printableSettingValue(const std::string& raw)
                {
                    const std::string value = trimSettingValue(raw);
                    return value.empty() ? std::string("<empty>") : value;
                }

                std::string maskedSecretSettingValue(const std::string& raw)
                {
                    return trimSettingValue(raw).empty()
                        ? std::string("not_set")
                        : std::string("***configured***");
                }

                const char* boolToString(bool value)
                {
                    return value ? "true" : "false";
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
                std::filesystem::path pluginHomeDir)
                : ConsumingDeviceAgent(deviceInfo, /*enableOutput*/ true),
                  m_pluginHomeDir(std::move(pluginHomeDir)),
                  m_cameraName(resolveCameraName(deviceInfo)),
                  m_cameraId(resolveStableCameraId(deviceInfo)),
                  m_objectDetector(std::make_unique<ObjectDetector>()),
                  m_objectTracker(std::make_unique<ObjectTracker>()),
                  m_workerThread(&DeviceAgent::workerThreadRun, this),
                  m_workerShouldStop(false),
                  m_healthPollThread(&DeviceAgent::healthPollThreadRun, this), // P P1.1
                  m_configPollThread(&DeviceAgent::configPollThreadRun, this)  // P2.2
            {
                logutil::log(
                    logutil::Level::info,
                    "DeviceAgent camera_name=\"" + m_cameraName + "\""
                        " camera_id=\"" + m_cameraId + "\"");
                logutil::log(
                    logutil::Level::info,
                    "Plugin log file: " + logutil::configuredLogFilePath());
            }

            DeviceAgent::~DeviceAgent()
            {
                // Stop frame worker thread
                {
                    std::unique_lock<std::mutex> lk(m_frameQueueMutex);
                    m_workerShouldStop = true;
                }
                m_frameQueueCV.notify_one();
                if (m_workerThread.joinable())
                    m_workerThread.join();

                // P P1.1 – stop health poll thread
                m_healthPollShouldStop.store(true);
                m_healthPollCV.notify_one();
                if (m_healthPollThread.joinable())
                    m_healthPollThread.join();

                // P2.2 – stop config poll thread
                m_configPollShouldStop.store(true);
                m_configPollCV.notify_one();
                if (m_configPollThread.joinable())
                    m_configPollThread.join();
            }

            // ============================================================
            // P P1.1 – Health poll thread: probes GET /health every N seconds,
            // emits Nx diagnostic events on status transitions.
            // ============================================================
            void DeviceAgent::healthPollThreadRun()
            {
                // Wait a short initial delay so the service has time to warm up
                // before the first probe is attempted.
                {
                    std::unique_lock<std::mutex> lk(m_healthPollMutex);
                    m_healthPollCV.wait_for(
                        lk,
                        std::chrono::seconds(kHealthPollInitialDelaySec),
                        [this]() { return m_healthPollShouldStop.load(); });
                    if (m_healthPollShouldStop.load())
                        return;
                }

                std::string lastStatus; //< empty = no probe done yet

                while (true)
                {
                    const HealthCheckResult result = m_objectDetector->checkHealth();

                    const std::string& newStatus = result.reachable
                        ? result.status
                        : "not_reachable";

                    // Build detail string shared across event types
                    std::string details;
                    if (!result.reachable)
                    {
                        details = "error=" + result.raw_error;
                    }
                    else
                    {
                        if (!result.reason_codes.empty())
                        {
                            details += "reason=[";
                            for (const auto& rc : result.reason_codes)
                                details += rc + " ";
                            details += "] ";
                        }
                        for (const auto& [dep, depStatus] : result.dependencies)
                            details += dep + "=" + depStatus + " ";
                    }
                    if (details.empty())
                        details = "status=" + newStatus;

                    logutil::logThrottled(
                        logutil::Level::info,
                        "device_agent.health_poll." + m_cameraName,
                        std::chrono::seconds(60),
                        "Health probe camera=\"" + m_cameraName + "\" " + details);

                    if (newStatus != lastStatus)
                    {
                        if (newStatus == "healthy")
                        {
                            // Recovery event — only emit if we had a prior non-healthy state
                            if (!lastStatus.empty() && lastStatus != "healthy")
                            {
                                pushPluginDiagnosticEvent(
                                    nx::sdk::IPluginDiagnosticEvent::Level::info,
                                    "Analytics service recovered",
                                    ("status=healthy camera=\"" + m_cameraName + "\"").c_str());
                                logutil::log(
                                    logutil::Level::info,
                                    "Health poll: service recovered camera=\"" + m_cameraName + "\"");
                            }
                        }
                        else if (newStatus == "degraded")
                        {
                            pushPluginDiagnosticEvent(
                                nx::sdk::IPluginDiagnosticEvent::Level::warning,
                                "Analytics service degraded",
                                details.c_str());
                            logutil::log(
                                logutil::Level::warn,
                                "Health poll: service degraded camera=\"" + m_cameraName +
                                    "\" " + details);
                        }
                        else // not_ready | not_reachable | unknown
                        {
                            pushPluginDiagnosticEvent(
                                nx::sdk::IPluginDiagnosticEvent::Level::error,
                                "Analytics service not available",
                                (details + " camera=\"" + m_cameraName + "\"").c_str());
                            logutil::log(
                                logutil::Level::error,
                                "Health poll: service not available camera=\"" +
                                    m_cameraName + "\" " + details);
                        }
                        lastStatus = newStatus;
                    }

                    // Wait for next poll interval (interruptible by stop signal)
                    {
                        std::unique_lock<std::mutex> lk(m_healthPollMutex);
                        const int intervalSec = std::max(
                            5, m_healthPollIntervalSec.load(std::memory_order_relaxed));
                        m_healthPollCV.wait_for(
                            lk,
                            std::chrono::seconds(intervalSec),
                            [this]() { return m_healthPollShouldStop.load(); });
                        if (m_healthPollShouldStop.load())
                            break;
                    }
                }
            }

            // ============================================================
            // P2.2 – Config poll thread: queries GET /config/{camera_id}
            // every N seconds; applies frame_period if provided.
            // ============================================================
            void DeviceAgent::configPollThreadRun()
            {
                // Short initial delay (service is warming up at plugin start).
                {
                    std::unique_lock<std::mutex> lk(m_configPollMutex);
                    m_configPollCV.wait_for(
                        lk,
                        std::chrono::seconds(kConfigPollInitialDelaySec),
                        [this]() { return m_configPollShouldStop.load(); });
                    if (m_configPollShouldStop.load())
                        return;
                }

                // P2.3 – Register this camera with the service so the service knows its
                // display name. Best-effort: a failure just logs and the poll loop continues.
                {
                    const bool ok = m_objectDetector->registerCamera(m_cameraId, m_cameraName);
                    if (ok)
                        logutil::log(logutil::Level::info,
                            "Camera registered with service camera_id=\"" + m_cameraId +
                                "\" display_name=\"" + m_cameraName + "\"");
                    else
                        logutil::logThrottled(
                            logutil::Level::warn,
                            "device_agent.register_camera." + m_cameraId,
                            std::chrono::seconds(60),
                            "Camera registration failed (service may not be ready yet) id=\"" + m_cameraId + "\"");
                }

                while (true)
                {
                    const CameraConfigFetch cfg = m_objectDetector->fetchCameraConfig(m_cameraId);

                    if (!cfg.reachable)
                    {
                        logutil::logThrottled(
                            logutil::Level::warn,
                            "device_agent.config_poll." + m_cameraId,
                            std::chrono::seconds(120),
                            "Config poll unreachable camera_id=\"" + m_cameraId +
                                "\" name=\"" + m_cameraName + "\" error=" + cfg.raw_error);
                    }
                    else
                    {
                        std::string summary;

                        if (cfg.frame_period >= 1)
                        {
                            const int prev = m_detectionFramePeriod.load(std::memory_order_relaxed);
                            if (cfg.frame_period != prev)
                            {
                                m_detectionFramePeriod.store(cfg.frame_period, std::memory_order_relaxed);
                                summary += "frame_period=" + std::to_string(cfg.frame_period) + "(applied) ";
                            }
                            else
                            {
                                summary += "frame_period=" + std::to_string(cfg.frame_period) + " ";
                            }
                        }

                        if (cfg.confidence_threshold >= 0.0f)
                            summary += "conf=" + std::to_string(cfg.confidence_threshold) + "(service-side) ";

                        if (cfg.iou_threshold >= 0.0f)
                            summary += "iou=" + std::to_string(cfg.iou_threshold) + "(service-side) ";

                        if (summary.empty())
                            summary = "(no overrides)";

                        logutil::logThrottled(
                            logutil::Level::info,
                            "device_agent.config_poll." + m_cameraId,
                            std::chrono::seconds(300),
                            "Per-camera config camera_id=\"" + m_cameraId +
                                "\" name=\"" + m_cameraName + "\" " + summary);
                    }

                    // Wait for next interval (interruptible by stop signal).
                    {
                        std::unique_lock<std::mutex> lk(m_configPollMutex);
                        const int intervalSec = std::max(
                            30, m_configPollIntervalSec.load(std::memory_order_relaxed));
                        m_configPollCV.wait_for(
                            lk,
                            std::chrono::seconds(intervalSec),
                            [this]() { return m_configPollShouldStop.load(); });
                        if (m_configPollShouldStop.load())
                            break;
                    }
                }
            }

            std::string DeviceAgent::manifestString() const
            {
                return /*suppress newline*/ 1 + R"json(
{
    "typeLibrary": {
        "objectTypes": [
            {
                "id": ")json" +
                       kPersonObjectType + R"json(",
                "name": "HUMAN DETECTED"
            },
            {
                "id": ")json" +
                       kCatObjectType + R"json(",
                "name": "Cat"
            },
            {
                "id": ")json" +
                       kDogObjectType + R"json(",
                "name": "Dog"
            }
        ],
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
            },
            {
                "id": ")json" +
                       kZoneViolationEventType + R"json(",
                "name": "Zone violation",
                "flags": "stateDependent"
            }
        ]
    },
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
        },
        {
            "eventTypeId": ")json" +
                       kDetectionEventType + R"json("
        },
        {
            "eventTypeId": ")json" +
                       kProlongedDetectionEventType + R"json("
        },
        {
            "eventTypeId": ")json" +
                       kFallDetectedEventType + R"json("
        },
        {
            "eventTypeId": ")json" +
                       kZoneViolationEventType + R"json("
        }
    ],
    "deviceAgentSettingsModel": {
        "type": "Settings",
        "items": [
            {
                "type": "CheckBox",
                "name": "enabled",
                "caption": "Enable Detection",
                "defaultValue": true
            },
            {
                "type": "SpinBox",
                "name": "detection_frame_period",
                "caption": "Detection Frame Period",
                "defaultValue": 2,
                "minValue": 1,
                "maxValue": 60,
                "description": "Run detection every Nth frame (1 = every frame)."
            },
            {
                "type": "SpinBox",
                "name": "target_enqueue_fps",
                "caption": "Target Enqueue FPS",
                "defaultValue": 2,
                "minValue": 1,
                "maxValue": 60,
                "description": "Max frame enqueue rate to AI worker. Use low values on CPU-only AI Box."
            },
            {
                "type": "SpinBox",
                "name": "frame_queue_max_size",
                "caption": "Frame Queue Max Size",
                "defaultValue": 1,
                "minValue": 1,
                "maxValue": 100,
                "description": "Max buffered frames before dropping oldest (1 = prefer freshest frame)."
            },
            {
                "type": "SpinBox",
                "name": "metrics_log_period_sec",
                "caption": "Metrics Period (sec)",
                "defaultValue": 10,
                "description": "Periodic pipeline metrics interval."
            },
            {
                "type": "TextField",
                "name": "service_host",
                "caption": "Service Host",
                "defaultValue": "127.0.0.1",
                "description": "Host name or IP address of the Python analytics service."
            },
            {
                "type": "SpinBox",
                "name": "service_port",
                "caption": "Service Port",
                "defaultValue": 18000,
                "description": "TCP port of the Python analytics service."
            },
            {
                "type": "PasswordField",
                "name": "service_api_key",
                "caption": "Service API Key",
                "defaultValue": "",
                "description": "Optional X-API-Key header value sent to the analytics service."
            },
            {
                "type": "CheckBox",
                "name": "service_use_https",
                "caption": "Use HTTPS",
                "defaultValue": false,
                "description": "Enable HTTPS when connecting to the analytics service."
            },
            {
                "type": "SpinBox",
                "name": "service_connect_timeout_ms",
                "caption": "Connect Timeout (ms)",
                "defaultValue": 2000,
                "description": "Socket connect timeout for analytics service requests."
            },
            {
                "type": "SpinBox",
                "name": "service_read_timeout_ms",
                "caption": "Read Timeout (ms)",
                "defaultValue": 15000,
                "description": "Response read timeout for analytics service requests."
            },
            {
                "type": "SpinBox",
                "name": "service_write_timeout_ms",
                "caption": "Write Timeout (ms)",
                "defaultValue": 2000,
                "description": "Request body write timeout for analytics service requests."
            },
            {
                "type": "SpinBox",
                "name": "service_retry_count",
                "caption": "Retry Count",
                "defaultValue": 3,
                "description": "Number of retry attempts after the initial request."
            },
            {
                "type": "SpinBox",
                "name": "service_retry_backoff_ms",
                "caption": "Retry Backoff (ms)",
                "defaultValue": 250,
                "description": "Delay between analytics service retry attempts."
            },
            {
                "type": "CheckBox",
                "name": "debug_dump_enabled",
                "caption": "Debug Frame Dump Enabled",
                "defaultValue": false,
                "description": "Enable frame dumping to disk for debugging (default off)."
            },
            {
                "type": "TextField",
                "name": "debug_dump_dir",
                "caption": "Debug Dump Directory",
                "defaultValue": "debug_frames",
                "description": "Relative path for debug frame dump directory."
            },
            {
                "type": "CheckBox",
                "name": "debug_dump_input",
                "caption": "Debug Dump Input Frames",
                "defaultValue": false,
                "description": "Dump input frames (when debug enabled)."
            },
            {
                "type": "CheckBox",
                "name": "debug_dump_output",
                "caption": "Debug Dump Output Frames",
                "defaultValue": false,
                "description": "Dump output frames with bounding boxes (when debug enabled)."
            },
            {
                "type": "SpinBox",
                "name": "debug_dump_every_n_frames",
                "caption": "Debug Dump Every N Frames",
                "defaultValue": 1,
                "description": "Sample rate for frame dumping (1 = every frame, 10 = every 10th frame)."
            },
            {
                "type": "SpinBox",
                "name": "queue_depth_warn_pct",
                "caption": "Queue Depth Warning (%)",
                "defaultValue": 80,
                "minValue": 0,
                "maxValue": 100,
                "description": "Emit a warning diagnostic when queue depth exceeds this percentage of its maximum capacity (0 = disabled)."
            },
            {
                "type": "SpinBox",
                "name": "drop_rate_warn_per_sec",
                "caption": "Drop Rate Warning (frames/sec)",
                "defaultValue": 5,
                "minValue": 0,
                "description": "Emit a warning diagnostic when the frame drop rate exceeds this threshold in frames per second (0 = disabled)."
            },
            {
                "type": "SpinBox",
                "name": "health_poll_interval_sec",
                "caption": "Health Poll Interval (sec)",
                "defaultValue": 30,
                "minValue": 5,
                "description": "How often the plugin polls GET /health on the analytics service and emits diagnostic events on status changes."
            }
        ]
    }
}
)json";
            }

            void DeviceAgent::updateRenderedTrackState(const DetectionList& detections)
            {
                std::unique_lock<std::mutex> lk(m_renderStateMutex);
                const auto now = std::chrono::steady_clock::now();
                const int targetEnqueueFps = std::max(
                    1, m_targetEnqueueFps.load(std::memory_order_relaxed));
                const auto holdWindow = std::chrono::milliseconds(renderTrackHoldMs(targetEnqueueFps));

                for (const auto& detection: detections)
                {
                    if (!detection)
                        continue;

                    m_renderedTrackStates[detection->trackId] = RenderedDetectionState{
                        detection,
                        now};
                }

                for (auto it = m_renderedTrackStates.begin(); it != m_renderedTrackStates.end();)
                {
                    if (now - it->second.lastSeen > holdWindow)
                        it = m_renderedTrackStates.erase(it);
                    else
                        ++it;
                }

                if (!detections.empty() || !m_renderedTrackStates.empty())
                    m_lastRenderedTrackStateUpdateTime = now;
            }

            Ptr<ObjectMetadataPacket> DeviceAgent::renderCurrentObjectMetadataPacket(
                int64_t timestampUs)
            {
                DetectionList detections;
                {
                    std::unique_lock<std::mutex> lk(m_renderStateMutex);
                    if (m_lastRenderedTrackStateUpdateTime ==
                        std::chrono::steady_clock::time_point::min())
                    {
                        return nullptr;
                    }

                    const int targetEnqueueFps = std::max(
                        1, m_targetEnqueueFps.load(std::memory_order_relaxed));
                    const int ttlMs = renderTrackHoldMs(targetEnqueueFps);
                    if (std::chrono::steady_clock::now() - m_lastRenderedTrackStateUpdateTime >
                        std::chrono::milliseconds(ttlMs))
                    {
                        return nullptr;
                    }

                    const auto now = std::chrono::steady_clock::now();
                    for (auto it = m_renderedTrackStates.begin(); it != m_renderedTrackStates.end();)
                    {
                        if (now - it->second.lastSeen > std::chrono::milliseconds(ttlMs))
                        {
                            it = m_renderedTrackStates.erase(it);
                            continue;
                        }

                        const auto& state = it->second;
                        if (state.detection)
                            detections.push_back(state.detection);
                        ++it;
                    }
                }

                if (detections.empty())
                    return nullptr;

                return detectionsToObjectMetadataPacket(detections, timestampUs);
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

                if (!m_detectionEnabled.load(std::memory_order_relaxed))
                {
                    MetadataPacketList cleanupPackets;
                    const bool shouldCleanup =
                        m_detectionDisableCleanupPending.exchange(false, std::memory_order_relaxed);
                    if (shouldCleanup)
                        cleanupPackets = buildDisabledCleanupPackets(videoFrame->timestampUs());

                    {
                        std::unique_lock<std::mutex> lk(m_metadataQueueMutex);
                        m_latestObjectMetadataPacket = nullptr;
                        if (shouldCleanup)
                        {
                            m_metadataQueue.clear();
                            for (auto& packet: cleanupPackets)
                                m_metadataQueue.push_back(std::move(packet));
                        }
                    }

                    ++m_frameIndex;
                    return true;
                }

                const int detectionFramePeriod = std::max(
                    1, m_detectionFramePeriod.load(std::memory_order_relaxed));
                const int targetEnqueueFps = std::max(
                    1, m_targetEnqueueFps.load(std::memory_order_relaxed));
                const size_t frameQueueMaxSize = std::max<size_t>(
                    1, m_frameQueueMaxSize.load(std::memory_order_relaxed));

                bool hasActiveRenderedTracks = false;
                {
                    std::unique_lock<std::mutex> lk(m_renderStateMutex);
                    if (!m_renderedTrackStates.empty() &&
                        m_lastRenderedTrackStateUpdateTime !=
                            std::chrono::steady_clock::time_point::min())
                    {
                        const int ttlMs = renderTrackHoldMs(targetEnqueueFps);
                        hasActiveRenderedTracks =
                            now - m_lastRenderedTrackStateUpdateTime <=
                            std::chrono::milliseconds(ttlMs);
                    }
                }

                const int effectiveTargetEnqueueFps = hasActiveRenderedTracks
                    ? targetEnqueueFps
                    : std::max(targetEnqueueFps, kAcquireBoostEnqueueFps);
                m_lastEffectiveEnqueueFps.store(
                    effectiveTargetEnqueueFps, std::memory_order_relaxed);

                bool shouldEnqueue = (m_frameIndex % detectionFramePeriod == 0);
                if (shouldEnqueue && effectiveTargetEnqueueFps > 0)
                {
                    const auto minIntervalMs =
                        std::chrono::milliseconds(1000 / effectiveTargetEnqueueFps);
                    if (m_lastEnqueueTime != std::chrono::steady_clock::time_point::min() &&
                        now - m_lastEnqueueTime < minIntervalMs)
                    {
                        shouldEnqueue = false;
                    }
                }

                if (shouldEnqueue)
                {
                    if (!isVideoFrameDecodable(videoFrame))
                    {
                        ++m_droppedFrameCount;
                        ++m_frameIndex;
                        return true;
                    }

                    m_lastEnqueueTime = now;
                    try
                    {
                        Frame frame(videoFrame, m_frameIndex);
                        std::vector<uint8_t> jpegBytes = encodeFrameToJpeg(frame, 640);

                        FrameJob job;
                        job.jpegBytes = std::move(jpegBytes);
                        job.frame = std::make_shared<Frame>(frame);
                        job.cameraId = m_cameraId; //< stable UUID, not display name
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
                        static std::chrono::steady_clock::time_point lastEncodingDiag;
                        const auto diagNow = std::chrono::steady_clock::now();
                        if (lastEncodingDiag == std::chrono::steady_clock::time_point::min() ||
                            diagNow - lastEncodingDiag >= std::chrono::seconds(10))
                        {
                            pushPluginDiagnosticEvent(
                                nx::sdk::IPluginDiagnosticEvent::Level::error,
                                "Frame encoding error",
                                e.what());
                            lastEncodingDiag = diagNow;
                        }
                    }
                }

                {
                    const auto objectMetadataPacket =
                        renderCurrentObjectMetadataPacket(videoFrame->timestampUs());
                    std::unique_lock<std::mutex> lk(m_metadataQueueMutex);
                    m_latestObjectMetadataPacket = objectMetadataPacket;
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
                            ", effective_enqueue_fps=" +
                            std::to_string(m_lastEffectiveEnqueueFps.load(std::memory_order_relaxed)) +
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
                            ", effective_enqueue_fps=" +
                            std::to_string(m_lastEffectiveEnqueueFps.load(std::memory_order_relaxed)) +
                            ", avg_infer_ms=" + std::to_string(avgInferMs);
                        pushPluginDiagnosticEvent(
                            nx::sdk::IPluginDiagnosticEvent::Level::info,
                            "Pipeline metrics",
                            diag.c_str());
                        m_lastMetricsDiagTime = now;
                    }

                    // P P1.4 – threshold-based warning diagnostics
                    const size_t queueMax =
                        std::max(size_t{1}, m_frameQueueMaxSize.load(std::memory_order_relaxed));
                    const int queuePct = static_cast<int>(queueLen * 100 / queueMax);
                    const double dropRate = (periodSec > 0.0)
                        ? (static_cast<double>(deltaDropped) / periodSec)
                        : 0.0;

                    const int warnPct = m_queueDepthWarnPct.load(std::memory_order_relaxed);
                    if (warnPct > 0 && queuePct >= warnPct)
                    {
                        if (m_lastQueueDepthThresholdWarnTime ==
                                std::chrono::steady_clock::time_point::min() ||
                            now - m_lastQueueDepthThresholdWarnTime >=
                                std::chrono::seconds(kThresholdWarnThrottleSec))
                        {
                            const std::string details =
                                "queue=" + std::to_string(queueLen) +
                                "/" + std::to_string(queueMax) +
                                " (" + std::to_string(queuePct) + "%" +
                                " >= threshold " + std::to_string(warnPct) + "%)";
                            logutil::log(
                                logutil::Level::warn,
                                "Queue depth threshold exceeded: " + details);
                            pushPluginDiagnosticEvent(
                                nx::sdk::IPluginDiagnosticEvent::Level::warning,
                                "Queue depth threshold exceeded",
                                details.c_str());
                            m_lastQueueDepthThresholdWarnTime = now;
                        }
                    }

                    const int warnRate = m_dropRateWarnPerSec.load(std::memory_order_relaxed);
                    if (warnRate > 0 && dropRate >= static_cast<double>(warnRate))
                    {
                        if (m_lastDropRateThresholdWarnTime ==
                                std::chrono::steady_clock::time_point::min() ||
                            now - m_lastDropRateThresholdWarnTime >=
                                std::chrono::seconds(kThresholdWarnThrottleSec))
                        {
                            const std::string details =
                                "drop_rate=" + std::to_string(dropRate).substr(0, 5) +
                                " frames/sec" +
                                " >= threshold " + std::to_string(warnRate) +
                                " frames/sec"
                                " (dropped=" + std::to_string(deltaDropped) +
                                " in " + std::to_string(periodSec).substr(0, 4) + "s)";
                            logutil::log(
                                logutil::Level::warn,
                                "Drop rate threshold exceeded: " + details);
                            pushPluginDiagnosticEvent(
                                nx::sdk::IPluginDiagnosticEvent::Level::warning,
                                "Drop rate threshold exceeded",
                                details.c_str());
                            m_lastDropRateThresholdWarnTime = now;
                        }
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
                if (m_latestObjectMetadataPacket)
                    metadataPackets->push_back(m_latestObjectMetadataPacket.releasePtr());
                while (!m_metadataQueue.empty())
                {
                    metadataPackets->push_back(m_metadataQueue.front().releasePtr());
                    m_metadataQueue.pop_front();
                }
                return true;
            }

            nx::sdk::Result<const nx::sdk::ISettingsResponse*> DeviceAgent::settingsReceived()
            {
                const std::string rawEnabled = settingValue("enabled");
                const std::string rawDetectionPeriod = settingValue("detection_frame_period");
                const std::string rawEnqueueFps = settingValue("target_enqueue_fps");
                const std::string rawQueueMax = settingValue("frame_queue_max_size");
                const std::string rawMetricsPeriod = settingValue("metrics_log_period_sec");
                const std::string rawServiceHost = settingValue("service_host");
                const std::string rawServicePort = settingValue("service_port");
                const std::string rawServiceApiKey = settingValue("service_api_key");
                const std::string rawServiceUseHttps = settingValue("service_use_https");
                const std::string rawServiceConnectTimeoutMs =
                    settingValue("service_connect_timeout_ms");
                const std::string rawServiceReadTimeoutMs =
                    settingValue("service_read_timeout_ms");
                const std::string rawServiceWriteTimeoutMs =
                    settingValue("service_write_timeout_ms");
                const std::string rawServiceRetryCount = settingValue("service_retry_count");
                const std::string rawServiceRetryBackoffMs =
                    settingValue("service_retry_backoff_ms");

                logutil::log(
                    logutil::Level::info,
                    "Raw settings from Nx: enabled=" +
                        printableSettingValue(rawEnabled) +
                        ", detection_frame_period=" +
                        printableSettingValue(rawDetectionPeriod) +
                        ", target_enqueue_fps=" + printableSettingValue(rawEnqueueFps) +
                        ", frame_queue_max_size=" + printableSettingValue(rawQueueMax) +
                        ", metrics_log_period_sec=" + printableSettingValue(rawMetricsPeriod) +
                        ", service_host=" + printableSettingValue(rawServiceHost) +
                        ", service_port=" + printableSettingValue(rawServicePort) +
                        ", service_api_key=" + maskedSecretSettingValue(rawServiceApiKey) +
                        ", service_use_https=" + printableSettingValue(rawServiceUseHttps) +
                        ", service_connect_timeout_ms=" +
                            printableSettingValue(rawServiceConnectTimeoutMs) +
                        ", service_read_timeout_ms=" +
                            printableSettingValue(rawServiceReadTimeoutMs) +
                        ", service_write_timeout_ms=" +
                            printableSettingValue(rawServiceWriteTimeoutMs) +
                        ", service_retry_count=" +
                            printableSettingValue(rawServiceRetryCount) +
                        ", service_retry_backoff_ms=" +
                            printableSettingValue(rawServiceRetryBackoffMs));

                const bool detectionEnabled = parseBoolSettingValue(rawEnabled, true);
                const int requestedDetectionPeriod = parseIntSettingValue(
                    rawDetectionPeriod,
                    kDefaultDetectionFramePeriod,
                    1,
                    60);
                const int requestedEnqueueFps = parseIntSettingValue(
                    rawEnqueueFps,
                    kDefaultTargetEnqueueFps,
                    1,
                    60);
                const int requestedQueueMax = parseIntSettingValue(
                    rawQueueMax,
                    static_cast<int>(kDefaultFrameQueueMaxSize),
                    1,
                    100);
                const int detectionPeriod = requestedDetectionPeriod;
                const int enqueueFps = requestedEnqueueFps;
                const int queueMax = requestedQueueMax;
                const int metricsPeriod = parseIntSettingValue(
                    rawMetricsPeriod,
                    kDefaultMetricsLogPeriodSec,
                    1,
                    300);
                AiServiceClientConfig serviceConfig;
                serviceConfig.host = parseTextSettingValue(
                    rawServiceHost,
                    AiServiceClientConfig::kDefaultHost);
                serviceConfig.port = parseIntSettingValue(
                    rawServicePort,
                    AiServiceClientConfig::kDefaultPort,
                    1,
                    65535);
                serviceConfig.apiKey = parseTextSettingValue(rawServiceApiKey, "");
                serviceConfig.useHttps = parseBoolSettingValue(
                    rawServiceUseHttps,
                    AiServiceClientConfig::kDefaultUseHttps);
                serviceConfig.connectTimeoutMs = parseIntSettingValue(
                    rawServiceConnectTimeoutMs,
                    AiServiceClientConfig::kDefaultConnectTimeoutMs,
                    100,
                    std::numeric_limits<int>::max());
                serviceConfig.readTimeoutMs = parseIntSettingValue(
                    rawServiceReadTimeoutMs,
                    AiServiceClientConfig::kDefaultReadTimeoutMs,
                    100,
                    std::numeric_limits<int>::max());
                serviceConfig.writeTimeoutMs = parseIntSettingValue(
                    rawServiceWriteTimeoutMs,
                    AiServiceClientConfig::kDefaultWriteTimeoutMs,
                    100,
                    std::numeric_limits<int>::max());
                serviceConfig.retryCount = parseIntSettingValue(
                    rawServiceRetryCount,
                    AiServiceClientConfig::kDefaultRetryCount,
                    0,
                    std::numeric_limits<int>::max());
                serviceConfig.retryBackoffMs = parseIntSettingValue(
                    rawServiceRetryBackoffMs,
                    AiServiceClientConfig::kDefaultRetryBackoffMs,
                    0,
                    std::numeric_limits<int>::max());

                const bool previousDetectionEnabled =
                    m_detectionEnabled.load(std::memory_order_relaxed);
                {
                    std::unique_lock<std::mutex> lifecycleLock(m_lifecycleStateMutex);
                    m_detectionEnabled.store(detectionEnabled, std::memory_order_relaxed);
                }
                m_detectionFramePeriod.store(detectionPeriod, std::memory_order_relaxed);
                m_targetEnqueueFps.store(enqueueFps, std::memory_order_relaxed);
                m_frameQueueMaxSize.store(static_cast<size_t>(queueMax), std::memory_order_relaxed);
                m_metricsLogPeriodSec.store(metricsPeriod, std::memory_order_relaxed);
                m_lastEffectiveEnqueueFps.store(
                    detectionEnabled ? enqueueFps : 0,
                    std::memory_order_relaxed);
                m_objectDetector->setServiceConfig(serviceConfig);

                if (!detectionEnabled)
                {
                    clearPendingFrameQueue();
                    m_detectionDisableCleanupPending.store(true, std::memory_order_relaxed);
                }
                else if (!previousDetectionEnabled)
                {
                    m_lastEnqueueTime = std::chrono::steady_clock::time_point::min();
                }

                logutil::log(
                    logutil::Level::info,
                    "Applied settings: enabled=" +
                        std::string(detectionEnabled ? "true" : "false") +
                        ", detection_frame_period=" +
                        std::to_string(m_detectionFramePeriod.load(std::memory_order_relaxed)) +
                        ", target_enqueue_fps=" +
                        std::to_string(m_targetEnqueueFps.load(std::memory_order_relaxed)) +
                        ", frame_queue_max_size=" +
                        std::to_string(m_frameQueueMaxSize.load(std::memory_order_relaxed)) +
                        ", metrics_log_period_sec=" +
                        std::to_string(m_metricsLogPeriodSec.load(std::memory_order_relaxed)) +
                        ", service_host=" + serviceConfig.host +
                        ", service_port=" + std::to_string(serviceConfig.port) +
                        ", service_use_https=" + std::string(boolToString(serviceConfig.useHttps)) +
                        ", service_api_key=" + maskedSecretSettingValue(serviceConfig.apiKey) +
                        ", service_connect_timeout_ms=" +
                            std::to_string(serviceConfig.connectTimeoutMs) +
                        ", service_read_timeout_ms=" +
                            std::to_string(serviceConfig.readTimeoutMs) +
                        ", service_write_timeout_ms=" +
                            std::to_string(serviceConfig.writeTimeoutMs) +
                        ", service_retry_count=" +
                            std::to_string(serviceConfig.retryCount) +
                        ", service_retry_backoff_ms=" +
                            std::to_string(serviceConfig.retryBackoffMs));

                // Parse debug settings
                const std::string rawDebugDumpEnabled = settingValue("debug_dump_enabled");
                const std::string rawDebugDumpDir = settingValue("debug_dump_dir");
                const std::string rawDebugDumpInput = settingValue("debug_dump_input");
                const std::string rawDebugDumpOutput = settingValue("debug_dump_output");
                const std::string rawDebugDumpEveryNFrames = settingValue("debug_dump_every_n_frames");

                const bool debugDumpEnabled = parseBoolSettingValue(rawDebugDumpEnabled, false);
                const std::string debugDumpDir = parseTextSettingValue(
                    rawDebugDumpDir,
                    DebugDumpConfig::kDefaultDumpDir);
                const bool debugDumpInput = parseBoolSettingValue(rawDebugDumpInput, false);
                const bool debugDumpOutput = parseBoolSettingValue(rawDebugDumpOutput, false);
                const int debugDumpEveryNFrames = parseIntSettingValue(
                    rawDebugDumpEveryNFrames,
                    DebugDumpConfig::kDefaultEveryNFrames,
                    1,
                    1000);
                bool debugDumpDirFallback = false;

                DebugDumpConfig debugConfig;
                debugConfig.enabled = debugDumpEnabled;
                debugConfig.rootDir = resolveRelativeDebugDumpRoot(
                    m_pluginHomeDir,
                    debugDumpDir,
                    &debugDumpDirFallback).string();
                debugConfig.dumpInput = debugDumpInput;
                debugConfig.dumpOutput = debugDumpOutput;
                debugConfig.everyNFrames = debugDumpEveryNFrames;

                {
                    std::lock_guard<std::mutex> lk(m_debugConfigMutex);
                    m_debugConfig = debugConfig;
                }
                m_objectDetector->setDebugDumpConfig(debugConfig);

                if (debugDumpEnabled)
                {
                    if (debugDumpDirFallback)
                    {
                        logutil::log(
                            logutil::Level::warn,
                            "Invalid debug_dump_dir rejected; using default relative directory: " +
                                std::string(DebugDumpConfig::kDefaultDumpDir));
                    }

                    logutil::log(
                        logutil::Level::info,
                        "Debug frame dump enabled: dir=" + debugConfig.rootDir +
                            ", input=" + std::string(debugDumpInput ? "true" : "false") +
                            ", output=" + std::string(debugDumpOutput ? "true" : "false") +
                            ", every_n_frames=" + std::to_string(debugDumpEveryNFrames));
                }

                // P P1.4 – parse warning thresholds
                const int queueDepthWarnPct = parseIntSettingValue(
                    settingValue("queue_depth_warn_pct"),
                    kDefaultQueueDepthWarnPct,
                    0,
                    100);
                const int dropRateWarnPerSec = parseIntSettingValue(
                    settingValue("drop_rate_warn_per_sec"),
                    kDefaultDropRateWarnPerSec,
                    0,
                    std::numeric_limits<int>::max());

                m_queueDepthWarnPct.store(queueDepthWarnPct, std::memory_order_relaxed);
                m_dropRateWarnPerSec.store(dropRateWarnPerSec, std::memory_order_relaxed);

                logutil::log(
                    logutil::Level::info,
                    "Threshold settings: queue_depth_warn_pct=" +
                        std::to_string(queueDepthWarnPct) +
                        (queueDepthWarnPct == 0 ? " (disabled)" : "%") +
                        ", drop_rate_warn_per_sec=" +
                        std::to_string(dropRateWarnPerSec) +
                        (dropRateWarnPerSec == 0 ? " (disabled)" : " frames/sec"));

                // P P1.1 – health poll interval
                const int healthPollIntervalSec = parseIntSettingValue(
                    settingValue("health_poll_interval_sec"),
                    kDefaultHealthPollIntervalSec,
                    5,
                    3600);
                m_healthPollIntervalSec.store(healthPollIntervalSec, std::memory_order_relaxed);
                logutil::log(
                    logutil::Level::info,
                    "Health poll interval=" + std::to_string(healthPollIntervalSec) + "s");

                // Kick the health poll thread so it re-reads config without waiting
                // out the remainder of the current sleep interval.
                m_healthPollCV.notify_one();

                // P2.2 – kick the config poll thread on settings change
                m_configPollCV.notify_one();

                return nullptr;
            }

            void DeviceAgent::doSetNeededMetadataTypes(
                nx::sdk::Result<void> *outValue,
                const nx::sdk::analytics::IMetadataTypes * /*neededMetadataTypes*/)
            {
                pushPluginDiagnosticEvent(
                    nx::sdk::IPluginDiagnosticEvent::Level::info,
                    "PLUGIN VERSION",
                    "yolo26_people_analytics_plugin.dll build=2025-12-14 v2");

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
                    if (!job.frame)
                        throw std::runtime_error("FrameJob is missing frame data for tracking");

                    if (!m_detectionEnabled.load(std::memory_order_relaxed))
                        return result;

                    // Call Python AI service with JPEG bytes
                    DetectionList detections = m_objectDetector->run(job.cameraId, job.jpegBytes);

                    // The Python service is already the authoritative source for track_id and
                    // track state. The plugin renders those detections directly and uses only
                    // stable, non-degraded tracks to drive lifecycle events.
                    {
                        std::unique_lock<std::mutex> lifecycleLock(m_lifecycleStateMutex);
                        if (!m_detectionEnabled.load(std::memory_order_relaxed))
                            return result;

                        updateRenderedTrackState(detections);

                        const auto participatesInLifecycle =
                            [](const std::shared_ptr<Detection>& detection) -> bool
                        {
                            return detection
                                && detection->classLabel == "person"
                                && detection->stable
                                && !detection->degraded;
                        };

                        bool hasStablePerson = false;
                        std::set<nx::sdk::Uuid> currentFallDetectedTrackIds;
                        std::set<nx::sdk::Uuid> currentZoneViolationTrackIds; // P2.1
                        // P2.1 — map track → zone info for event descriptions
                        std::map<nx::sdk::Uuid, std::string> trackZoneType;
                        std::map<nx::sdk::Uuid, std::string> trackZoneId;
                        EventList newTrackEvents;
                        for (const auto &detection : detections)
                        {
                            if (!participatesInLifecycle(detection))
                                continue;

                            hasStablePerson = true;
                            if (m_seenPersonIds.insert(detection->trackId).second)
                            {
                                newTrackEvents.push_back(std::make_shared<Event>(Event{
                                    EventType::object_detected,
                                    job.timestampUs,
                                    "person"}));
                            }

                            if (detection->fallDetected)
                                currentFallDetectedTrackIds.insert(detection->trackId);

                            // P2.1 — collect newly violated zone tracks
                            if (detection->zoneViolation && !detection->zoneType.empty())
                            {
                                currentZoneViolationTrackIds.insert(detection->trackId);
                                trackZoneType[detection->trackId] = detection->zoneType;
                                trackZoneId[detection->trackId]   = detection->zoneId;
                            }
                        }

                        if (!newTrackEvents.empty())
                        {
                            const auto newTrackEventPackets =
                                eventsToEventMetadataPacketList(newTrackEvents, job.timestampUs);
                            result.insert(
                                result.end(),
                                std::make_move_iterator(newTrackEventPackets.begin()),
                                std::make_move_iterator(newTrackEventPackets.end()));
                        }

                        if (hasStablePerson != m_personDetectionActive)
                        {
                            EventList personEvents;
                            personEvents.push_back(std::make_shared<Event>(Event{
                                hasStablePerson ? EventType::detection_started : EventType::detection_finished,
                                job.timestampUs,
                                "person"}));

                            const auto personEventPackets =
                                eventsToEventMetadataPacketList(personEvents, job.timestampUs);
                            result.insert(
                                result.end(),
                                std::make_move_iterator(personEventPackets.begin()),
                                std::make_move_iterator(personEventPackets.end()));

                            if (!hasStablePerson)
                                m_seenPersonIds.clear();

                            m_personDetectionActive = hasStablePerson;
                        }

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

                        // P2.1 — Zone violation events (stateful, mirrors fall pattern)
                        for (const auto &trackId : currentZoneViolationTrackIds)
                        {
                            if (m_activeZoneViolationTrackIds.count(trackId) > 0)
                                continue; // already active — don't re-fire

                            const std::string zt = trackZoneType.count(trackId)
                                ? trackZoneType.at(trackId) : "restricted";
                            const std::string zid = trackZoneId.count(trackId)
                                ? trackZoneId.at(trackId) : "";

                            auto eventMetadata = nx::sdk::makePtr<nx::sdk::analytics::EventMetadata>();
                            eventMetadata->setCaption("Zone violation");
                            eventMetadata->setDescription(
                                "Person " + nx::sdk::UuidHelper::toStdString(trackId) +
                                " entered " + zt + " zone" +
                                (zid.empty() ? "" : " (id=" + zid.substr(0, 8) + ")"));
                            eventMetadata->setIsActive(true);
                            eventMetadata->setTypeId(kZoneViolationEventType);

                            auto eventPacket = nx::sdk::makePtr<nx::sdk::analytics::EventMetadataPacket>();
                            eventPacket->addItem(eventMetadata.get());
                            eventPacket->setTimestampUs(job.timestampUs);
                            result.push_back(eventPacket);

                            m_activeZoneViolationTrackIds.insert(trackId);
                        }

                        std::vector<nx::sdk::Uuid> zoneToClear;
                        for (const auto &activeTrackId : m_activeZoneViolationTrackIds)
                        {
                            if (currentZoneViolationTrackIds.count(activeTrackId) > 0)
                                continue;
                            // Track left zone or disappeared — clear the active event
                            auto eventMetadata = nx::sdk::makePtr<nx::sdk::analytics::EventMetadata>();
                            eventMetadata->setCaption("Zone violation cleared");
                            eventMetadata->setDescription(
                                "Person " + nx::sdk::UuidHelper::toStdString(activeTrackId) +
                                " left restricted zone");
                            eventMetadata->setIsActive(false);
                            eventMetadata->setTypeId(kZoneViolationEventType);

                            auto eventPacket = nx::sdk::makePtr<nx::sdk::analytics::EventMetadataPacket>();
                            eventPacket->addItem(eventMetadata.get());
                            eventPacket->setTimestampUs(job.timestampUs);
                            result.push_back(eventPacket);

                            zoneToClear.push_back(activeTrackId);
                        }
                        for (const auto &trackId : zoneToClear)
                            m_activeZoneViolationTrackIds.erase(trackId);
                    }
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

            DeviceAgent::MetadataPacketList DeviceAgent::buildDisabledCleanupPackets(
                int64_t timestampUs)
            {
                MetadataPacketList result;
                std::unique_lock<std::mutex> lifecycleLock(m_lifecycleStateMutex);

                if (m_personDetectionActive)
                {
                    EventList personEvents;
                    personEvents.push_back(std::make_shared<Event>(Event{
                        EventType::detection_finished,
                        timestampUs,
                        "person"}));

                    const auto personEventPackets =
                        eventsToEventMetadataPacketList(personEvents, timestampUs);
                    result.insert(
                        result.end(),
                        std::make_move_iterator(personEventPackets.begin()),
                        std::make_move_iterator(personEventPackets.end()));
                }

                for (const auto& activeTrackId: m_activeFallDetectedTrackIds)
                {
                    auto eventMetadata = nx::sdk::makePtr<nx::sdk::analytics::EventMetadata>();
                    eventMetadata->setCaption("Fall cleared");
                    eventMetadata->setDescription(
                        "Person " + nx::sdk::UuidHelper::toStdString(activeTrackId) +
                        " is no longer fallen");
                    eventMetadata->setIsActive(false);
                    eventMetadata->setTypeId(kFallDetectedEventType);

                    auto eventPacket = nx::sdk::makePtr<nx::sdk::analytics::EventMetadataPacket>();
                    eventPacket->addItem(eventMetadata.get());
                    eventPacket->setTimestampUs(timestampUs);
                    result.push_back(eventPacket);
                }

                m_personDetectionActive = false;
                m_activeFallDetectedTrackIds.clear();
                m_seenPersonIds.clear();
                m_currentPersons = 0;

                {
                    std::unique_lock<std::mutex> renderLock(m_renderStateMutex);
                    m_renderedTrackStates.clear();
                    m_lastRenderedTrackStateUpdateTime =
                        std::chrono::steady_clock::time_point::min();
                }

                return result;
            }

            void DeviceAgent::clearPendingFrameQueue()
            {
                std::unique_lock<std::mutex> lk(m_frameQueueMutex);
                m_frameQueue.clear();
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
                const auto participatesInLifecycle =
                    [](const std::shared_ptr<Detection>& detection) -> bool
                {
                    return detection
                        && detection->classLabel == "person"
                        && detection->stable
                        && !detection->degraded;
                };

                // PASS 1: count persons in this frame.
                m_currentPersons = 0;
                for (const std::shared_ptr<Detection> &detection : detections)
                {
                    if (participatesInLifecycle(detection))
                        ++m_currentPersons;
                }

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

            // Identity fields: only publish after a successful match so Nx does not
            // aggregate "Unknown" with the resolved name across the object lifetime.
            if (detection->recognized)
            {
                if (!detection->personName.empty())
                {
                    objectMetadata->addAttribute(makePtr<Attribute>(
                        IAttribute::Type::string,
                        "Person name",
                        detection->personName));
                }
                if (!detection->personGender.empty())
                {
                    objectMetadata->addAttribute(makePtr<Attribute>(
                        IAttribute::Type::string,
                        "Gender",
                        detection->personGender));
                }
                if (detection->personAge >= 0)
                {
                    objectMetadata->addAttribute(makePtr<Attribute>(
                        IAttribute::Type::string,
                        "Age",
                        std::to_string(detection->personAge)));
                }
            }

            if (!detection->zoneName.empty())
            {
                objectMetadata->addAttribute(makePtr<Attribute>(
                    IAttribute::Type::string,
                    "Zone Name",
                    detection->zoneName));
            }
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

            // ── Legacy synchronous path ──────────────────────────────────────────────
            // NOT called during normal operation. The active path is:
            //   pushUncompressedVideoFrame() → worker queue → processFrameJob()
            // This method is kept to satisfy the ConsumingDeviceAgent interface but
            // is never invoked because doSetNeededMetadataTypes() always succeeds.
            //
            // The local m_objectTracker call below is intentionally isolated here.
            // The authoritative track_id comes from the Python service (via
            // processFrameJob). Do NOT call m_objectTracker from the worker path.
            // TODO P2: Remove this method and m_objectTracker entirely once
            //          ConsumingDeviceAgent no longer requires it.
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
                    Frame frame(videoFrame, m_frameIndex);
                    reinitializeObjectTrackerOnFrameSizeChanges(frame);

                    logutil::logThrottled(
                        logutil::Level::warn,
                        "device_agent.process_frame.legacy_path",
                        std::chrono::seconds(30),
                        "Legacy processFrame path invoked — this should not happen in normal operation");

                    // Local tracker used only here; Python service is authoritative.
                    std::vector<uint8_t> jpegBytes = encodeFrameToJpeg(frame, 640);
                    DetectionList detections = m_objectDetector->run(m_cameraId, jpegBytes);
                    const auto trackingResult = m_objectTracker->run(frame, detections);

                    const auto &objectMetadataPacket =
                        detectionsToObjectMetadataPacket(trackingResult.detections, frame.timestampUs);

                    const auto &eventMetadataPacketList =
                        eventsToEventMetadataPacketList(trackingResult.events, frame.timestampUs);

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
