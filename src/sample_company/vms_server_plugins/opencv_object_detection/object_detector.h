// object_detector.h
// Copyright 2018-present Network Optix, Inc.
// Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#pragma once

#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <nx/sdk/uuid.h>

#include "detection.h"
#include "transport_client.h"

namespace sample_company {
namespace vms_server_plugins {
namespace opencv_object_detection {

struct AiServiceClientConfig
{
    static constexpr const char* kDefaultHost = "127.0.0.1";
    static constexpr int kDefaultPort = 18000;
    static constexpr bool kDefaultUseHttps = false;
    static constexpr int kDefaultConnectTimeoutMs = 2000;
    static constexpr int kDefaultReadTimeoutMs = 15000;
    static constexpr int kDefaultWriteTimeoutMs = 2000;
    static constexpr int kDefaultRetryCount = 3;
    static constexpr int kDefaultRetryBackoffMs = 250;

    // Transport settings for the local/remote analytics service. Auth uses X-API-Key.
    std::string host = kDefaultHost;
    int port = kDefaultPort;
    std::string apiKey;
    bool useHttps = kDefaultUseHttps;
    int connectTimeoutMs = kDefaultConnectTimeoutMs;
    int readTimeoutMs = kDefaultReadTimeoutMs;
    int writeTimeoutMs = kDefaultWriteTimeoutMs;
    int retryCount = kDefaultRetryCount;
    int retryBackoffMs = kDefaultRetryBackoffMs;
};

struct DebugDumpConfig
{
    static constexpr const char* kDefaultDumpDir = "debug_frames";
    static constexpr int kDefaultEveryNFrames = 1;

    bool enabled = false;
    std::string rootDir = kDefaultDumpDir;
    bool dumpInput = false;
    bool dumpOutput = false;
    int everyNFrames = kDefaultEveryNFrames;
};

// Result of a GET /health probe — used by the plugin health poll thread.
struct HealthCheckResult
{
    bool reachable = false;
    std::string status = "unknown"; //< "healthy" | "degraded" | "not_ready" | "unknown"
    std::vector<std::string> reason_codes;
    std::map<std::string, std::string> dependencies;
    std::string raw_error; //< Non-empty when reachable=false or parse failed
};

// Result of a GET /config/{camera_id} probe — used by the P2.2 config poll thread.
struct CameraConfigFetch
{
    bool reachable = false;
    std::string raw_error;
    // Nullable per-camera overrides; -1 means "not provided / use service default".
    float confidence_threshold = -1.0f;
    float iou_threshold        = -1.0f;
    int   frame_period         = -1;
    std::string raw_json;
};

class ObjectDetector
{
public:
    ObjectDetector();
    explicit ObjectDetector(const AiServiceClientConfig& serviceConfig);

    void ensureInitialized();
    bool isTerminated() const;
    void terminate();
    void setServiceConfig(const AiServiceClientConfig& serviceConfig);
    void setDebugDumpConfig(const DebugDumpConfig& debugConfig);

    // Call the configured local/remote analytics service with camera id and encoded frame payload.
    // frameWidth/frameHeight (P1-4) must be the jpegBytes' actual encoded pixel
    // dimensions; the caller (DeviceAgent) already knows them from the encode
    // step, which avoids a redundant cv::imdecode here just to recover them.
    DetectionList run(
        const std::string& cameraId,
        const std::vector<uint8_t>& jpegBytes,
        int frameWidth,
        int frameHeight);

    // Probe GET /health — lightweight, never throws. Used by health poll thread.
    HealthCheckResult checkHealth() const;

    // Probe GET /config/{cameraId} — lightweight, never throws. Used by P2.2 config poll thread.
    CameraConfigFetch fetchCameraConfig(const std::string& cameraId) const;

    // P2.3 — Register/update camera metadata in the service via PUT /admin/camera-configs.
    // Returns true on HTTP 200/201, false on any failure. Never throws.
    // confidenceThreshold in [0,1] is optional; omitted when < 0.
    bool registerCamera(
        const std::string& cameraId,
        const std::string& displayName,
        float confidenceThreshold = -1.0f) const;

private:
    AiServiceClientConfig serviceConfig() const;
    DebugDumpConfig debugDumpConfig() const;
    DetectionList callPythonService(
        const std::string& cameraId,
        const std::vector<uint8_t>& jpegBytes,
        int frameWidth,
        int frameHeight);

private:
    mutable std::mutex m_serviceConfigMutex;
    AiServiceClientConfig m_serviceConfig;
    mutable std::mutex m_debugConfigMutex;
    DebugDumpConfig m_debugConfig;
    bool m_terminated = false;
};

// P1-4 — ITransportClient implementation for the current, default JSON+base64
// transport. Deliberately a thin wrapper around ObjectDetector::run(): the
// existing method already owns circuit breaking, retries, response parsing
// and debug dumping, all proven in production, so this class only adapts
// that call to the Strategy interface DeviceAgent depends on. See
// MultipartBinaryTransport (multipart_binary_transport.h) for the additive,
// opt-in alternative.
class JsonBase64Transport: public ITransportClient
{
public:
    explicit JsonBase64Transport(ObjectDetector& detector): m_detector(detector) {}

    DetectionList sendFrame(
        const std::string& cameraId,
        const std::vector<uint8_t>& jpegBytes,
        int frameWidth,
        int frameHeight) override
    {
        return m_detector.run(cameraId, jpegBytes, frameWidth, frameHeight);
    }

private:
    ObjectDetector& m_detector;
};

} // namespace opencv_object_detection
} // namespace vms_server_plugins
} // namespace sample_company
