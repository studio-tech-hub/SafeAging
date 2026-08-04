#include "multipart_binary_transport.h"
#include "detection_box_normalizer.h"
#include "exceptions.h"
#include "logging_utils.h"

#ifdef _MSC_VER
#pragma warning(push, 0)
#pragma warning(disable: 26439 26495 6294 6201 6262)
#endif

#include "httplib.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "json.hpp"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>

namespace sample_company {
namespace vms_server_plugins {
namespace opencv_object_detection {

using json = nlohmann::json;

namespace {

    // ── Small helpers duplicated from object_detector.cpp's anonymous namespace ──
    // These have internal linkage there (invisible across translation units), and
    // are intentionally re-implemented here rather than shared, to keep this
    // additive P1-4 transport fully isolated from the proven JSON+base64 path.
    // See detection_box_normalizer.h for the one piece of logic that *is* shared
    // (pure math, independently unit-tested).

    std::string normalizeCameraKey(const std::string& cameraId)
    {
        return cameraId.empty() ? "unknown_camera" : cameraId;
    }

    bool isTransientHttpStatus(int status)
    {
        return status == 408 || status == 429 || (status >= 500 && status <= 599);
    }

    std::string serviceEndpoint(const AiServiceClientConfig& config)
    {
        return std::string(config.useHttps ? "https://" : "http://") +
            config.host + ":" + std::to_string(config.port);
    }

    std::string responseBodySnippet(const httplib::Response& response)
    {
        if (response.body.empty())
            return {};

        constexpr size_t kMaxBodyChars = 160;
        std::string body = response.body.substr(0, kMaxBodyChars);
        if (response.body.size() > kMaxBodyChars)
            body += "...";
        return ", body=\"" + body + "\"";
    }

    std::string transportErrorDetails(
        const httplib::Result& result,
        const AiServiceClientConfig& config)
    {
        std::string details =
            "transport error=" + httplib::to_string(result.error()) +
            ", target=" + serviceEndpoint(config);

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        if (config.useHttps)
        {
            if (result.ssl_error() != 0)
                details += ", ssl_error=" + std::to_string(result.ssl_error());
            if (result.ssl_openssl_error() != 0)
                details += ", openssl_error=" + std::to_string(result.ssl_openssl_error());
        }
#endif

        return details;
    }

    template<typename ClientT>
    void configureServiceClient(ClientT& client, const AiServiceClientConfig& config)
    {
        client.set_keep_alive(true);
        client.set_connection_timeout(std::chrono::milliseconds(config.connectTimeoutMs));
        client.set_read_timeout(std::chrono::milliseconds(config.readTimeoutMs));
        client.set_write_timeout(std::chrono::milliseconds(config.writeTimeoutMs));
    }

    // Stable per-(cameraId, trackId) UUID cache, mirroring object_detector.cpp's
    // uuidFromTrackId(). Kept as a separate instance here (not shared) — safe
    // because only one transport is ever active per camera at a time; switching
    // transport_mode for a live camera may momentarily re-mint UUIDs for its
    // in-flight tracks, which at worst re-fires a "new object" lifecycle event
    // once, not a correctness issue.
    nx::sdk::Uuid uuidFromTrackId(const std::string& cameraId, int trackId)
    {
        struct TrackCacheKey
        {
            std::string cameraId;
            int trackId = 0;
            bool operator==(const TrackCacheKey& other) const
            {
                return trackId == other.trackId && cameraId == other.cameraId;
            }
        };
        struct TrackCacheKeyHash
        {
            size_t operator()(const TrackCacheKey& key) const
            {
                size_t h1 = std::hash<std::string>{}(key.cameraId);
                size_t h2 = std::hash<int>{}(key.trackId);
                return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
            }
        };
        struct TrackCacheEntry
        {
            nx::sdk::Uuid uuid;
            std::chrono::steady_clock::time_point lastSeen;
        };

        static std::mutex m;
        static std::unordered_map<TrackCacheKey, TrackCacheEntry, TrackCacheKeyHash> map;
        static size_t accessCount = 0;

        constexpr std::chrono::minutes kTrackUuidTtl{5};
        constexpr size_t kTrackUuidMaxSize = 10000;
        constexpr size_t kCleanupPeriod = 256;

        std::lock_guard<std::mutex> lk(m);
        const auto now = std::chrono::steady_clock::now();

        ++accessCount;
        if (accessCount % kCleanupPeriod == 0)
        {
            for (auto it = map.begin(); it != map.end();)
            {
                if (now - it->second.lastSeen > kTrackUuidTtl)
                    it = map.erase(it);
                else
                    ++it;
            }
            if (map.size() > kTrackUuidMaxSize)
            {
                std::vector<std::pair<TrackCacheKey, std::chrono::steady_clock::time_point>> entries;
                entries.reserve(map.size());
                for (const auto& kv : map)
                    entries.push_back({kv.first, kv.second.lastSeen});
                std::sort(entries.begin(), entries.end(),
                    [](const auto& a, const auto& b) { return a.second < b.second; });
                const size_t toRemove = map.size() - kTrackUuidMaxSize;
                for (size_t i = 0; i < toRemove; ++i)
                    map.erase(entries[i].first);
            }
        }

        const TrackCacheKey key{cameraId.empty() ? "unknown_camera" : cameraId, trackId};
        auto it = map.find(key);
        if (it != map.end())
        {
            it->second.lastSeen = now;
            return it->second.uuid;
        }

        thread_local std::mt19937_64 rng(std::random_device{}());
        uint8_t bytes[nx::sdk::Uuid::kSize];
        for (size_t i = 0; i < nx::sdk::Uuid::kSize; i += sizeof(uint64_t))
        {
            const uint64_t value = rng();
            const size_t copySize = std::min(sizeof(value), nx::sdk::Uuid::kSize - i);
            std::memcpy(bytes + i, &value, copySize);
        }
        bytes[6] = static_cast<uint8_t>((bytes[6] & 0x0F) | 0x40);
        bytes[8] = static_cast<uint8_t>((bytes[8] & 0x3F) | 0x80);
        nx::sdk::Uuid u(bytes);

        map.emplace(key, TrackCacheEntry{u, now});
        return u;
    }

    // Input-only debug dump: the raw JPEG bytes are already exactly what would
    // be written to disk, so this writes them directly with no decode step —
    // simpler and cheaper than object_detector.cpp's cv::imwrite(cv::imdecode(...))
    // round-trip. Output-frame dumping (annotated with boxes) is not supported
    // for this transport yet; see DebugDumpConfig::dumpOutput handling below.
    void dumpRawJpegBytes(
        const std::string& rootDir,
        const std::string& cameraId,
        uint64_t seq,
        const std::vector<uint8_t>& jpegBytes)
    {
        std::error_code ec;
        const std::filesystem::path inputDir = std::filesystem::path(rootDir) / "input";
        std::filesystem::create_directories(inputDir, ec);
        if (ec)
        {
            logutil::logThrottled(
                logutil::Level::warn,
                "multipart_binary_transport.dump_dir_failed",
                std::chrono::seconds(30),
                "Failed to create debug dump input directory: " + inputDir.string() +
                    " (" + ec.message() + ")");
            return;
        }

        std::string fileName = cameraId.empty() ? "unknown_camera" : cameraId;
        for (char& ch : fileName)
        {
            const unsigned char u = static_cast<unsigned char>(ch);
            if (!std::isalnum(u) && ch != '-' && ch != '_')
                ch = '_';
        }
        constexpr uint64_t kMaxFrameFiles = 100;
        fileName += "_" + std::to_string(seq % kMaxFrameFiles) + ".jpg";

        std::ofstream out((inputDir / fileName).string(), std::ios::binary | std::ios::trunc);
        if (!out)
        {
            logutil::logThrottled(
                logutil::Level::warn,
                "multipart_binary_transport.dump_write_failed",
                std::chrono::seconds(30),
                "Failed to open debug dump file for write: " + (inputDir / fileName).string());
            return;
        }
        out.write(reinterpret_cast<const char*>(jpegBytes.data()), static_cast<std::streamsize>(jpegBytes.size()));
    }

} // namespace (anonymous)

MultipartBinaryTransport::MultipartBinaryTransport():
    m_circuitBreaker(/*failureThreshold*/ 5, /*cooldown*/ std::chrono::seconds{15})
{
}

void MultipartBinaryTransport::setServiceConfig(const AiServiceClientConfig& serviceConfig)
{
    std::lock_guard<std::mutex> lk(m_serviceConfigMutex);
    m_serviceConfig = serviceConfig;
}

AiServiceClientConfig MultipartBinaryTransport::serviceConfig() const
{
    std::lock_guard<std::mutex> lk(m_serviceConfigMutex);
    return m_serviceConfig;
}

void MultipartBinaryTransport::setDebugDumpConfig(const DebugDumpConfig& debugConfig)
{
    std::lock_guard<std::mutex> lk(m_debugConfigMutex);
    m_debugConfig = debugConfig;
}

DebugDumpConfig MultipartBinaryTransport::debugDumpConfig() const
{
    std::lock_guard<std::mutex> lk(m_debugConfigMutex);
    return m_debugConfig;
}

DetectionList MultipartBinaryTransport::sendFrame(
    const std::string& cameraId,
    const std::vector<uint8_t>& jpegBytes,
    int frameWidth,
    int frameHeight)
{
    DetectionList result;

    if (jpegBytes.empty())
        throw ObjectDetectionError("JPEG bytes are empty");
    if (frameWidth <= 0 || frameHeight <= 0)
    {
        throw ObjectDetectionError(
            "Invalid frame dimensions passed to MultipartBinaryTransport::sendFrame: " +
            std::to_string(frameWidth) + "x" + std::to_string(frameHeight));
    }

    const std::string normalizedCameraId = normalizeCameraKey(cameraId);
    const AiServiceClientConfig config = serviceConfig();
    const DebugDumpConfig debugConfig = debugDumpConfig();
    const std::string serviceTarget = serviceEndpoint(config);

    try
    {
        std::string breakerReason;
        bool halfOpenTransition = false;
        if (!m_circuitBreaker.allowRequest(normalizedCameraId, &breakerReason, &halfOpenTransition))
        {
            logutil::logThrottled(
                logutil::Level::warn,
                "multipart_binary_transport.circuit_open." + normalizedCameraId,
                std::chrono::seconds(30),
                "Circuit breaker OPEN for camera \"" + normalizedCameraId +
                    "\", fail-fast: " + breakerReason);
            throw ObjectDetectionError(
                "Circuit breaker open for camera \"" + normalizedCameraId + "\"");
        }

        if (halfOpenTransition)
        {
            logutil::logThrottled(
                logutil::Level::info,
                "multipart_binary_transport.circuit_half_open." + normalizedCameraId,
                std::chrono::seconds(10),
                "Circuit breaker HALF_OPEN probe for camera \"" + normalizedCameraId + "\"");
        }

        if (debugConfig.enabled && debugConfig.dumpInput)
        {
            static std::atomic<uint64_t> s_dumpSeq{0};
            const uint64_t seq = ++s_dumpSeq;
            if ((seq - 1) % std::max(1, debugConfig.everyNFrames) == 0)
                dumpRawJpegBytes(debugConfig.rootDir, normalizedCameraId, seq, jpegBytes);
        }
        if (debugConfig.enabled && debugConfig.dumpOutput)
        {
            logutil::logThrottled(
                logutil::Level::warn,
                "multipart_binary_transport.output_dump_unsupported",
                std::chrono::seconds(300),
                "debug_dump_output is not yet supported for transport_mode=binary; "
                "switch to transport_mode=json_base64 to use annotated output frame dumps");
        }

        const httplib::UploadFormDataItems items = {
            {"camera_id", normalizedCameraId, "", ""},
            {
                "image",
                std::string(reinterpret_cast<const char*>(jpegBytes.data()), jpegBytes.size()),
                "frame.jpg",
                "image/jpeg",
            },
        };

        const httplib::Headers authHeaders = config.apiKey.empty()
            ? httplib::Headers{}
            : httplib::Headers{{"X-API-Key", config.apiKey}};

        std::string responseBody;
        bool requestSucceeded = false;
        std::string lastTransientError;
        const int maxAttempts = std::max(1, config.retryCount + 1);

        auto executeRequest = [&](auto& client)
        {
            if (!client.is_valid())
                throw ObjectDetectionError("Failed to create HTTP client for " + serviceTarget);

            configureServiceClient(client, config);

            for (int attempt = 1; attempt <= maxAttempts; ++attempt)
            {
                auto res = client.Post("/infer/binary", authHeaders, items);
                if (res && res->status == 200)
                {
                    responseBody = res->body;
                    requestSucceeded = true;
                    const bool recovered = m_circuitBreaker.onSuccess(normalizedCameraId);
                    if (recovered)
                    {
                        logutil::log(
                            logutil::Level::info,
                            "Circuit breaker CLOSED (recovered) for camera \"" +
                                normalizedCameraId + "\" target=" + serviceTarget +
                                " transport=binary");
                    }
                    break;
                }

                std::string err;
                if (!res)
                {
                    err = transportErrorDetails(res, config);
                }
                else if (isTransientHttpStatus(res->status))
                {
                    err = "transient service response target=" + serviceTarget +
                        " status=" + std::to_string(res->status) + responseBodySnippet(*res);
                }
                else
                {
                    m_circuitBreaker.onTransientFailure(normalizedCameraId);
                    throw ObjectDetectionError(
                        "Service response error target=" + serviceTarget +
                        " status=" + std::to_string(res->status) + responseBodySnippet(*res));
                }

                lastTransientError = err;
                if (attempt < maxAttempts)
                {
                    logutil::logThrottled(
                        logutil::Level::warn,
                        "multipart_binary_transport.retry." + normalizedCameraId,
                        std::chrono::seconds(10),
                        "Transient infer error (binary transport) for camera \"" +
                            normalizedCameraId + "\" attempt " + std::to_string(attempt) + "/" +
                            std::to_string(maxAttempts) + ": " + err);

                    if (config.retryBackoffMs > 0)
                        std::this_thread::sleep_for(std::chrono::milliseconds(config.retryBackoffMs));
                    continue;
                }

                const bool opened = m_circuitBreaker.onTransientFailure(normalizedCameraId);
                if (opened)
                {
                    logutil::logThrottled(
                        logutil::Level::warn,
                        "multipart_binary_transport.circuit_open.transition." + normalizedCameraId,
                        std::chrono::seconds(10),
                        "Circuit breaker OPEN for camera \"" + normalizedCameraId +
                            "\" after repeated transient infer failures (binary transport) target=" +
                            serviceTarget);
                }
            }
        };

        if (config.useHttps)
        {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
            httplib::SSLClient client(config.host, config.port);
            executeRequest(client);
#else
            throw ObjectDetectionError(
                "HTTPS transport requested for " + serviceTarget +
                " but the plugin was built without SSL client support");
#endif
        }
        else
        {
            httplib::Client client(config.host, config.port);
            executeRequest(client);
        }

        if (!requestSucceeded)
        {
            throw ObjectDetectionError(
                "Transient infer failure after retries for " + serviceTarget +
                " (binary transport): " + lastTransientError);
        }

        json j;
        try
        {
            j = json::parse(responseBody);
        }
        catch (const std::exception& e)
        {
            throw ObjectDetectionError(
                std::string("Failed to parse JSON response from ") + serviceTarget + ": " + e.what());
        }

        if (!j.is_array())
            throw ObjectDetectionError("Response from " + serviceTarget + " is not a JSON array");

        for (const auto& item : j)
        {
            try
            {
                const std::string classLabel = item.value("cls", "person");
                const float score = item.value("score", 0.0f);
                const float x = item.value("x", 0.0f);
                const float y = item.value("y", 0.0f);
                const float w = item.value("w", 0.0f);
                const float h = item.value("h", 0.0f);
                const int trackId = item.value("track_id", 0);
                const bool fallDetected = item.value("fall_detected", false);
                const bool stable = item.value("stable", true);
                const bool degraded = item.value("degraded", false);

                const auto jsonStr = [&item](const char* key) -> std::string {
                    auto it = item.find(key);
                    if (it != item.end() && it->is_string())
                        return it->get<std::string>();
                    return std::string{};
                };
                const auto jsonBool = [&item](const char* key, bool def) -> bool {
                    auto it = item.find(key);
                    if (it != item.end() && it->is_boolean())
                        return it->get<bool>();
                    return def;
                };
                const auto jsonInt = [&item](const char* key, int def) -> int {
                    auto it = item.find(key);
                    if (it != item.end() && it->is_number_integer())
                        return it->get<int>();
                    return def;
                };

                const bool zoneViolation = jsonBool("zone_violation", false);
                const std::string zoneType = jsonStr("zone_type");
                const std::string zoneId = jsonStr("zone_id");
                const std::string zoneName = jsonStr("zone_name");
                const bool recognized = jsonBool("recognized", false);
                const std::string personName = jsonStr("person_name");
                const std::string personGender = jsonStr("person_gender");
                const std::string personId = jsonStr("person_id");
                const int personNo = jsonInt("person_no", -1);
                const int personAge = jsonInt("person_age", -1);

                const NormalizedBox box = normalizeDetectionBox(x, y, w, h, frameWidth, frameHeight);
                if (!box.valid)
                    continue;

                auto detection = std::make_shared<Detection>(Detection{
                    nx::sdk::analytics::Rect(box.x, box.y, box.w, box.h),
                    classLabel,
                    score,
                    trackId > 0
                        ? uuidFromTrackId(normalizedCameraId, trackId)
                        : nx::sdk::Uuid{},
                    fallDetected,
                    stable,
                    degraded,
                    zoneViolation,
                    zoneType,
                    zoneId,
                    zoneName,
                    recognized,
                    personName,
                    personGender,
                    personId,
                    personNo,
                    personAge,
                });

                result.push_back(detection);
            }
            catch (const std::exception& e)
            {
                logutil::logThrottled(
                    logutil::Level::warn,
                    "multipart_binary_transport.bad_detection_item",
                    std::chrono::seconds(30),
                    std::string("Skip invalid detection item (binary transport): ") + e.what());
                continue;
            }
        }

        logutil::logThrottled(
            logutil::Level::debug,
            "multipart_binary_transport.detections",
            std::chrono::seconds(10),
            "Binary transport detections=" + std::to_string(result.size()));

        return result;
    }
    catch (const ObjectDetectionError&)
    {
        m_circuitBreaker.releaseHalfOpenProbe(normalizedCameraId);
        throw;
    }
    catch (const std::exception& e)
    {
        m_circuitBreaker.releaseHalfOpenProbe(normalizedCameraId);
        throw ObjectDetectionError(std::string("MultipartBinaryTransport::sendFrame error: ") + e.what());
    }
}

} // namespace opencv_object_detection
} // namespace vms_server_plugins
} // namespace sample_company
