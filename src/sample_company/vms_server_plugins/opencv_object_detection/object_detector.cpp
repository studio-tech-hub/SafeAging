#include "object_detector.h"
#include "exceptions.h"
#include "logging_utils.h"

#ifdef _MSC_VER
#pragma warning(push, 0)
#pragma warning(disable: 26439 26495 6294 6201 6262)
#endif

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "httplib.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "json.hpp"
#include <atomic>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <random>
#include <vector>
#include <array>
#include <thread>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace sample_company {
    namespace vms_server_plugins {
        namespace opencv_object_detection {

            using namespace std::string_literals;
            using namespace cv;

            using json = nlohmann::json;

            //-------------------------------------------------------------------------------------------------
            // Base64 helper (encode buffer -> base64 string)

            // (cÃ³ thá»ƒ Ä‘á»ƒ trong anonymous namespace)
            namespace {

                static const std::string kBase64Chars =
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "abcdefghijklmnopqrstuvwxyz"
                    "0123456789+/";

                std::string base64Encode(const unsigned char* data, size_t len)
                {
                    std::string out;
                    out.reserve(((len + 2) / 3) * 4);

                    unsigned char char_array_3[3];
                    unsigned char char_array_4[4];
                    int i = 0;

                    while (len--)
                    {
                        char_array_3[i++] = *(data++);
                        if (i == 3)
                        {
                            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
                            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) +
                                ((char_array_3[1] & 0xf0) >> 4);
                            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) +
                                ((char_array_3[2] & 0xc0) >> 6);
                            char_array_4[3] = char_array_3[2] & 0x3f;

                            for (i = 0; i < 4; ++i)
                                out.push_back(kBase64Chars[char_array_4[i]]);
                            i = 0;
                        }
                    }

                    if (i)
                    {
                        for (int j = i; j < 3; ++j)
                            char_array_3[j] = '\0';

                        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
                        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) +
                            ((char_array_3[1] & 0xf0) >> 4);
                        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) +
                            ((char_array_3[2] & 0xc0) >> 6);
                        char_array_4[3] = char_array_3[2] & 0x3f;

                        for (int j = 0; j < i + 1; ++j)
                            out.push_back(kBase64Chars[char_array_4[j]]);

                        while (i++ < 3)
                            out.push_back('=');
                    }

                    return out;
                }

                struct DebugBbox
                {
                    cv::Rect rect;
                    std::string label;
                    float score = 0.0f;
                };

                std::string sanitizeForFileName(const std::string& value)
                {
                    if (value.empty())
                        return "unknown_camera";

                    std::string out = value;
                    for (char& ch : out)
                    {
                        const unsigned char u = static_cast<unsigned char>(ch);
                        if (!std::isalnum(u) && ch != '-' && ch != '_')
                            ch = '_';
                    }
                    return out;
                }

                nx::sdk::Uuid makeRandomUuid()
                {
                    thread_local std::mt19937_64 rng(std::random_device{}());
                    uint8_t bytes[nx::sdk::Uuid::kSize];

                    for (size_t i = 0; i < nx::sdk::Uuid::kSize; i += sizeof(uint64_t))
                    {
                        const uint64_t value = rng();
                        const size_t copySize =
                            std::min(sizeof(value), nx::sdk::Uuid::kSize - i);
                        std::memcpy(bytes + i, &value, copySize);
                    }

                    // RFC 4122 version 4 style UUID bits.
                    bytes[6] = static_cast<uint8_t>((bytes[6] & 0x0F) | 0x40);
                    bytes[8] = static_cast<uint8_t>((bytes[8] & 0x3F) | 0x80);
                    return nx::sdk::Uuid(bytes);
                }

                std::filesystem::path frameDumpRootDir(
                    const std::filesystem::path& pluginHomeDir,
                    const std::string& relativeDir)
                {
                    if (relativeDir.empty())
                        return pluginHomeDir / DebugDumpConfig::kDefaultDumpDir;
                    return pluginHomeDir / relativeDir;
                }

                struct FrameDumpContext
                {
                    std::filesystem::path rootDir;
                    bool inputDirsCreated = false;
                    bool outputDirsCreated = false;
                    std::mutex mutex;

                    void ensureInputDirCreated()
                    {
                        if (inputDirsCreated)
                            return;
                        std::lock_guard<std::mutex> lk(mutex);
                        if (inputDirsCreated)
                            return;
                        std::error_code ec;
                        std::filesystem::create_directories(rootDir / "input", ec);
                        if (!ec)
                            inputDirsCreated = true;
                    }

                    void ensureOutputDirCreated()
                    {
                        if (outputDirsCreated)
                            return;
                        std::lock_guard<std::mutex> lk(mutex);
                        if (outputDirsCreated)
                            return;
                        std::error_code ec;
                        std::filesystem::create_directories(rootDir / "output", ec);
                        if (!ec)
                            outputDirsCreated = true;
                    }
                };

                uint64_t nextFrameDumpSeq()
                {
                    static std::mutex m;
                    static uint64_t seq = 0;
                    std::lock_guard<std::mutex> lk(m);
                    ++seq;
                    return seq;
                }

                std::filesystem::path frameDumpPath(
                    const std::filesystem::path& rootDir,
                    const std::string& cameraId,
                    const char* type,
                    uint64_t seq)
                {
                    constexpr uint64_t kMaxFrameFiles = 100;
                    const uint64_t slot = seq % kMaxFrameFiles;

                    std::string fileName = sanitizeForFileName(cameraId);
                    fileName += "_";
                    fileName += std::to_string(slot);
                    fileName += ".jpg";

                    return rootDir / type / fileName;
                }

                void dumpInputFrame(
                    FrameDumpContext& ctx,
                    const std::string& cameraId,
                    uint64_t seq,
                    const cv::Mat& frameBgr)
                {
                    if (frameBgr.empty())
                        return;

                    ctx.ensureInputDirCreated();

                    try
                    {
                        cv::imwrite(frameDumpPath(ctx.rootDir, cameraId, "input", seq).string(), frameBgr);
                    }
                    catch (const std::exception& e)
                    {
                        logutil::logThrottled(
                            logutil::Level::warn,
                            "object_detector.frame_dump.input_write_failed",
                            std::chrono::seconds(30),
                            std::string("Failed to write input frame: ") + e.what());
                    }
                }

                void dumpOutputFrame(
                    FrameDumpContext& ctx,
                    const std::string& cameraId,
                    uint64_t seq,
                    const cv::Mat& frameBgr,
                    const std::vector<DebugBbox>& boxes)
                {
                    if (frameBgr.empty())
                        return;

                    ctx.ensureOutputDirCreated();

                    try
                    {
                        cv::Mat rendered = frameBgr.clone();
                        for (const auto& box : boxes)
                        {
                            cv::rectangle(rendered, box.rect, cv::Scalar(0, 255, 0), 2);

                            std::string text = box.label + " " + std::to_string(box.score);
                            int baseLine = 0;
                            const cv::Size textSize =
                                cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                            const int textX = std::max(0, box.rect.x);
                            const int textY = std::max(textSize.height + 4, box.rect.y - 6);

                            cv::rectangle(
                                rendered,
                                cv::Rect(textX, textY - textSize.height - 4, textSize.width + 6, textSize.height + 6),
                                cv::Scalar(0, 255, 0),
                                cv::FILLED);
                            cv::putText(
                                rendered,
                                text,
                                cv::Point(textX + 3, textY - 3),
                                cv::FONT_HERSHEY_SIMPLEX,
                                0.5,
                                cv::Scalar(0, 0, 0),
                                1,
                                cv::LINE_AA);
                        }

                        cv::imwrite(frameDumpPath(ctx.rootDir, cameraId, "output", seq).string(), rendered);
                    }
                    catch (const std::exception& e)
                    {
                        logutil::logThrottled(
                            logutil::Level::warn,
                            "object_detector.frame_dump.output_write_failed",
                            std::chrono::seconds(30),
                            std::string("Failed to write output frame: ") + e.what());
                    }
                }

                static nx::sdk::Uuid uuidFromTrackId(const std::string& cameraId, int trackId)
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

                            std::sort(
                                entries.begin(),
                                entries.end(),
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

                    nx::sdk::Uuid u = makeRandomUuid();
                    map.emplace(key, TrackCacheEntry{u, now});
                    return u;
                }

                // Service-side HTTP client and circuit breaker state.
                enum class CircuitState
                {
                    closed,
                    open,
                    halfOpen
                };

                struct CircuitBreakerEntry
                {
                    CircuitState state = CircuitState::closed;
                    int consecutiveFailures = 0;
                    bool halfOpenProbeInFlight = false;
                    std::chrono::steady_clock::time_point openUntil =
                        std::chrono::steady_clock::time_point::min();
                    std::chrono::steady_clock::time_point lastSeen =
                        std::chrono::steady_clock::now();
                };

                constexpr int kCircuitFailureThreshold = 5;
                constexpr std::chrono::seconds kCircuitOpenCooldown{15};
                constexpr size_t kCircuitMapMaxSize = 256;

                std::mutex g_circuitMutex;
                std::unordered_map<std::string, CircuitBreakerEntry> g_circuitByCamera;
                size_t g_circuitAccessCount = 0;

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
                        {
                            details +=
                                ", openssl_error=" + std::to_string(result.ssl_openssl_error());
                        }
                    }
#endif

                    return details;
                }

                template<typename ClientT>
                void configureServiceClient(
                    ClientT& client,
                    const AiServiceClientConfig& config)
                {
                    // The plugin may call a local or remote analytics service; auth uses X-API-Key.
                    client.set_keep_alive(true);
                    client.set_connection_timeout(
                        std::chrono::milliseconds(config.connectTimeoutMs));
                    client.set_read_timeout(std::chrono::milliseconds(config.readTimeoutMs));
                    client.set_write_timeout(std::chrono::milliseconds(config.writeTimeoutMs));
                }

                template<typename ClientT>
                httplib::Result postInferRequest(
                    ClientT& client,
                    const httplib::Headers& headers,
                    const std::string& jsonBody)
                {
                    if (headers.empty())
                        return client.Post("/infer", jsonBody, "application/json");

                    return client.Post("/infer", headers, jsonBody, "application/json");
                }

                void cleanupCircuitStateIfNeeded(const std::chrono::steady_clock::time_point now)
                {
                    ++g_circuitAccessCount;
                    if (g_circuitAccessCount % 256 != 0)
                        return;

                    for (auto it = g_circuitByCamera.begin(); it != g_circuitByCamera.end();)
                    {
                        if (now - it->second.lastSeen > std::chrono::minutes(30))
                            it = g_circuitByCamera.erase(it);
                        else
                            ++it;
                    }

                    if (g_circuitByCamera.size() <= kCircuitMapMaxSize)
                        return;

                    std::vector<std::pair<std::string, std::chrono::steady_clock::time_point>> ages;
                    ages.reserve(g_circuitByCamera.size());
                    for (const auto& kv : g_circuitByCamera)
                        ages.push_back({kv.first, kv.second.lastSeen});
                    std::sort(
                        ages.begin(),
                        ages.end(),
                        [](const auto& a, const auto& b) { return a.second < b.second; });

                    const size_t toRemove = g_circuitByCamera.size() - kCircuitMapMaxSize;
                    for (size_t i = 0; i < toRemove; ++i)
                        g_circuitByCamera.erase(ages[i].first);
                }

                bool circuitBreakerAllowRequest(
                    const std::string& cameraId,
                    std::string* outReason,
                    bool* outHalfOpenTransition)
                {
                    const auto now = std::chrono::steady_clock::now();
                    std::lock_guard<std::mutex> lk(g_circuitMutex);
                    cleanupCircuitStateIfNeeded(now);

                    auto& state = g_circuitByCamera[cameraId];
                    state.lastSeen = now;
                    if (outHalfOpenTransition)
                        *outHalfOpenTransition = false;

                    if (state.state == CircuitState::open)
                    {
                        if (now < state.openUntil)
                        {
                            if (outReason)
                            {
                                const auto remainMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    state.openUntil - now).count();
                                *outReason = "cooldown_ms=" + std::to_string(std::max<int64_t>(0, remainMs));
                            }
                            return false;
                        }

                        state.state = CircuitState::halfOpen;
                        state.halfOpenProbeInFlight = true;
                        if (outHalfOpenTransition)
                            *outHalfOpenTransition = true;
                        return true;
                    }

                    if (state.state == CircuitState::halfOpen)
                    {
                        if (state.halfOpenProbeInFlight)
                        {
                            if (outReason)
                                *outReason = "half_open_probe_in_flight";
                            return false;
                        }
                        state.halfOpenProbeInFlight = true;
                        return true;
                    }

                    return true;
                }

                bool circuitBreakerOnSuccess(const std::string& cameraId)
                {
                    const auto now = std::chrono::steady_clock::now();
                    std::lock_guard<std::mutex> lk(g_circuitMutex);
                    auto& state = g_circuitByCamera[cameraId];
                    const bool recovered = (state.state != CircuitState::closed);
                    state.state = CircuitState::closed;
                    state.consecutiveFailures = 0;
                    state.halfOpenProbeInFlight = false;
                    state.openUntil = std::chrono::steady_clock::time_point::min();
                    state.lastSeen = now;
                    return recovered;
                }

                bool circuitBreakerOnTransientFailure(const std::string& cameraId)
                {
                    const auto now = std::chrono::steady_clock::now();
                    std::lock_guard<std::mutex> lk(g_circuitMutex);
                    auto& state = g_circuitByCamera[cameraId];
                    state.lastSeen = now;

                    if (state.state == CircuitState::halfOpen)
                    {
                        state.state = CircuitState::open;
                        state.halfOpenProbeInFlight = false;
                        state.consecutiveFailures = 0;
                        state.openUntil = now + kCircuitOpenCooldown;
                        return true;
                    }

                    ++state.consecutiveFailures;
                    if (state.consecutiveFailures >= kCircuitFailureThreshold)
                    {
                        state.state = CircuitState::open;
                        state.halfOpenProbeInFlight = false;
                        state.consecutiveFailures = 0;
                        state.openUntil = now + kCircuitOpenCooldown;
                        return true;
                    }

                    return false;
                }

                void circuitBreakerReleaseHalfOpenProbe(const std::string& cameraId)
                {
                    const auto now = std::chrono::steady_clock::now();
                    std::lock_guard<std::mutex> lk(g_circuitMutex);
                    auto it = g_circuitByCamera.find(cameraId);
                    if (it == g_circuitByCamera.end())
                        return;

                    it->second.lastSeen = now;
                    if (it->second.state == CircuitState::halfOpen)
                        it->second.halfOpenProbeInFlight = false;
                }

            } // namespace (anonymous)

            //-------------------------------------------------------------------------------------------------
            // ObjectDetector implementation

            ObjectDetector::ObjectDetector():
                ObjectDetector(AiServiceClientConfig{})
            {
            }

            ObjectDetector::ObjectDetector(const AiServiceClientConfig& serviceConfig):
                m_serviceConfig(serviceConfig)
            {
            }

            void ObjectDetector::ensureInitialized()
            {
                if (isTerminated())
                {
                    throw ObjectDetectorIsTerminatedError(
                        "Object detector initialization error: object detector is terminated.");
                }

                // No local model is loaded in the plugin anymore. Initialization only validates
                // that the detector has not been terminated before HTTP requests start.
            }

            bool ObjectDetector::isTerminated() const
            {
                return m_terminated;
            }

            void ObjectDetector::terminate()
            {
                m_terminated = true;
            }

            void ObjectDetector::setServiceConfig(const AiServiceClientConfig& serviceConfig)
            {
                std::lock_guard<std::mutex> lk(m_serviceConfigMutex);
                m_serviceConfig = serviceConfig;
            }

            AiServiceClientConfig ObjectDetector::serviceConfig() const
            {
                std::lock_guard<std::mutex> lk(m_serviceConfigMutex);
                return m_serviceConfig;
            }

            void ObjectDetector::setDebugDumpConfig(const DebugDumpConfig& debugConfig)
            {
                std::lock_guard<std::mutex> lk(m_debugConfigMutex);
                m_debugConfig = debugConfig;
            }

            DebugDumpConfig ObjectDetector::debugDumpConfig() const
            {
                std::lock_guard<std::mutex> lk(m_debugConfigMutex);
                return m_debugConfig;
            }

            // ============================================================
            // P P1.1 – Lightweight GET /health probe for the health poll thread.
            // Never throws; all errors are captured in HealthCheckResult.raw_error.
            // ============================================================
            HealthCheckResult ObjectDetector::checkHealth() const
            {
                HealthCheckResult result;
                const AiServiceClientConfig config = serviceConfig();

                // Helper lambda: configure timeouts and fire GET /health.
                // Mirrors the HTTPS/HTTP branching used in the infer path.
                const auto doGet = [&](auto& client) -> httplib::Result
                {
                    client.set_connection_timeout(
                        config.connectTimeoutMs / 1000,
                        (config.connectTimeoutMs % 1000) * 1000000);
                    client.set_read_timeout(5, 0); // short read timeout for health check
                    const httplib::Headers headers = config.apiKey.empty()
                        ? httplib::Headers{}
                        : httplib::Headers{{"X-API-Key", config.apiKey}};
                    return client.Get("/health", headers);
                };

                try
                {
                    const std::string scheme = config.useHttps ? "https" : "http";
                    httplib::Result res{nullptr, httplib::Error::Unknown};

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
                    if (config.useHttps)
                    {
                        httplib::SSLClient client(config.host, config.port);
                        configureServiceClient(client, config);
                        res = doGet(client);
                    }
                    else
#endif
                    {
                        httplib::Client client(config.host, config.port);
                        res = doGet(client);
                    }

                    if (!res)
                    {
                        result.reachable = false;
                        result.raw_error = "transport_error=" +
                            httplib::to_string(res.error()) +
                            " target=" + scheme + "://" + config.host +
                            ":" + std::to_string(config.port);
                        return result;
                    }

                    if (res->status != 200 && res->status != 503)
                    {
                        result.reachable = false;
                        result.raw_error = "unexpected_status=" +
                            std::to_string(res->status);
                        return result;
                    }

                    result.reachable = true;

                    const auto j = json::parse(res->body);
                    result.status = j.value("status", "unknown");

                    if (j.contains("reason_codes") && j["reason_codes"].is_array())
                    {
                        for (const auto& code : j["reason_codes"])
                        {
                            if (code.is_string())
                                result.reason_codes.push_back(code.get<std::string>());
                        }
                    }

                    if (j.contains("dependencies") && j["dependencies"].is_object())
                    {
                        for (const auto& [dep, val] : j["dependencies"].items())
                        {
                            if (val.is_string())
                                result.dependencies[dep] = val.get<std::string>();
                        }
                    }
                }
                catch (const std::exception& e)
                {
                    result.reachable = false;
                    result.raw_error = std::string("exception: ") + e.what();
                }

                return result;
            }

            // ============================================================
            // P2.2 – Fetch per-camera config from GET /config/{cameraId}
            // ============================================================
            CameraConfigFetch ObjectDetector::fetchCameraConfig(const std::string& cameraId) const
            {
                CameraConfigFetch result;
                const AiServiceClientConfig config = serviceConfig();

                const std::string path = "/config/" + cameraId;

                const auto doGet = [&](auto& client) -> httplib::Result
                {
                    client.set_connection_timeout(
                        config.connectTimeoutMs / 1000,
                        (config.connectTimeoutMs % 1000) * 1000000);
                    client.set_read_timeout(5, 0);
                    const httplib::Headers headers = config.apiKey.empty()
                        ? httplib::Headers{}
                        : httplib::Headers{{"X-API-Key", config.apiKey}};
                    return client.Get(path.c_str(), headers);
                };

                try
                {
                    httplib::Result res{nullptr, httplib::Error::Unknown};

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
                    if (config.useHttps)
                    {
                        httplib::SSLClient client(config.host, config.port);
                        configureServiceClient(client, config);
                        res = doGet(client);
                    }
                    else
#endif
                    {
                        httplib::Client client(config.host, config.port);
                        res = doGet(client);
                    }

                    if (!res)
                    {
                        result.reachable = false;
                        result.raw_error = "transport_error=" + httplib::to_string(res.error());
                        return result;
                    }

                    if (res->status != 200)
                    {
                        result.reachable = false;
                        result.raw_error = "status=" + std::to_string(res->status);
                        return result;
                    }

                    result.reachable = true;
                    result.raw_json = res->body;

                    const auto j = json::parse(res->body);

                    if (j.contains("confidence_threshold") && !j["confidence_threshold"].is_null())
                        result.confidence_threshold = j["confidence_threshold"].get<float>();

                    if (j.contains("iou_threshold") && !j["iou_threshold"].is_null())
                        result.iou_threshold = j["iou_threshold"].get<float>();

                    if (j.contains("frame_period") && !j["frame_period"].is_null())
                        result.frame_period = j["frame_period"].get<int>();
                }
                catch (const std::exception& e)
                {
                    result.reachable = false;
                    result.raw_error = std::string("exception: ") + e.what();
                }

                return result;
            }

            // ============================================================
            // P2.3 – Register camera with service: PUT /admin/camera-configs/{cameraId}
            // Stores the human-readable display name in the `extra` JSONB field so the
            // service can show a friendly name in its admin API.  Best-effort; never throws.
            // ============================================================
            bool ObjectDetector::registerCamera(
                const std::string& cameraId,
                const std::string& displayName,
                float confidenceThreshold) const
            {
                const AiServiceClientConfig config = serviceConfig();
                const std::string path = "/admin/camera-configs/" + cameraId;

                json body;
                body["extra"] = {{"display_name", displayName}};
                if (confidenceThreshold >= 0.0f && confidenceThreshold <= 1.0f)
                    body["confidence_threshold"] = confidenceThreshold;
                const std::string bodyStr = body.dump();

                const auto doPut = [&](auto& client) -> httplib::Result
                {
                    client.set_connection_timeout(
                        config.connectTimeoutMs / 1000,
                        (config.connectTimeoutMs % 1000) * 1000000);
                    client.set_read_timeout(5, 0);
                    httplib::Headers headers = {{"Content-Type", "application/json"}};
                    if (!config.apiKey.empty())
                        headers.emplace("X-API-Key", config.apiKey);
                    return client.Put(path.c_str(), headers, bodyStr, "application/json");
                };

                try
                {
                    httplib::Result res{nullptr, httplib::Error::Unknown};

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
                    if (config.useHttps)
                    {
                        httplib::SSLClient client(config.host, config.port);
                        configureServiceClient(client, config);
                        res = doPut(client);
                    }
                    else
#endif
                    {
                        httplib::Client client(config.host, config.port);
                        res = doPut(client);
                    }

                    if (!res)
                        return false;

                    return res->status == 200 || res->status == 201;
                }
                catch (const std::exception&)
                {
                    return false;
                }
            }

            // ============================================================
            // FLOW 2: New method - run inference on JPEG bytes
            // ============================================================
            DetectionList ObjectDetector::run(
                const std::string& cameraId,
                const std::vector<uint8_t>& jpegBytes,
                int frameWidth,
                int frameHeight)
            {
                if (isTerminated())
                    return {};

                try
                {
                    if (jpegBytes.empty())
                        throw ObjectDetectionError("JPEG bytes are empty");
                    
                    return callPythonService(cameraId, jpegBytes, frameWidth, frameHeight);
                }
                catch (const ObjectDetectionError&)
                {
                    throw;  // Re-throw detection errors
                }
                catch (const std::exception& e)
                {
                    throw ObjectDetectionError(std::string("Error in run(cameraId, jpegBytes): ") + e.what());
                }
            }
            
            // ============================================================
            // Call the Python analytics service using the current JSON payload contract
            // Timeouts and retries are configured at runtime by the plugin settings.
            // ============================================================
            DetectionList ObjectDetector::callPythonService(
                const std::string& cameraId,
                const std::vector<uint8_t>& jpegBytes,
                int frameWidth,
                int frameHeight)
            {
                DetectionList result;
                const std::string normalizedCameraId = normalizeCameraKey(cameraId);
                const AiServiceClientConfig config = serviceConfig();
                const DebugDumpConfig debugConfig = debugDumpConfig();
                const std::string serviceTarget = serviceEndpoint(config);

                if (frameWidth <= 0 || frameHeight <= 0)
                {
                    throw ObjectDetectionError(
                        "Invalid frame dimensions passed to callPythonService: " +
                        std::to_string(frameWidth) + "x" + std::to_string(frameHeight));
                }

                try
                {
                    std::string breakerReason;
                    bool halfOpenTransition = false;
                    if (!circuitBreakerAllowRequest(
                            normalizedCameraId,
                            &breakerReason,
                            &halfOpenTransition))
                    {
                        logutil::logThrottled(
                            logutil::Level::warn,
                            "object_detector.flow2.circuit_open." + normalizedCameraId,
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
                            "object_detector.flow2.circuit_half_open." + normalizedCameraId,
                            std::chrono::seconds(10),
                            "Circuit breaker HALF_OPEN probe for camera \"" + normalizedCameraId + "\"");
                    }

                    std::string b64 = base64Encode(jpegBytes.data(), jpegBytes.size());
                    if (b64.empty())
                        throw ObjectDetectionError("Failed to base64 encode JPEG bytes");

                    // P1-4: frameW/frameH now come from the caller (which already knows
                    // its own encoded JPEG dimensions) instead of decoding jpegBytes here
                    // on every single request. The JPEG is only decoded below, lazily,
                    // when debug frame dumping is actually enabled (default off) — that
                    // is the only remaining consumer that needs a decoded cv::Mat.
                    const int frameW = frameWidth;
                    const int frameH = frameHeight;

                    // Handle debug frame dumping - only if debug is enabled
                    uint64_t dumpSeq = 0;
                    std::unique_ptr<FrameDumpContext> dumpCtx;
                    cv::Mat decodedJpegForDump;
                    if (debugConfig.enabled && (debugConfig.dumpInput || debugConfig.dumpOutput))
                    {
                        dumpSeq = nextFrameDumpSeq();
                        // Apply frame sampling: only dump if (dumpSeq - 1) % everyNFrames == 0
                        if ((dumpSeq - 1) % std::max(1, debugConfig.everyNFrames) == 0)
                        {
                            decodedJpegForDump = cv::imdecode(jpegBytes, cv::IMREAD_COLOR);
                            if (decodedJpegForDump.empty())
                            {
                                logutil::logThrottled(
                                    logutil::Level::warn,
                                    "object_detector.flow2.dump_decode_failed",
                                    std::chrono::seconds(30),
                                    "Failed to decode JPEG bytes for debug frame dump; skipping dump for this frame");
                            }
                            else
                            {
                                dumpCtx = std::make_unique<FrameDumpContext>();
                                dumpCtx->rootDir = debugConfig.rootDir;
                                if (debugConfig.dumpInput)
                                {
                                    dumpInputFrame(*dumpCtx, normalizedCameraId, dumpSeq, decodedJpegForDump);
                                }
                            }
                        }
                    }

                    json req;
                    req["camera_id"] = normalizedCameraId;
                    req["image"] = b64;
                    const std::string jsonBody = req.dump();

                    const httplib::Headers authHeaders = config.apiKey.empty()
                        ? httplib::Headers{}
                        : httplib::Headers{{"X-API-Key", config.apiKey}};

                    std::string responseBody;
                    bool requestSucceeded = false;
                    std::string lastTransientError;
                    const int maxAttempts = std::max(1, config.retryCount + 1);
                    static std::atomic<int> s_reqCount{0};
                    const int requestCount = ++s_reqCount;
                    if ((requestCount % 120) == 0)
                    {
                        logutil::log(
                            logutil::Level::info,
                            "FLOW2 infer requests processed=" + std::to_string(requestCount));
                    }

                    auto executeRequest = [&](auto& client)
                    {
                        if (!client.is_valid())
                        {
                            throw ObjectDetectionError(
                                "Failed to create HTTP client for " + serviceTarget);
                        }

                        configureServiceClient(client, config);

                        for (int attempt = 1; attempt <= maxAttempts; ++attempt)
                        {
                            auto res = postInferRequest(client, authHeaders, jsonBody);
                            if (res && res->status == 200)
                            {
                                responseBody = res->body;
                                requestSucceeded = true;
                                const bool recovered = circuitBreakerOnSuccess(normalizedCameraId);
                                if (recovered)
                                {
                                    logutil::log(
                                        logutil::Level::info,
                                        "Circuit breaker CLOSED (recovered) for camera \"" +
                                            normalizedCameraId + "\" target=" + serviceTarget);
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
                                    " status=" + std::to_string(res->status) +
                                    responseBodySnippet(*res);
                            }
                            else
                            {
                                circuitBreakerOnTransientFailure(normalizedCameraId);
                                throw ObjectDetectionError(
                                    "Service response error target=" + serviceTarget +
                                    " status=" + std::to_string(res->status) +
                                    responseBodySnippet(*res));
                            }

                            lastTransientError = err;
                            if (attempt < maxAttempts)
                            {
                                logutil::logThrottled(
                                    logutil::Level::warn,
                                    "object_detector.flow2.retry." + normalizedCameraId,
                                    std::chrono::seconds(10),
                                    "Transient infer error for camera \"" + normalizedCameraId +
                                        "\" attempt " + std::to_string(attempt) + "/" +
                                        std::to_string(maxAttempts) + ": " + err);

                                if (config.retryBackoffMs > 0)
                                {
                                    std::this_thread::sleep_for(
                                        std::chrono::milliseconds(config.retryBackoffMs));
                                }
                                continue;
                            }

                            const bool opened = circuitBreakerOnTransientFailure(normalizedCameraId);
                            if (opened)
                            {
                                logutil::logThrottled(
                                    logutil::Level::warn,
                                    "object_detector.flow2.circuit_open.transition." +
                                        normalizedCameraId,
                                    std::chrono::seconds(10),
                                    "Circuit breaker OPEN for camera \"" + normalizedCameraId +
                                        "\" after repeated transient infer failures target=" +
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
                            ": " + lastTransientError);
                    }

                    json j;
                    try
                    {
                        j = json::parse(responseBody);
                    }
                    catch (const std::exception& e)
                    {
                        throw ObjectDetectionError(
                            std::string("Failed to parse JSON response from ") +
                            serviceTarget + ": " + e.what());
                    }

                    if (!j.is_array())
                    {
                        throw ObjectDetectionError(
                            "Response from " + serviceTarget + " is not a JSON array");
                    }

                    std::vector<DebugBbox> debugBoxes;
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
        // Optional fields may be serialised as JSON null (Pydantic Optional[...] = None),
        // so guard against null/wrong-type which would otherwise throw in nlohmann::value().
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

        // P2.1 — zone violation fields (optional; default to no violation)
        const bool zoneViolation = jsonBool("zone_violation", false);
        const std::string zoneType = jsonStr("zone_type");
        const std::string zoneId   = jsonStr("zone_id");
        const std::string zoneName = jsonStr("zone_name");
        // Face recognition identity fields (optional)
        const bool recognized = jsonBool("recognized", false);
        const std::string personName = jsonStr("person_name");
        const std::string personGender = jsonStr("person_gender");
        const std::string personId = jsonStr("person_id");
        const int personNo = jsonInt("person_no", -1);
        const int personAge = jsonInt("person_age", -1);

                            if (w <= 0.0f || h <= 0.0f)
                                continue;

                            float xNorm = x / static_cast<float>(frameW);
                            float yNorm = y / static_cast<float>(frameH);
                            float wNorm = w / static_cast<float>(frameW);
                            float hNorm = h / static_cast<float>(frameH);

                            if (xNorm < 0.0f) xNorm = 0.0f;
                            if (yNorm < 0.0f) yNorm = 0.0f;
                            if (xNorm + wNorm > 1.0f) wNorm = 1.0f - xNorm;
                            if (yNorm + hNorm > 1.0f) hNorm = 1.0f - yNorm;
                            if (wNorm <= 0.0f || hNorm <= 0.0f)
                                continue;

                            const int x1 = std::max(0, static_cast<int>(std::round(x)));
                            const int y1 = std::max(0, static_cast<int>(std::round(y)));
                            const int x2 = std::min(frameW, static_cast<int>(std::round(x + w)));
                            const int y2 = std::min(frameH, static_cast<int>(std::round(y + h)));
                            if (x2 > x1 && y2 > y1)
                            {
                                debugBoxes.push_back(DebugBbox{
                                    cv::Rect(x1, y1, x2 - x1, y2 - y1),
                                    classLabel,
                                    score});
                            }

        auto detection = std::make_shared<Detection>(Detection{
            nx::sdk::analytics::Rect(xNorm, yNorm, wNorm, hNorm),
            classLabel,
            score,
            trackId > 0
                ? uuidFromTrackId(normalizedCameraId, trackId)
                : nx::sdk::Uuid{},
            fallDetected,
            stable,
            degraded,
            zoneViolation,  // P2.1
            zoneType,       // P2.1
            zoneId,         // P2.1
            zoneName,       // P2.1
            recognized,     // face recognition
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
                                "object_detector.flow2.bad_detection_item",
                                std::chrono::seconds(30),
                                std::string("Skip invalid detection item: ") + e.what());
                            continue;  // Skip bad items
                        }
                    }

                    logutil::logThrottled(
                        logutil::Level::debug,
                        "object_detector.flow2.detections",
                        std::chrono::seconds(10),
                        "FLOW2 detections=" + std::to_string(result.size()));
                    
                    // Dump output frame if debug is enabled
                    if (dumpCtx && debugConfig.dumpOutput)
                    {
                        dumpOutputFrame(*dumpCtx, normalizedCameraId, dumpSeq, decodedJpegForDump, debugBoxes);
                    }
                    
                    return result;
                }
                catch (const ObjectDetectionError&)
                {
                    circuitBreakerReleaseHalfOpenProbe(normalizedCameraId);
                    throw;
                }
                catch (const std::exception& e)
                {
                    circuitBreakerReleaseHalfOpenProbe(normalizedCameraId);
                    throw ObjectDetectionError(std::string("callPythonService error: ") + e.what());
                }
            }

        } // namespace opencv_object_detection
    } // namespace vms_server_plugins
} // namespace sample_company

