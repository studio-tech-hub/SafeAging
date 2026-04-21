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

                std::filesystem::path frameDumpRootDir()
                {
                    static std::filesystem::path root =
                        std::filesystem::path(R"(D:\Part-time\SafeAgingV4\SafeAging\debug_frames)");
                    static bool initialized = false;
                    static std::mutex initMutex;

                    std::lock_guard<std::mutex> lk(initMutex);
                    if (!initialized)
                    {
                        std::error_code ec;
                        std::filesystem::create_directories(root / "input", ec);
                        std::filesystem::create_directories(root / "output", ec);
                        logutil::log(
                            logutil::Level::info,
                            "Frame dump root: " + root.string());
                        initialized = true;
                    }

                    return root;
                }

                uint64_t nextFrameDumpSeq()
                {
                    static std::mutex m;
                    static uint64_t seq = 0;
                    std::lock_guard<std::mutex> lk(m);
                    ++seq;
                    return seq;
                }

                std::filesystem::path frameDumpPath(
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

                    return frameDumpRootDir() / type / fileName;
                }

                void dumpInputFrame(
                    const std::string& cameraId,
                    uint64_t seq,
                    const cv::Mat& frameBgr)
                {
                    if (frameBgr.empty())
                        return;

                    try
                    {
                        cv::imwrite(frameDumpPath(cameraId, "input", seq).string(), frameBgr);
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
                    const std::string& cameraId,
                    uint64_t seq,
                    const cv::Mat& frameBgr,
                    const std::vector<DebugBbox>& boxes)
                {
                    if (frameBgr.empty())
                        return;

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

                        cv::imwrite(frameDumpPath(cameraId, "output", seq).string(), rendered);
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
                constexpr std::array<int, 2> kTransientRetryBackoffMs{{150, 400}};

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

            ObjectDetector::ObjectDetector() = default;

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

            // ============================================================
            // FLOW 2: New method - run inference on JPEG bytes
            // ============================================================
            DetectionList ObjectDetector::run(const std::string& cameraId, const std::vector<uint8_t>& jpegBytes)
            {
                if (isTerminated())
                    return {};

                try
                {
                    if (jpegBytes.empty())
                        throw ObjectDetectionError("JPEG bytes are empty");
                    
                    return callPythonService(cameraId, jpegBytes);
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
            // Uses short timeout for MVP (fail-fast)
            // ============================================================
            DetectionList ObjectDetector::callPythonService(
                const std::string& cameraId,
                const std::vector<uint8_t>& jpegBytes)
            {
                DetectionList result;
                const std::string normalizedCameraId = normalizeCameraKey(cameraId);

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

                    const cv::Mat decodedJpeg = cv::imdecode(jpegBytes, cv::IMREAD_COLOR);
                    if (decodedJpeg.empty())
                        throw ObjectDetectionError("Failed to decode JPEG bytes to determine frame dimensions");

                    const int frameW = decodedJpeg.cols;
                    const int frameH = decodedJpeg.rows;
                    const uint64_t dumpSeq = nextFrameDumpSeq();
                    dumpInputFrame(normalizedCameraId, dumpSeq, decodedJpeg);

                    json req;
                    req["camera_id"] = normalizedCameraId;
                    req["image"] = b64;
                    const std::string jsonBody = req.dump();

                    thread_local httplib::Client cli("127.0.0.1", 18000);
                    cli.set_keep_alive(true);
                    
                    // Longer timeouts for high-res GPU inference (avoid client cancel -> 500)
                    cli.set_connection_timeout(2, 0);  // 2s
                    cli.set_read_timeout(15, 0);       // 15s
                    cli.set_write_timeout(2, 0);       // 2s
                    
                    static int s_reqCount = 0;
                    if ((++s_reqCount % 120) == 0)
                    {
                        logutil::log(
                            logutil::Level::info,
                            "FLOW2 infer requests processed=" + std::to_string(s_reqCount));
                    }
                    
                    std::string responseBody;
                    bool requestSucceeded = false;
                    std::string lastTransientError;
                    constexpr int kMaxAttempts = static_cast<int>(kTransientRetryBackoffMs.size()) + 1;
                    for (int attempt = 1; attempt <= kMaxAttempts; ++attempt)
                    {
                        auto res = cli.Post("/infer", jsonBody, "application/json");
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
                                        normalizedCameraId + "\"");
                            }
                            break;
                        }

                        bool transient = false;
                        std::string err;
                        if (!res)
                        {
                            transient = true;
                            err = "no response from /infer endpoint";
                        }
                        else if (isTransientHttpStatus(res->status))
                        {
                            transient = true;
                            err = "transient HTTP status=" + std::to_string(res->status);
                        }
                        else
                        {
                            circuitBreakerOnTransientFailure(normalizedCameraId);
                            throw ObjectDetectionError("HTTP error " + std::to_string(res->status));
                        }

                        lastTransientError = err;
                        if (attempt < kMaxAttempts)
                        {
                            logutil::logThrottled(
                                logutil::Level::warn,
                                "object_detector.flow2.retry." + normalizedCameraId,
                                std::chrono::seconds(10),
                                "Transient infer error for camera \"" + normalizedCameraId +
                                    "\" attempt " + std::to_string(attempt) + "/" +
                                    std::to_string(kMaxAttempts) + ": " + err);

                            const int backoffMs = kTransientRetryBackoffMs[attempt - 1];
                            std::this_thread::sleep_for(std::chrono::milliseconds(backoffMs));
                            continue;
                        }

                        const bool opened = circuitBreakerOnTransientFailure(normalizedCameraId);
                        if (opened)
                        {
                            logutil::logThrottled(
                                logutil::Level::warn,
                                "object_detector.flow2.circuit_open.transition." + normalizedCameraId,
                                std::chrono::seconds(10),
                                "Circuit breaker OPEN for camera \"" + normalizedCameraId +
                                    "\" after repeated transient infer failures");
                        }
                    }

                    if (!requestSucceeded)
                    {
                        throw ObjectDetectionError(
                            "Transient infer failure after retries: " + lastTransientError);
                    }

                    json j;
                    try
                    {
                        j = json::parse(responseBody);
                    }
                    catch (const std::exception& e)
                    {
                        throw ObjectDetectionError(std::string("Failed to parse JSON response: ") + e.what());
                    }

                    if (!j.is_array())
                        throw ObjectDetectionError("Response is not a JSON array");

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
                                degraded
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

