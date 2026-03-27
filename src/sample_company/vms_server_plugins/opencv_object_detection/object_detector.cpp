#include "object_detector.h"
#include "exceptions.h"
#include "frame.h"

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
#include <vector>
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

            // (có thể để trong anonymous namespace)
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

                std::string matToBase64Jpeg(const cv::Mat& frame)
                {
                    cv::Mat bgr;

                    if (frame.empty())
                        throw ObjectDetectionError("Empty frame Mat");

                    if (frame.type() == CV_8UC3)
                    {
                        bgr = frame;
                    }
                    else if (frame.type() == CV_8UC4)
                    {
                        cv::cvtColor(frame, bgr, cv::COLOR_BGRA2BGR);
                    }
                    else if (frame.type() == CV_8UC1)
                    {
                        cv::cvtColor(frame, bgr, cv::COLOR_GRAY2BGR);
                    }
                    else
                    {
                        throw ObjectDetectionError("Unsupported Mat type=" + std::to_string(frame.type()));
                    }

                    // Ensure Mat is contiguous in memory
                    if (!bgr.isContinuous())
                    {
                        bgr = bgr.clone();
                    }

                    if (bgr.empty())
                        throw ObjectDetectionError("bgr Mat is empty after processing");

                    // Skip imencode entirely - just use raw BGR with minimal compression
                    // This avoids issues with missing JPEG/PNG encoders
                    const uint8_t* data = bgr.data;
                    size_t dataSize = bgr.total() * bgr.elemSize();
                    
                    std::cerr << "[C++ encode] Using RAW_BGR: " << bgr.cols << "x" << bgr.rows 
                              << " data=" << dataSize << " bytes" << std::endl;
                    
                    // Simple format: magic + width + height + raw BGR data
                    std::vector<uchar> buf;
                    
                    // Add magic "BGR" header
                    buf.push_back('B');
                    buf.push_back('G');
                    buf.push_back('R');
                    
                    // Add dimensions (4 bytes each, little-endian)
                    uint32_t w = bgr.cols;
                    uint32_t h = bgr.rows;
                    buf.push_back((w >> 0) & 0xFF);
                    buf.push_back((w >> 8) & 0xFF);
                    buf.push_back((w >> 16) & 0xFF);
                    buf.push_back((w >> 24) & 0xFF);
                    buf.push_back((h >> 0) & 0xFF);
                    buf.push_back((h >> 8) & 0xFF);
                    buf.push_back((h >> 16) & 0xFF);
                    buf.push_back((h >> 24) & 0xFF);
                    
                    // Add raw BGR data
                    buf.insert(buf.end(), data, data + dataSize);
                    
                    std::cerr << "[C++ encode] Total buffer: " << buf.size() << " bytes (header=11)" << std::endl;
                    
                    if (buf.empty())
                        throw ObjectDetectionError("Encoded buffer is empty");
                    
                    return base64Encode(buf.data(), buf.size());
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
                        std::cerr << "[FrameDump] root=" << root.string() << std::endl;
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
                    constexpr uint64_t kMaxFrameFiles = 150;
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
                        std::cerr << "[FrameDump] Failed to write input frame: " << e.what() << std::endl;
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
                        std::cerr << "[FrameDump] Failed to write output frame: " << e.what() << std::endl;
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

                    nx::sdk::Uuid u = nx::sdk::UuidHelper::randomUuid();
                    map.emplace(key, TrackCacheEntry{u, now});
                    return u;
                }

                // Gọi Python service, trả về DetectionList (danh sách Detection của plugin)
                DetectionList callPythonService(const Frame& frame)
                {
                    DetectionList result;

                    const Mat& image = frame.cvMat;
                    if (image.empty())
                        return result;

                    static std::chrono::steady_clock::time_point lastCall = std::chrono::steady_clock::now();
                    auto now = std::chrono::steady_clock::now();
                    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastCall).count();

                    if (ms < 200)
                        return {};
                    lastCall = now;

                    cv::Mat sendImg = image;
                    const int targetW = 640;
                    if (image.cols > targetW)
                    {
                        const float scale = static_cast<float>(targetW) / static_cast<float>(image.cols);
                        const int newW = targetW;
                        const int newH = std::max(1, static_cast<int>(std::round(image.rows * scale)));
                        cv::resize(image, sendImg, cv::Size(newW, newH));
                    }

                    const int imgW = sendImg.cols;
                    const int imgH = sendImg.rows;
                    const std::string cameraId = "nx_camera";
                    const uint64_t dumpSeq = nextFrameDumpSeq();
                    dumpInputFrame(cameraId, dumpSeq, sendImg);

                    std::string b64;
                    try
                    {
                        b64 = matToBase64Jpeg(sendImg);
                    }
                    catch (const std::exception& e)
                    {
                        throw ObjectDetectionError(std::string("Failed to encode image to base64: ") + e.what());
                    }

                    if (b64.empty())
                        throw ObjectDetectionError("b64 empty after encode - image may be invalid");

                    json req;
                    req["camera_id"] = cameraId;
                    req["image"] = b64;

                    thread_local httplib::Client cli("127.0.0.1", 18000);
                    cli.set_keep_alive(true);
                    cli.set_connection_timeout(1, 500000);
                    cli.set_read_timeout(2, 500000);
                    cli.set_write_timeout(1, 0);

                    auto res = cli.Post("/infer", req.dump(), "application/json");
                    if (!res || res->status != 200)
                        return {};

                    json j;
                    try
                    {
                        j = json::parse(res->body);
                    }
                    catch (...)
                    {
                        return {};
                    }

                    if (!j.is_array())
                        return {};

                    std::vector<DebugBbox> debugBoxes;
                    for (const auto& item : j)
                    {
                        const std::string classLabel = item.value("cls", "person");
                        const float score = item.value("score", 0.0f);

                        const float x = item.value("x", 0.0f);
                        const float y = item.value("y", 0.0f);
                        const float w = item.value("w", 0.0f);
                        const float h = item.value("h", 0.0f);

                        if (w <= 0.0f || h <= 0.0f)
                            continue;

                        float xNorm = x / static_cast<float>(imgW);
                        float yNorm = y / static_cast<float>(imgH);
                        float wNorm = w / static_cast<float>(imgW);
                        float hNorm = h / static_cast<float>(imgH);

                        if (xNorm < 0.0f) xNorm = 0.0f;
                        if (yNorm < 0.0f) yNorm = 0.0f;
                        if (xNorm + wNorm > 1.0f) wNorm = 1.0f - xNorm;
                        if (yNorm + hNorm > 1.0f) hNorm = 1.0f - yNorm;
                        if (wNorm <= 0.0f || hNorm <= 0.0f)
                            continue;

                        const int x1 = std::max(0, static_cast<int>(std::round(x)));
                        const int y1 = std::max(0, static_cast<int>(std::round(y)));
                        const int x2 = std::min(imgW, static_cast<int>(std::round(x + w)));
                        const int y2 = std::min(imgH, static_cast<int>(std::round(y + h)));
                        if (x2 > x1 && y2 > y1)
                        {
                            debugBoxes.push_back(DebugBbox{
                                cv::Rect(x1, y1, x2 - x1, y2 - y1),
                                classLabel,
                                score});
                        }

                        const int trackId = item.value("track_id", 0);
                        nx::sdk::Uuid trackUuid = uuidFromTrackId(cameraId, trackId);

                        auto detection = std::make_shared<Detection>(Detection{
                            nx::sdk::analytics::Rect(xNorm, yNorm, wNorm, hNorm),
                            classLabel,
                            score,
                            trackUuid
                            });

                        result.push_back(detection);
                    }

                    dumpOutputFrame(cameraId, dumpSeq, sendImg, debugBoxes);
                    return result;
                }

            } // namespace (anonymous)

            //-------------------------------------------------------------------------------------------------
            // ObjectDetector implementation

            ObjectDetector::ObjectDetector(std::filesystem::path modelPath) :
                m_modelPath(std::move(modelPath))
            {
            }

            void ObjectDetector::ensureInitialized()
            {
                if (isTerminated())
                {
                    throw ObjectDetectorIsTerminatedError(
                        "Object detector initialization error: object detector is terminated.");
                }

                // Không load model trong C++ nữa, chỉ cần đánh dấu là "loaded".
                m_netLoaded = true;
            }

            bool ObjectDetector::isTerminated() const
            {
                return m_terminated;
            }

            void ObjectDetector::terminate()
            {
                m_terminated = true;
            }

            DetectionList ObjectDetector::run(const Frame& frame)
            {
                if (isTerminated())
                    return {};

                try
                {
                    return runImpl(frame);
                }
                catch (const ObjectDetectionError&)
                {
                    // ĐỂ CHO device_agent.cpp bắt và push event
                    throw;
                }
                catch (const cv::Exception& e)
                {
                    throw ObjectDetectionError(std::string("OpenCV error: ") + e.what());
                }
                catch (const std::exception& e)
                {
                    throw ObjectDetectionError(std::string("Std error: ") + e.what());
                }
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
                    
                    return callPythonServiceMultipart(cameraId, jpegBytes);
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
            // FLOW 2: HTTP multipart/form-data call to Python service
            // Uses short timeout for MVP (fail-fast)
            // ============================================================
            DetectionList ObjectDetector::callPythonServiceMultipart(
                const std::string& cameraId,
                const std::vector<uint8_t>& jpegBytes)
            {
                DetectionList result;

                try
                {
                    std::string b64 = base64Encode(jpegBytes.data(), jpegBytes.size());
                    if (b64.empty())
                        throw ObjectDetectionError("Failed to base64 encode JPEG bytes");

                    const cv::Mat decodedJpeg = cv::imdecode(jpegBytes, cv::IMREAD_COLOR);
                    if (decodedJpeg.empty())
                        throw ObjectDetectionError("Failed to decode JPEG bytes to determine frame dimensions");

                    const int frameW = decodedJpeg.cols;
                    const int frameH = decodedJpeg.rows;
                    const uint64_t dumpSeq = nextFrameDumpSeq();
                    dumpInputFrame(cameraId, dumpSeq, decodedJpeg);

                    json req;
                    req["camera_id"] = cameraId;
                    req["image"] = b64;
                    const std::string jsonBody = req.dump();

                    thread_local httplib::Client cli("127.0.0.1", 18000);
                    cli.set_keep_alive(true);
                    cli.set_connection_timeout(2, 0);
                    cli.set_read_timeout(15, 0);
                    cli.set_write_timeout(2, 0);

                    auto res = cli.Post("/infer", jsonBody, "application/json");
                    if (!res)
                        throw ObjectDetectionError("No response from /infer endpoint");
                    if (res->status != 200)
                        throw ObjectDetectionError("HTTP error " + std::to_string(res->status));

                    json j;
                    try
                    {
                        j = json::parse(res->body);
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

                            const bool fallDetected = item.value("fall_detected", false);

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

                            const int trackId = item.value("track_id", 0);
                            nx::sdk::Uuid trackUuid = uuidFromTrackId(cameraId, trackId);

                            auto detection = std::make_shared<Detection>(Detection{
                                nx::sdk::analytics::Rect(xNorm, yNorm, wNorm, hNorm),
                                classLabel,
                                score,
                                trackUuid,
                                fallDetected
                            });

                            result.push_back(detection);
                        }
                        catch (const std::exception& e)
                        {
                            std::cerr << "[FLOW2 C++] Error parsing detection item: " << e.what() << std::endl;
                            continue;
                        }
                    }

                    dumpOutputFrame(cameraId, dumpSeq, decodedJpeg, debugBoxes);
                    return result;
                }
                catch (const ObjectDetectionError&)
                {
                    throw;
                }
                catch (const std::exception& e)
                {
                    throw ObjectDetectionError(std::string("callPythonServiceMultipart error: ") + e.what());
                }
            }

            //-------------------------------------------------------------------------------------------------
            // private

            // Hàm loadModel() cũ không còn dùng nữa, nhưng giữ lại cho đủ định nghĩa (nếu header còn khai báo).
            void ObjectDetector::loadModel()
            {
                // KHÔNG còn dùng OpenCV DNN / ONNX nữa.
            }

            DetectionList ObjectDetector::runImpl(const Frame& frame)
            {
                if (isTerminated())
                {
                    throw ObjectDetectorIsTerminatedError(
                        "Object detection error: object detector is terminated.");
                }

                // Thay toàn bộ logic ONNX cũ bằng gọi Python service:
                return callPythonService(frame);
            }

        } // namespace opencv_object_detection
    } // namespace vms_server_plugins
} // namespace sample_company
