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

                static nx::sdk::Uuid uuidFromTrackId(int trackId)
                {
                    static std::mutex m;
                    static std::unordered_map<int, nx::sdk::Uuid> map;

                    std::lock_guard<std::mutex> lk(m);

                    auto it = map.find(trackId);
                    if (it != map.end())
                        return it->second;

                    // tạo 1 UUID mới và cache lại cho trackId này
                    nx::sdk::Uuid u = nx::sdk::UuidHelper::randomUuid();
                    map.emplace(trackId, u);
                    return u;
                }

                // Gọi Python service, trả về DetectionList (danh sách Detection của plugin)
                DetectionList callPythonService(const Frame& frame, const std::string& cameraId)
                {
                    DetectionList result;

                    const Mat& image = frame.cvMat;
                    if (image.empty())
                        return result;

                // Resize để giảm thời gian imencode/base64 và tăng FPS tổng
                    cv::Mat sendImg = image;
                    const int targetW = 640; // bạn có thể thử 416 nếu máy yếu
                    if (image.cols > targetW)
                    {
                        float scale = (float)targetW / (float)image.cols;
                        int newW = targetW;
                        int newH = std::max(1, (int)std::round(image.rows * scale));
                        cv::resize(image, sendImg, cv::Size(newW, newH));
                    }

                    const int imgW = sendImg.cols;
                    const int imgH = sendImg.rows;
                    
                    std::cerr << "[C++ infer] Encoding sendImg " << imgW << "x" << imgH 
                              << " type=" << sendImg.type() << " continuous=" << sendImg.isContinuous() << std::endl;
                    
                    std::string b64;
                    try
                    {
                        std::cerr << "[C++ infer] Calling matToBase64Jpeg..." << std::endl;
                        b64 = matToBase64Jpeg(sendImg);
                        std::cerr << "[C++ infer] matToBase64Jpeg returned, b64 size=" << b64.size() << std::endl;
                    }
                    catch (const std::exception& e)
                    {
                        std::cerr << "[C++ infer] matToBase64Jpeg threw exception: " << e.what() << std::endl;
                        throw ObjectDetectionError(std::string("Failed to encode image to base64: ") + e.what());
                    }

                    if (b64.empty())
                    {
                        std::cerr << "[C++ infer] ERROR: b64 is empty after encoding!" << std::endl;
                        throw ObjectDetectionError("b64 empty after imencode - image may be invalid");
                    }
                    
                    std::cerr << "[C++ infer] b64 size OK: " << b64.size() << " bytes" << std::endl;


                    // 2. JSON request body
                    json req;
                    req["camera_id"] = cameraId;
                    req["image"] = b64;

                    // 3. HTTP client -> POST /infer
                    // Reuse client để đỡ tạo kết nối liên tục mỗi frame
                    thread_local httplib::Client cli("127.0.0.1", 18000);
                    cli.set_keep_alive(true); // Keep connection alive để tái sử dụng

                    // Fail fast to avoid blocking the Nx analytics pipeline.
                    cli.set_connection_timeout(0, 250000); // 250ms
                    cli.set_read_timeout(0, 400000);       // 400ms
                    cli.set_write_timeout(0, 250000);      // 250ms


                    static int s_reqCount = 0;
                    if ((++s_reqCount % 20) == 0)
                    {
                        std::cerr << "[C++] calling /infer count=" << s_reqCount << std::endl;
                    }

                    auto res = cli.Post("/infer", req.dump(), "application/json");

                    // ❗ res là pointer-like
                    if (!res)
                    {
                        throw ObjectDetectionError(
                            "Python /infer failed (no response). Service may be unavailable or timed out.");
                    }

                    if (res->status != 200)
                    {
                        throw ObjectDetectionError(
                            "Python /infer returned HTTP status " + std::to_string(res->status));
                    }

                    json j;
                    try
                    {
                        j = json::parse(res->body);
                    }
                    catch (const std::exception& e)
                    {
                        throw ObjectDetectionError(
                            std::string("Invalid JSON from Python /infer: ") + e.what());
                    }

                    if (!j.is_array())
                        throw ObjectDetectionError("Unexpected JSON payload from Python /infer (expected array).");

                    // 5. Mỗi phần tử là 1 detection:
                    //    { "cls": "person", "score": 0.9, "x": 180.0, "y": 270.6, "w": 120.0, "h": 360.8, "track_id": 1 }
                    for (const auto& item : j)
                    {
                        const std::string classLabel = item.value("cls", "person");
                        const float score = item.value("score", 0.0f);

                        float x = item.value("x", 0.0f);
                        float y = item.value("y", 0.0f);
                        float w = item.value("w", 0.0f);
                        float h = item.value("h", 0.0f);

                        if (w <= 0.0f || h <= 0.0f)
                            continue;

                        // Chuyển từ toạ độ pixel sang normalized [0..1]
                        float xNorm = x / static_cast<float>(imgW);
                        float yNorm = y / static_cast<float>(imgH);
                        float wNorm = w / static_cast<float>(imgW);
                        float hNorm = h / static_cast<float>(imgH);

                        // Clamp lại cho chắc
                        if (xNorm < 0.0f) xNorm = 0.0f;
                        if (yNorm < 0.0f) yNorm = 0.0f;
                        if (xNorm + wNorm > 1.0f) wNorm = 1.0f - xNorm;
                        if (yNorm + hNorm > 1.0f) hNorm = 1.0f - yNorm;

                        if (wNorm <= 0.0f || hNorm <= 0.0f)
                            continue;

                        // 🔹 Lấy track_id từ JSON -> UUID ổn định
                        const int trackId = item.value("track_id", 0);
                        const bool fallDetected = item.value("fall_detected", false);
                        nx::sdk::Uuid trackUuid = uuidFromTrackId(trackId);

                        auto detection = std::make_shared<Detection>(Detection{
                            nx::sdk::analytics::Rect(xNorm, yNorm, wNorm, hNorm),
                            classLabel,
                            score,
                            trackUuid,
                            fallDetected
                            });

                        result.push_back(detection);
                    }

                    static int s_log = 0;
                    if ((++s_log % 100) == 0)
                    {
                        std::cerr << "[C++] detections=" << result.size()
                            << " img=" << imgW << "x" << imgH << std::endl;
                    }

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

            DetectionList ObjectDetector::run(const Frame& frame, const std::string& cameraId)
            {
                if (isTerminated())
                    return {};

                try
                {
                    return runImpl(frame, cameraId);
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
                    // Base64 encode JPEG for JSON request
                    std::string b64 = base64Encode(jpegBytes.data(), jpegBytes.size());
                    
                    if (b64.empty())
                        throw ObjectDetectionError("Failed to base64 encode JPEG bytes");
                    
                    // Create JSON request
                    json req;
                    req["camera_id"] = cameraId;
                    req["image"] = b64;
                    
                    std::string jsonBody = req.dump();
                    
                    // HTTP client (thread-local, reused)
                    thread_local httplib::Client cli("127.0.0.1", 18000);
                    cli.set_keep_alive(true);
                    
                    // ⚠️ SHORT TIMEOUT FOR MVP: fail-fast if AI service is slow
                    // 1.5 seconds total (fail-fast instead of blocking Nx)
                    cli.set_connection_timeout(0, 500000);  // 500ms
                    cli.set_read_timeout(1, 0);             // 1s
                    cli.set_write_timeout(0, 500000);       // 500ms
                    
                    static int s_reqCount = 0;
                    if ((++s_reqCount % 20) == 0)
                    {
                        std::cerr << "[FLOW2 C++] Calling /infer with JPEG, count=" << s_reqCount 
                                  << " jpegSize=" << jpegBytes.size() << " bytes" << std::endl;
                    }
                    
                    // POST /infer endpoint
                    auto res = cli.Post("/infer", jsonBody, "application/json");
                    
                    if (!res)
                    {
                        static int s_fail = 0;
                        if ((++s_fail % 200) == 0)
                        {
                            std::cerr << "[FLOW2 C++] /infer failed (no response)" << std::endl;
                            std::cerr << "[FLOW2 C++] Python service at 127.0.0.1:18000 may not be running." << std::endl;
                        }
                        throw ObjectDetectionError("No response from /infer endpoint");
                    }
                    
                    if (res->status != 200)
                    {
                        static int s_bad = 0;
                        if ((++s_bad % 200) == 0)
                        {
                            std::cerr << "[FLOW2 C++] /infer status=" << res->status 
                                     << " body=" << res->body.substr(0, 100) << std::endl;
                        }
                        throw ObjectDetectionError("HTTP error " + std::to_string(res->status));
                    }
                    
                    // Parse JSON response
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

                    // Determine the exact image size that was sent to /infer.
                    // This avoids bbox normalization errors when frame height is not 480.
                    const cv::Mat decodedJpeg = cv::imdecode(jpegBytes, cv::IMREAD_COLOR);
                    if (decodedJpeg.empty())
                        throw ObjectDetectionError("Failed to decode JPEG bytes to determine frame dimensions");
                    const int frameW = decodedJpeg.cols;
                    const int frameH = decodedJpeg.rows;
                    
                    // Parse each detection
                    for (const auto& item : j)
                    {
                        try
                        {
                            const std::string classLabel = item.value("cls", "person");
                            const float score = item.value("score", 0.0f);
                            
                            float x = item.value("x", 0.0f);
                            float y = item.value("y", 0.0f);
                            float w = item.value("w", 0.0f);
                            float h = item.value("h", 0.0f);
                            
                            const bool fallDetected = item.value("fall_detected", false);  // FLOW 2
                            
                            if (w <= 0.0f || h <= 0.0f)
                                continue;
                            
                            // Normalize coordinates
                            float xNorm = x / static_cast<float>(frameW);
                            float yNorm = y / static_cast<float>(frameH);
                            float wNorm = w / static_cast<float>(frameW);
                            float hNorm = h / static_cast<float>(frameH);
                            
                            // Clamp
                            if (xNorm < 0.0f) xNorm = 0.0f;
                            if (yNorm < 0.0f) yNorm = 0.0f;
                            if (xNorm + wNorm > 1.0f) wNorm = 1.0f - xNorm;
                            if (yNorm + hNorm > 1.0f) hNorm = 1.0f - yNorm;
                            
                            if (wNorm <= 0.0f || hNorm <= 0.0f)
                                continue;
                            
                            // Get track ID
                            const int trackId = item.value("track_id", 0);
                            nx::sdk::Uuid trackUuid = uuidFromTrackId(trackId);
                            
                            // FLOW 2: Include fall_detected flag
                            auto detection = std::make_shared<Detection>(Detection{
                                nx::sdk::analytics::Rect(xNorm, yNorm, wNorm, hNorm),
                                classLabel,
                                score,
                                trackUuid,
                                fallDetected  // FLOW 2
                            });
                            
                            result.push_back(detection);
                        }
                        catch (const std::exception& e)
                        {
                            std::cerr << "[FLOW2 C++] Error parsing detection item: " << e.what() << std::endl;
                            continue;  // Skip bad items
                        }
                    }
                    
                    static int s_log = 0;
                    if ((++s_log % 100) == 0)
                    {
                        std::cerr << "[FLOW2 C++] detections=" << result.size() << std::endl;
                    }
                    
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

            DetectionList ObjectDetector::runImpl(const Frame& frame, const std::string& cameraId)
            {
                if (isTerminated())
                {
                    throw ObjectDetectorIsTerminatedError(
                        "Object detection error: object detector is terminated.");
                }

                // Thay toàn bộ logic ONNX cũ bằng gọi Python service:
                return callPythonService(frame, cameraId);
            }

        } // namespace opencv_object_detection
    } // namespace vms_server_plugins
} // namespace sample_company
