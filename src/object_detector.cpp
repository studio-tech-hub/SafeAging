#include "object_detector.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "httplib.h"
#include "json.hpp"

#include "utils/image_codec.h"

namespace mycompany::yolov8_flow2 {

using json = nlohmann::json;

namespace {

std::string trim(std::string value)
{
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), value.end());
    return value;
}

float clamp01(float value)
{
    return std::max(0.0F, std::min(1.0F, value));
}

std::optional<int64_t> parseTrackId(const json& item)
{
    if (!item.contains("track_id"))
        return std::nullopt;

    const auto& value = item.at("track_id");
    try
    {
        if (value.is_number_integer())
            return value.get<int64_t>();
        if (value.is_number_float())
            return static_cast<int64_t>(std::llround(value.get<double>()));
        if (value.is_string())
        {
            const std::string s = value.get<std::string>();
            if (!s.empty())
                return std::stoll(s);
        }
    }
    catch (...)
    {
        return std::nullopt;
    }

    return std::nullopt;
}

} // namespace

ObjectDetector::ObjectDetector(DetectorConfig config):
    m_config(std::move(config)),
    m_endpoint(parseServiceUrl(m_config.serviceUrl))
{
}

DetectionList ObjectDetector::run(const std::string& cameraId, const cv::Mat& bgrFrame)
{
    const auto now = std::chrono::steady_clock::now();
    if (m_circuitOpen && now < m_circuitRetryAt)
        return {};

    if (m_circuitOpen && now >= m_circuitRetryAt)
    {
        m_circuitOpen = false;
        m_consecutiveFailures = 0;
    }

    try
    {
        auto detections = callService(cameraId, bgrFrame);
        onSuccess();
        return detections;
    }
    catch (const std::exception& e)
    {
        onFailure(e.what());
        return {};
    }
}

DetectionList ObjectDetector::callService(const std::string& cameraId, const cv::Mat& bgrFrame)
{
    const auto encoded = utils::encodeFrameAsBase64Jpeg(
        bgrFrame,
        m_config.sendWidth,
        m_config.jpegQuality);

    if (encoded.width <= 0 || encoded.height <= 0)
        throw std::runtime_error("encoded frame dimensions are invalid");

    json req;
    req["camera_id"] = cameraId;
    req["image"] = encoded.base64Jpeg;

    httplib::Client client(m_endpoint.host, m_endpoint.port);
    client.set_keep_alive(true);
    client.set_connection_timeout(
        m_config.connectTimeoutMs / 1000,
        (m_config.connectTimeoutMs % 1000) * 1000);
    client.set_read_timeout(
        m_config.readTimeoutMs / 1000,
        (m_config.readTimeoutMs % 1000) * 1000);
    client.set_write_timeout(
        m_config.writeTimeoutMs / 1000,
        (m_config.writeTimeoutMs % 1000) * 1000);

    const auto res = client.Post(m_endpoint.inferPath.c_str(), req.dump(), "application/json");
    if (!res)
        throw std::runtime_error("AI service did not respond");
    if (res->status != 200)
        throw std::runtime_error("AI service returned HTTP " + std::to_string(res->status));

    json responseJson = json::parse(res->body);
    if (!responseJson.is_array())
        throw std::runtime_error("AI response must be a JSON array");

    DetectionList detections;
    detections.reserve(responseJson.size());
    for (const auto& item : responseJson)
    {
        const float xPx = item.value("x", 0.0F);
        const float yPx = item.value("y", 0.0F);
        const float wPx = item.value("w", 0.0F);
        const float hPx = item.value("h", 0.0F);
        if (wPx <= 0.0F || hPx <= 0.0F)
            continue;

        float xNorm = xPx / static_cast<float>(encoded.width);
        float yNorm = yPx / static_cast<float>(encoded.height);
        float wNorm = wPx / static_cast<float>(encoded.width);
        float hNorm = hPx / static_cast<float>(encoded.height);

        xNorm = clamp01(xNorm);
        yNorm = clamp01(yNorm);
        wNorm = clamp01(wNorm);
        hNorm = clamp01(hNorm);
        if (xNorm + wNorm > 1.0F)
            wNorm = std::max(0.0F, 1.0F - xNorm);
        if (yNorm + hNorm > 1.0F)
            hNorm = std::max(0.0F, 1.0F - yNorm);
        if (wNorm <= 0.0F || hNorm <= 0.0F)
            continue;

        Detection detection;
        detection.bbox = nx::sdk::analytics::Rect(xNorm, yNorm, wNorm, hNorm);
        detection.classLabel = item.value("cls", item.value("class", std::string("person")));
        detection.confidence = item.value("score", item.value("confidence", 0.0F));
        detection.fallDetected = item.value("fall_detected", false);
        detection.aiTrackId = parseTrackId(item);

        detections.push_back(std::move(detection));
    }

    return detections;
}

ObjectDetector::ServiceEndpoint ObjectDetector::parseServiceUrl(const std::string& serviceUrl) const
{
    std::string input = trim(serviceUrl);
    if (input.empty())
        throw std::runtime_error("AI service URL is empty");

    constexpr const char* kHttpPrefix = "http://";
    constexpr const char* kHttpsPrefix = "https://";
    if (input.rfind(kHttpsPrefix, 0) == 0)
        throw std::runtime_error("https:// is not supported by this build, use http://");
    if (input.rfind(kHttpPrefix, 0) != 0)
        input = std::string(kHttpPrefix) + input;

    const size_t hostBegin = std::string(kHttpPrefix).size();
    const size_t slashPos = input.find('/', hostBegin);
    const std::string hostPort = input.substr(hostBegin, slashPos == std::string::npos ? std::string::npos : slashPos - hostBegin);
    std::string path = (slashPos == std::string::npos) ? "" : input.substr(slashPos);

    const size_t colonPos = hostPort.rfind(':');
    ServiceEndpoint endpoint;
    if (colonPos == std::string::npos)
    {
        endpoint.host = hostPort;
        endpoint.port = 80;
    }
    else
    {
        endpoint.host = hostPort.substr(0, colonPos);
        endpoint.port = std::stoi(hostPort.substr(colonPos + 1));
    }

    if (endpoint.host.empty())
        throw std::runtime_error("invalid AI service URL host");
    if (endpoint.port <= 0 || endpoint.port > 65535)
        throw std::runtime_error("invalid AI service URL port");

    if (path.empty())
        endpoint.inferPath = "/infer";
    else if (path == "/")
        endpoint.inferPath = "/infer";
    else if (path.size() >= 6 && path.substr(path.size() - 6) == "/infer")
        endpoint.inferPath = path;
    else
        endpoint.inferPath = path + "/infer";

    return endpoint;
}

void ObjectDetector::onFailure(const std::string& reason)
{
    ++m_consecutiveFailures;
    if (m_consecutiveFailures >= std::max(1, m_config.circuitFailureThreshold))
    {
        m_circuitOpen = true;
        m_circuitRetryAt =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(std::max(1, m_config.circuitOpenMs));
    }

    const auto now = std::chrono::steady_clock::now();
    const auto sinceLastLog =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastLogAt).count();
    if (m_lastLogAt.time_since_epoch().count() == 0 || sinceLastLog >= m_config.logThrottleMs)
    {
        std::cerr << "[ObjectDetector] inference failure: " << reason
                  << " (consecutive_failures=" << m_consecutiveFailures
                  << ", circuit_open=" << (m_circuitOpen ? "true" : "false") << ")"
                  << std::endl;
        m_lastLogAt = now;
    }
}

void ObjectDetector::onSuccess()
{
    m_consecutiveFailures = 0;
    m_circuitOpen = false;
}

} // namespace mycompany::yolov8_flow2
