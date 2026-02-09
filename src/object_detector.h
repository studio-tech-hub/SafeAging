#pragma once

#include <chrono>
#include <string>

#include <opencv2/core.hpp>

#include "detection.h"

namespace mycompany::yolov8_flow2 {

struct DetectorConfig
{
    std::string serviceUrl = "http://127.0.0.1:18000";
    int connectTimeoutMs = 250;
    int readTimeoutMs = 400;
    int writeTimeoutMs = 250;
    int sendWidth = 640;
    int jpegQuality = 80;
    int circuitFailureThreshold = 3;
    int circuitOpenMs = 3000;
    int logThrottleMs = 5000;
};

class ObjectDetector
{
public:
    explicit ObjectDetector(DetectorConfig config);
    DetectionList run(const std::string& cameraId, const cv::Mat& bgrFrame);

private:
    struct ServiceEndpoint
    {
        std::string host;
        int port = 80;
        std::string inferPath = "/infer";
    };

    DetectionList callService(const std::string& cameraId, const cv::Mat& bgrFrame);
    ServiceEndpoint parseServiceUrl(const std::string& serviceUrl) const;
    void onFailure(const std::string& reason);
    void onSuccess();

private:
    DetectorConfig m_config;
    ServiceEndpoint m_endpoint;

    int m_consecutiveFailures = 0;
    bool m_circuitOpen = false;
    std::chrono::steady_clock::time_point m_circuitRetryAt{};
    std::chrono::steady_clock::time_point m_lastLogAt{};
};

} // namespace mycompany::yolov8_flow2

