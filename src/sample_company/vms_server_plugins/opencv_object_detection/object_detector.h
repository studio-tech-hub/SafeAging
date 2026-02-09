// object_detector.h
// Copyright 2018-present Network Optix, Inc.
// Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/dnn.hpp>

#include <nx/sdk/helpers/uuid_helper.h>
#include <nx/sdk/uuid.h>

#include "detection.h"
#include "frame.h"

namespace sample_company {
namespace vms_server_plugins {
namespace opencv_object_detection {

class ObjectDetector
{
public:
    // modelPath: ĐƯỜNG DẪN ĐẦY ĐỦ tới file .onnx
    explicit ObjectDetector(std::filesystem::path modelPath);

    void ensureInitialized();
    bool isTerminated() const;
    void terminate();
    
    // FLOW 2: Run inference on JPEG bytes via HTTP /infer endpoint
    // Signature: run(cameraId, jpegBytes) -> DetectionList
    // Throws ObjectDetectionError on HTTP error / timeout / JSON parse error
    DetectionList run(const std::string& cameraId, const std::vector<uint8_t>& jpegBytes);
    
    // Legacy: Run inference on Frame (still available)
    DetectionList run(const Frame& frame);

private:
    void loadModel();
    
    // FLOW 2: Call Python AI service via HTTP multipart/form-data
    DetectionList callPythonServiceMultipart(
        const std::string& cameraId, 
        const std::vector<uint8_t>& jpegBytes);
    
    DetectionList runImpl(const Frame& frame);

private:
    bool m_netLoaded = false;
    bool m_terminated = false;

    // Full path tới file model, ví dụ:
    // C:\Program Files\...\plugins\yolov8_people_analytics_plugin\yolov8n.onnx
    const std::filesystem::path m_modelPath;

    std::unique_ptr<cv::dnn::Net> m_net;
};

} // namespace opencv_object_detection
} // namespace vms_server_plugins
} // namespace sample_company
