// object_detector.h
// Copyright 2018-present Network Optix, Inc.
// Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#pragma once

#include <string>
#include <vector>

#include <nx/sdk/uuid.h>

#include "detection.h"

namespace sample_company {
namespace vms_server_plugins {
namespace opencv_object_detection {

class ObjectDetector
{
public:
    ObjectDetector();

    void ensureInitialized();
    bool isTerminated() const;
    void terminate();

    // Call the local Python analytics service using a camera id and encoded frame payload.
    DetectionList run(const std::string& cameraId, const std::vector<uint8_t>& jpegBytes);

private:
    DetectionList callPythonService(
        const std::string& cameraId,
        const std::vector<uint8_t>& jpegBytes);

private:
    bool m_terminated = false;
};

} // namespace opencv_object_detection
} // namespace vms_server_plugins
} // namespace sample_company
