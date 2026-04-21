// Copyright 2018-present Network Optix, Inc. Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nx/sdk/analytics/rect.h>
#include <nx/sdk/uuid.h>

namespace sample_company {
namespace vms_server_plugins {
namespace opencv_object_detection {

// Object classes exposed by the current plugin/service contract.
extern const std::vector<std::string> kClassesToDetect;
extern const std::map<std::string, std::string> kClassesToDetectPluralCapitalized;

/**
 * Stores information about detection (one box per frame).
 */
struct Detection
{
    const nx::sdk::analytics::Rect boundingBox;
    const std::string classLabel;
    const float confidence;
    const nx::sdk::Uuid trackId;
    const bool fallDetected;  // Fall detection flag from the Python service
    const bool stable;        // Service-confirmed track that plugin may use for lifecycle events
    const bool degraded;      // Fallback/raw detection; render only, no lifecycle events
};

using DetectionList = std::vector<std::shared_ptr<Detection>>;

} // namespace opencv_object_detection
} // namespace vms_server_plugins
} // namespace sample_company
