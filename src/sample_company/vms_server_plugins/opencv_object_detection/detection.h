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
    const bool fallDetected;    //< Fall detection flag from the Python service
    const bool stable;          //< Service-confirmed track; may use for lifecycle events
    const bool degraded;        //< Fallback/raw detection; render only, no lifecycle events
    // P2.1 — zone violation data from zone_engine
    const bool zoneViolation = false;       //< Person entered a forbidden/entry/exit zone
    const std::string zoneType{};           //< "forbidden" | "entry" | "exit" | ""
    const std::string zoneId{};             //< UUID string of the violated zone, or ""
    // Face recognition identity from the service (rendered on the bounding box).
    const bool recognized = false;          //< True if matched to a known person
    const std::string personName{};         //< "Ông A" / "Bà B" / "Unknown"
    const std::string personGender{};       //< "Nam" | "Nữ" | ""
    const std::string personId{};           //< Person UUID string, or ""
    const int personNo = -1;                //< Stable per-camera person index; -1 = none
    const int personAge = -1;               //< Real-time age from date of birth; -1 = unknown
};

using DetectionList = std::vector<std::shared_ptr<Detection>>;

} // namespace opencv_object_detection
} // namespace vms_server_plugins
} // namespace sample_company
