#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nx/sdk/analytics/rect.h>
#include <nx/sdk/uuid.h>

namespace mycompany::yolov8_flow2 {

struct Detection
{
    nx::sdk::analytics::Rect bbox;
    std::string classLabel;
    float confidence = 0.0F;
    bool fallDetected = false;
    std::optional<int64_t> aiTrackId;
    nx::sdk::Uuid trackId;
};

using DetectionList = std::vector<Detection>;

} // namespace mycompany::yolov8_flow2

