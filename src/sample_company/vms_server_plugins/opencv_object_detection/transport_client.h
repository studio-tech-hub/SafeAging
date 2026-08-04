// transport_client.h
// Copyright 2018-present Network Optix, Inc.
// Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/
//
// P1-4 — Strategy interface for the plugin -> analytics-service frame
// transport. JsonBase64Transport (object_detector.h, default) sends
// JSON+base64 via the existing, proven ObjectDetector::run() path.
// MultipartBinaryTransport (multipart_binary_transport.h, additive/opt-in
// via the "transport_mode" plugin setting) sends raw JPEG bytes over
// multipart/form-data to remove base64's ~33% payload inflation and both
// sides' encode/decode CPU cost.
//
// DeviceAgent depends only on this interface (selected once per
// settingsReceived() call, default json_base64) — swapping transports
// requires no change to processFrameJob() or any other calling code.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "detection.h"

namespace sample_company {
namespace vms_server_plugins {
namespace opencv_object_detection {

class ITransportClient
{
public:
    virtual ~ITransportClient() = default;

    // Sends one already-JPEG-encoded frame to the analytics service and
    // returns the parsed detections.
    //
    // frameWidth/frameHeight must be the JPEG's actual encoded pixel
    // dimensions (i.e. after any downscale applied before encoding, not the
    // original camera frame size) — the analytics service returns detection
    // boxes in that same pixel space, and implementations use these
    // dimensions to normalize boxes into [0,1] Nx Rect coordinates without
    // needing to re-decode the JPEG on this hot path.
    virtual DetectionList sendFrame(
        const std::string& cameraId,
        const std::vector<uint8_t>& jpegBytes,
        int frameWidth,
        int frameHeight) = 0;
};

} // namespace opencv_object_detection
} // namespace vms_server_plugins
} // namespace sample_company
