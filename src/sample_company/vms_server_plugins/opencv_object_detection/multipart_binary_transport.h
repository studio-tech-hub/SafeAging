// multipart_binary_transport.h
// Copyright 2018-present Network Optix, Inc.
// Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/
//
// P1-4 — additive, opt-in ITransportClient implementation that posts raw JPEG
// bytes to POST /infer/binary via multipart/form-data instead of wrapping
// them in a JSON+base64 body. Selected via the "transport_mode" plugin
// setting (default remains json_base64 / JsonBase64Transport in
// object_detector.h — this class is never used unless explicitly configured).

#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "circuit_breaker.h"
#include "object_detector.h" // AiServiceClientConfig, DebugDumpConfig
#include "transport_client.h"

namespace sample_company {
namespace vms_server_plugins {
namespace opencv_object_detection {

class MultipartBinaryTransport: public ITransportClient
{
public:
    MultipartBinaryTransport();

    void setServiceConfig(const AiServiceClientConfig& serviceConfig);
    void setDebugDumpConfig(const DebugDumpConfig& debugConfig);

    DetectionList sendFrame(
        const std::string& cameraId,
        const std::vector<uint8_t>& jpegBytes,
        int frameWidth,
        int frameHeight) override;

private:
    AiServiceClientConfig serviceConfig() const;
    DebugDumpConfig debugDumpConfig() const;

private:
    mutable std::mutex m_serviceConfigMutex;
    AiServiceClientConfig m_serviceConfig;
    mutable std::mutex m_debugConfigMutex;
    DebugDumpConfig m_debugConfig;

    // Independent from ObjectDetector's own circuit breaker: the two
    // transports are never both active for the same camera at the same time
    // (transport_mode is selected once at settings-apply time), so separate
    // per-transport breaker state is simpler than sharing one and carries no
    // real downside. Reuses the already-unit-tested safeaging::CircuitBreakerMap
    // (see src/tests/test_circuit_breaker.cpp) instead of duplicating
    // object_detector.cpp's hand-rolled, internal-linkage equivalent.
    safeaging::DefaultCircuitBreakerMap m_circuitBreaker;
};

} // namespace opencv_object_detection
} // namespace vms_server_plugins
} // namespace sample_company
