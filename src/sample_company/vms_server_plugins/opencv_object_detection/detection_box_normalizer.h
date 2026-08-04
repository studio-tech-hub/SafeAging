// detection_box_normalizer.h
// Copyright 2018-present Network Optix, Inc.
// Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/
//
// Pure, header-only helper: converts one absolute-pixel detection box (as
// returned by the analytics service's /infer* JSON) into normalized [0,1]
// coordinates clamped to the frame. No OpenCV / Nx SDK / httplib dependency,
// so it can be unit-tested in isolation the same way as circuit_breaker.h —
// see src/tests/test_detection_box_normalizer.cpp.
//
// object_detector.cpp's existing JSON+base64 path (callPythonService) keeps
// its own inline copy of this same math rather than being refactored to call
// this helper, to avoid touching that proven, already-deployed code path as
// part of the additive P1-4 change. MultipartBinaryTransport uses this
// helper directly. Consider consolidating both call sites onto this helper
// in a follow-up once the binary transport has completed field validation.

#pragma once

namespace sample_company {
namespace vms_server_plugins {
namespace opencv_object_detection {

struct NormalizedBox
{
    float x = 0.0f;
    float y = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
    bool valid = false; //< false means: skip this detection (degenerate box)
};

// x, y, w, h are absolute pixel coordinates in [0, frameWidth) x [0, frameHeight)
// space, as returned by the analytics service. Mirrors the clamping rules used
// by object_detector.cpp's callPythonService(): negative origins are clamped to
// 0, and a box that would overflow the frame on the right/bottom edge is
// shrunk (not rejected) so partially-visible people at the frame edge still
// render a box instead of being silently dropped.
inline NormalizedBox normalizeDetectionBox(
    float x, float y, float w, float h, int frameWidth, int frameHeight)
{
    if (frameWidth <= 0 || frameHeight <= 0 || w <= 0.0f || h <= 0.0f)
        return NormalizedBox{};

    const float fw = static_cast<float>(frameWidth);
    const float fh = static_cast<float>(frameHeight);

    float xNorm = x / fw;
    float yNorm = y / fh;
    float wNorm = w / fw;
    float hNorm = h / fh;

    if (xNorm < 0.0f)
        xNorm = 0.0f;
    if (yNorm < 0.0f)
        yNorm = 0.0f;
    if (xNorm + wNorm > 1.0f)
        wNorm = 1.0f - xNorm;
    if (yNorm + hNorm > 1.0f)
        hNorm = 1.0f - yNorm;

    if (wNorm <= 0.0f || hNorm <= 0.0f)
        return NormalizedBox{};

    return NormalizedBox{xNorm, yNorm, wNorm, hNorm, true};
}

} // namespace opencv_object_detection
} // namespace vms_server_plugins
} // namespace sample_company
