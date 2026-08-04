// test_detection_box_normalizer.cpp
// Standalone C++ unit tests for normalizeDetectionBox() (P1-4).
// No Nx SDK, no OpenCV, no httplib required.
//
// Compile (g++/clang++):
//   g++ -std=c++17 -I../sample_company/vms_server_plugins/opencv_object_detection \
//       test_detection_box_normalizer.cpp -o test_detection_box_normalizer
//   ./test_detection_box_normalizer
//
// On Windows (MSVC):
//   cl /std:c++17 /I..\sample_company\vms_server_plugins\opencv_object_detection \
//      test_detection_box_normalizer.cpp /link /OUT:test_detection_box_normalizer.exe

#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>

#include "detection_box_normalizer.h"

using sample_company::vms_server_plugins::opencv_object_detection::NormalizedBox;
using sample_company::vms_server_plugins::opencv_object_detection::normalizeDetectionBox;

// ── Minimal test harness (mirrors test_circuit_breaker.cpp) ──────────────────

static int s_passed = 0, s_failed = 0;

#define TEST(name) \
    void name(); \
    struct _Reg_##name { _Reg_##name() { runTest(#name, name); } } _reg_##name; \
    void name()

static void runTest(const char* name, const std::function<void()>& fn)
{
    try {
        fn();
        std::printf("  PASS  %s\n", name);
        ++s_passed;
    } catch (const std::exception& e) {
        std::printf("  FAIL  %s: %s\n", name, e.what());
        ++s_failed;
    } catch (...) {
        std::printf("  FAIL  %s: unknown exception\n", name);
        ++s_failed;
    }
}

#define ASSERT(expr) \
    do { if (!(expr)) throw std::runtime_error("Assertion failed: " #expr " (line " + std::to_string(__LINE__) + ")"); } while(0)

static bool nearlyEqual(float a, float b, float eps = 1e-4f)
{
    return std::fabs(a - b) < eps;
}

#define ASSERT_NEAR(a, b) \
    do { if (!nearlyEqual((a), (b))) throw std::runtime_error("ASSERT_NEAR: " #a " != " #b " (line " + std::to_string(__LINE__) + ")"); } while(0)


// ── Test cases ────────────────────────────────────────────────────────────────

TEST(simple_centered_box)
{
    // 100x100 box inside a 1000x2000 (w x h) frame, at origin (100, 200).
    const NormalizedBox box = normalizeDetectionBox(100.0f, 200.0f, 100.0f, 100.0f, 1000, 2000);
    ASSERT(box.valid);
    ASSERT_NEAR(box.x, 0.10f);
    ASSERT_NEAR(box.y, 0.10f);
    ASSERT_NEAR(box.w, 0.10f);
    ASSERT_NEAR(box.h, 0.05f);
}

TEST(full_frame_box)
{
    const NormalizedBox box = normalizeDetectionBox(0.0f, 0.0f, 640.0f, 480.0f, 640, 480);
    ASSERT(box.valid);
    ASSERT_NEAR(box.x, 0.0f);
    ASSERT_NEAR(box.y, 0.0f);
    ASSERT_NEAR(box.w, 1.0f);
    ASSERT_NEAR(box.h, 1.0f);
}

TEST(negative_origin_clamped_to_zero)
{
    // Detector can report a small negative x/y right at the frame edge.
    const NormalizedBox box = normalizeDetectionBox(-5.0f, -5.0f, 50.0f, 50.0f, 640, 480);
    ASSERT(box.valid);
    ASSERT(box.x >= 0.0f);
    ASSERT(box.y >= 0.0f);
}

TEST(overflow_right_edge_shrinks_width_not_rejected)
{
    // Box starts inside the frame but extends past the right edge.
    const NormalizedBox box = normalizeDetectionBox(600.0f, 100.0f, 100.0f, 50.0f, 640, 480);
    ASSERT(box.valid);
    ASSERT_NEAR(box.x + box.w, 1.0f); // shrunk to exactly the frame boundary
}

TEST(overflow_bottom_edge_shrinks_height_not_rejected)
{
    const NormalizedBox box = normalizeDetectionBox(100.0f, 450.0f, 50.0f, 100.0f, 640, 480);
    ASSERT(box.valid);
    ASSERT_NEAR(box.y + box.h, 1.0f);
}

TEST(zero_width_is_invalid)
{
    const NormalizedBox box = normalizeDetectionBox(10.0f, 10.0f, 0.0f, 50.0f, 640, 480);
    ASSERT(!box.valid);
}

TEST(zero_height_is_invalid)
{
    const NormalizedBox box = normalizeDetectionBox(10.0f, 10.0f, 50.0f, 0.0f, 640, 480);
    ASSERT(!box.valid);
}

TEST(negative_width_is_invalid)
{
    const NormalizedBox box = normalizeDetectionBox(10.0f, 10.0f, -50.0f, 50.0f, 640, 480);
    ASSERT(!box.valid);
}

TEST(box_entirely_past_right_edge_is_invalid)
{
    // x already >= frameWidth: after clamping, wNorm collapses to <= 0.
    const NormalizedBox box = normalizeDetectionBox(700.0f, 100.0f, 50.0f, 50.0f, 640, 480);
    ASSERT(!box.valid);
}

TEST(zero_frame_width_is_invalid)
{
    const NormalizedBox box = normalizeDetectionBox(10.0f, 10.0f, 50.0f, 50.0f, 0, 480);
    ASSERT(!box.valid);
}

TEST(zero_frame_height_is_invalid)
{
    const NormalizedBox box = normalizeDetectionBox(10.0f, 10.0f, 50.0f, 50.0f, 640, 0);
    ASSERT(!box.valid);
}

TEST(negative_frame_dimensions_invalid)
{
    const NormalizedBox box = normalizeDetectionBox(10.0f, 10.0f, 50.0f, 50.0f, -640, 480);
    ASSERT(!box.valid);
}

TEST(portrait_frame_dimensions_handled_independently)
{
    // Regression guard: width/height normalization must not be swapped for
    // portrait (h > w) frames — a common source of bugs when someone
    // accidentally normalizes both axes by the same dimension.
    const NormalizedBox box = normalizeDetectionBox(90.0f, 400.0f, 180.0f, 800.0f, 360, 1280);
    ASSERT(box.valid);
    ASSERT_NEAR(box.x, 0.25f);       // 90 / 360
    ASSERT_NEAR(box.y, 0.3125f);     // 400 / 1280
    ASSERT_NEAR(box.w, 0.5f);        // 180 / 360
    ASSERT_NEAR(box.h, 0.625f);      // 800 / 1280
}


// ── Entry point ───────────────────────────────────────────────────────────────

int main()
{
    std::printf("\n=== SafeAging Detection Box Normalizer Unit Tests ===\n\n");
    // All tests are registered via static initialisers above
    std::printf("\n--- Results ---\n");
    std::printf("  Passed: %d\n", s_passed);
    std::printf("  Failed: %d\n", s_failed);
    std::printf("==========================================\n\n");
    return s_failed == 0 ? 0 : 1;
}
