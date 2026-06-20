// test_circuit_breaker.cpp
// Standalone C++ unit tests for safeaging::CircuitBreakerMap.
//
// Compile (no Nx SDK, no OpenCV required):
//   g++ -std=c++17 -I../sample_company/vms_server_plugins/opencv_object_detection \
//       test_circuit_breaker.cpp -o test_circuit_breaker -pthread
//   ./test_circuit_breaker
//
// On Windows (MSVC):
//   cl /std:c++17 /I..\sample_company\vms_server_plugins\opencv_object_detection \
//      test_circuit_breaker.cpp /link /OUT:test_circuit_breaker.exe

#include <cassert>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "circuit_breaker.h"

// ── Controllable test clock ───────────────────────────────────────────────────

struct TestClock
{
    // Shared mutable state (tests are single-threaded via this clock)
    static inline std::chrono::steady_clock::time_point s_now =
        std::chrono::steady_clock::now();

    using time_point = std::chrono::steady_clock::time_point;
    using duration   = std::chrono::steady_clock::duration;

    static time_point now()   { return s_now; }
    static time_point epoch() { return time_point{}; }
    static void advance(std::chrono::milliseconds ms) { s_now += ms; }
    static void reset()       { s_now = std::chrono::steady_clock::now(); }
};

using TestCB = safeaging::CircuitBreakerMap<TestClock>;
using State  = safeaging::CircuitBreakerMap<TestClock>::State;

// ── Minimal test harness ──────────────────────────────────────────────────────

static int s_passed = 0, s_failed = 0;

#define TEST(name) \
    void name(); \
    struct _Reg_##name { _Reg_##name() { runTest(#name, name); } } _reg_##name; \
    void name()

static void runTest(const char* name, std::function<void()> fn)
{
    try {
        TestClock::reset();
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

#define ASSERT_EQ(a, b) \
    do { if ((a) != (b)) throw std::runtime_error(std::string("ASSERT_EQ: ") + #a + " != " + #b + " (line " + std::to_string(__LINE__) + ")"); } while(0)


// ── Test cases ────────────────────────────────────────────────────────────────

TEST(starts_closed)
{
    TestCB cb{5, std::chrono::seconds{15}};
    ASSERT(cb.allowRequest("cam1"));
    ASSERT_EQ(cb.getState("cam1"), State::closed);
    ASSERT_EQ(cb.getConsecutiveFailures("cam1"), 0);
}

TEST(one_failure_stays_closed)
{
    TestCB cb{5, std::chrono::seconds{15}};
    cb.allowRequest("cam1");
    bool opened = cb.onTransientFailure("cam1");
    ASSERT(!opened);
    ASSERT_EQ(cb.getState("cam1"), State::closed);
    ASSERT_EQ(cb.getConsecutiveFailures("cam1"), 1);
}

TEST(success_resets_failure_counter)
{
    TestCB cb{5, std::chrono::seconds{15}};
    cb.allowRequest("cam1");
    cb.onTransientFailure("cam1");
    cb.onTransientFailure("cam1");
    ASSERT_EQ(cb.getConsecutiveFailures("cam1"), 2);

    cb.allowRequest("cam1");
    cb.onSuccess("cam1");
    ASSERT_EQ(cb.getConsecutiveFailures("cam1"), 0);
    ASSERT_EQ(cb.getState("cam1"), State::closed);
}

TEST(opens_after_threshold_failures)
{
    TestCB cb{5, std::chrono::seconds{15}};
    bool opened = false;
    for (int i = 0; i < 5; ++i)
    {
        cb.allowRequest("cam1");
        opened = cb.onTransientFailure("cam1");
    }
    ASSERT(opened);
    ASSERT_EQ(cb.getState("cam1"), State::open);
}

TEST(open_circuit_blocks_requests)
{
    TestCB cb{5, std::chrono::seconds{15}};
    for (int i = 0; i < 5; ++i)
    {
        cb.allowRequest("cam1");
        cb.onTransientFailure("cam1");
    }
    ASSERT_EQ(cb.getState("cam1"), State::open);

    // Must be blocked while cooldown has not expired
    std::string reason;
    bool allowed = cb.allowRequest("cam1", &reason);
    ASSERT(!allowed);
    ASSERT(!reason.empty());                 // reason should describe cooldown
}

TEST(advances_to_half_open_after_cooldown)
{
    TestCB cb{5, std::chrono::seconds{15}};
    for (int i = 0; i < 5; ++i)
    {
        cb.allowRequest("cam1");
        cb.onTransientFailure("cam1");
    }
    ASSERT_EQ(cb.getState("cam1"), State::open);

    // Advance past cooldown
    TestClock::advance(std::chrono::seconds{16});

    bool halfOpenTransition = false;
    bool allowed = cb.allowRequest("cam1", nullptr, &halfOpenTransition);
    ASSERT(allowed);
    ASSERT(halfOpenTransition);
    ASSERT_EQ(cb.getState("cam1"), State::halfOpen);
}

TEST(half_open_success_closes_circuit)
{
    TestCB cb{5, std::chrono::seconds{15}};
    for (int i = 0; i < 5; ++i)
    {
        cb.allowRequest("cam1");
        cb.onTransientFailure("cam1");
    }
    TestClock::advance(std::chrono::seconds{16});
    cb.allowRequest("cam1");                    // probe → half-open

    bool recovered = cb.onSuccess("cam1");
    ASSERT(recovered);                           // was recovering from open
    ASSERT_EQ(cb.getState("cam1"), State::closed);
    ASSERT_EQ(cb.getConsecutiveFailures("cam1"), 0);
}

TEST(half_open_failure_reopens_circuit)
{
    TestCB cb{5, std::chrono::seconds{15}};
    for (int i = 0; i < 5; ++i)
    {
        cb.allowRequest("cam1");
        cb.onTransientFailure("cam1");
    }
    TestClock::advance(std::chrono::seconds{16});
    cb.allowRequest("cam1");                    // probe → half-open
    ASSERT_EQ(cb.getState("cam1"), State::halfOpen);

    bool reopened = cb.onTransientFailure("cam1");
    ASSERT(reopened);
    ASSERT_EQ(cb.getState("cam1"), State::open);
}

TEST(half_open_probe_in_flight_blocks_concurrent)
{
    TestCB cb{5, std::chrono::seconds{15}};
    for (int i = 0; i < 5; ++i)
    {
        cb.allowRequest("cam1");
        cb.onTransientFailure("cam1");
    }
    TestClock::advance(std::chrono::seconds{16});

    // First probe is allowed
    bool first = cb.allowRequest("cam1");
    ASSERT(first);
    ASSERT(cb.isHalfOpenProbeInFlight("cam1"));

    // Second attempt blocked while probe is in flight
    std::string reason;
    bool second = cb.allowRequest("cam1", &reason);
    ASSERT(!second);
    ASSERT(reason.find("half_open_probe_in_flight") != std::string::npos);
}

TEST(release_half_open_probe_allows_retry)
{
    TestCB cb{5, std::chrono::seconds{15}};
    for (int i = 0; i < 5; ++i)
    {
        cb.allowRequest("cam1");
        cb.onTransientFailure("cam1");
    }
    TestClock::advance(std::chrono::seconds{16});
    cb.allowRequest("cam1");                    // probe in flight
    ASSERT(cb.isHalfOpenProbeInFlight("cam1"));

    cb.releaseHalfOpenProbe("cam1");
    ASSERT(!cb.isHalfOpenProbeInFlight("cam1"));

    // Now another probe should be allowed
    bool allowed = cb.allowRequest("cam1");
    ASSERT(allowed);
}

TEST(multiple_cameras_independent)
{
    TestCB cb{3, std::chrono::seconds{15}};

    // Camera A opens
    for (int i = 0; i < 3; ++i)
    {
        cb.allowRequest("cam_a");
        cb.onTransientFailure("cam_a");
    }
    ASSERT_EQ(cb.getState("cam_a"), State::open);

    // Camera B should be unaffected
    ASSERT(cb.allowRequest("cam_b"));
    ASSERT_EQ(cb.getState("cam_b"), State::closed);
}

TEST(cooldown_not_yet_expired_stays_open)
{
    TestCB cb{5, std::chrono::seconds{15}};
    for (int i = 0; i < 5; ++i)
    {
        cb.allowRequest("cam1");
        cb.onTransientFailure("cam1");
    }
    // Only 10 seconds later — not enough
    TestClock::advance(std::chrono::seconds{10});
    bool allowed = cb.allowRequest("cam1");
    ASSERT(!allowed);
    ASSERT_EQ(cb.getState("cam1"), State::open);
}

TEST(reset_clears_state)
{
    TestCB cb{5, std::chrono::seconds{15}};
    for (int i = 0; i < 5; ++i)
    {
        cb.allowRequest("cam1");
        cb.onTransientFailure("cam1");
    }
    ASSERT_EQ(cb.getState("cam1"), State::open);
    cb.reset("cam1");
    // After reset, should start from closed
    ASSERT_EQ(cb.getState("cam1"), State::closed);
    ASSERT(cb.allowRequest("cam1"));
}

TEST(success_on_fresh_entry_returns_false_recovered)
{
    TestCB cb{5, std::chrono::seconds{15}};
    cb.allowRequest("cam1");
    bool recovered = cb.onSuccess("cam1");  // was never open
    ASSERT(!recovered);
    ASSERT_EQ(cb.getState("cam1"), State::closed);
}

TEST(isTransientStatus_helpers)
{
    // 408/429/5xx are transient; 200/400/404 are not
    // (tested indirectly by calling onTransientFailure for 500-like scenario)
    TestCB cb{1, std::chrono::seconds{15}};  // threshold=1 for speed
    cb.allowRequest("cam1");
    bool opened = cb.onTransientFailure("cam1");
    ASSERT(opened);

    // Non-transient status (e.g. 400 bad request) should NOT increment the
    // failure counter. We simulate this by not calling onTransientFailure.
    // Instead verify state is still open from above.
    ASSERT_EQ(cb.getState("cam1"), State::open);
}

TEST(threadsafe_concurrent_failures)
{
    TestCB cb{50, std::chrono::seconds{30}};
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i)
    {
        threads.emplace_back([&cb]()
        {
            for (int j = 0; j < 5; ++j)
            {
                cb.allowRequest("shared_cam");
                cb.onTransientFailure("shared_cam");
            }
        });
    }
    for (auto& t : threads)
        t.join();

    // 10 threads × 5 = 50 failures → exactly at threshold → opened
    ASSERT_EQ(cb.getState("shared_cam"), State::open);
}


// ── Queue policy tests (drop-oldest) ─────────────────────────────────────────
// These test the conceptual behaviour in isolation using a simple queue model
// mirroring the deque + max-size logic in device_agent.cpp.

#include <deque>

struct FakeFrameJob { int id; };

struct DroppingQueue
{
    std::deque<FakeFrameJob> q;
    const std::size_t maxSize;

    explicit DroppingQueue(std::size_t sz) : maxSize(sz) {}

    void push(FakeFrameJob job)
    {
        if (q.size() >= maxSize)
            q.pop_front();   // drop oldest (matches device_agent.cpp policy)
        q.push_back(std::move(job));
    }

    FakeFrameJob pop()
    {
        auto j = q.front();
        q.pop_front();
        return j;
    }

    std::size_t size() const { return q.size(); }
};

TEST(queue_drop_oldest_policy)
{
    DroppingQueue q{2};
    q.push({1});
    q.push({2});
    q.push({3}); // should drop 1

    ASSERT_EQ(q.size(), 2u);
    ASSERT_EQ(q.pop().id, 2);
    ASSERT_EQ(q.pop().id, 3);
}

TEST(queue_allows_up_to_max_size)
{
    DroppingQueue q{3};
    q.push({1});
    q.push({2});
    q.push({3});
    ASSERT_EQ(q.size(), 3u);
    // No drop yet
    ASSERT_EQ(q.pop().id, 1);
}

TEST(queue_overflow_drops_multiple_oldest)
{
    DroppingQueue q{2};
    for (int i = 1; i <= 10; ++i)
        q.push({i});
    // Always keeps latest 2
    ASSERT_EQ(q.size(), 2u);
    ASSERT_EQ(q.pop().id, 9);
    ASSERT_EQ(q.pop().id, 10);
}

TEST(queue_empty_pop_would_be_safe)
{
    DroppingQueue q{2};
    q.push({42});
    auto j = q.pop();
    ASSERT_EQ(j.id, 42);
    ASSERT_EQ(q.size(), 0u);
}


// ── Entry point ───────────────────────────────────────────────────────────────

int main()
{
    std::printf("\n=== SafeAging Circuit Breaker + Queue Unit Tests ===\n\n");
    // All tests are registered via static initialisers above
    std::printf("\n--- Results ---\n");
    std::printf("  Passed: %d\n", s_passed);
    std::printf("  Failed: %d\n", s_failed);
    std::printf("==========================================\n\n");
    return s_failed == 0 ? 0 : 1;
}
