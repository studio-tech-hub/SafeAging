// circuit_breaker.h
// Header-only, policy-based circuit breaker map used by ObjectDetector.
// No Nx SDK or OpenCV headers required — fully testable in isolation.
//
// Usage in production:
//   CircuitBreakerMap<> cb;          // uses std::chrono::steady_clock
//
// Usage in tests (controllable time):
//   CircuitBreakerMap<TestClock> cb;
//   TestClock::advance(std::chrono::seconds{20}); // skip cooldown

#pragma once

#include <algorithm>
#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace safeaging {

// ── Default clock policy (production) ────────────────────────────────────────

struct SteadyClock
{
    using time_point = std::chrono::steady_clock::time_point;
    using duration   = std::chrono::steady_clock::duration;
    static time_point now() { return std::chrono::steady_clock::now(); }
    static time_point epoch() { return time_point{}; }
};


// ── CircuitBreakerMap<Clock> ──────────────────────────────────────────────────

template<typename ClockT = SteadyClock>
class CircuitBreakerMap
{
public:
    enum class State { closed, open, halfOpen };

    struct Entry
    {
        State state = State::closed;
        int   consecutiveFailures    = 0;
        bool  halfOpenProbeInFlight  = false;
        typename ClockT::time_point openUntil = ClockT::epoch();
        typename ClockT::time_point lastSeen  = ClockT::epoch();
    };

    explicit CircuitBreakerMap(
        int               failureThreshold = 5,
        std::chrono::seconds cooldown      = std::chrono::seconds{15},
        std::size_t       maxMapSize       = 256)
        : m_failureThreshold(failureThreshold)
        , m_cooldown(cooldown)
        , m_maxMapSize(maxMapSize)
    {}

    // ── Public API ────────────────────────────────────────────────────────────

    /// Returns true if the caller is allowed to make a request for this camera.
    /// May transition from open → half-open if the cooldown has expired.
    bool allowRequest(
        const std::string& cameraId,
        std::string*       outReason             = nullptr,
        bool*              outHalfOpenTransition = nullptr)
    {
        const auto now = ClockT::now();
        std::lock_guard<std::mutex> lk(m_mutex);
        cleanup(now);

        auto& s = m_map[cameraId];
        s.lastSeen = now;
        if (outHalfOpenTransition)
            *outHalfOpenTransition = false;

        if (s.state == State::open)
        {
            if (now < s.openUntil)
            {
                if (outReason)
                {
                    auto remainMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                        s.openUntil - now).count();
                    *outReason = "cooldown_ms=" + std::to_string(std::max<long long>(0, remainMs));
                }
                return false;
            }
            // Cooldown expired → transition to half-open
            s.state = State::halfOpen;
            s.halfOpenProbeInFlight = true;
            if (outHalfOpenTransition)
                *outHalfOpenTransition = true;
            return true;
        }

        if (s.state == State::halfOpen)
        {
            if (s.halfOpenProbeInFlight)
            {
                if (outReason)
                    *outReason = "half_open_probe_in_flight";
                return false;
            }
            s.halfOpenProbeInFlight = true;
            return true;
        }

        return true; // closed → always allow
    }

    /// Mark a request as successful. Transitions to closed if in half-open.
    /// Returns true if state actually changed (i.e. was recovering from open).
    bool onSuccess(const std::string& cameraId)
    {
        const auto now = ClockT::now();
        std::lock_guard<std::mutex> lk(m_mutex);
        auto& s = m_map[cameraId];
        const bool recovered = (s.state != State::closed);
        s.state                 = State::closed;
        s.consecutiveFailures   = 0;
        s.halfOpenProbeInFlight = false;
        s.openUntil             = ClockT::epoch();
        s.lastSeen              = now;
        return recovered;
    }

    /// Mark a transient failure (5xx, timeout, connection refused).
    /// Increments failure counter; opens the circuit once threshold is reached.
    /// Also re-opens immediately if in half-open state.
    /// Returns true if the circuit just opened.
    bool onTransientFailure(const std::string& cameraId)
    {
        const auto now = ClockT::now();
        std::lock_guard<std::mutex> lk(m_mutex);
        auto& s = m_map[cameraId];
        s.lastSeen = now;

        if (s.state == State::halfOpen)
        {
            s.state               = State::open;
            s.halfOpenProbeInFlight = false;
            s.consecutiveFailures = 0;
            s.openUntil           = now + m_cooldown;
            return true;
        }

        ++s.consecutiveFailures;
        if (s.consecutiveFailures >= m_failureThreshold)
        {
            s.state               = State::open;
            s.halfOpenProbeInFlight = false;
            s.consecutiveFailures = 0;
            s.openUntil           = now + m_cooldown;
            return true;
        }
        return false;
    }

    /// Called when a half-open probe was cancelled (e.g. request timed out
    /// before receiving a response) so the next attempt can try again.
    void releaseHalfOpenProbe(const std::string& cameraId)
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        auto it = m_map.find(cameraId);
        if (it == m_map.end())
            return;
        it->second.lastSeen = ClockT::now();
        if (it->second.state == State::halfOpen)
            it->second.halfOpenProbeInFlight = false;
    }

    // ── Inspection (test helpers) ─────────────────────────────────────────────

    State getState(const std::string& cameraId) const
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        auto it = m_map.find(cameraId);
        return it == m_map.end() ? State::closed : it->second.state;
    }

    int getConsecutiveFailures(const std::string& cameraId) const
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        auto it = m_map.find(cameraId);
        return it == m_map.end() ? 0 : it->second.consecutiveFailures;
    }

    bool isHalfOpenProbeInFlight(const std::string& cameraId) const
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        auto it = m_map.find(cameraId);
        return it != m_map.end() && it->second.halfOpenProbeInFlight;
    }

    void reset(const std::string& cameraId)
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_map.erase(cameraId);
    }

    std::size_t size() const
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        return m_map.size();
    }

private:
    void cleanup(const typename ClockT::time_point& now)
    {
        // Lightweight GC: only check every 256 calls
        if ((++m_accessCount % 256) != 0)
            return;

        for (auto it = m_map.begin(); it != m_map.end();)
        {
            if (now - it->second.lastSeen > std::chrono::minutes{30})
                it = m_map.erase(it);
            else
                ++it;
        }

        if (m_map.size() <= m_maxMapSize)
            return;

        using Pair = std::pair<std::string, typename ClockT::time_point>;
        std::vector<Pair> ages;
        ages.reserve(m_map.size());
        for (const auto& kv : m_map)
            ages.push_back({kv.first, kv.second.lastSeen});
        std::sort(ages.begin(), ages.end(), [](const Pair& a, const Pair& b){
            return a.second < b.second;
        });
        const std::size_t toRemove = m_map.size() - m_maxMapSize;
        for (std::size_t i = 0; i < toRemove; ++i)
            m_map.erase(ages[i].first);
    }

    const int            m_failureThreshold;
    const std::chrono::seconds m_cooldown;
    const std::size_t    m_maxMapSize;
    mutable std::mutex   m_mutex;
    std::unordered_map<std::string, Entry> m_map;
    std::size_t          m_accessCount = 0;
};

// ── Convenience alias ─────────────────────────────────────────────────────────
using DefaultCircuitBreakerMap = CircuitBreakerMap<SteadyClock>;

} // namespace safeaging
