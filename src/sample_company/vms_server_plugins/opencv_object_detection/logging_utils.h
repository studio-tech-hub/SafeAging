// logging_utils.h
#pragma once

#include <chrono>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>

namespace sample_company {
namespace vms_server_plugins {
namespace opencv_object_detection {
namespace logutil {

enum class Level
{
    debug = 0,
    info = 1,
    warn = 2,
    error = 3
};

inline const char* levelName(Level level)
{
    switch (level)
    {
        case Level::debug: return "DEBUG";
        case Level::info: return "INFO";
        case Level::warn: return "WARN";
        case Level::error: return "ERROR";
        default: return "INFO";
    }
}

inline Level configuredLevel()
{
    static const Level level = []()
    {
        const char* env = std::getenv("SAFEAGING_LOG_LEVEL");
        if (!env)
            return Level::info;

        std::string value(env);
        for (char& c : value)
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

        if (value == "DEBUG")
            return Level::debug;
        if (value == "INFO")
            return Level::info;
        if (value == "WARN" || value == "WARNING")
            return Level::warn;
        if (value == "ERROR")
            return Level::error;

        return Level::info;
    }();

    return level;
}

inline bool shouldLog(Level level)
{
    return static_cast<int>(level) >= static_cast<int>(configuredLevel());
}

inline void log(Level level, const std::string& message)
{
    if (!shouldLog(level))
        return;

    std::clog << "[SafeAging][" << levelName(level) << "] " << message << std::endl;
}

struct ThrottleState
{
    std::chrono::steady_clock::time_point lastEmitted;
    uint64_t suppressed = 0;
    bool initialized = false;
};

inline bool shouldEmit(
    const std::string& key,
    std::chrono::milliseconds interval,
    uint64_t* suppressedOut = nullptr)
{
    static std::mutex mutex;
    static std::unordered_map<std::string, ThrottleState> states;

    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(mutex);

    auto& state = states[key];
    if (!state.initialized || now - state.lastEmitted >= interval)
    {
        if (suppressedOut)
            *suppressedOut = state.suppressed;
        state.lastEmitted = now;
        state.suppressed = 0;
        state.initialized = true;
        return true;
    }

    ++state.suppressed;
    return false;
}

inline void logThrottled(
    Level level,
    const std::string& key,
    std::chrono::milliseconds interval,
    const std::string& message)
{
    if (!shouldLog(level))
        return;

    uint64_t suppressed = 0;
    if (!shouldEmit(key, interval, &suppressed))
        return;

    if (suppressed > 0)
    {
        log(level, message + " (+" + std::to_string(suppressed) + " suppressed)");
        return;
    }

    log(level, message);
}

} // namespace logutil
} // namespace opencv_object_detection
} // namespace vms_server_plugins
} // namespace sample_company

