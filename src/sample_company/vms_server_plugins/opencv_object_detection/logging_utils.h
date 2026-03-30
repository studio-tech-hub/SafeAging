// logging_utils.h
#pragma once

#include <chrono>
#include <cctype>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
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

inline std::mutex& sinkMutex()
{
    static std::mutex mutex;
    return mutex;
}

inline std::string configuredLogFilePath()
{
    static const std::string path = []()
    {
        const char* env = std::getenv("SAFEAGING_LOG_FILE");
        if (env && *env)
            return std::string(env);

#ifdef _WIN32
        return std::string(R"(D:\SafeAging\logs\plugin.log)");
#else
        return std::string("/tmp/safeaging_plugin.log");
#endif
    }();

    return path;
}

inline std::string makeTimestamp()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);

    std::tm localTime{};
#ifdef _WIN32
    localtime_s(&localTime, &time);
#else
    localtime_r(&time, &localTime);
#endif

    std::ostringstream stream;
    stream << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S");
    return stream.str();
}

inline void appendToLogFileUnlocked(const std::string& line)
{
    struct FileSink
    {
        std::ofstream stream;
        bool initialized = false;
        bool available = false;
        bool warned = false;
    };

    static FileSink sink;
    if (!sink.initialized)
    {
        sink.initialized = true;

        const std::filesystem::path logPath(configuredLogFilePath());
        std::error_code ec;
        if (logPath.has_parent_path())
            std::filesystem::create_directories(logPath.parent_path(), ec);

        sink.stream.open(logPath, std::ios::app);
        sink.available = sink.stream.is_open();

        if (!sink.available && !sink.warned)
        {
            sink.warned = true;
            std::clog << "[SafeAging][WARN] Failed to open plugin log file: "
                      << logPath.string() << std::endl;
        }
    }

    if (!sink.available)
        return;

    sink.stream << line << std::endl;
    sink.stream.flush();
}

inline void log(Level level, const std::string& message)
{
    if (!shouldLog(level))
        return;

    const std::string consoleLine =
        "[SafeAging][" + std::string(levelName(level)) + "] " + message;
    const std::string fileLine = "[" + makeTimestamp() + "] " + consoleLine;

    std::lock_guard<std::mutex> lock(sinkMutex());
    std::clog << consoleLine << std::endl;
    appendToLogFileUnlocked(fileLine);
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

