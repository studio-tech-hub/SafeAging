// logging.h
// Lightweight logging helper for NX plugin (levels + throttling)
// Designed to reduce noise in high-frequency processing loops.

#pragma once

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>

namespace sample_company {
namespace vms_server_plugins {
namespace opencv_object_detection {

enum class LogLevel { Debug, Info, Warn, Error };

class Logger
{
public:
    static LogLevel getLogLevel()
    {
        static LogLevel level = parseLogLevel();
        return level;
    }

    static bool shouldLog(LogLevel level)
    {
        return level >= getLogLevel();
    }

    static void log(LogLevel level, const std::string& message)
    {
        if (!shouldLog(level))
            return;

        const char* prefix = "";
        switch (level)
        {
            case LogLevel::Debug: prefix = "[DEBUG] "; break;
            case LogLevel::Info:  prefix = "[INFO] "; break;
            case LogLevel::Warn:  prefix = "[WARN] "; break;
            case LogLevel::Error: prefix = "[ERROR] "; break;
        }

        std::cerr << prefix << message << std::endl;
    }

    static void logThrottled(LogLevel level,
        const std::string& key,
        std::chrono::seconds interval,
        const std::string& message)
    {
        if (!shouldLog(level))
            return;

        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lk(throttleMutex_);

        auto& entry = throttleState_[key];
        if (now - entry.lastLog >= interval)
        {
            std::string msg = message;
            if (entry.suppressedCount > 0)
                msg += " (suppressed " + std::to_string(entry.suppressedCount) + " similar messages)";

            log(level, msg);
            entry.lastLog = now;
            entry.suppressedCount = 0;
        }
        else
        {
            ++entry.suppressedCount;
        }

        if (throttleState_.size() > 100)
            cleanupThrottleState(now);
    }

private:
    struct ThrottleEntry
    {
        std::chrono::steady_clock::time_point lastLog = std::chrono::steady_clock::now();
        size_t suppressedCount = 0;
    };

    static LogLevel parseLogLevel()
    {
        const char* env = std::getenv("PLUGIN_LOG_LEVEL");
        if (!env)
            return LogLevel::Info;

        std::string levelStr(env);
        if (levelStr == "DEBUG") return LogLevel::Debug;
        if (levelStr == "INFO") return LogLevel::Info;
        if (levelStr == "WARN") return LogLevel::Warn;
        if (levelStr == "ERROR") return LogLevel::Error;
        return LogLevel::Info;
    }

    static void cleanupThrottleState(std::chrono::steady_clock::time_point now)
    {
        for (auto it = throttleState_.begin(); it != throttleState_.end();)
        {
            if (now - it->second.lastLog > std::chrono::minutes(5))
                it = throttleState_.erase(it);
            else
                ++it;
        }
    }

    static std::mutex throttleMutex_;
    static std::unordered_map<std::string, ThrottleEntry> throttleState_;
};

inline std::mutex Logger::throttleMutex_;
inline std::unordered_map<std::string, Logger::ThrottleEntry> Logger::throttleState_;

} // namespace opencv_object_detection
} // namespace vms_server_plugins
} // namespace sample_company
