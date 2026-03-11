#include "logging.h"

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include "nx_print.h"

namespace mycompany::yolov8_flow2 {

namespace {

LogLevel parseLogLevel(const std::string& value)
{
    std::string lower;
    lower.reserve(value.size());
    for (unsigned char c : value)
        lower.push_back(static_cast<char>(std::tolower(c)));

    if (lower == "debug")
        return LogLevel::Debug;
    if (lower == "info")
        return LogLevel::Info;
    if (lower == "warn" || lower == "warning")
        return LogLevel::Warn;
    if (lower == "error")
        return LogLevel::Error;
    return LogLevel::Info;
}

LogLevel getMinLogLevel()
{
    static LogLevel level = []() {
        const char* raw = std::getenv("NX_AI_LOG_LEVEL");
        if (!raw || raw[0] == '\0')
            return LogLevel::Info;
        return parseLogLevel(raw);
    }();
    return level;
}

bool shouldLog(LogLevel level)
{
    const LogLevel minLevel = getMinLogLevel();
    return static_cast<int>(level) >= static_cast<int>(minLevel);
}

const char* levelString(LogLevel level)
{
    switch (level)
    {
        case LogLevel::Debug:
            return "DEBUG";
        case LogLevel::Info:
            return "INFO";
        case LogLevel::Warn:
            return "WARN";
        case LogLevel::Error:
            return "ERROR";
    }
    return "INFO";
}

void writeLog(LogLevel level, const std::string& tag, const std::string& msg)
{
    NX_PRINT("[%s][%s] %s", levelString(level), tag.c_str(), msg.c_str());
}

struct ThrottleState
{
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> lastEmit;
    std::mutex mutex;
};

ThrottleState& throttleState()
{
    static ThrottleState s;
    return s;
}

} // namespace

void log(LogLevel level, const std::string& tag, const std::string& msg)
{
    if (!shouldLog(level))
        return;
    writeLog(level, tag, msg);
}

void logThrottled(
    LogLevel level,
    const std::string& tag,
    const std::string& key,
    std::chrono::milliseconds interval,
    const std::string& msg)
{
    if (!shouldLog(level))
        return;

    auto& state = throttleState();
    const auto now = std::chrono::steady_clock::now();
    bool doEmit = false;

    {
        std::lock_guard<std::mutex> lock(state.mutex);
        auto it = state.lastEmit.find(key);
        if (it == state.lastEmit.end())
        {
            state.lastEmit[key] = now;
            doEmit = true;
        }
        else
        {
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second);
            if (elapsed >= interval)
            {
                it->second = now;
                doEmit = true;
            }
        }
    }

    if (doEmit)
        writeLog(level, tag, msg);
}

} // namespace mycompany::yolov8_flow2
