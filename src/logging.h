#pragma once

#include <chrono>
#include <string>

namespace mycompany::yolov8_flow2 {

enum class LogLevel
{
    Debug,
    Info,
    Warn,
    Error
};

void log(LogLevel level, const std::string& tag, const std::string& msg);

void logThrottled(
    LogLevel level,
    const std::string& tag,
    const std::string& key,
    std::chrono::milliseconds interval,
    const std::string& msg);

} // namespace mycompany::yolov8_flow2
