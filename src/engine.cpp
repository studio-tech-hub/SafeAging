#include "engine.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <utility>

#include "device_agent.h"

namespace mycompany::yolov8_flow2 {

namespace {

std::string envString(const char* key, const std::string& defaultValue)
{
    const char* value = std::getenv(key);
    if (!value || std::string(value).empty())
        return defaultValue;
    return value;
}

int envInt(const char* key, int defaultValue, int minValue, int maxValue)
{
    const char* value = std::getenv(key);
    if (!value)
        return defaultValue;
    try
    {
        const int parsed = std::stoi(value);
        return std::clamp(parsed, minValue, maxValue);
    }
    catch (...)
    {
        return defaultValue;
    }
}

double envDouble(const char* key, double defaultValue, double minValue, double maxValue)
{
    const char* value = std::getenv(key);
    if (!value)
        return defaultValue;
    try
    {
        const double parsed = std::stod(value);
        return std::max(minValue, std::min(maxValue, parsed));
    }
    catch (...)
    {
        return defaultValue;
    }
}

} // namespace

Engine::Engine(std::filesystem::path pluginHomeDir):
    nx::sdk::analytics::Engine(/*enableOutput*/ true),
    m_pluginHomeDir(std::move(pluginHomeDir)),
    m_config(loadConfigFromEnvironment())
{
}

Engine::~Engine() = default;

std::string Engine::manifestString() const
{
    return /*suppress newline*/ 1 + R"json(
{
    "capabilities": "needUncompressedVideoFrames_yuv420"
}
)json";
}

void Engine::doObtainDeviceAgent(
    nx::sdk::Result<nx::sdk::analytics::IDeviceAgent*>* outResult,
    const nx::sdk::IDeviceInfo* deviceInfo)
{
    *outResult = new DeviceAgent(deviceInfo, m_pluginHomeDir, m_config);
}

DeviceAgentConfig Engine::loadConfigFromEnvironment() const
{
    DeviceAgentConfig config;
    config.detector.serviceUrl = envString("NX_AI_SERVICE_URL", "http://127.0.0.1:18000");
    config.detector.connectTimeoutMs = envInt("NX_AI_TIMEOUT_CONNECT_MS", 250, 50, 5000);
    config.detector.readTimeoutMs = envInt("NX_AI_TIMEOUT_READ_MS", 400, 50, 5000);
    config.detector.writeTimeoutMs = envInt("NX_AI_TIMEOUT_WRITE_MS", 250, 50, 5000);
    config.detector.sendWidth = envInt("NX_AI_SEND_WIDTH", 640, 160, 3840);
    config.detector.jpegQuality = envInt("NX_AI_JPEG_QUALITY", 80, 40, 95);
    config.detector.circuitFailureThreshold = envInt("NX_AI_CIRCUIT_FAILS", 3, 1, 20);
    config.detector.circuitOpenMs = envInt("NX_AI_CIRCUIT_OPEN_MS", 3000, 200, 60000);
    config.detector.logThrottleMs = envInt("NX_AI_LOG_THROTTLE_MS", 5000, 200, 60000);

    config.sampleFps = envDouble("NX_AI_SAMPLE_FPS", 5.0, 0.1, 60.0);
    config.maxQueueSize = static_cast<size_t>(envInt("NX_AI_QUEUE_SIZE", 4, 1, 120));
    config.fallFinishGraceUs =
        static_cast<int64_t>(envInt("NX_AI_FALL_FINISH_MS", 3000, 0, 120000)) * 1000;
    config.syntheticTrackTtlUs =
        static_cast<int64_t>(envInt("NX_AI_SYNTH_TRACK_TTL_MS", 2000, 100, 120000)) * 1000;
    config.trackMapTtlUs =
        static_cast<int64_t>(envInt("NX_AI_TRACK_MAP_TTL_MS", 60000, 1000, 3'600'000)) * 1000;
    config.logThrottleMs = envInt("NX_AI_LOG_THROTTLE_MS", 5000, 200, 60000);
    return config;
}

} // namespace mycompany::yolov8_flow2
