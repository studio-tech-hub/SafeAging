// engine.cpp
// Copyright 2018-present Network Optix, Inc.
// Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#include "engine.h"

#include "device_agent.h"

namespace sample_company {
namespace vms_server_plugins {
namespace opencv_object_detection {

using namespace nx::sdk;
using namespace nx::sdk::analytics;

Engine::Engine(std::filesystem::path pluginHomeDir):
    nx::sdk::analytics::Engine(/*enableOutput*/ true),
    m_pluginHomeDir(std::move(pluginHomeDir))
{
}

Engine::~Engine()
{
}

void Engine::doObtainDeviceAgent(
    Result<IDeviceAgent*>* outResult,
    const IDeviceInfo* deviceInfo)
{
    *outResult = new DeviceAgent(
        deviceInfo,
        m_pluginHomeDir);
}

std::string Engine::manifestString() const
{
    // Request BGR frames — native OpenCV layout; avoids YUV420 plane issues on some
    // ARM decoders (null Y plane / missing Y plane spam on QCS6490-class boxes).
    return /*suppress newline*/ 1 + R"json(
{
    "capabilities": "needUncompressedVideoFrames_bgr"
}
)json";
}

} // namespace opencv_object_detection
} // namespace vms_server_plugins
} // namespace sample_company
