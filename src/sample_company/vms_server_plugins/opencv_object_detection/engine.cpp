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
    // Request YUV420 frames from NX so the plugin can convert them to OpenCV Mats
    // before forwarding them to the Python analytics service.
    return /*suppress newline*/ 1 + R"json(
{
    "capabilities": "needUncompressedVideoFrames_yuv420"
}
)json";
}

} // namespace opencv_object_detection
} // namespace vms_server_plugins
} // namespace sample_company
