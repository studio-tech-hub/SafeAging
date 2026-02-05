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
    // Model nằm cùng thư mục plugin:
    // C:\Program Files\Network Optix\Nx Meta\MediaServer\plugins\yolov8_people_analytics_plugin\yolov8n.onnx
    m_modelPath = m_pluginHomeDir / "yolov5s.onnx";

    // Nếu bạn để trong subfolder "models" thì đổi lại:
    // m_modelPath = m_pluginHomeDir / "models" / "yolov8n.onnx";
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
        m_pluginHomeDir,
        m_modelPath);
}

std::string Engine::manifestString() const
{
    // Request YUV420 format (same as internal NX server format, more efficient)
    // YV12 format is YUV 4:2:0 planar, which we properly convert to BGR for OpenCV
    return /*suppress newline*/ 1 + R"json(
{
    "capabilities": "needUncompressedVideoFrames_yuv420"
}
)json";
}

} // namespace opencv_object_detection
} // namespace vms_server_plugins
} // namespace sample_company
