#pragma once

#include <filesystem>

#include <nx/sdk/analytics/helpers/engine.h>

#include "device_agent.h"

namespace mycompany::yolov8_flow2 {

class Engine: public nx::sdk::analytics::Engine
{
public:
    explicit Engine(std::filesystem::path pluginHomeDir);
    ~Engine() override;

protected:
    std::string manifestString() const override;
    void doObtainDeviceAgent(
        nx::sdk::Result<nx::sdk::analytics::IDeviceAgent*>* outResult,
        const nx::sdk::IDeviceInfo* deviceInfo) override;

private:
    DeviceAgentConfig loadConfigFromEnvironment() const;

private:
    std::filesystem::path m_pluginHomeDir;
    DeviceAgentConfig m_config;
};

} // namespace mycompany::yolov8_flow2

