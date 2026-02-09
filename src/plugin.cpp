#include "plugin.h"

#include <filesystem>

#include "engine.h"

namespace mycompany::yolov8_flow2 {

nx::sdk::Result<nx::sdk::analytics::IEngine*> Plugin::doObtainEngine()
{
    const auto utilityProvider = this->utilityProvider();
    const std::filesystem::path pluginHomeDir = utilityProvider->homeDir();
    return new Engine(pluginHomeDir);
}

std::string Plugin::manifestString() const
{
    return /*suppress newline*/ 1 + R"json(
{
    "id": "mycompany.yolov8_flow2",
    "name": "YOLOv8 FLOW2 Analytics",
    "description": "Nx plugin receives frames, sends to AI service /infer, and publishes object/event metadata.",
    "version": "2.0.0",
    "vendor": "mycompany",
    "engineSettingsModel": {
        "type": "Settings",
        "items": []
    },
    "deviceAgentSettingsModel": {
        "type": "Settings",
        "items": []
    }
}
)json";
}

} // namespace mycompany::yolov8_flow2

extern "C" NX_PLUGIN_API nx::sdk::IPlugin* createNxPlugin()
{
    return new mycompany::yolov8_flow2::Plugin();
}
