#pragma once

#include <nx/sdk/analytics/helpers/plugin.h>

namespace mycompany::yolov8_flow2 {

class Plugin: public nx::sdk::analytics::Plugin
{
protected:
    nx::sdk::Result<nx::sdk::analytics::IEngine*> doObtainEngine() override;
    std::string manifestString() const override;
};

} // namespace mycompany::yolov8_flow2

