// Copyright 2018-present Network Optix, Inc. Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#include "detection.h"

namespace sample_company {
namespace vms_server_plugins {
namespace opencv_object_detection {

const std::vector<std::string> kClassesToDetect{"cat", "dog", "person"};
const std::map<std::string, std::string> kClassesToDetectPluralCapitalized{
    {"cat", "Cats"}, {"dog", "Dogs"}, {"person", "People"}};

} // namespace opencv_object_detection
} // namespace vms_server_plugins
} // namespace sample_company
