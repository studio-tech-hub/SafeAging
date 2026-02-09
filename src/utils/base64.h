#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mycompany::yolov8_flow2::utils {

std::string base64Encode(const uint8_t* data, size_t len);
std::string base64Encode(const std::vector<uint8_t>& data);

} // namespace mycompany::yolov8_flow2::utils
