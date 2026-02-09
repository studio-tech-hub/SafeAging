#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace mycompany::yolov8_flow2::utils {

struct EncodedImage
{
    std::string base64Jpeg;
    int width = 0;
    int height = 0;
};

EncodedImage encodeFrameAsBase64Jpeg(
    const cv::Mat& bgrFrame,
    int targetWidth,
    int jpegQuality);

} // namespace mycompany::yolov8_flow2::utils

