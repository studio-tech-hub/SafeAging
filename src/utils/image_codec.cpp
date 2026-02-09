#include "utils/image_codec.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "utils/base64.h"

namespace mycompany::yolov8_flow2::utils {

EncodedImage encodeFrameAsBase64Jpeg(
    const cv::Mat& bgrFrame,
    int targetWidth,
    int jpegQuality)
{
    if (bgrFrame.empty())
        throw std::runtime_error("encodeFrameAsBase64Jpeg: empty frame");

    cv::Mat toEncode = bgrFrame;
    if (targetWidth > 0 && bgrFrame.cols > targetWidth)
    {
        const double scale = static_cast<double>(targetWidth) / static_cast<double>(bgrFrame.cols);
        const int resizedHeight =
            std::max(1, static_cast<int>(std::llround(static_cast<double>(bgrFrame.rows) * scale)));
        cv::resize(bgrFrame, toEncode, cv::Size(targetWidth, resizedHeight), 0.0, 0.0, cv::INTER_AREA);
    }

    if (!toEncode.isContinuous())
        toEncode = toEncode.clone();

    std::vector<uint8_t> jpegBytes;
    const int clampedQuality = std::clamp(jpegQuality, 40, 95);
    const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, clampedQuality};
    if (!cv::imencode(".jpg", toEncode, jpegBytes, params))
        throw std::runtime_error("encodeFrameAsBase64Jpeg: cv::imencode failed");

    EncodedImage result;
    result.width = toEncode.cols;
    result.height = toEncode.rows;
    result.base64Jpeg = base64Encode(jpegBytes);
    return result;
}

} // namespace mycompany::yolov8_flow2::utils
