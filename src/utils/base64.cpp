#include "utils/base64.h"

namespace mycompany::yolov8_flow2::utils {

namespace {

constexpr char kAlphabet[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

} // namespace

std::string base64Encode(const uint8_t* data, size_t len)
{
    std::string out;
    out.reserve(((len + 2U) / 3U) * 4U);

    uint8_t block3[3] = {0, 0, 0};
    uint8_t block4[4] = {0, 0, 0, 0};
    int i = 0;

    while (len--)
    {
        block3[i++] = *(data++);
        if (i == 3)
        {
            block4[0] = (block3[0] & 0xfcU) >> 2;
            block4[1] = static_cast<uint8_t>(((block3[0] & 0x03U) << 4) + ((block3[1] & 0xf0U) >> 4));
            block4[2] = static_cast<uint8_t>(((block3[1] & 0x0fU) << 2) + ((block3[2] & 0xc0U) >> 6));
            block4[3] = block3[2] & 0x3fU;

            for (i = 0; i < 4; ++i)
                out.push_back(kAlphabet[block4[i]]);
            i = 0;
        }
    }

    if (i > 0)
    {
        for (int j = i; j < 3; ++j)
            block3[j] = 0;

        block4[0] = (block3[0] & 0xfcU) >> 2;
        block4[1] = static_cast<uint8_t>(((block3[0] & 0x03U) << 4) + ((block3[1] & 0xf0U) >> 4));
        block4[2] = static_cast<uint8_t>(((block3[1] & 0x0fU) << 2) + ((block3[2] & 0xc0U) >> 6));
        block4[3] = block3[2] & 0x3fU;

        for (int j = 0; j < i + 1; ++j)
            out.push_back(kAlphabet[block4[j]]);

        while (i++ < 3)
            out.push_back('=');
    }

    return out;
}

std::string base64Encode(const std::vector<uint8_t>& data)
{
    if (data.empty())
        return {};
    return base64Encode(data.data(), data.size());
}

} // namespace mycompany::yolov8_flow2::utils

