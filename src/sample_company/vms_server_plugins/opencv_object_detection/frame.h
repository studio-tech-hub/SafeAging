// Copyright 2018-present Network Optix, Inc. Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <nx/sdk/analytics/i_uncompressed_video_frame.h>

namespace sample_company {
    namespace vms_server_plugins {
        namespace opencv_object_detection {

            /**
             * Stores frame data and cv::Mat. Note, there is no copying of image data in the constructor.
             */
            struct Frame
            {
                const int width;
                const int height;
                const int64_t timestampUs;
                const int64_t index;
                cv::Mat cvMat;

            public:
                Frame(const nx::sdk::analytics::IUncompressedVideoFrame* frame, int64_t index) :
                    width(frame->width()),
                    height(frame->height()),
                    timestampUs(frame->timestampUs()),
                    index(index),
                    cvMat()  // Default empty Mat
                {
                    using PixelFormat = nx::sdk::analytics::IUncompressedVideoFrame::PixelFormat;
                    const PixelFormat pf = frame->pixelFormat();
                    const int w = width;
                    const int h = height;

                    if (pf == PixelFormat::bgr)
                    {
                        cv::Mat temp(h, w, CV_8UC3, (void*)frame->data(0), (size_t)frame->lineSize(0));
                        cvMat = temp.clone();  // Clone to ensure data is owned by this Mat
                    }
                    else if (pf == PixelFormat::bgra)
                    {
                        cv::Mat bgra(h, w, CV_8UC4, (void*)frame->data(0), (size_t)frame->lineSize(0));
                        if (!bgra.empty())
                        {
                            cv::cvtColor(bgra, cvMat, cv::COLOR_BGRA2BGR);
                            if (cvMat.empty())
                                throw std::runtime_error("cvtColor(BGRA->BGR) produced empty Mat");
                        }
                        else
                            throw std::runtime_error("Failed to create bgra Mat from frame data");
                    }
                    else if (pf == PixelFormat::rgba)
                    {
                        cv::Mat rgba(h, w, CV_8UC4, (void*)frame->data(0), (size_t)frame->lineSize(0));
                        if (!rgba.empty())
                        {
                            cv::cvtColor(rgba, cvMat, cv::COLOR_RGBA2BGR);
                            if (cvMat.empty())
                                throw std::runtime_error("cvtColor(RGBA->BGR) produced empty Mat");
                        }
                        else
                            throw std::runtime_error("Failed to create rgba Mat from frame data");
                    }
                    else if (pf == PixelFormat::rgb)
                    {
                        cv::Mat rgb(h, w, CV_8UC3, (void*)frame->data(0), (size_t)frame->lineSize(0));
                        if (!rgb.empty())
                        {
                            cv::cvtColor(rgb, cvMat, cv::COLOR_RGB2BGR);
                            if (cvMat.empty())
                                throw std::runtime_error("cvtColor(RGB->BGR) produced empty Mat");
                        }
                        else
                            throw std::runtime_error("Failed to create rgb Mat from frame data");
                    }
                    else if (pf == PixelFormat::argb)
                    {
                        cv::Mat argb(h, w, CV_8UC4, (void*)frame->data(0), (size_t)frame->lineSize(0));
                        if (!argb.empty())
                        {
                            std::vector<cv::Mat> ch(4);
                            cv::split(argb, ch); // A, R, G, B
                            cv::merge(std::vector<cv::Mat>{ ch[3], ch[2], ch[1] }, cvMat); // BGR
                            if (cvMat.empty())
                                throw std::runtime_error("ARGB->BGR conversion produced empty Mat");
                        }
                        else
                            throw std::runtime_error("Failed to create argb Mat from frame data");
                    }
                    else if (pf == PixelFormat::abgr)
                    {
                        cv::Mat abgr(h, w, CV_8UC4, (void*)frame->data(0), (size_t)frame->lineSize(0));
                        if (!abgr.empty())
                        {
                            std::vector<cv::Mat> ch(4);
                            cv::split(abgr, ch); // A, B, G, R
                            cv::merge(std::vector<cv::Mat>{ ch[1], ch[2], ch[3] }, cvMat); // BGR
                            if (cvMat.empty())
                                throw std::runtime_error("ABGR->BGR conversion produced empty Mat");
                        }
                        else
                            throw std::runtime_error("Failed to create abgr Mat from frame data");
                    }
                    else if (pf == PixelFormat::yuv420)
                    {
                        // YUV420 (I420): Y plane + U plane + V plane
                        // Use per-plane strides to avoid color artifacts on padded buffers.
                        try
                        {
                            const int ySize = w * h;
                            const int uvW = w / 2;
                            const int uvH = h / 2;
                            const int uvSize = uvW * uvH;

                            const uint8_t* yData = reinterpret_cast<const uint8_t*>(frame->data(0));
                            const uint8_t* uData = reinterpret_cast<const uint8_t*>(frame->data(1));
                            const uint8_t* vData = reinterpret_cast<const uint8_t*>(frame->data(2));

                            const int yStride = frame->lineSize(0);
                            const int uStride = frame->lineSize(1);
                            const int vStride = frame->lineSize(2);

                            if (!yData)
                                throw std::runtime_error("YUV420 conversion failed: missing Y plane");

                            std::vector<uint8_t> i420Buffer(ySize + 2 * uvSize);

                            // Copy Y plane row by row (handle padded line size).
                            for (int row = 0; row < h; ++row)
                            {
                                std::memcpy(
                                    i420Buffer.data() + row * w,
                                    yData + row * yStride,
                                    static_cast<size_t>(w));
                            }

                            // Preferred path: separate U/V planes provided by SDK.
                            if (uData && vData && uStride > 0 && vStride > 0)
                            {
                                uint8_t* uDst = i420Buffer.data() + ySize;
                                uint8_t* vDst = uDst + uvSize;
                                for (int row = 0; row < uvH; ++row)
                                {
                                    std::memcpy(
                                        uDst + row * uvW,
                                        uData + row * uStride,
                                        static_cast<size_t>(uvW));
                                    std::memcpy(
                                        vDst + row * uvW,
                                        vData + row * vStride,
                                        static_cast<size_t>(uvW));
                                }
                            }
                            else
                            {
                                // Fallback: packed I420 buffer in plane 0 (Y + U + V).
                                const uint8_t* packed = reinterpret_cast<const uint8_t*>(frame->data(0));
                                const uint8_t* srcU = packed + ySize;
                                const uint8_t* srcV = srcU + uvSize;
                                std::memcpy(i420Buffer.data() + ySize, srcU, static_cast<size_t>(uvSize));
                                std::memcpy(i420Buffer.data() + ySize + uvSize, srcV, static_cast<size_t>(uvSize));
                            }

                            // Convert I420 directly to BGR.
                            cv::Mat i420(h * 3 / 2, w, CV_8UC1, i420Buffer.data());
                            cv::cvtColor(i420, cvMat, cv::COLOR_YUV2BGR_I420);
                            
                            if (cvMat.empty())
                                throw std::runtime_error("cvtColor(YUV2BGR_I420) produced empty Mat");
                        }
                        catch (const cv::Exception& e)
                        {
                            throw std::runtime_error(std::string("YUV420 OpenCV conversion failed: ") + e.what());
                        }
                        catch (const std::exception& e)
                        {
                            throw std::runtime_error(std::string("YUV420 conversion failed: ") + e.what());
                        }
                    }
                    else
                    {
                        throw std::runtime_error("Unsupported pixelFormat=" + std::to_string(static_cast<int>(pf)) +
                                                 " (expected: yuv420,argb,abgr,rgba,bgra,rgb,bgr)");
                    }
                }
            };

        } // namespace opencv_object_detection
    } // namespace vms_server_plugins
} // namespace sample_company
