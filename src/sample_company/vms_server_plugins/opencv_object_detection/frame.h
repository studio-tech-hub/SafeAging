// Copyright 2018-present Network Optix, Inc. Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>
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
                    const int pf = (int)frame->pixelFormat();
                    const int w = width;
                    const int h = height;
                    constexpr int PF_RGB24 = 2;
                    constexpr int PF_BGR24 = 3;
                    constexpr int PF_BGRA32 = 4;
                    constexpr int PF_RGBA32 = 5;
                    constexpr int PF_YV12 = 6;    // YUV 4:2:0 planar (common for IP cameras)

                    if (pf == PF_BGR24)
                    {
                        cv::Mat temp(h, w, CV_8UC3, (void*)frame->data(0), (size_t)frame->lineSize(0));
                        cvMat = temp.clone();  // Clone to ensure data is owned by this Mat
                    }
                    else if (pf == PF_BGRA32)
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
                    else if (pf == PF_RGBA32)
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
                    else if (pf == PF_RGB24)
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
                    else if (pf == PF_YV12)
                    {
                        // YV12: YUV 4:2:0 planar format
                        // Layout: Y plane (w*h) + V plane (w*h/4) + U plane (w*h/4)
                        // Following official NX Meta plugin pattern for proper color conversion
                        try
                        {
                            // Create a single contiguous buffer with all three planes
                            // YV12 format: Y (full), V (quarter), U (quarter)
                            const uint8_t* data = (const uint8_t*)frame->data(0);
                            int ySize = w * h;
                            int uvSize = ySize / 4;
                            
                            // Create I420 format (Y + U + V) from YV12 (Y + V + U)
                            // by copying planes in correct order for OpenCV
                            std::vector<uint8_t> i420Buffer;
                            i420Buffer.reserve(ySize + 2 * uvSize);
                            
                            // Copy Y plane
                            i420Buffer.insert(i420Buffer.end(), data, data + ySize);
                            
                            // Copy U plane (second quarter in YV12 is V, so skip and get U)
                            i420Buffer.insert(i420Buffer.end(), data + ySize + uvSize, data + ySize + 2 * uvSize);
                            
                            // Copy V plane (first quarter after Y in YV12 is V)
                            i420Buffer.insert(i420Buffer.end(), data + ySize, data + ySize + uvSize);
                            
                            // Create Mat with I420 layout and convert to BGR
                            cv::Mat i420(h * 3 / 2, w, CV_8UC1, i420Buffer.data());
                            cv::cvtColor(i420, cvMat, cv::COLOR_YUV2BGR_I420);
                            
                            if (cvMat.empty())
                                throw std::runtime_error("cvtColor(YUV2BGR_I420) produced empty Mat");
                        }
                        catch (const cv::Exception& e)
                        {
                            throw std::runtime_error(std::string("YV12 OpenCV conversion failed: ") + e.what());
                        }
                        catch (const std::exception& e)
                        {
                            throw std::runtime_error(std::string("YV12 conversion failed: ") + e.what());
                        }
                    }
                    else
                    {
                        throw std::runtime_error("Unsupported pixelFormat=" + std::to_string(pf) + 
                                                 " (expected: 2=RGB24, 3=BGR24, 4=BGRA32, 5=RGBA32, 6=YV12)");
                    }
                }
            };

        } // namespace opencv_object_detection
    } // namespace vms_server_plugins
} // namespace sample_company
