// Copyright 2018-present Network Optix, Inc. Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#pragma once

#include <stdexcept>

#include <opencv2/core/core.hpp>

namespace sample_company {
namespace vms_server_plugins {
namespace opencv_object_detection {

class Error: public std::runtime_error { using std::runtime_error::runtime_error; };

class ObjectDetectorError: public Error { using Error::Error; };

class ObjectDetectorInitializationError: public ObjectDetectorError
    { using ObjectDetectorError::ObjectDetectorError; };

class ObjectDetectorIsTerminatedError: public ObjectDetectorError
    { using ObjectDetectorError::ObjectDetectorError; };

class ObjectDetectionError: public ObjectDetectorError
    { using ObjectDetectorError::ObjectDetectorError; };

// P2-4: ObjectTrackerError / ObjectTrackingError, and the cvExceptionToStdString()
// helper they used for formatting, removed along with the legacy ObjectTracker
// they existed solely to support (object_tracker.cpp was their only call/throw
// site; device_agent.cpp's catch(const ObjectTrackingError&) was their only
// catch site, removed in the same change) — see the P2-4 commit.

} // namespace opencv_object_detection
} // namespace vms_server_plugins
} // namespace sample_company
