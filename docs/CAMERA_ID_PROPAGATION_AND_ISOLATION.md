# Camera ID Propagation and Isolation

## Problem Statement

The plugin was using a hardcoded camera ID ("nx_camera") for all device agents, causing cross-camera data pollution in the Python service. This led to:

- Track UUID collisions between different cameras
- Incorrect analytics aggregation across devices
- Loss of per-device state isolation

## Solution Overview

Replace hardcoded "nx_camera" with real device identifiers sourced from NX SDK's IDeviceInfo, ensuring each camera has unique state in the Python service.

## Implementation Details

### Camera ID Sourcing Strategy

1. **Primary Source**: `deviceInfo->id()` from NX SDK IDeviceInfo interface
2. **Normalization**: Convert to lowercase alphanumeric + underscores only
3. **Fallback**: If ID is empty or null, use deterministic fallback based on deviceInfo pointer

### Code Changes

#### device_agent.h
- Added `std::string m_cameraId;` member variable
- Added `std::string getCameraIdFromDeviceInfo(const nx::sdk::IDeviceInfo* deviceInfo);` private method

#### device_agent.cpp
- Constructor initializes `m_cameraId` using `getCameraIdFromDeviceInfo(deviceInfo)`
- FrameJob assignment changed from `job.cameraId = "nx_camera"` to `job.cameraId = m_cameraId`
- Added `getCameraIdFromDeviceInfo()` implementation with normalization and fallbacks

### Data Flow

```
NX Server DeviceInfo.id() → DeviceAgent::getCameraIdFromDeviceInfo() → m_cameraId → FrameJob.cameraId → Python Service
```

## Validation Checklist

- [ ] Build succeeds without errors
- [ ] Plugin loads in NX VMS
- [ ] Camera ID appears in Python service logs (not "nx_camera")
- [ ] Multiple cameras show distinct IDs in service requests
- [ ] Track UUIDs are isolated per camera (no cross-camera collisions)
- [ ] Fallback logic works for devices without proper IDs

## Testing

1. Deploy plugin to NX VMS with multiple cameras
2. Monitor Python service logs for cameraId values
3. Verify track continuity within each camera
4. Confirm no track mixing between cameras

## Rollback Plan

If issues arise, temporarily revert FrameJob assignment back to hardcoded "nx_camera" while investigating root cause.</content>
<parameter name="filePath">d:\Part-time\SafeAgingV2\SafeAging\docs\CAMERA_ID_PROPAGATION_AND_ISOLATION.md