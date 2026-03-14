# Pixel Format Capability Alignment

## Problem Statement

The plugin's static manifest declared a BGR uncompressed video frame capability (`needUncompressedVideoFrames_bgr`), but the runtime engine manifest returned a YUV420 capability (`needUncompressedVideoFrames_yuv420`). This mismatch caused instability, warnings, and inconsistent decode paths as the NX Server negotiated frame formats based on the static manifest but the plugin expected YUV420 frames.

## Observed Symptoms

- "Unsupported pixel format" warnings in plugin logs
- Inconsistent frame decode behavior across plugin restarts
- Potential instability in video processing pipeline

## Root Cause

Capability declaration mismatch between:
- Static manifest: `config/manifest.json` declared `needUncompressedVideoFrames_bgr`
- Runtime manifest: `Engine::manifestString()` returned `needUncompressedVideoFrames_yuv420`

The NX Server uses the static manifest for capability negotiation, but the plugin's runtime behavior expected YUV420 frames, leading to format mismatches.

## Business Logic Impact

1. **Capability Negotiation**: NX Server chooses uncompressed frame format based on plugin capabilities
2. **Decode Pipeline**: Plugin converts incoming frames to OpenCV BGR internally for processing
3. **Consistency Requirement**: Static and runtime manifests must declare identical capabilities to ensure stable negotiation

## Decision Rationale for yuv420

- **Efficiency**: YUV420 matches internal NX server format, reducing unnecessary conversions
- **Implementation Alignment**: Engine code already expects and handles YUV420 → BGR conversion
- **Existing Code**: Comments in `engine.cpp` explicitly mention YUV420 as the intended format
- **Minimal Risk**: No changes to decode logic required, only capability alignment

## Exact Files Changed

### 1. config/manifest.json
- **Line**: 23 (capabilities array)
- **Change**: `"needUncompressedVideoFrames_bgr"` → `"needUncompressedVideoFrames_yuv420"`

### 2. docs/PIXEL_FORMAT_CAPABILITY_ALIGNMENT.md (new file)
- **Purpose**: Document the issue and fix for future maintainers

## Use Cases Covered

1. **Normal Startup**: Server negotiates one consistent format (yuv420) across all cameras
2. **Streaming Across Cameras**: All cameras feed yuv420 frames deterministically
3. **Long-Running Runtime**: No repeated mismatch warnings or decode instability

## Edge Cases Handled

1. **Existing Comments**: Updated capability string removes confusion about "bgr" format
2. **Future Mistakes**: Documentation explicitly warns against capability drift
3. **Dual Declarations**: Single canonical capability prevents conflicting declarations
4. **Unexpected Formats**: Existing fallback conversion paths remain intact

## Verification Checklist

1. **Manifest Consistency**
   - [ ] `config/manifest.json` contains `"needUncompressedVideoFrames_yuv420"`
   - [ ] `Engine::manifestString()` returns `"needUncompressedVideoFrames_yuv420"`

2. **No Mismatch Warnings**
   - [ ] Plugin loads without "unsupported pixel format" warnings
   - [ ] No capability drift errors in server logs

3. **Decode Stability**
   - [ ] YUV420 frames received from RTSP sources
   - [ ] OpenCV BGR conversion produces correct output
   - [ ] Fall detection pipeline works without regression

4. **Behavioral Integrity**
   - [ ] Event IDs, object types, REST calls unchanged
   - [ ] Detection logic unaffected
   - [ ] Frame conversion safety preserved

## Operational Guidance

### How to Avoid This Mismatch in Future Changes

1. **Single Source of Truth**: Always update both manifest locations together
2. **Code Review Checklist**: Verify static and runtime manifests match before commit
3. **Testing**: Load plugin and check for format warnings in logs
4. **Documentation**: Update this doc when changing pixel format capabilities

### Tuning Considerations

- If performance issues arise with yuv420, consider bgr as alternative (requires updating both manifests)
- Test thoroughly with actual camera feeds before changing formats
- Monitor plugin logs for conversion performance

## Summary

This fix unifies capability declarations to yuv420, eliminating mismatch warnings and ensuring stable decode pipeline. The change is purely declarative with no behavioral impact on detection or event logic.