# Logging Level and Rate Limit Guide

## 1. Problem Summary

The plugin previously used `std::cerr` directly for many messages, including high-frequency per-frame logs and diagnostic events. This caused log flooding in long-running deployments, made operational logs hard to read, and created unnecessary overhead in critical processing paths.

## 2. Root Cause

- **Unstructured logging**: `std::cerr` used directly in hot loops (e.g., per-frame inference, per-frame parsing).
- **No throttling**: repeated messages were emitted every frame or on every failure, causing log spam.
- **Diagnostic event misuse**: `pushPluginDiagnosticEvent` was used for non-error telemetry (e.g., frame arrival), which is not what NX diagnostic events are intended for.

## 3. Logging Design & Policies

### 3.1 Log Levels

We introduced a lightweight `Logger` helper with the following levels:
- **DEBUG**: Detailed troubleshooting data. Throttled to avoid flood.
- **INFO**: High-level operational state changes / lifecycle events.
- **WARN**: Recoverable issues that could indicate degraded performance.
- **ERROR**: Actual failures requiring attention.

### 3.2 Configuring Log Level

Log level is controlled by environment variable:
- `PLUGIN_LOG_LEVEL=DEBUG|INFO|WARN|ERROR` (default: `INFO`)

Example:
```bash
set PLUGIN_LOG_LEVEL=DEBUG
```

### 3.3 Rate Limiting / Throttling

Certain messages are rate-limited by a unique key and interval.
- Frequent (per-frame) debug messages: 30s window.
- Repeated errors: 60s window with suppressed count summary.
- Throttle state is bounded and cleaned up after 5 minutes of inactivity.

## 4. Mapping Table: Old Logs → New Behavior

| Old Location | Old Behavior | New Behavior | Level | Throttle |
|-------------|--------------|--------------|-------|----------|
| `object_detector.cpp` per-frame encoding info | `std::cerr` every frame | `Logger::logThrottled(Debug)` | DEBUG | 30s | 
| `/infer` call count | `std::cerr` every 20 frames | `Logger::log(Info)` | INFO | none (limited by modulo) |
| `/infer` failure (no response) | `std::cerr` every 200 failures | `Logger::log(Error)` | ERROR | implicit (rate off 200 fails) |
| Frame queue full | diagnostic event every 20 occurrences | `Logger::log(Warn)` + one diagnostic event | WARN | first occurrence only |
| Frame arrival / pixel format | diagnostic event + `std::cerr` | `Logger::logThrottled(Debug)` | DEBUG | 30s |

## 5. Diagnostic Event Policy

### Kept
- **Fatal errors**: (e.g., worker thread exception, plugin terminated state) remain as diagnostic events.

### Reduced / Removed
- **Frame arrival telemetry**: changed to DEBUG logs.
- **Per-frame / infer-cycle logs**: removed from diagnostic events.
- **Non-actionable warnings**: only reported once for the frame queue being full.

## 6. Configuration

### Environment Variable
- `PLUGIN_LOG_LEVEL` (default: `INFO`)

Example settings:
- `PLUGIN_LOG_LEVEL=DEBUG` – enable verbose runtime logs
- `PLUGIN_LOG_LEVEL=WARN` – only warnings/errors

## 7. Verification Checklist

1. [ ] Run plugin in normal mode (default INFO) and confirm no per-frame log flood.
2. [ ] Enable `PLUGIN_LOG_LEVEL=DEBUG` and confirm debug logs appear but are throttled.
3. [ ] Simulate repeated `/infer` failures and confirm errors are logged once + summary counts.
4. [ ] Confirm only real errors generate `pushPluginDiagnosticEvent` calls.
5. [ ] Confirm no functional differences in detection/tracking behavior.

## 8. Troubleshooting Tips

- If you see no logs at all, verify `PLUGIN_LOG_LEVEL` is set (defaults to INFO).
- If a specific log is missing, ensure its key is unique (used for throttling).
- Increase log level to DEBUG for deeper investigation, but expect throttle behavior to suppress very frequent messages.
