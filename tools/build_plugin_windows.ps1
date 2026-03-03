<#
.SYNOPSIS
Build YOLOv8 FLOW2 Nx plugin on Windows using CMake presets.

.DESCRIPTION
Configures and builds target `yolov8_flow2_plugin` with presets:
- windows-vs2022-x64
- windows-vs2022-x64-release

The script temporarily sets `NX_METADATA_SDK_DIR` for CMake preset resolution.

.PARAMETER NxMetadataSdkDir
Path to unpacked Nx Metadata SDK directory.
Defaults to environment variable `NX_METADATA_SDK_DIR`.

.PARAMETER ConfigurePreset
CMake configure preset name. Default: windows-vs2022-x64

.PARAMETER BuildPreset
CMake build preset name. Default: windows-vs2022-x64-release
#>
[CmdletBinding()]
param(
    [string]$NxMetadataSdkDir = $env:NX_METADATA_SDK_DIR,
    [string]$ConfigurePreset = "windows-vs2022-x64",
    [string]$BuildPreset = "windows-vs2022-x64-release"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildPath = Join-Path $repoRoot "build_flow2"

if (-not (Get-Command cmake -ErrorAction SilentlyContinue))
{
    throw "cmake is not found in PATH. Install CMake and retry."
}

if ([string]::IsNullOrWhiteSpace($NxMetadataSdkDir))
{
    throw "NX_METADATA_SDK_DIR is missing. Pass -NxMetadataSdkDir or set env:NX_METADATA_SDK_DIR."
}

$resolvedSdkDir = (Resolve-Path $NxMetadataSdkDir -ErrorAction Stop).Path

$previousSdkEnv = [Environment]::GetEnvironmentVariable("NX_METADATA_SDK_DIR", "Process")
try
{
    [Environment]::SetEnvironmentVariable("NX_METADATA_SDK_DIR", $resolvedSdkDir, "Process")

    Write-Host "Repository: $repoRoot"
    Write-Host "SDK path:   $resolvedSdkDir"
    Write-Host "Build dir:  $buildPath"
    Write-Host "Configure preset: $ConfigurePreset"
    Write-Host "Build preset:     $BuildPreset"

    & cmake --preset $ConfigurePreset
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed for preset '$ConfigurePreset'." }

    & cmake --build --preset $BuildPreset
    if ($LASTEXITCODE -ne 0) { throw "cmake build failed for preset '$BuildPreset'." }
}
finally
{
    [Environment]::SetEnvironmentVariable("NX_METADATA_SDK_DIR", $previousSdkEnv, "Process")
}

$dll = Get-ChildItem -Path $buildPath -Recurse -Filter "yolov8_flow2_plugin.dll" -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $dll)
{
    throw "Build finished but yolov8_flow2_plugin.dll was not found under $buildPath."
}

Write-Host "DLL ready: $($dll.FullName)"
Write-Host "Quick command: .\\tools\\build_plugin_windows.ps1 -NxMetadataSdkDir `"<path-to-nx-metadata-sdk>`""
