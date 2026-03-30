<#
.SYNOPSIS
Build Nx plugin DLL on Windows using the same flow as manual command line.

.DESCRIPTION
This script reproduces the known-good manual build flow:
1) set CONAN_VCVARS_AUTO=0
2) call <metadataSdkDir>\call_vcvars64.bat
3) call VS vcvars64.bat --vcvars_ver=14.29.30133
4) cmake -S config -B build\yolov8_people_analytics_plugin ...
5) cmake --build ...

It does not use CMake presets.

.PARAMETER NxMetadataSdkDir
Path to unpacked Nx Metadata SDK directory (the folder containing call_vcvars64.bat).
Example:
  D:\metavms-metadata_sdk-6.0.6.41837-universal\metadata_sdk

.PARAMETER VcvarsVersion
MSVC toolset version required by this project. Default: 14.29.30133

.PARAMETER VsVcvarsPath
Path to Visual Studio vcvars64.bat.

.PARAMETER CmakePath
Optional path to cmake.exe. If omitted, cmake must be available in PATH.

.PARAMETER BuildType
CMake build type. Default: Release
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$NxMetadataSdkDir = $env:NX_METADATA_SDK_DIR,

    [Parameter(Mandatory = $false)]
    [string]$VcvarsVersion = "14.29.30133",

    [Parameter(Mandatory = $false)]
    [string]$VsVcvarsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",

    [Parameter(Mandatory = $false)]
    [string]$CmakePath,

    [Parameter(Mandatory = $false)]
    [ValidateSet("Release", "Debug", "RelWithDebInfo", "MinSizeRel")]
    [string]$BuildType = "Release"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($CmakePath))
{
    $cmakeCommand = Get-Command cmake -ErrorAction SilentlyContinue
    if (-not $cmakeCommand)
    {
        throw "cmake is not found in PATH. Install CMake, add it to PATH, or pass -CmakePath."
    }

    $resolvedCmake = $cmakeCommand.Source
}
else
{
    $resolvedCmake = (Resolve-Path $CmakePath -ErrorAction Stop).Path
}

if ([string]::IsNullOrWhiteSpace($NxMetadataSdkDir))
{
    throw "NxMetadataSdkDir is missing. Pass -NxMetadataSdkDir or set env:NX_METADATA_SDK_DIR."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$srcDir = Join-Path $repoRoot "config"
$buildDir = Join-Path $repoRoot "build\yolov8_people_analytics_plugin"

$resolvedSdkDir = (Resolve-Path $NxMetadataSdkDir -ErrorAction Stop).Path
$resolvedVsVcvars = (Resolve-Path $VsVcvarsPath -ErrorAction Stop).Path

$sdkCallVcvars = Join-Path $resolvedSdkDir "call_vcvars64.bat"
if (-not (Test-Path $sdkCallVcvars))
{
    throw "Cannot find call_vcvars64.bat under '$resolvedSdkDir'. Expected: $sdkCallVcvars"
}

Write-Host "Repository:       $repoRoot"
Write-Host "Source dir:       $srcDir"
Write-Host "Build dir:        $buildDir"
Write-Host "SDK dir:          $resolvedSdkDir"
Write-Host "SDK call vcvars:  $sdkCallVcvars"
Write-Host "VS vcvars:        $resolvedVsVcvars"
Write-Host "CMake:            $resolvedCmake"
Write-Host "MSVC version:     $VcvarsVersion"
Write-Host "Build type:       $BuildType"

# Keep everything in one cmd.exe session so vcvars env is preserved.
$cmdParts = @(
    "set CONAN_VCVARS_AUTO=0",
    "call `"$sdkCallVcvars`"",
    "call `"$resolvedVsVcvars`" --vcvars_ver=$VcvarsVersion",
    "where cl",
    "`"$resolvedCmake`" --version",
    "`"$resolvedCmake`" -S `"$srcDir`" -B `"$buildDir`" -G Ninja -DCMAKE_BUILD_TYPE=$BuildType -DmetadataSdkDir=`"$resolvedSdkDir`" -DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe",
    "`"$resolvedCmake`" --build `"$buildDir`""
)
$cmdLine = $cmdParts -join " && "

& cmd.exe /d /s /c $cmdLine
if ($LASTEXITCODE -ne 0)
{
    throw "Build failed (exit code $LASTEXITCODE)."
}

$dll = Get-ChildItem -Path $buildDir -Recurse -Filter "yolov8_people_analytics_plugin.dll" -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $dll)
{
    throw "Build finished but yolov8_people_analytics_plugin.dll was not found under $buildDir."
}

Write-Host "DLL ready: $($dll.FullName)"
Write-Host "Quick command: .\tools\build_plugin_windows.ps1 -NxMetadataSdkDir `"<path-to-nx-metadata-sdk>\metadata_sdk`" [-CmakePath `"<path-to-cmake.exe>`"]"
