<#
.SYNOPSIS
Deploy YOLOv8 FLOW2 plugin DLL and manifest into Nx Meta Media Server plugin directory.

.DESCRIPTION
Copies:
- yolov8_flow2_plugin.dll
- src/manifest.json (as manifest.json)

Then restarts Nx Media Server service by default.

.PARAMETER DllPath
Path to built DLL. If omitted, the newest DLL under build_flow2 is used.

.PARAMETER NxMediaServerRoot
Nx Meta Media Server installation root.

.PARAMETER NxPluginDir
Exact plugin directory override. If omitted, auto-detects from NxMediaServerRoot.

.PARAMETER PluginName
Plugin folder name when auto-detecting directory.

.PARAMETER SkipServiceRestart
Skip Media Server service restart.
#>
[CmdletBinding()]
param(
    [string]$DllPath = "",
    [string]$NxMediaServerRoot = "C:\\Program Files\\Network Optix\\Nx Meta\\Media Server",
    [string]$NxPluginDir = "",
    [string]$PluginName = "yolov8_flow2_plugin",
    [switch]$SkipServiceRestart
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-PluginDir([string]$serverRoot, [string]$pluginDirOverride, [string]$pluginName)
{
    if (-not [string]::IsNullOrWhiteSpace($pluginDirOverride))
    {
        if ([System.IO.Path]::IsPathRooted($pluginDirOverride))
        {
            return $pluginDirOverride
        }

        return (Join-Path $serverRoot $pluginDirOverride)
    }

    $baseCandidates = @(
        (Join-Path $serverRoot "plugins"),
        (Join-Path $serverRoot "bin\\plugins")
    )

    foreach ($base in $baseCandidates)
    {
        if (Test-Path $base)
        {
            return (Join-Path $base $pluginName)
        }
    }

    return (Join-Path $baseCandidates[0] $pluginName)
}

function Find-NxMediaService()
{
    $preferredNames = @(
        "networkoptix-mediaserver",
        "NetworkOptixMediaServer",
        "NxMetaMediaServer",
        "nx_mediaserver"
    )

    foreach ($name in $preferredNames)
    {
        $svc = Get-Service -Name $name -ErrorAction SilentlyContinue
        if ($svc) { return $svc }
    }

    $matches = Get-Service | Where-Object {
        $_.DisplayName -match "Media Server" -and
        ($_.DisplayName -match "Nx|Network Optix" -or $_.Name -match "nx|optix|media")
    }

    if ($matches.Count -gt 0)
    {
        $running = $matches | Where-Object Status -eq "Running" | Select-Object -First 1
        if ($running) { return $running }
        return ($matches | Select-Object -First 1)
    }

    return $null
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$manifestSource = Join-Path $repoRoot "src\\manifest.json"

if (-not (Test-Path $manifestSource))
{
    throw "Manifest file is missing: $manifestSource"
}

if ([string]::IsNullOrWhiteSpace($DllPath))
{
    $candidate = Get-ChildItem -Path (Join-Path $repoRoot "build_flow2") -Recurse -Filter "yolov8_flow2_plugin.dll" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $candidate)
    {
        throw "DLL was not provided and no yolov8_flow2_plugin.dll was found under build_flow2."
    }

    $DllPath = $candidate.FullName
}

$resolvedDllPath = (Resolve-Path $DllPath -ErrorAction Stop).Path
$pluginTargetDir = Resolve-PluginDir $NxMediaServerRoot $NxPluginDir $PluginName

Write-Host "Nx root:      $NxMediaServerRoot"
Write-Host "Plugin dir:   $pluginTargetDir"
Write-Host "DLL source:   $resolvedDllPath"
Write-Host "Manifest src: $manifestSource"

New-Item -ItemType Directory -Path $pluginTargetDir -Force | Out-Null

$dllTarget = Join-Path $pluginTargetDir "yolov8_flow2_plugin.dll"
$manifestTarget = Join-Path $pluginTargetDir "manifest.json"

Copy-Item -Path $resolvedDllPath -Destination $dllTarget -Force
Copy-Item -Path $manifestSource -Destination $manifestTarget -Force

Write-Host "Copied DLL:      $dllTarget"
Write-Host "Copied manifest: $manifestTarget"

if ($SkipServiceRestart)
{
    Write-Host "Skip service restart requested."
    exit 0
}

$service = Find-NxMediaService
if (-not $service)
{
    throw "Nx Media Server service was not auto-detected. Use -SkipServiceRestart or restart service manually."
}

Write-Host "Restarting service: $($service.Name) ($($service.DisplayName))"
Restart-Service -Name $service.Name -Force
Write-Host "Service restarted."
