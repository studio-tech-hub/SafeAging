<#
.SYNOPSIS
Validate FLOW2 runtime environment for plugin + AI service.

.DESCRIPTION
Checks:
- Plugin files in Nx plugin directory
- Nx Media Server service visibility
- Required environment variables
- AI service /health endpoint
- Optional /infer test with a sample image

.PARAMETER AiServiceUrl
AI service base URL. Defaults to NX_AI_SERVICE_URL or http://127.0.0.1:18000.

.PARAMETER NxMediaServerRoot
Nx Meta Media Server installation root.

.PARAMETER PluginName
Plugin folder name.

.PARAMETER SampleImagePath
Optional image path for POST /infer test.

.PARAMETER SkipInfer
Skip /infer test even when SampleImagePath is provided.
#>
[CmdletBinding()]
param(
    [string]$AiServiceUrl = "",
    [string]$NxMediaServerRoot = "C:\\Program Files\\Network Optix\\Nx Meta\\Media Server",
    [string]$PluginName = "yolov8_flow2_plugin",
    [string]$SampleImagePath = "",
    [switch]$SkipInfer
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-EnvValue([string]$name)
{
    foreach ($scope in @("Process", "Machine", "User"))
    {
        $value = [Environment]::GetEnvironmentVariable($name, $scope)
        if (-not [string]::IsNullOrWhiteSpace($value))
        {
            return $value
        }
    }

    return $null
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

    return (Get-Service | Where-Object {
        $_.DisplayName -match "Media Server" -and
        ($_.DisplayName -match "Nx|Network Optix" -or $_.Name -match "nx|optix|media")
    } | Select-Object -First 1)
}

function Get-DetectionCountFromInferResponse([object]$response)
{
    function ConvertTo-Count([object]$value)
    {
        if ($null -eq $value)
        {
            return -1
        }

        if ($value -is [string])
        {
            return -1
        }

        if ($value -is [System.Array])
        {
            return $value.Count
        }

        if ($value -is [System.Collections.IEnumerable])
        {
            return @($value).Count
        }

        return -1
    }

    if ($response -is [System.Array])
    {
        return $response.Count
    }

    if ($response -is [System.Collections.IDictionary] -and $response.ContainsKey("detections"))
    {
        return ConvertTo-Count($response["detections"])
    }

    if ($response.PSObject -and $response.PSObject.Properties.Match("detections").Count -gt 0)
    {
        return ConvertTo-Count($response.detections)
    }

    return -1
}

$expectedEnv = [ordered]@{
    "NX_AI_SERVICE_URL" = "http://127.0.0.1:18000"
    "NX_AI_TIMEOUT_CONNECT_MS" = "2000"
    "NX_AI_TIMEOUT_READ_MS" = "2000"
    "NX_AI_TIMEOUT_WRITE_MS" = "2000"
    "NX_AI_SAMPLE_FPS" = "5.0"
    "NX_AI_QUEUE_SIZE" = "4"
    "NX_AI_SEND_WIDTH" = "640"
    "NX_AI_JPEG_QUALITY" = "80"
    "NX_AI_RETRY_COUNT" = "1"
    "NX_AI_CIRCUIT_FAILS" = "3"
    "NX_AI_CIRCUIT_OPEN_MS" = "3000"
    "NX_AI_FALL_FINISH_MS" = "1500"
    "NX_AI_SYNTH_TRACK_TTL_MS" = "2000"
    "NX_AI_TRACK_MAP_TTL_MS" = "60000"
    "NX_AI_LOG_THROTTLE_MS" = "5000"
    "NX_AI_DEBUG_STATS_MS" = "5000"
}

if ([string]::IsNullOrWhiteSpace($AiServiceUrl))
{
    $AiServiceUrl = Get-EnvValue "NX_AI_SERVICE_URL"
    if ([string]::IsNullOrWhiteSpace($AiServiceUrl))
    {
        $AiServiceUrl = "http://127.0.0.1:18000"
    }
}

$AiServiceUrl = $AiServiceUrl.TrimEnd("/")
if ($AiServiceUrl.EndsWith("/infer"))
{
    $AiServiceUrl = $AiServiceUrl.Substring(0, $AiServiceUrl.Length - "/infer".Length)
}

Write-Host "=== FLOW2 Environment Check ==="
Write-Host "Nx Media Server root: $NxMediaServerRoot"
Write-Host "AI service URL:       $AiServiceUrl"
Write-Host "NX_AI_SAMPLE_FPS:     $(Get-EnvValue "NX_AI_SAMPLE_FPS")"
Write-Host "NX_AI_QUEUE_SIZE:     $(Get-EnvValue "NX_AI_QUEUE_SIZE")"
Write-Host ""

Write-Host "[1/4] Checking plugin files..."
$pluginDirCandidates = @(
    (Join-Path $NxMediaServerRoot ("plugins\\" + $PluginName)),
    (Join-Path $NxMediaServerRoot ("bin\\plugins\\" + $PluginName))
)

$pluginFound = $false
foreach ($candidate in $pluginDirCandidates)
{
    if (-not (Test-Path $candidate))
    {
        continue
    }

    $pluginFound = $true
    Write-Host "Found plugin dir: $candidate"

    $dllPath = Join-Path $candidate "yolov8_flow2_plugin.dll"
    $manifestPath = Join-Path $candidate "manifest.json"

    if (Test-Path $dllPath) { Write-Host "Found DLL:      $dllPath" } else { Write-Warning "Missing DLL: $dllPath" }
    if (Test-Path $manifestPath) { Write-Host "Found manifest: $manifestPath" } else { Write-Warning "Missing manifest: $manifestPath" }
}

if (-not $pluginFound)
{
    Write-Warning "Plugin directory not found under Nx root."
}
Write-Host ""

Write-Host "[2/4] Checking Nx Media Server service..."
$service = Find-NxMediaService
if ($service)
{
    Write-Host "Service: $($service.Name) ($($service.DisplayName)) - Status: $($service.Status)"
}
else
{
    Write-Warning "Nx Media Server service was not auto-detected."
}
Write-Host ""

Write-Host "[3/4] Checking environment variables..."
foreach ($item in $expectedEnv.GetEnumerator())
{
    $name = $item.Key
    $defaultValue = $item.Value
    $value = Get-EnvValue $name

    if ([string]::IsNullOrWhiteSpace($value))
    {
        Write-Warning "$name is not set (default: $defaultValue)"
        continue
    }

    if ($value -eq $defaultValue)
    {
        Write-Host "$name = $value"
    }
    else
    {
        Write-Host "$name = $value (default: $defaultValue)"
    }
}
Write-Host ""

Write-Host "[4/4] Checking AI service..."
try
{
    $health = Invoke-RestMethod -Uri "$AiServiceUrl/health" -Method Get -TimeoutSec 8
    Write-Host "Health OK: $($health.status)"
}
catch
{
    Write-Error "Health check failed: $($_.Exception.Message)"
    exit 1
}

if ($SkipInfer)
{
    Write-Host "Infer test skipped by -SkipInfer."
    exit 0
}

if ([string]::IsNullOrWhiteSpace($SampleImagePath))
{
    Write-Warning "Infer test skipped: -SampleImagePath not provided."
    exit 0
}

if (-not (Test-Path $SampleImagePath))
{
    Write-Warning "Infer test skipped: image not found at $SampleImagePath"
    exit 0
}

$bytes = [System.IO.File]::ReadAllBytes((Resolve-Path $SampleImagePath))
$b64 = [System.Convert]::ToBase64String($bytes)
$payload = @{
    camera_id = "nx_env_check"
    image = $b64
} | ConvertTo-Json -Compress

try
{
    $response = Invoke-RestMethod -Uri "$AiServiceUrl/infer" -Method Post -TimeoutSec 20 -ContentType "application/json" -Body $payload
    $count = Get-DetectionCountFromInferResponse $response
    if ($count -lt 0)
    {
        Write-Error "Infer response schema is invalid. Expected array or object with detections[]."
        exit 1
    }

    Write-Host "Infer OK: received $count detection(s)."
}
catch
{
    Write-Error "Infer test failed: $($_.Exception.Message)"
    exit 1
}

Write-Host ""
Write-Host "Environment check completed."
