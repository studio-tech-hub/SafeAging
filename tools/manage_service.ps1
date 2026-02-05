#!/usr/bin/env pwsh
# Script to test and manage YOLOv8 People Analytics Service

param(
    [string]$Action = "status",
    [string]$CameraId = "default"
)

$SERVICE_URL = "http://127.0.0.1:18000"

function Test-Health {
    Write-Host "🏥 Checking service health..." -ForegroundColor Cyan
    try {
        $resp = Invoke-RestMethod "$SERVICE_URL/health" -TimeoutSec 5
        Write-Host "✅ Service is RUNNING" -ForegroundColor Green
        Write-Host "   Status: $($resp.status)"
        Write-Host "   Uptime: $($resp.service_uptime_seconds)s"
        return $true
    } catch {
        Write-Host "❌ Service is DOWN" -ForegroundColor Red
        Write-Host "   Error: $_"
        return $false
    }
}

function Get-Status {
    Write-Host "📊 Service Statistics..." -ForegroundColor Cyan
    try {
        $resp = Invoke-RestMethod "$SERVICE_URL/status" -TimeoutSec 5
        Write-Host "✅ Status retrieved" -ForegroundColor Green
        Write-Host ""
        Write-Host "Service Information:" -ForegroundColor Yellow
        Write-Host "  Uptime: $($resp.uptime_seconds)s"
        Write-Host "  Total Requests: $($resp.total_requests)"
        Write-Host "  Total Errors: $($resp.total_errors)"
        Write-Host "  Error Rate: $($resp.error_rate)%"
        Write-Host ""
        Write-Host "Active Cameras: $($resp.active_cameras)" -ForegroundColor Yellow
        foreach ($cam_id in $resp.cameras.PSObject.Properties.Name) {
            $cam = $resp.cameras.$cam_id
            Write-Host "  Camera '$cam_id':" -ForegroundColor Cyan
            Write-Host "    Active Tracks: $($cam.tracks)"
            Write-Host "    Unique Persons: $($cam.unique_persons)" -ForegroundColor Green
            Write-Host "    Avg Inference: $($cam.avg_inference_ms)ms"
        }
    } catch {
        Write-Host "❌ Failed to get status" -ForegroundColor Red
        Write-Host "   Error: $_"
    }
}

function Reset-Camera {
    param([string]$CamId)
    Write-Host "🔄 Resetting camera '$CamId'..." -ForegroundColor Cyan
    try {
        $resp = Invoke-RestMethod "$SERVICE_URL/reset/$CamId" -Method Post -TimeoutSec 5
        Write-Host "✅ Reset successful!" -ForegroundColor Green
        Write-Host "   Previous count: $($resp.previous_count)"
        Write-Host "   Current count: $($resp.current_count)"
        Write-Host "   Status: $($resp.status)"
    } catch {
        Write-Host "❌ Reset failed" -ForegroundColor Red
        Write-Host "   Error: $_"
    }
}

function Reset-All {
    Write-Host "🔄 Resetting ALL cameras..." -ForegroundColor Yellow
    try {
        $resp = Invoke-RestMethod "$SERVICE_URL/reset_all" -Method Post -TimeoutSec 5
        Write-Host "✅ Reset successful!" -ForegroundColor Green
        Write-Host "   Cameras reset: $($resp.cameras_reset)"
        Write-Host "   Total persons cleared: $($resp.total_persons_cleared)"
    } catch {
        Write-Host "❌ Reset failed" -ForegroundColor Red
        Write-Host "   Error: $_"
    }
}

function Show-Help {
    Write-Host "YOLOv8 People Analytics Service Manager" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: ./manage_service.ps1 -Action <action> [-CameraId <id>]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Actions:" -ForegroundColor Green
    Write-Host "  health       Check if service is running"
    Write-Host "  status       Get service statistics and person count"
    Write-Host "  reset        Reset count for specific camera (default: 'default')"
    Write-Host "               Example: ./manage_service.ps1 -Action reset -CameraId default"
    Write-Host "  reset_all    Reset count for ALL cameras"
    Write-Host "  help         Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  ./manage_service.ps1 -Action health"
    Write-Host "  ./manage_service.ps1 -Action status"
    Write-Host "  ./manage_service.ps1 -Action reset"
    Write-Host "  ./manage_service.ps1 -Action reset -CameraId 'camera_1'"
    Write-Host "  ./manage_service.ps1 -Action reset_all"
}

# Main
Write-Host ""
switch ($Action.ToLower()) {
    "health" {
        Test-Health
    }
    "status" {
        if (Test-Health) {
            Write-Host ""
            Get-Status
        }
    }
    "reset" {
        Reset-Camera $CameraId
    }
    "reset_all" {
        Reset-All
    }
    "help" {
        Show-Help
    }
    default {
        Write-Host "Unknown action: $Action" -ForegroundColor Red
        Write-Host ""
        Show-Help
    }
}
Write-Host ""
