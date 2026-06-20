# Upload pre-built ONNX model to AI Box (plugin .so is built ON the box).
#
# Usage:
#   .\tools\upload_artifacts_to_box.ps1 -BoxIp 192.168.1.210
#
param(
    [string]$BoxIp = "192.168.1.210",
    [string]$BoxUser = "root",
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"

$onnx = Join-Path $RepoRoot "models\yolo26n.onnx"
if (-not (Test-Path $onnx)) {
    Write-Host "ONNX not found. Export first:" -ForegroundColor Yellow
    Write-Host "  .\.venv\Scripts\python.exe tools\export_yolo26_onnx.py"
    exit 1
}

$remoteDir = "/root/SafeAging/models"
Write-Host "Uploading $onnx → ${BoxUser}@${BoxIp}:${remoteDir}/"
ssh "${BoxUser}@${BoxIp}" "mkdir -p $remoteDir"
scp $onnx "${BoxUser}@${BoxIp}:${remoteDir}/yolo26n.onnx"

Write-Host ""
Write-Host "ONNX uploaded. Next on the box:" -ForegroundColor Green
Write-Host "  1) scp -r SafeAging repo if not yet copied"
Write-Host "  2) scp -r metavms-metadata_sdk-6.0.6.41837-universal root@${BoxIp}:/root/"
Write-Host "  3) bash tools/build_plugin_aibox.sh --install-deps --sdk-dir ... --install"
Write-Host "  4) docker compose up (MODEL_PATH=/app/models/yolo26n.onnx)"
