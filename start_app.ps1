$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendScript = Join-Path $projectRoot "start_backend.ps1"
$frontendScript = Join-Path $projectRoot "start_frontend.ps1"

if (-not (Test-Path $backendScript)) {
    throw "Backend start script not found at $backendScript"
}

if (-not (Test-Path $frontendScript)) {
    throw "Frontend start script not found at $frontendScript"
}

Start-Process powershell.exe -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", $backendScript
Start-Sleep -Seconds 3
Start-Process powershell.exe -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", $frontendScript

Write-Host "Started backend and frontend in separate PowerShell windows." -ForegroundColor Green
Write-Host "Frontend URL: http://127.0.0.1:5173" -ForegroundColor Green
