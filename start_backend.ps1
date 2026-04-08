param(
    [int]$Port = 9000,
    [switch]$ForceRestart
)

$ErrorActionPreference = "Stop"

function Get-PortOwner {
    param([int]$ListenPort)

    $match = netstat -ano | Select-String ":$ListenPort"
    foreach ($line in $match) {
        $parts = ($line.ToString() -split "\s+") | Where-Object { $_ }
        if ($parts.Length -ge 5 -and $parts[3] -eq "LISTENING") {
            $procId = [int]$parts[-1]
            try {
                return Get-CimInstance Win32_Process -Filter "ProcessId = $procId"
            } catch {
                return $null
            }
        }
    }
    return $null
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot ".venv313\Scripts\python.exe"
$useVenv = Test-Path $venvPython

$existing = Get-PortOwner -ListenPort $Port
if ($null -ne $existing) {
    $commandLine = [string]$existing.CommandLine
    $isProjectBackend = $commandLine -like "*webapp.app:app*" -and $commandLine -like "*$projectRoot*"

    if ($isProjectBackend -and -not $ForceRestart) {
        Write-Host "Backend is already running on http://127.0.0.1:$Port" -ForegroundColor Yellow
        Write-Host "If you want to restart it, run: powershell -ExecutionPolicy Bypass -File .\start_backend.ps1 -ForceRestart" -ForegroundColor Yellow
        exit 0
    }

    if ($isProjectBackend -and $ForceRestart) {
        Stop-Process -Id $existing.ProcessId -Force
        Start-Sleep -Seconds 1
    } else {
        throw "Port $Port is already in use by process $($existing.ProcessId) ($($existing.Name)). Use a different port or stop that process first."
    }
}

if ($useVenv) {
    $env:PYTHONPATH = "$projectRoot\.venv313\Lib\site-packages;$projectRoot"
    Write-Host "Starting backend with project environment: $venvPython" -ForegroundColor Cyan
    & $venvPython -m uvicorn webapp.app:app --host 127.0.0.1 --port $Port
} else {
    $env:PYTHONPATH = $projectRoot
    Write-Host "Project environment not found. Falling back to py -3.13." -ForegroundColor Yellow
    py -3.13 -m uvicorn webapp.app:app --host 127.0.0.1 --port $Port
}
