param(
    [int]$Port = 9000
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$match = netstat -ano | Select-String ":$Port"

foreach ($line in $match) {
    $parts = ($line.ToString() -split "\s+") | Where-Object { $_ }
    if ($parts.Length -ge 5 -and $parts[3] -eq "LISTENING") {
        $procId = [int]$parts[-1]
        $process = Get-CimInstance Win32_Process -Filter "ProcessId = $procId"
        $commandLine = [string]$process.CommandLine
        if ($commandLine -like "*webapp.app:app*" -and $commandLine -like "*$projectRoot*") {
            Stop-Process -Id $procId -Force
            Write-Host "Stopped backend on port $Port (PID $procId)." -ForegroundColor Green
            exit 0
        }
    }
}

Write-Host "No project backend was found on port $Port." -ForegroundColor Yellow
