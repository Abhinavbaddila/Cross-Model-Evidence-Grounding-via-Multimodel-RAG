$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontendRoot = Join-Path $projectRoot "webapp\frontend"
Set-Location $frontendRoot

& npm.cmd run dev -- --host 127.0.0.1 --port 5173
