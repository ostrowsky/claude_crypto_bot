$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$startupDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup"
$cmdPath = Join-Path $startupDir "gpt_crypto_rl_headless.cmd"
$workerScript = Join-Path $root "start_rl_worker_bg.ps1"

if (-not (Test-Path $startupDir)) {
    New-Item -ItemType Directory -Force -Path $startupDir | Out-Null
}
if (-not (Test-Path $workerScript)) {
    throw "Worker launcher not found: $workerScript"
}

$cmd = @"
@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$workerScript"
"@

Set-Content -Path $cmdPath -Value $cmd -Encoding ASCII

[pscustomobject]@{
    Installed = $true
    StartupFile = $cmdPath
    WorkerScript = $workerScript
} | Format-List

