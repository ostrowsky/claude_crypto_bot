param(
    [string]$WorkerTaskName = "GPT Crypto RL Headless",
    [string]$ReportTaskName = "GPT Crypto RL Daily Report",
    [string]$DailyReportTime = "00:05"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$workerScript = Join-Path $root "headless_loop.ps1"
$reportScript = Join-Path $root "rl_daily_report.ps1"

if (-not (Test-Path $workerScript)) {
    throw "Worker script not found: $workerScript"
}
if (-not (Test-Path $reportScript)) {
    throw "Report script not found: $reportScript"
}

$workerCmd = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' + $workerScript + '"'
$reportCmd = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' + $reportScript + '" --previous-day'

$workerArgs = @(
    "/Create", "/F",
    "/SC", "ONLOGON",
    "/TN", $WorkerTaskName,
    "/TR", $workerCmd,
    "/RL", "LIMITED"
)
$reportArgs = @(
    "/Create", "/F",
    "/SC", "DAILY",
    "/ST", $DailyReportTime,
    "/TN", $ReportTaskName,
    "/TR", $reportCmd,
    "/RL", "LIMITED"
)

$workerOutput = & schtasks.exe @workerArgs 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to register worker task. schtasks exit code=$LASTEXITCODE`n$workerOutput"
}

$reportOutput = & schtasks.exe @reportArgs 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to register report task. schtasks exit code=$LASTEXITCODE`n$reportOutput"
}

[pscustomobject]@{
    WorkerTask = $WorkerTaskName
    ReportTask = $ReportTaskName
    DailyReportTime = $DailyReportTime
    WorkerScript = $workerScript
    ReportScript = $reportScript
} | Format-List
