param(
    [string]$WorkerTaskName = "GPT Crypto RL Headless",
    [string]$ReportTaskName = "GPT Crypto RL Daily Report"
)

$ErrorActionPreference = "Stop"

schtasks.exe /Delete /F /TN $WorkerTaskName | Out-Null
schtasks.exe /Delete /F /TN $ReportTaskName | Out-Null

[pscustomobject]@{
    WorkerTaskRemoved = $WorkerTaskName
    ReportTaskRemoved = $ReportTaskName
} | Format-List

