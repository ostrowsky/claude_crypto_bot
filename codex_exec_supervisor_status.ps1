$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$statusFile = Join-Path $root ".runtime\codex_exec_supervisor_status.json"
$jobRoot = Join-Path $root ".runtime\codex_jobs"
$queueDir = Join-Path $jobRoot "queue"
$statusDir = Join-Path $jobRoot "status"

$status = $null
if (Test-Path $statusFile) {
    try {
        $status = Get-Content $statusFile -Raw | ConvertFrom-Json
    } catch {
        $status = $null
    }
}

$fresh = $false
$procAlive = $false
if ($status -and $status.updated_at) {
    try {
        $updated = [datetime]::Parse($status.updated_at).ToUniversalTime()
        $fresh = (((Get-Date).ToUniversalTime() - $updated).TotalSeconds -lt 30)
    } catch {
        $fresh = $false
    }
}

if ($status -and $status.pid) {
    try {
        $procAlive = [bool](Get-Process -Id ([int]$status.pid) -ErrorAction Stop)
    } catch {
        $procAlive = $false
    }
}

$queueCount = 0
if (Test-Path $queueDir) {
    $queueCount = @(Get-ChildItem $queueDir -Filter *.json -ErrorAction SilentlyContinue).Count
}

$latestJob = $null
if (Test-Path $statusDir) {
    $latestFile = Get-ChildItem $statusDir -Filter *.json -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($latestFile) {
        try {
            $latestJob = Get-Content $latestFile.FullName -Raw | ConvertFrom-Json
        } catch {
            $latestJob = $null
        }
    }
}

[pscustomobject]@{
    Running = [bool]($status -and $fresh -and $procAlive)
    UpdatedAt = if ($status) { $status.updated_at } else { $null }
    State = if ($status) { $status.state } else { $null }
    Pid = if ($status) { $status.pid } else { $null }
    QueueCount = $queueCount
    LatestJobId = if ($latestJob) { $latestJob.job_id } else { $null }
    LatestJobType = if ($latestJob) { $latestJob.job_type } else { $null }
    LatestJobState = if ($latestJob) { $latestJob.state } else { $null }
    StatusFile = $statusFile
} | Format-List

if ($status) {
    Write-Host ""
    Write-Host "supervisor status:"
    $status | ConvertTo-Json -Depth 8
}

if ($latestJob) {
    Write-Host ""
    Write-Host "latest job:"
    $latestJob | ConvertTo-Json -Depth 10
}
