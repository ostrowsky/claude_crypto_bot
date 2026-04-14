$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$runtimeDir = Join-Path $root ".runtime"
$filesRuntimeDir = Join-Path $root "files\.runtime"
$pidFile = Join-Path $runtimeDir "market_agent_bg.json"
$statusFile = Join-Path $runtimeDir "market_agent_status.json"
if (-not (Test-Path $statusFile)) {
    $statusFile = Join-Path $filesRuntimeDir "market_agent_status.json"
}

$state = $null
if (Test-Path $pidFile) {
    try {
        $state = Get-Content $pidFile -Raw | ConvertFrom-Json
    } catch {
        $state = $null
    }
}

$pythonPid = if ($state) { [int]$state.python_pid } else { 0 }
$wrapperPid = if ($state) { [int]$state.wrapper_pid } else { 0 }
$proc = $null
if ($pythonPid -gt 0) {
    $proc = Get-Process -Id $pythonPid -ErrorAction SilentlyContinue
}
if (-not $proc -and $wrapperPid -gt 0) {
    $proc = Get-Process -Id $wrapperPid -ErrorAction SilentlyContinue
}

$runtime = $null
if (Test-Path $statusFile) {
    try {
        $runtime = Get-Content $statusFile -Raw | ConvertFrom-Json
    } catch {
        $runtime = $null
    }
}

$heartbeatFresh = $false
if ($runtime -and $runtime.worker -and $runtime.worker.last_heartbeat) {
    try {
        $lastHeartbeat = [datetime]::Parse($runtime.worker.last_heartbeat).ToUniversalTime()
        $pollSec = 60
        if ($runtime.worker.poll_sec) {
            $pollSec = [math]::Max(5, [int]$runtime.worker.poll_sec)
        }
        $graceSec = [math]::Max(180, $pollSec * 3)
        $heartbeatFresh = ((Get-Date).ToUniversalTime() - $lastHeartbeat).TotalSeconds -lt $graceSec
    } catch {
        $heartbeatFresh = $false
    }
}

$running = [bool]$proc -or $heartbeatFresh

[pscustomobject]@{
    Running   = $running
    PythonPid = if ($pythonPid -gt 0) { $pythonPid } else { 0 }
    WrapperPid = if ($wrapperPid -gt 0) { $wrapperPid } else { 0 }
    StartedAt = if ($state) { $state.started_at } else { $null }
    Stdout    = if ($state) { $state.stdout } else { $null }
    Stderr    = if ($state) { $state.stderr } else { $null }
    StatusFile = $statusFile
} | Format-List
