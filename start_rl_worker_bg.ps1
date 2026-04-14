param(
    [switch]$ForceRestart = $true
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "pyembed\python.exe"
$workdir = Join-Path $root "files"
$script = Join-Path $workdir "rl_headless_worker.py"
$loopScript = Join-Path $root "headless_loop.ps1"
$runtimeDir = Join-Path $root ".runtime"
$pidFile = Join-Path $runtimeDir "rl_worker_bg.json"
$heartbeatFile = Join-Path $runtimeDir "rl_worker_wrapper_heartbeat.json"
$stdout = Join-Path $runtimeDir "rl_worker_wrapper_stdout.log"
$stderr = Join-Path $runtimeDir "rl_worker_wrapper_stderr.log"

if (-not (Test-Path $runtimeDir)) {
    New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
}

function Get-RLWorkerProcesses {
    if (-not (Test-Path $pidFile)) {
        return @()
    }
    try {
        $state = Get-Content $pidFile -Raw | ConvertFrom-Json
        $out = @()
        foreach ($pid in @($state.wrapper_pid, $state.python_pid)) {
            if ($pid) {
                $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
                if ($proc) {
                    $out += $proc
                }
            }
        }
        return $out
    } catch {
        return @()
    }
}

function Stop-StaleWorker {
    if (-not (Test-Path $pidFile)) {
        return
    }
    try {
        $state = Get-Content $pidFile -Raw | ConvertFrom-Json
        foreach ($pid in @($state.wrapper_pid, $state.python_pid)) {
            if ($pid) {
                Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            }
        }
    } catch {
    }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    Remove-Item $heartbeatFile -Force -ErrorAction SilentlyContinue
}

Stop-StaleWorker
Start-Sleep -Seconds 1

Remove-Item $stdout -Force -ErrorAction SilentlyContinue
Remove-Item $stderr -Force -ErrorAction SilentlyContinue
Remove-Item $heartbeatFile -Force -ErrorAction SilentlyContinue

$env:PYTHONNOUSERSITE = "1"
Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue

$wrapperProc = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File", $loopScript, "--disable-collector") `
    -WindowStyle Hidden `
    -PassThru

if (-not $wrapperProc -or -not $wrapperProc.Id) {
    throw "Detached RL worker process did not start."
}

$wrapperPid = $wrapperProc.Id
$deadline = (Get-Date).AddSeconds(12)
$readyState = $null
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 500
    if (-not (Get-Process -Id $wrapperPid -ErrorAction SilentlyContinue)) {
        break
    }
    if (Test-Path $heartbeatFile) {
        try {
            $state = Get-Content $heartbeatFile -Raw | ConvertFrom-Json
            if ($state.wrapper_pid -eq $wrapperPid) {
                $readyState = $state
                break
            }
        } catch {
        }
    }
}

if (-not $readyState) {
    throw "Detached RL wrapper did not initialize heartbeat: pid=$wrapperPid"
}

$payload = [ordered]@{
    wrapper_pid = $readyState.wrapper_pid
    python_pid = $readyState.python_pid
    state = $readyState.state
    started_at = $readyState.started_at
    updated_at = $readyState.updated_at
    stdout = $stdout
    stderr = $stderr
}
$payload | ConvertTo-Json -Depth 5 | Set-Content -Path $pidFile -Encoding UTF8

[pscustomobject]@{
    WrapperPid = $wrapperPid
    PythonPid = $readyState.python_pid
    Stdout = $stdout
    Stderr = $stderr
} | Format-List
