param(
    [int]$RestartDelaySec = 15
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "pyembed\python.exe"
$script = Join-Path $root "files\rl_headless_worker.py"
$workdir = Join-Path $root "files"
$runtimeDir = Join-Path $root ".runtime"
$loopLog = Join-Path $runtimeDir "headless_loop.log"
$pidFile = Join-Path $runtimeDir "rl_worker_bg.json"
$heartbeatFile = Join-Path $runtimeDir "rl_worker_wrapper_heartbeat.json"

if (-not (Test-Path $python)) {
    throw "Python runtime not found: $python"
}
if (-not (Test-Path $script)) {
    throw "RL headless worker script not found: $script"
}
if (-not (Test-Path $runtimeDir)) {
    New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
}

$env:PYTHONNOUSERSITE = "1"
Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue

function Write-LoopState {
    param(
        [string]$State,
        [int]$PythonPid = 0,
        [string]$StartedAt = "",
        [int]$ExitCode = 0
    )

    $now = (Get-Date).ToUniversalTime().ToString("o")
    $payload = [ordered]@{
        wrapper_pid = $PID
        python_pid = $PythonPid
        state = $State
        started_at = $StartedAt
        updated_at = $now
        restart_delay_sec = $RestartDelaySec
        script = $script
        cwd = $workdir
        args = @($args)
    }
    if ($ExitCode -ne 0) {
        $payload.exit_code = $ExitCode
    }
    $json = $payload | ConvertTo-Json -Depth 4
    Set-Content -Path $pidFile -Value $json -Encoding UTF8
    Set-Content -Path $heartbeatFile -Value $json -Encoding UTF8
}

Push-Location $workdir
try {
    while ($true) {
        $started = (Get-Date).ToString("o")
        Add-Content -Path $loopLog -Encoding UTF8 -Value "$started headless_loop: worker start"
        $argList = @($script) + @($args)
        $proc = Start-Process -FilePath $python -ArgumentList $argList -WorkingDirectory $workdir -WindowStyle Hidden -PassThru
        Write-LoopState -State "running" -PythonPid $proc.Id -StartedAt $started

        while (-not $proc.HasExited) {
            Write-LoopState -State "running" -PythonPid $proc.Id -StartedAt $started
            try {
                Wait-Process -Id $proc.Id -Timeout 5 -ErrorAction Stop
            } catch {
                # timeout -> heartbeat refresh loop
            }
            try {
                $proc.Refresh()
            } catch {
            }
        }

        $code = 0
        try { $code = $proc.ExitCode } catch {}
        $stopped = (Get-Date).ToString("o")
        Write-LoopState -State "stopped" -PythonPid 0 -StartedAt $started -ExitCode $code
        Add-Content -Path $loopLog -Encoding UTF8 -Value "$stopped headless_loop: worker exit code=$code, restart in ${RestartDelaySec}s"
        Start-Sleep -Seconds $RestartDelaySec
    }
} finally {
    Pop-Location
}
