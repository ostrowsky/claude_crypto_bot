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

function Stop-AllRLWorkers {
    # Idempotency guard: kill ANY existing RL worker/wrapper for THIS repo by
    # command line, not just the PIDs recorded in the pid-file. A prior start
    # that threw on the heartbeat timeout (worker loads catboost > deadline)
    # leaves an UNTRACKED wrapper that keeps respawning workers -> duplicates
    # accumulate -> RAM exhaustion -> event-loop paging freeze. Match is scoped
    # to "$root" so the separate gpt_crypto_bot is never touched (CLAUDE.md s1).
    $rootEsc = $root.Replace('\', '\\')
    $patterns = @("$rootEsc.*rl_headless_worker", "$rootEsc.*headless_loop")
    foreach ($pat in $patterns) {
        try {
            Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='powershell.exe'" |
                Where-Object { $_.CommandLine -match $pat } |
                ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        } catch { }
    }
}

Stop-StaleWorker
Stop-AllRLWorkers
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
$deadline = (Get-Date).AddSeconds(25)   # was 12 — worker loads catboost models, can exceed 12s
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
    # Heartbeat not seen within the deadline. If the wrapper is still ALIVE it
    # is almost certainly mid model-load and will write the heartbeat shortly —
    # record its pid so the NEXT start can stop it (prevents the untracked-
    # orphan -> duplicate-worker leak) and warn instead of throwing, so a
    # combined launcher does not abort before starting the bot.
    if (Get-Process -Id $wrapperPid -ErrorAction SilentlyContinue) {
        [ordered]@{ wrapper_pid = $wrapperPid; python_pid = $null;
                    state = "starting"; started_at = (Get-Date).ToString("o");
                    updated_at = (Get-Date).ToString("o");
                    stdout = $stdout; stderr = $stderr } |
            ConvertTo-Json -Depth 5 | Set-Content -Path $pidFile -Encoding UTF8
        Write-Warning "RL wrapper $wrapperPid alive but heartbeat not ready in 25s (likely model load); recorded pid, continuing."
        return
    }
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
