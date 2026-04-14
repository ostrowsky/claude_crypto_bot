param(
    [string]$Token = $env:TELEGRAM_BOT_TOKEN,
    [switch]$ForceRestart = $true
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "pyembed\python.exe"
$workdir = Join-Path $root "files"
$stdout = Join-Path $root "agent_stdout.log"
$stderr = Join-Path $root "agent_stderr.log"
$runtimeDir = Join-Path $root ".runtime"
$pidFile = Join-Path $runtimeDir "market_agent_bg.json"

if (-not (Test-Path $runtimeDir)) {
    New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
}

function Stop-StaleWrapper {
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
}

if (Test-Path $pidFile) {
    try {
        $state = Get-Content $pidFile -Raw | ConvertFrom-Json
        $existingPid = [int]($state.python_pid | ForEach-Object { $_ })
        if ($existingPid -gt 0) {
            $existing = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
            if ($existing) {
                if (-not $ForceRestart) {
                    [pscustomobject]@{
                        PythonPid = $existing.Id
                        StartedAt = $existing.StartTime
                        Path = $existing.Path
                    } | Format-List
                    exit 0
                }
                Stop-Process -Id $existing.Id -Force -ErrorAction SilentlyContinue
            }
        }
    } catch {
    }
}

Stop-StaleWrapper
Start-Sleep -Seconds 1

Remove-Item $stdout -Force -ErrorAction SilentlyContinue
Remove-Item $stderr -Force -ErrorAction SilentlyContinue

$env:PYTHONNOUSERSITE = "1"
Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue
if ($Token) {
    $env:TELEGRAM_BOT_TOKEN = $Token
} else {
    Remove-Item Env:TELEGRAM_BOT_TOKEN -ErrorAction SilentlyContinue
}

$stdoutEsc = $stdout -replace '"', '\"'
$stderrEsc = $stderr -replace '"', '\"'
$workdirEsc = $workdir -replace "'", "''"
$pythonEsc = $python -replace "'", "''"
$stdoutPs = $stdout -replace "'", "''"
$stderrPs = $stderr -replace "'", "''"
$psCommand = @"
Set-Location -LiteralPath '$workdirEsc'
& '$pythonEsc' 'market_signal_agent.py' '--log-level' 'INFO' 1>> '$stdoutPs' 2>> '$stderrPs'
"@

$wrapper = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-Command", $psCommand) `
    -WindowStyle Hidden `
    -PassThru

if (-not $wrapper -or -not $wrapper.Id) {
    throw "Detached market agent wrapper did not start."
}

Start-Sleep -Seconds 2

@{
    wrapper_pid = $wrapper.Id
    python_pid = 0
    started_at = (Get-Date).ToString("o")
    stdout = $stdout
    stderr = $stderr
} | ConvertTo-Json | Set-Content -Path $pidFile -Encoding UTF8

[pscustomobject]@{
    WrapperPid = $wrapper.Id
    PythonPid = 0
    Stdout = $stdout
    Stderr = $stderr
} | Format-List
