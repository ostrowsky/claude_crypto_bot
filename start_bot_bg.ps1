param(
    [string]$Token = $env:TELEGRAM_BOT_TOKEN,
    [switch]$ForceRestart = $true
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "pyembed\python.exe"
$workdir = Join-Path $root "files"
$botScript = Join-Path $workdir "bot.py"
$stdout = Join-Path $root "bot_stdout.log"
$stderr = Join-Path $root "bot_stderr.log"
$runtimeDir = Join-Path $root ".runtime"
$pidFile = Join-Path $runtimeDir "bot_bg.json"
$runnerFile = Join-Path $runtimeDir "bot_bg_runner.cmd"
$launcherLog = Join-Path $runtimeDir "start_bot_bg.log"

if (-not (Test-Path $runtimeDir)) {
    New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
}

function Get-BotPythonProcesses {
    try {
        $botScriptEscaped = [regex]::Escape($botScript)
        $matches = Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'py.exe'" -ErrorAction Stop |
            Where-Object {
                $_.ExecutablePath -eq $python -and
                [string]$_.CommandLine -match $botScriptEscaped
            }
        $out = @()
        foreach ($match in $matches) {
            $proc = Get-Process -Id $match.ProcessId -ErrorAction SilentlyContinue
            if ($proc) {
                $out += $proc
            }
        }
        return @($out)
    } catch {
        return @()
    }
}

function Get-LiveProcessById {
    param([Nullable[int]]$Pid)
    if (-not $Pid) {
        return $null
    }
    try {
        return Get-Process -Id ([int]$Pid) -ErrorAction Stop
    } catch {
        return $null
    }
}

function Write-LauncherLog {
    param([string]$Message)
    $ts = (Get-Date).ToString("o")
    Add-Content -Path $launcherLog -Encoding UTF8 -Value "$ts start_bot_bg: $Message"
}

function Read-TailText {
    param(
        [string]$Path,
        [int]$Lines = 40
    )
    if (-not (Test-Path $Path)) {
        return ""
    }
    try {
        return ((Get-Content -Path $Path -Tail $Lines -ErrorAction Stop) -join [Environment]::NewLine)
    } catch {
        return ""
    }
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

if (-not $Token) {
    throw "TELEGRAM_BOT_TOKEN is not set."
}

$existing = @(Get-BotPythonProcesses)
if ($existing.Count -gt 0) {
    if (-not $ForceRestart) {
        $existing |
            Select-Object Id, ProcessName, StartTime, Path |
            Format-Table -AutoSize
        exit 0
    }
    $existing | Stop-Process -Force -ErrorAction SilentlyContinue
}

Stop-StaleWrapper
Start-Sleep -Seconds 1

Remove-Item $stdout -Force -ErrorAction SilentlyContinue
Remove-Item $stderr -Force -ErrorAction SilentlyContinue
Remove-Item $launcherLog -Force -ErrorAction SilentlyContinue
Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
Remove-Item $runnerFile -Force -ErrorAction SilentlyContinue

$env:TELEGRAM_BOT_TOKEN = $Token
$env:PYTHONNOUSERSITE = "1"
Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue

Write-LauncherLog "launch requested"

$runnerLines = @(
    "@echo off",
    "set PYTHONUTF8=",
    "set PYTHONNOUSERSITE=1",
    "set TELEGRAM_BOT_TOKEN=$Token",
    "cd /d $workdir",
    """$python"" ""$botScript"" 1>>""$stdout"" 2>>""$stderr"""
)
$runnerLines | Set-Content -Path $runnerFile -Encoding ASCII
Write-LauncherLog "runner file written: $runnerFile"

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "cmd.exe"
$psi.Arguments = "/c `"$runnerFile`""
$psi.WorkingDirectory = $root
$psi.UseShellExecute = $true
$psi.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden

$wrapper = [System.Diagnostics.Process]::Start($psi)
if (-not $wrapper -or -not $wrapper.Id) {
    Write-LauncherLog "Process.Start returned no wrapper process id"
    throw "Detached bot process did not start."
}

$wrapperPid = $wrapper.Id
Write-LauncherLog "wrapper launched pid=$wrapperPid"

$bot = $null
$launchObserved = $false
for ($i = 0; $i -lt 15; $i++) {
    Start-Sleep -Seconds 2
    $bot = Get-BotPythonProcesses | Sort-Object StartTime -Descending | Select-Object -First 1
    if ($bot) {
        Write-LauncherLog "bot python confirmed pid=$($bot.Id)"
        $launchObserved = $true
        break
    }
    if (Test-Path $stderr) {
        try {
            $stderrInfo = Get-Item $stderr -ErrorAction Stop
            if (((Get-Date) - $stderrInfo.LastWriteTime).TotalSeconds -le 15) {
                Write-LauncherLog "fresh stderr heartbeat observed without python pid confirmation"
                $launchObserved = $true
                break
            }
        } catch {
        }
    }
    if ($wrapper.HasExited) {
        Write-LauncherLog "wrapper exited before bot confirmation code=$($wrapper.ExitCode)"
        break
    }
    Write-LauncherLog "waiting for bot python attempt=$($i + 1)"
}

if (-not $launchObserved) {
    $stderrTail = Read-TailText -Path $stderr -Lines 60
    if ($stderrTail) {
        Write-LauncherLog "bot not confirmed, stderr tail: $stderrTail"
    } else {
        Write-LauncherLog "bot not confirmed, stderr empty"
    }
    throw "Bot python process did not start."
}

@{
    wrapper_pid = $wrapperPid
    python_pid = if ($bot) { $bot.Id } else { $null }
    started_at = (Get-Date).ToString("o")
    stdout = $stdout
    stderr = $stderr
} | ConvertTo-Json | Set-Content -Path $pidFile -Encoding UTF8

Write-LauncherLog "pid file written wrapper=$wrapperPid python=$(if ($bot) { $bot.Id } else { '' })"

[pscustomobject]@{
    WrapperPid = $wrapperPid
    PythonPid = if ($bot) { $bot.Id } else { $null }
    Stdout = $stdout
    Stderr = $stderr
} | Format-List
