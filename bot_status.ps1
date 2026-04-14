$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$stdout = Join-Path $root "bot_stdout.log"
$stderr = Join-Path $root "bot_stderr.log"
$runtimeDir = Join-Path $root ".runtime"
$pidFile = Join-Path $runtimeDir "bot_bg.json"
$launcherLog = Join-Path $runtimeDir "start_bot_bg.log"

$state = $null
if (Test-Path $pidFile) {
    try {
        $state = Get-Content $pidFile -Raw | ConvertFrom-Json
    } catch {
    }
}

$bot = $null
if ($state -and $state.python_pid) {
    try {
        $bot = Get-Process -Id $state.python_pid -ErrorAction Stop
    } catch {
        $bot = $null
    }
}

$wrapper = $null
if ($state -and $state.wrapper_pid) {
    try {
        $wrapper = Get-Process -Id $state.wrapper_pid -ErrorAction Stop
    } catch {
        $wrapper = $null
    }
}

$stderrFresh = $false
$stderrSeenAt = $null
if (Test-Path $stderr) {
    try {
        $stderrInfo = Get-Item $stderr -ErrorAction Stop
        $stderrSeenAt = $stderrInfo.LastWriteTime
        $stderrFresh = ((Get-Date) - $stderrInfo.LastWriteTime).TotalSeconds -le 30
    } catch {
        $stderrFresh = $false
    }
}

$launcherFresh = $false
if (Test-Path $launcherLog) {
    try {
        $launcherInfo = Get-Item $launcherLog -ErrorAction Stop
        $launcherFresh = ((Get-Date) - $launcherInfo.LastWriteTime).TotalSeconds -le 30
    } catch {
        $launcherFresh = $false
    }
}

$running = [bool]$bot -or [bool]$wrapper
if (-not $running -and ($stderrFresh -or $launcherFresh)) {
    $running = $true
}

[pscustomobject]@{
    Running = $running
    PythonPid = if ($bot) { $bot.Id } elseif ($state) { $state.python_pid } else { $null }
    Started = if ($bot) { $bot.StartTime } elseif ($wrapper) { $wrapper.StartTime } elseif ($state) { $state.started_at } elseif ($stderrSeenAt) { $stderrSeenAt } else { $null }
    WrapperPid = if ($wrapper) { $wrapper.Id } elseif ($state) { $state.wrapper_pid } else { $null }
    Stdout = $stdout
    Stderr = $stderr
} | Format-List

if (Test-Path $stderr) {
    Write-Host ""
    Write-Host "stderr tail:"
    Get-Content $stderr -Tail 20
}
