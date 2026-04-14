$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "pyembed\python.exe"
$script = Join-Path $root "files\report_rl_daily.py"
$workdir = Join-Path $root "files"

if (-not (Test-Path $python)) {
    throw "Python runtime not found: $python"
}
if (-not (Test-Path $script)) {
    throw "RL daily report script not found: $script"
}

$env:PYTHONNOUSERSITE = "1"
Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue

Push-Location $workdir
try {
    & $python $script @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
