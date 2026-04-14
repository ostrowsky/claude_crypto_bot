$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "pyembed\python.exe"
$workdir = Join-Path $root "files"
$script = Join-Path $workdir "collect_headless_once.py"

if (-not (Test-Path $python)) {
    throw "Python not found: $python"
}
if (-not (Test-Path $script)) {
    throw "Script not found: $script"
}

$env:PYTHONNOUSERSITE = "1"
Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue

Push-Location $workdir
try {
    & $python $script @args
    if ($LASTEXITCODE -ne 0) {
        throw "headless_collect_once failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
