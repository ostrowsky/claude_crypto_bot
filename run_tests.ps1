$ErrorActionPreference = "Stop"

$env:PYTHONNOUSERSITE = "1"
Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue

Set-Location "$PSScriptRoot\files"
& "$PSScriptRoot\pyembed\python.exe" -m unittest discover -s . -p "test_bot.py"
