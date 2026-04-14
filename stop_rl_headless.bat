@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "$root = '%~dp0'.TrimEnd('\');" ^
  "$pidFile = Join-Path $root '.runtime\rl_worker_bg.json';" ^
  "$ids = @();" ^
  "if (Test-Path $pidFile) {" ^
  "  try {" ^
  "    $state = Get-Content $pidFile -Raw | ConvertFrom-Json;" ^
  "    $ids += @($state.python_pid, $state.wrapper_pid);" ^
  "  } catch {}" ^
  "}" ^
  "$ids = $ids | Where-Object { $_ -and [int]$_ -gt 0 } | ForEach-Object { [int]$_ } | Select-Object -Unique;" ^
  "foreach ($id in $ids) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue };" ^
  "if (Test-Path $pidFile) { Remove-Item $pidFile -Force -ErrorAction SilentlyContinue };" ^
  "Write-Host ('Stopped RL worker PIDs: ' + ($(if ($ids) { $ids -join ', ' } else { 'none' })))"

endlocal
