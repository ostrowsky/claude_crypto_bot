@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "$statusFile = 'D:\Projects\gpt_crypto_bot\.runtime\codex_exec_supervisor_status.json';" ^
  "$spid = 0;" ^
  "if (Test-Path $statusFile) {" ^
  "  try { $spid = [int]((Get-Content $statusFile -Raw | ConvertFrom-Json).pid) } catch {}" ^
  "}" ^
  "if ($spid -gt 0) { Stop-Process -Id $spid -Force -ErrorAction SilentlyContinue; Write-Host ('Stopped supervisor PID: ' + $spid) } else { Write-Host 'Stopped supervisor PID: none' }"

endlocal
