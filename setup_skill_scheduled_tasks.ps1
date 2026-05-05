# Sets up Windows Scheduled Tasks for skill integration.
# Run from elevated PowerShell once.
#
# Tasks created:
#   1. CryptoBot_KlinesBackfill_Daily — daily klines refresh (06:00 local)
#      Keeps history/ cache fresh for ZigZag-based EX1 and skill evaluations.
#   2. CryptoBot_SignalEvaluator_Weekly — Sunday 03:30 local
#      Runs evaluator + posts digest to TG admin chat.
#
# Spec: docs/specs/features/signal-evaluator-integration-spec.md §8.

$ErrorActionPreference = "Stop"

$ROOT    = "D:\Projects\claude_crypto_bot"
$PYEMBED = "$ROOT\pyembed\python.exe"

if (-not (Test-Path $PYEMBED)) {
    Write-Error "pyembed not found at $PYEMBED"
    exit 1
}

# ── Task 1: Daily klines backfill ──────────────────────────────────────────
$task1Name = "CryptoBot_KlinesBackfill_Daily"
$task1Action = New-ScheduledTaskAction `
    -Execute $PYEMBED `
    -Argument "$ROOT\files\_backfill_klines_history.py --days 30 --tf 15m --skip-existing" `
    -WorkingDirectory $ROOT
$task1Trigger = New-ScheduledTaskTrigger -Daily -At 06:00
$task1Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15)

if (Get-ScheduledTask -TaskName $task1Name -ErrorAction SilentlyContinue) {
    Write-Host "[setup] removing existing $task1Name"
    Unregister-ScheduledTask -TaskName $task1Name -Confirm:$false
}
Register-ScheduledTask `
    -TaskName $task1Name `
    -Action $task1Action `
    -Trigger $task1Trigger `
    -Settings $task1Settings `
    -Description "Refresh history/<sym>_<tf>.csv klines cache for skill + EX1 ZigZag mode" | Out-Null
Write-Host "[setup] OK: $task1Name (daily 06:00)"

# ── Task 2: Weekly signal evaluator with TG digest ─────────────────────────
$task2Name = "CryptoBot_SignalEvaluator_Weekly"
$task2Action = New-ScheduledTaskAction `
    -Execute $PYEMBED `
    -Argument "$ROOT\files\_weekly_signal_eval_with_tg.py" `
    -WorkingDirectory $ROOT
$task2Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 03:30
$task2Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

if (Get-ScheduledTask -TaskName $task2Name -ErrorAction SilentlyContinue) {
    Write-Host "[setup] removing existing $task2Name"
    Unregister-ScheduledTask -TaskName $task2Name -Confirm:$false
}
Register-ScheduledTask `
    -TaskName $task2Name `
    -Action $task2Action `
    -Trigger $task2Trigger `
    -Settings $task2Settings `
    -Description "Weekly skill evaluation + TG digest for canonical metrics review" | Out-Null
Write-Host "[setup] OK: $task2Name (Sunday 03:30)"

Write-Host ""
Write-Host "[setup] Both tasks registered. Verify with:"
Write-Host "  Get-ScheduledTask CryptoBot_KlinesBackfill_Daily, CryptoBot_SignalEvaluator_Weekly"
Write-Host ""
Write-Host "[setup] Test runs (without waiting for schedule):"
Write-Host "  Start-ScheduledTask CryptoBot_KlinesBackfill_Daily"
Write-Host "  Start-ScheduledTask CryptoBot_SignalEvaluator_Weekly"
