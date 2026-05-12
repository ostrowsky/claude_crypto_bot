# Sets up Windows Scheduled Tasks for skill integration.
# Run from PowerShell once (no admin needed).
#
# Tasks created:
#   1. CryptoBot_KlinesBackfill_Daily      — 06:00 every day
#      Refresh history/<sym>_<tf>.csv cache; idempotent --skip-existing.
#   2. CryptoBot_SignalEval_Daily          — 08:00 every day
#      Last 24h window. Silent unless incidents (premature exits,
#      missed trends >= 5%, losing trades < -1.5%). NO scout trigger.
#   3. CryptoBot_SignalEvaluator_Weekly    — Sunday 03:30
#      Last 7d window. Full digest + scout auto-apply (low risk).
#   4. CryptoBot_SignalEval_Monthly        — 1st of month 04:00
#      Last 30d window. Strategic review + scout dry-run proposals.
#
# Spec: docs/specs/features/signal-evaluator-integration-spec.md §8a.

$ErrorActionPreference = "Stop"

$ROOT    = "D:\Projects\claude_crypto_bot"
$PYEMBED = "$ROOT\pyembed\python.exe"
$RUNNER  = "$ROOT\files\_weekly_signal_eval_with_tg.py"

if (-not (Test-Path $PYEMBED)) {
    Write-Error "pyembed not found at $PYEMBED"
    exit 1
}

function Register-OrUpdate {
    param(
        [string]$Name, [string]$CmdArgs, [Microsoft.Management.Infrastructure.CimInstance]$Trigger,
        [int]$LimitMin, [string]$Desc
    )
    if (Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue) {
        Write-Host "[setup] removing existing $Name"
        Unregister-ScheduledTask -TaskName $Name -Confirm:$false
    }
    $action = New-ScheduledTaskAction -Execute $PYEMBED -Argument $CmdArgs -WorkingDirectory $ROOT
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Minutes $LimitMin)
    Register-ScheduledTask -TaskName $Name -Action $action -Trigger $Trigger `
        -Settings $settings -Description $Desc | Out-Null
    Write-Host "[setup] OK: $Name"
}

# ── Task 1: Daily klines backfill ──────────────────────────────────────────
Register-OrUpdate `
    -Name "CryptoBot_KlinesBackfill_Daily" `
    -CmdArgs "$ROOT\files\_backfill_klines_history.py --days 30 --tf 15m --skip-existing" `
    -Trigger (New-ScheduledTaskTrigger -Daily -At 06:00) `
    -LimitMin 15 `
    -Desc "Refresh history/<sym>_<tf>.csv klines cache for skill + EX1 ZigZag mode"

# ── Task 2: Daily signal-evaluator (silent on no incidents) ────────────────
Register-OrUpdate `
    -Name "CryptoBot_SignalEval_Daily" `
    -CmdArgs "$RUNNER --cadence daily" `
    -Trigger (New-ScheduledTaskTrigger -Daily -At 08:00) `
    -LimitMin 15 `
    -Desc "Daily 24h window. Silent unless incidents (premature exits, big misses)"

# ── Task 3: Weekly signal-evaluator (full digest + scout auto-apply) ───────
Register-OrUpdate `
    -Name "CryptoBot_SignalEvaluator_Weekly" `
    -CmdArgs "$RUNNER --cadence weekly" `
    -Trigger (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 03:30) `
    -LimitMin 30 `
    -Desc "Weekly 7d skill evaluation + TG digest + scout low-risk auto-apply"

# ── Task 4: Monthly signal-evaluator (strategic review, dry-run scout) ─────
# PowerShell New-ScheduledTaskTrigger doesn't support -Monthly; use schtasks.exe.
$task4Name = "CryptoBot_SignalEval_Monthly"
$existing = Get-ScheduledTask -TaskName $task4Name -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "[setup] removing existing $task4Name"
    Unregister-ScheduledTask -TaskName $task4Name -Confirm:$false
}
$cmdLine = "`"$PYEMBED`" `"$RUNNER`" --cadence monthly"
$out = & schtasks /Create /TN $task4Name /TR $cmdLine /SC MONTHLY /D 1 /ST 04:00 /F 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "[setup] OK: $task4Name (monthly 1st @ 04:00)"
} else {
    Write-Warning "[setup] schtasks /Create $task4Name failed: $out"
}

Write-Host ""
Write-Host "[setup] All 4 tasks registered. Verify with:"
Write-Host "  Get-ScheduledTask CryptoBot_KlinesBackfill_Daily,"
Write-Host "                    CryptoBot_SignalEval_Daily,"
Write-Host "                    CryptoBot_SignalEvaluator_Weekly,"
Write-Host "                    CryptoBot_SignalEval_Monthly | Format-Table"
Write-Host ""
Write-Host "[setup] Test runs (without waiting):"
Write-Host "  Start-ScheduledTask CryptoBot_SignalEval_Daily"
Write-Host "  Start-ScheduledTask CryptoBot_SignalEvaluator_Weekly"
Write-Host "  Start-ScheduledTask CryptoBot_SignalEval_Monthly"
