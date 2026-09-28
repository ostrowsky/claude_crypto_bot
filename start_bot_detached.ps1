# Start the bot OUTSIDE the caller's process tree.
#
# Why (2026-09-28): the bot was restarted from a Claude Code session, so it lived
# in the Claude desktop app's process tree; when Windows updated the app
# (20:19 local) the bot died silently at 20:18:48 -- no traceback, no crash
# event -- while the RL worker, started from a user cmd window, survived.
#
# Usage from a tool / another app (the new process is created by the WMI service,
# not by the caller):
#   Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
#     CommandLine = 'powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File <root>\start_bot_detached.ps1' }
# The token is read here from the last runner file (or files\.env) and never
# appears on a command line. Log: .runtime\start_bot_detached.log
$root = $PSScriptRoot
$log = Join-Path $root ".runtime\start_bot_detached.log"
try {
    $tok = $null
    $runner = Join-Path $root ".runtime\bot_bg_runner.cmd"
    if (Test-Path $runner) {
        $m = Select-String -Path $runner -Pattern '^set TELEGRAM_BOT_TOKEN=(\S+)' | Select-Object -First 1
        if ($m) { $tok = $m.Matches[0].Groups[1].Value }
    }
    if (-not $tok) {
        $envf = Join-Path $root "files\.env"
        if (Test-Path $envf) {
            $m = Select-String -Path $envf -Pattern '^TELEGRAM_BOT_TOKEN=(\S+)' | Select-Object -First 1
            if ($m) { $tok = $m.Matches[0].Groups[1].Value }
        }
    }
    "$(Get-Date -Format o) token found: $([bool]$tok)" | Out-File $log -Encoding UTF8
    if (-not $tok) { throw "TELEGRAM_BOT_TOKEN not found in .runtime\bot_bg_runner.cmd or files\.env" }
    & (Join-Path $root "start_bot_bg.ps1") -Token $tok *>> $log
    "$(Get-Date -Format o) start_bot_bg.ps1 returned" | Out-File $log -Append -Encoding UTF8
} catch {
    "$(Get-Date -Format o) ERROR $_" | Out-File $log -Append -Encoding UTF8
}
