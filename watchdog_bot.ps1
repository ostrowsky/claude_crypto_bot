# watchdog_bot.ps1 — restart the bot if it died silently.
#
# Reason: the bot has gone down unnoticed several times (2026-07-23: log cut
# mid-line, no traceback, 8 days with zero signals). Runs every 10 min via the
# CryptoBot_Watchdog scheduled task.
#
# Safety: only ever inspects/starts processes whose command line contains THIS
# project root. The separate D:\Projects\gpt_crypto_bot is never touched, and
# nothing is ever killed — the watchdog only starts a bot when none is running.
# The Telegram token is read from the gitignored runner at runtime and is never
# logged or printed.

$ErrorActionPreference = "Stop"
$root   = "D:\Projects\claude_crypto_bot"
$runner = Join-Path $root ".runtime\bot_bg_runner.cmd"
$log    = Join-Path $root ".runtime\watchdog.log"

function Write-WLog([string]$msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $log -Value $line -Encoding UTF8
}

function Get-BotProc {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*$root*" -and $_.CommandLine -match 'bot\.py' } |
        Select-Object -First 1
}

# 1. Alive? Then do nothing (stay quiet — no log spam on the happy path).
if (Get-BotProc) { exit 0 }

Write-WLog "bot.py NOT running -> restarting"

if (-not (Test-Path $runner)) {
    Write-WLog "ERROR: runner missing: $runner"
    exit 1
}

# 2. Start detached; the runner injects the token and launches bot.py.
Start-Process -FilePath $env:ComSpec -ArgumentList '/c', $runner -WindowStyle Hidden
Start-Sleep -Seconds 25

$bot = Get-BotProc
if ($bot) { Write-WLog ("restart OK pid=" + $bot.ProcessId) }
else      { Write-WLog "restart FAILED (no bot.py after 25s)" }

# 3. Best-effort Telegram notice so a silent death becomes visible.
try {
    $tokLine = Get-Content $runner | Where-Object { $_ -match 'set TELEGRAM_BOT_TOKEN=' } | Select-Object -First 1
    $tok = ($tokLine -replace '.*set TELEGRAM_BOT_TOKEN=', '').Trim()
    $chatFile = Join-Path $root "files\.chat_ids"
    if ($tok -and (Test-Path $chatFile)) {
        $text = if ($bot) { "watchdog: bot was down, restarted (pid $($bot.ProcessId))" }
                else      { "watchdog: bot is DOWN and restart FAILED - needs attention" }
        # .chat_ids is a JSON array (e.g. [179184487]); fall back to one-per-line.
        $rawIds = (Get-Content $chatFile -Raw -Encoding UTF8).TrimStart([char]0xFEFF).Trim()
        $ids = @()
        try { $ids = @($rawIds | ConvertFrom-Json) } catch { $ids = @($rawIds -split "\r?\n") }
        foreach ($cid in $ids) {
            $cid = "$cid".Trim()
            if (-not $cid) { continue }
            $body = @{ chat_id = $cid; text = $text } | ConvertTo-Json -Compress
            Invoke-RestMethod -Uri "https://api.telegram.org/bot$tok/sendMessage" `
                -Method Post -Body $body -ContentType "application/json" -TimeoutSec 15 | Out-Null
        }
    }
} catch {
    Write-WLog ("tg notify failed: " + $_.Exception.Message)
}
