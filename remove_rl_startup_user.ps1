$ErrorActionPreference = "Stop"

$startupDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup"
$cmdPath = Join-Path $startupDir "gpt_crypto_rl_headless.cmd"

$removed = $false
if (Test-Path $cmdPath) {
    Remove-Item $cmdPath -Force
    $removed = $true
}

[pscustomobject]@{
    Removed = $removed
    StartupFile = $cmdPath
} | Format-List
