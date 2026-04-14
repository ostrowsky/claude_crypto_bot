$ErrorActionPreference = "Stop"

$root = "D:\Projects\gpt_crypto_bot"
$files = Join-Path $root "files"
$py = Join-Path $root "pyembed\python.exe"
$bot = Join-Path $files "bot.py"
$stdout = Join-Path $root "bot_stdout.log"
$stderr = Join-Path $root "bot_stderr.log"

Set-Location $files

& $py $bot 1>> $stdout 2>> $stderr
