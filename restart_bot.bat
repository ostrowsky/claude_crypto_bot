@echo off
REM Self-relaunch inside cmd /k so the window NEVER closes automatically
if "%1"=="--run" goto :MAIN
cmd /k "%~f0" --run
exit

:MAIN
chcp 65001 >nul
setlocal EnableDelayedExpansion

set "ROOT=D:\Projects\claude_crypto_bot"
set "PYTHON=%ROOT%\pyembed\python.exe"
set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
set "PSF=%PS% -NoProfile -ExecutionPolicy Bypass -File"

title Crypto Bot Restart

echo.
echo ================================================
echo       CRYPTO BOT -- FULL RESTART
echo ================================================
echo   %date%  %time%
echo   Project: %ROOT%
echo ================================================
echo.

REM -- 1. Stop RL Worker --
echo [1/6] Stopping RL worker...
call "%ROOT%\stop_rl_headless.bat"
timeout /t 2 /nobreak >nul
echo   Done.
echo.

REM -- 2. Stop Bot --
echo [2/6] Stopping Bot...
%PSF% "%ROOT%\stop_bot_bg.ps1"
timeout /t 3 /nobreak >nul
echo   Done.
echo.

REM -- 3. Read Telegram token --
echo [3/6] Reading Telegram token...
set "BOT_TOKEN="

REM Try runner cmd file first
if exist "%ROOT%\.runtime\bot_bg_runner.cmd" (
    for /f "tokens=2 delims==" %%A in ('findstr /i "TELEGRAM_BOT_TOKEN=" "%ROOT%\.runtime\bot_bg_runner.cmd" 2^>nul') do (
        if not defined BOT_TOKEN set "BOT_TOKEN=%%A"
    )
)

REM Fallback to environment variable
if not defined BOT_TOKEN (
    if not "%TELEGRAM_BOT_TOKEN%"=="" set "BOT_TOKEN=%TELEGRAM_BOT_TOKEN%"
)

if not defined BOT_TOKEN (
    echo.
    echo [ERROR] TELEGRAM_BOT_TOKEN not found.
    echo         Check: .runtime\bot_bg_runner.cmd  ^(line: set TELEGRAM_BOT_TOKEN=...^)
    echo         Or set Windows env var TELEGRAM_BOT_TOKEN
    echo.
    goto :STATUS
)
echo   Token: !BOT_TOKEN:~0,10!... (truncated)
echo.

REM -- 4. Check scheduled tasks --
echo [4/6] Checking scheduled tasks...
for %%T in (CryptoBot_DailyLearning_EOD CryptoBot_IntradaySnapshot) do (
    schtasks /query /tn "%%T" /fo LIST >nul 2>&1
    if !errorlevel!==0 (
        echo   %%T: found
        schtasks /query /tn "%%T" /fo LIST 2>nul | findstr /i "Disabled" >nul
        if !errorlevel!==0 (
            echo     Was disabled -- enabling...
            schtasks /change /tn "%%T" /enable >nul 2>&1
        )
    ) else (
        echo   [WARN] Task %%T not found
    )
)
echo.

REM -- 5. Start RL Worker --
echo [5/6] Starting RL worker...
%PSF% "%ROOT%\start_rl_worker_bg.ps1"
echo   RL worker: exit code !errorlevel!
timeout /t 2 /nobreak >nul
echo.

REM -- 6. Start Bot --
echo [6/6] Starting Bot...
%PSF% "%ROOT%\start_bot_bg.ps1" -Token "!BOT_TOKEN!"
echo   Bot: exit code !errorlevel!
timeout /t 5 /nobreak >nul
echo.

:STATUS
echo ================================================
echo   STATUS CHECK
echo ================================================
echo.

echo -- Bot --
%PSF% "%ROOT%\bot_status.ps1"
echo.

echo -- RL Worker --
if exist "%ROOT%\.runtime\rl_worker_bg.json" (
    type "%ROOT%\.runtime\rl_worker_bg.json"
) else (
    echo   [WARN] .runtime\rl_worker_bg.json not found
)
echo.

echo -- Scheduled Tasks --
for %%T in (CryptoBot_DailyLearning_EOD CryptoBot_IntradaySnapshot) do (
    echo   [task: %%T]
    schtasks /query /tn "%%T" /fo LIST 2>nul | findstr /i "Status"
    schtasks /query /tn "%%T" /fo LIST 2>nul | findstr /i "Next Run"
)

echo.
echo ================================================
echo   Done.  %date%  %time%
echo ================================================
echo.
echo (window stays open -- close manually when ready)
echo.
