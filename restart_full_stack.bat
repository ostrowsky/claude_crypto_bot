@echo off
REM ===========================================================================
REM  Full stack restart that FAILS LOUDLY.
REM
REM  restart_bot.bat re-launches itself inside `cmd /k` and exits immediately,
REM  so the caller always sees success while the real work happens in a window
REM  nobody watches. On 2026-08-13 that hid a restart in which bot.py never
REM  started; on 07-23 the bot stayed dead for 8 days.
REM
REM  This script keeps one process, checks every step, and returns a non-zero
REM  exit code when anything fails to come up.
REM
REM    restart_full_stack.bat                 quiet, script-friendly
REM    restart_full_stack.bat --pause         keep the window open at the end
REM    restart_full_stack.bat --skip-tests    emergency restart, no test gate
REM ===========================================================================
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"
set "PYTHON=%ROOT%\pyembed\python.exe"
set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
set "PSF=%PS% -NoProfile -ExecutionPolicy Bypass -File"

set "KEEP_OPEN=0"
set "RUN_TESTS=1"
:parse
if "%~1"=="" goto :parsed
if /I "%~1"=="--pause" set "KEEP_OPEN=1"
if /I "%~1"=="--skip-tests" set "RUN_TESTS=0"
shift
goto :parse
:parsed

set "BUILD_COMMIT=unknown"
set "BUILD_DATE=unknown"
for /f "usebackq delims=" %%i in (`git rev-parse --short HEAD 2^>nul`) do set "BUILD_COMMIT=%%i"
set "BUILD_DATE=%date% %time%"

echo ============================================================
echo   CRYPTO BOT - FULL STACK RESTART
echo   Commit: %BUILD_COMMIT%   Started: %BUILD_DATE%
echo   Root:  %ROOT%
echo ============================================================
echo.

REM -- 1/6 test regression gate ------------------------------------------
REM  The suite is not green (40 known failures of 757). Gating on green would
REM  block every restart, so the gate blocks only on NEW failures against the
REM  recorded baseline. Use --skip-tests when the bot is down and must return.
if "%RUN_TESTS%"=="0" (
    echo [1/6] Test gate SKIPPED by request.
) else (
    echo [1/6] Test regression gate...
    "%PYTHON%" "%ROOT%\files\run_test_suite.py"
    if errorlevel 1 (
        echo.
        echo   NEW test failures - restart aborted. Fix them, or rerun with
        echo   --skip-tests if the bot must come back up first.
        goto :fail
    )
    echo   No new failures.
)
echo.

REM -- 2/6 stop ----------------------------------------------------------
echo [2/6] Stopping RL worker and bot...
call "%ROOT%\stop_rl_headless.bat" >nul 2>&1
%PSF% "%ROOT%\stop_bot_bg.ps1" >nul 2>&1
ping -n 4 127.0.0.1 >nul
echo   Stopped.
echo.

REM -- 3/6 token ---------------------------------------------------------
REM  Read from the generated runner file; never printed, never committed
REM  (CLAUDE.md section 13).
echo [3/6] Reading Telegram token...
set "BOT_TOKEN="
if exist "%ROOT%\.runtime\bot_bg_runner.cmd" (
    for /f "tokens=2 delims==" %%A in ('findstr /i "TELEGRAM_BOT_TOKEN=" "%ROOT%\.runtime\bot_bg_runner.cmd" 2^>nul') do (
        if not defined BOT_TOKEN set "BOT_TOKEN=%%A"
    )
)
if not defined BOT_TOKEN if not "%TELEGRAM_BOT_TOKEN%"=="" set "BOT_TOKEN=%TELEGRAM_BOT_TOKEN%"
if not defined BOT_TOKEN (
    echo   [ERROR] TELEGRAM_BOT_TOKEN not found in .runtime\bot_bg_runner.cmd
    echo           nor in the environment. Cannot start the bot.
    goto :fail
)
echo   Token found.
echo.

REM -- 4/6 start ---------------------------------------------------------
echo [4/6] Starting RL worker...
%PSF% "%ROOT%\start_rl_worker_bg.ps1"
if errorlevel 1 (
    echo   [ERROR] RL worker launcher returned %errorlevel%.
    goto :fail
)
ping -n 3 127.0.0.1 >nul

echo       Starting bot...
%PSF% "%ROOT%\start_bot_bg.ps1" -Token "!BOT_TOKEN!"
if errorlevel 1 (
    echo   [ERROR] Bot launcher returned %errorlevel%.
    goto :fail
)
echo.

REM -- 5/6 settle --------------------------------------------------------
echo [5/6] Waiting for processes to settle...
ping -n 13 127.0.0.1 >nul
echo.

REM -- 6/6 verify --------------------------------------------------------
REM  This is the step restart_bot.bat never had: proof, not hope.
echo [6/6] Verifying...
echo   -- bot --
%PSF% "%ROOT%\bot_status.ps1" -FailIfNotRunning
if errorlevel 1 (
    echo   [ERROR] Bot did not come up. See bot_stderr.log.
    goto :fail
)
echo   -- RL worker --
%PSF% "%ROOT%\rl_worker_status.ps1" -FailIfNotRunning
if errorlevel 1 (
    echo   [ERROR] RL worker did not come up.
    goto :fail
)

echo.
echo ============================================================
echo   RESTART OK - both workers verified running.
echo   Logs: bot_stderr.log, .runtime\rl_worker_runtime.log
echo ============================================================
if "%KEEP_OPEN%"=="1" pause
endlocal
exit /b 0

:fail
echo.
echo ============================================================
echo   RESTART FAILED
echo ============================================================
if "%KEEP_OPEN%"=="1" pause
endlocal
exit /b 1
