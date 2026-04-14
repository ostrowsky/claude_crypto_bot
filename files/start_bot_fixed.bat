@echo off
chcp 65001 >nul
setlocal

title OST Crypto Signals Bot

set "ROOT=%~dp0"
cd /d "%ROOT%"

cls
echo ==========================================
echo      OST CRYPTO SIGNALS BOT LAUNCHER
echo ==========================================
echo.

echo [CHECK] Python...
where py >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON=py"
) else (
    where python >nul 2>&1
    if %errorlevel%==0 (
        set "PYTHON=python"
    ) else (
        echo [ERROR] Python not found
        pause
        exit /b 1
    )
)

for /f "tokens=*" %%v in ('%PYTHON% --version 2^>^&1') do set "PYVER=%%v"
echo [OK] %PYVER%
echo.

echo [CHECK] Project files...
set "MISSING=0"
for %%f in (bot.py config.py monitor.py strategy.py indicators.py watchlist.json) do (
    if not exist "%%f" (
        echo [ERROR] Missing file: %%f
        set "MISSING=1"
    )
)

if "%MISSING%"=="1" (
    echo.
    echo [ERROR] Some files are missing
    pause
    exit /b 1
)

if not exist logs mkdir logs

echo Select mode:
echo 1 - Full launch
echo 2 - Bot only
echo 3 - RL only
echo 0 - Exit
echo.

set /p MODE=Your choice [1]:
if "%MODE%"=="" set "MODE=1"

if "%MODE%"=="0" exit /b 0
if "%MODE%"=="2" goto BOT
if "%MODE%"=="3" goto RL
goto FULL

:FULL
echo.
echo [START] Full launch...

if exist rl_headless_worker.py (
    echo [START] RL worker in separate window...
    start "RL Worker" cmd /k "%PYTHON% rl_headless_worker.py >> logs\rl_worker.log 2>&1"
) else (
    echo [WARN] rl_headless_worker.py not found, skipping RL worker
)

timeout /t 2 /nobreak >nul
goto BOT

:BOT
echo.
echo [START] Bot...
echo Log: logs\bot.log
echo.

%PYTHON% bot.py >> logs\bot.log 2>&1
set "ERR=%errorlevel%"

echo.
echo Exit code: %ERR%
echo Check log file: logs\bot.log
pause
exit /b %ERR%

:RL
echo.
if not exist rl_headless_worker.py (
    echo [ERROR] rl_headless_worker.py not found
    pause
    exit /b 1
)

echo [START] RL worker...
echo Log: logs\rl_worker.log
echo.

%PYTHON% rl_headless_worker.py >> logs\rl_worker.log 2>&1
set "ERR=%errorlevel%"

echo.
echo Exit code: %ERR%
echo Check log file: logs\rl_worker.log
pause
exit /b %ERR%
