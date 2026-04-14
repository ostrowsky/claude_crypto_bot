@echo off
setlocal

cd /d "%~dp0"
set "ROOT=%~dp0"
set "PY=%ROOT%pyembed\python.exe"
set "SUPERVISOR=%ROOT%files\local_exec_supervisor.py"
set "REQUEST_JOB=%ROOT%files\request_local_job.py"

if not exist "%PY%" (
  echo ERROR: Python not found: "%PY%"
  exit /b 1
)

echo [1/6] Restarting local Codex supervisor...
call "%ROOT%stop_codex_exec_supervisor.bat" >nul 2>nul
start "GPT Crypto Codex Supervisor" "%PY%" "%SUPERVISOR%" --poll-sec 5 --log-level INFO
timeout /t 3 /nobreak >nul

echo [2/6] Checking CatBoost in project Python...
"%PY%" -c "import sys; import six, catboost, dateutil, pandas"
if errorlevel 1 (
  echo CatBoost runtime is missing or broken. Installing dependency stack and reinstalling catboost...
  "%PY%" -m pip install --upgrade --force-reinstall six python-dateutil pytz tzdata pandas catboost
  if errorlevel 1 (
    echo ERROR: CatBoost dependency installation failed.
    exit /b 1
  )
) else (
  echo CatBoost import is healthy.
)

echo [3/6] Training CatBoost-aware ranker...
"%PY%" "%ROOT%files\train_candidate_ranker.py" --require-catboost
if errorlevel 1 (
  echo ERROR: Ranker training failed.
  exit /b 1
)

echo [4/6] Queueing trading bot start...
"%PY%" "%REQUEST_JOB%" run_trade_bot
if errorlevel 1 (
  echo ERROR: Failed to queue trade bot start.
  exit /b 1
)

echo [5/6] Queueing RL headless worker start...
"%PY%" "%REQUEST_JOB%" run_rl_headless
if errorlevel 1 (
  echo ERROR: Failed to queue RL worker start.
  exit /b 1
)

echo [6/6] Waiting for processes to initialize...
timeout /t 8 /nobreak >nul

echo.
echo === Trade Bot Status ===
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ROOT%bot_status.ps1"

echo.
echo === RL Worker Status ===
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ROOT%rl_worker_status.ps1"

echo.
echo Pipeline launch completed.
echo If this was the first CatBoost install, the setup may take a little longer than usual.

endlocal
