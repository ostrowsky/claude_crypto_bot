@echo off
REM ---------------------------------------------------------------------------
REM kickoff_data_collection.bat
REM
REM Launched DETACHED by restart_bot.bat so hypothesis-validation data starts
REM flowing immediately instead of waiting for the next Task Scheduler slot.
REM
REM Chain (each step independent; a failure does not abort the rest):
REM   1. Intraday feature snapshot  -> top_gainer_dataset.jsonl
REM   2. Klines backfill            -> price history for label resolution
REM   3. Full L0..L7 pipeline       -> metrics_daily, health, attribution,
REM                                    drift, blind critic, notify
REM
REM All output appended to a rolling log so late failures are auditable.
REM Manual run / test:  D:\Projects\claude_crypto_bot\kickoff_data_collection.bat
REM ---------------------------------------------------------------------------
setlocal
set "ROOT=D:\Projects\claude_crypto_bot"
set "PY=%ROOT%\pyembed\python.exe"
set "FILES=%ROOT%\files"
set "LOG=%ROOT%\.runtime\pipeline\restart-kickoff.log"

if not exist "%ROOT%\.runtime\pipeline" mkdir "%ROOT%\.runtime\pipeline" >/dev/null 2>&1

echo. >> "%LOG%"
echo ============================================================ >> "%LOG%"
echo === kickoff_data_collection %DATE% %TIME% === >> "%LOG%"
echo ============================================================ >> "%LOG%"

cd /d "%ROOT%"

echo [kickoff] 1/3 intraday snapshot... >> "%LOG%"
"%PY%" "%FILES%\daily_learning.py" --snapshot >> "%LOG%" 2>&1
echo [kickoff]   snapshot exit=%ERRORLEVEL% >> "%LOG%"

echo [kickoff] 2/3 klines backfill... >> "%LOG%"
"%PY%" "%FILES%\_backfill_klines_history.py" >> "%LOG%" 2>&1
echo [kickoff]   klines exit=%ERRORLEVEL% >> "%LOG%"

echo [kickoff] 3/3 full pipeline (L0..L7 + notify)... >> "%LOG%"
"%PY%" "%FILES%\pipeline_run.py" --daily >> "%LOG%" 2>&1
echo [kickoff]   pipeline exit=%ERRORLEVEL% >> "%LOG%"

echo [kickoff] DONE %DATE% %TIME% >> "%LOG%"
endlocal
