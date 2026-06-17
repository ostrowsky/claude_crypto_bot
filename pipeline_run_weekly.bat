@echo off
REM Weekly pipeline: runs pipeline_run.py --weekly = daily steps + L2 hypothesis
REM generation + L3 validation of the pending queue. Without this, L2 never fires
REM and no new hypotheses are generated (the loop's generation half goes dormant).
setlocal
set "REPO=D:\Projects\claude_crypto_bot"
set "LOG=%REPO%\.runtime\pipeline\cron-weekly.log"

cd /d "%REPO%"

echo. >> "%LOG%"
echo === %DATE% %TIME%  pipeline_run.py --weekly === >> "%LOG%"
"%REPO%\pyembed\python.exe" "%REPO%\files\pipeline_run.py" --weekly >> "%LOG%" 2>&1
echo === exit code %ERRORLEVEL% === >> "%LOG%"
endlocal
