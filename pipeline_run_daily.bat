@echo off
REM Daily pipeline runner — invoked by Task Scheduler "CryptoBot_Pipeline_Daily".
REM
REM Runs L1 + L4 + L5 + L7 + attribution. Output appended to a rolling log so
REM we can audit late failures without needing the Task Scheduler history GUI.
REM
REM Manage:
REM   schtasks /Query /TN "CryptoBot_Pipeline_Daily" /V /FO LIST
REM   schtasks /Run   /TN "CryptoBot_Pipeline_Daily"   ( run now, for testing )
REM   schtasks /Delete /TN "CryptoBot_Pipeline_Daily" /F   ( unschedule )

setlocal
set "REPO=D:\Projects\claude_crypto_bot"
set "LOG=%REPO%\.runtime\pipeline\cron-daily.log"

cd /d "%REPO%"

echo. >> "%LOG%"
echo === %DATE% %TIME%  pipeline_run.py --daily === >> "%LOG%"
"%REPO%\pyembed\python.exe" "%REPO%\files\pipeline_run.py" --daily >> "%LOG%" 2>&1
echo === exit code %ERRORLEVEL% === >> "%LOG%"
endlocal
