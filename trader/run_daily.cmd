@echo off
REM Demo TRADER (separate app, trader/) -- phase D daily: ledger sync, daily revenue, report.
REM demo-ledger-d-1013-spec.md. Runs 00:20 UTC. Reads the demo account (signed, demo host only)
REM and public klines; writes only trader\runtime\. No order is placed.

setlocal
set ROOT=D:\Projects\claude_crypto_bot
set PY=%ROOT%\pyembed\python.exe
set LOG=%ROOT%\trader\runtime\reports\daily.log

cd /d "%ROOT%\trader"
if not exist "%ROOT%\trader\runtime\reports" mkdir "%ROOT%\trader\runtime\reports"

echo ==== %DATE% %TIME% ==== >> "%LOG%"
"%PY%" trade_ledger.py sync >> "%LOG%" 2>&1
if errorlevel 1 echo LEDGER SYNC FAILED with errorlevel %errorlevel% >> "%LOG%"
"%PY%" daily_revenue.py compute >> "%LOG%" 2>&1
if errorlevel 1 echo DAILY REVENUE FAILED with errorlevel %errorlevel% >> "%LOG%"
"%PY%" trader_report.py >> "%LOG%" 2>&1
if errorlevel 1 echo REPORT FAILED with errorlevel %errorlevel% >> "%LOG%"
echo ---- %DATE% %TIME% done ---- >> "%LOG%"

endlocal
exit /b 0
