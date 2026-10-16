@echo off
REM Demo TRADER (separate app, trader/) -- phase A0 + equity, every 5 minutes (Task Scheduler).
REM demo-connection-a-1010-spec.md. Public market data + one read-only account call; never an order.
REM Equity first (7 s): the microstructure pass takes ~2 min of network waits and must not
REM starve it. Output is appended, never truncated.

setlocal
set ROOT=D:\Projects\claude_crypto_bot
set PY=%ROOT%\pyembed\python.exe
set LOG=%ROOT%\trader\runtime\recorder.log

cd /d "%ROOT%\trader"

echo ==== %DATE% %TIME% ==== >> "%LOG%"
"%PY%" equity_snapshots.py >> "%LOG%" 2>&1
if errorlevel 1 echo EQUITY SNAPSHOT FAILED with errorlevel %errorlevel% >> "%LOG%"
"%PY%" microstructure_recorder.py snapshot >> "%LOG%" 2>&1
if errorlevel 1 echo MICROSTRUCTURE SNAPSHOT FAILED with errorlevel %errorlevel% >> "%LOG%"
echo ---- %DATE% %TIME% done ---- >> "%LOG%"

endlocal
exit /b 0
