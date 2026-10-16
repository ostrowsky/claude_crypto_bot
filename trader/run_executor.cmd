@echo off
REM Demo TRADER (separate app, trader/) -- phase C executor, long-running.
REM demo-execution-c-1013-spec.md. TRADING_ENABLED=False -> C1 paper mode (no order, no key).
REM The task starts this every 5 minutes; a second copy exits at once (lock + cmdline check).

setlocal
set ROOT=D:\Projects\claude_crypto_bot
set PY=%ROOT%\pyembed\python.exe
set LOG=%ROOT%\trader\runtime\executor\executor.log

cd /d "%ROOT%\trader"
if not exist "%ROOT%\trader\runtime\executor" mkdir "%ROOT%\trader\runtime\executor"

echo ==== %DATE% %TIME% start ==== >> "%LOG%"
"%PY%" -u executor.py run >> "%LOG%" 2>&1
echo ---- %DATE% %TIME% exit %errorlevel% ---- >> "%LOG%"

endlocal
exit /b 0
