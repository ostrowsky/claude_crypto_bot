@echo off
REM Demo TRADER (separate app, trader/) -- E7 shadow portfolios step, every 5 minutes.
REM e7-shadow-portfolio-1012-spec.md. Paper only: reads the bot's event log and public klines,
REM no key, no order. Output is appended, never truncated.

setlocal
set ROOT=D:\Projects\claude_crypto_bot
set PY=%ROOT%\pyembed\python.exe
set LOG=%ROOT%\trader\runtime\shadow\shadow.log

cd /d "%ROOT%\trader"
if not exist "%ROOT%\trader\runtime\shadow" mkdir "%ROOT%\trader\runtime\shadow"

echo ==== %DATE% %TIME% ==== >> "%LOG%"
"%PY%" shadow_portfolio.py step >> "%LOG%" 2>&1
if errorlevel 1 echo SHADOW STEP FAILED with errorlevel %errorlevel% >> "%LOG%"
"%PY%" shadow_readout.py >> "%LOG%" 2>&1
echo ---- %DATE% %TIME% done ---- >> "%LOG%"

endlocal
exit /b 0
