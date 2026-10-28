@echo off
REM Demo TRADER (separate app, trader/) -- W1 shadow scanner of the whole spot market, every 5 minutes.
REM wide-scan-shadow-w1-1028-spec.md. Public GET only, no key, no order. Works only when a new 15m bar
REM has closed; the other runs return at once. Task CryptoBot_TraderWideScan, limit 10 min, IgnoreNew.
REM Rollback: trader_config.WIDE_SCAN_ENABLED = False (the step then does nothing).

setlocal
set ROOT=D:\Projects\claude_crypto_bot
set PY=%ROOT%\pyembed\python.exe
set LOG=%ROOT%\trader\runtime\scan\wide_scan.log

cd /d "%ROOT%\trader"
if not exist "%ROOT%\trader\runtime\scan" mkdir "%ROOT%\trader\runtime\scan"

echo ==== %DATE% %TIME% ==== >> "%LOG%"
"%PY%" wide_scan.py step >> "%LOG%" 2>&1
if errorlevel 1 echo WIDE SCAN STEP FAILED with errorlevel %errorlevel% >> "%LOG%"
"%PY%" wide_scan.py readout > nul 2>> "%LOG%"
echo ---- %DATE% %TIME% done ---- >> "%LOG%"

endlocal
exit /b 0
