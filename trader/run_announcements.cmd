@echo off
REM Demo TRADER (separate app, trader/) -- LA exchange announcements, every 5 minutes.
REM listing-announcements-la-1028-spec.md. Public read-only GET (Binance CMS, OKX, Bybit), no key, no order.
REM Task CryptoBot_TraderAnnouncements, limit 5 min, IgnoreNew. Rollback: trader_config.ANNOUNCEMENTS_ENABLED = False.

setlocal
set ROOT=D:\Projects\claude_crypto_bot
set PY=%ROOT%\pyembed\python.exe
set LOG=%ROOT%\trader\runtime\announcements\announcements.log

cd /d "%ROOT%\trader"
if not exist "%ROOT%\trader\runtime\announcements" mkdir "%ROOT%\trader\runtime\announcements"

echo ==== %DATE% %TIME% ==== >> "%LOG%"
"%PY%" announcements.py step >> "%LOG%" 2>&1
if errorlevel 1 echo ANNOUNCEMENTS STEP FAILED with errorlevel %errorlevel% >> "%LOG%"

endlocal
exit /b 0
