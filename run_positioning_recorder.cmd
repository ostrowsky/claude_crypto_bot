@echo off
REM Positioning recorder: one snapshot, then resolve whatever has come due.
REM
REM Scheduled every 2 hours. Binance serves open interest, taker flow and
REM long/short positioning for 30 DAYS ONLY -- anything not written down today is
REM gone for good, and 30 days is far too short to tell a real relationship from
REM a coincidence. This task exists to build the window the API will never give
REM back.
REM
REM snapshot first, resolve second, deliberately: resolve rewrites the whole
REM store, so letting it run on a file the snapshot has just appended to keeps
REM both operating on one consistent state.
REM
REM Output is appended, never truncated -- the log is the record of whether this
REM actually ran, which is the failure mode section 5 of CLAUDE.md was written
REM about.

setlocal
set ROOT=D:\Projects\claude_crypto_bot
set PY=%ROOT%\pyembed\python.exe
set LOG=%ROOT%\.runtime\positioning_recorder.log

cd /d "%ROOT%\files"

echo. >> "%LOG%"
echo ==== %DATE% %TIME% snapshot ==== >> "%LOG%"
"%PY%" positioning_recorder.py snapshot >> "%LOG%" 2>&1
if errorlevel 1 echo SNAPSHOT FAILED with errorlevel %errorlevel% >> "%LOG%"

echo ---- %DATE% %TIME% resolve ---- >> "%LOG%"
"%PY%" positioning_recorder.py resolve --horizon 8 >> "%LOG%" 2>&1
if errorlevel 1 echo RESOLVE FAILED with errorlevel %errorlevel% >> "%LOG%"

endlocal
exit /b 0
