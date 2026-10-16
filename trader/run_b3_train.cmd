@echo off
REM Demo TRADER phase B3 (wide-universe-b-1014-spec.md 4b): candidates -> lazy grid -> TRAIN only (2023-2025).
REM Waits for the history backfill to finish (BACKFILL_SUMMARY.json). The TEST half is NOT opened here.
set ROOT=D:\Projects\claude_crypto_bot
set PY=%ROOT%\pyembed\python.exe
set LOG=%ROOT%\trader\runtime\experiments\b3_train.log
cd /d "%ROOT%\trader"
if not exist "%ROOT%\trader\runtime\experiments" mkdir "%ROOT%\trader\runtime\experiments"
echo ==== %DATE% %TIME% waiting for backfill ==== >> "%LOG%"
:wait
if exist "%ROOT%\trader\runtime\history\BACKFILL_SUMMARY.json" goto go
ping -n 601 127.0.0.1 >nul
goto wait
:go
echo ==== %DATE% %TIME% backfill top-up (pairs added to the universe later, e.g. PAXG; newest bars) ==== >> "%LOG%"
"%PY%" -u history_store.py backfill --since 2023-01-01 --workers 4 >> "%LOG%" 2>&1
echo ==== %DATE% %TIME% gen ==== >> "%LOG%"
"%PY%" -u experiment_b3.py gen --processes 3 >> "%LOG%" 2>&1
echo ==== %DATE% %TIME% grid ==== >> "%LOG%"
"%PY%" -u experiment_b3.py grid >> "%LOG%" 2>&1
echo ==== %DATE% %TIME% train ==== >> "%LOG%"
"%PY%" -u experiment_b3.py train >> "%LOG%" 2>&1
echo ==== %DATE% %TIME% done ==== >> "%LOG%"
