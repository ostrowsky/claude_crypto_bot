@echo off
REM Demo TRADER phase B1: 15m history of the whole spot universe (wide-universe-b-1014-spec.md).
REM Public klines only, no key. Resumable: rerun continues after the last stored bar.
cd /d "D:\Projects\claude_crypto_bot\trader"
if not exist "D:\Projects\claude_crypto_bot\trader\runtime\history" mkdir "D:\Projects\claude_crypto_bot\trader\runtime\history"
"D:\Projects\claude_crypto_bot\pyembed\python.exe" -u history_store.py backfill --since 2023-01-01 --workers 4 >> "D:\Projects\claude_crypto_bot\trader\runtime\history\backfill.log" 2>&1
