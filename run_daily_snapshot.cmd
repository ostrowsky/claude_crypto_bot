@echo off
cd /d D:\Projects\claude_crypto_bot\files
D:\Projects\claude_crypto_bot\pyembed\python.exe daily_learning.py --snapshot >> ..\bot_learning.log 2>&1
