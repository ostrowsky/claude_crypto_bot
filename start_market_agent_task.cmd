@echo off
setlocal
set PYTHONNOUSERSITE=1
set PYTHONUTF8=
cd /d D:\Projects\gpt_crypto_bot\files
D:\Projects\gpt_crypto_bot\pyembed\python.exe market_signal_agent.py --log-level INFO --poll-sec 15 1>>D:\Projects\gpt_crypto_bot\agent_stdout.log 2>>D:\Projects\gpt_crypto_bot\agent_stderr.log
