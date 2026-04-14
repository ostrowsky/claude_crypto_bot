@echo off
cd /d "%~dp0"
start "GPT Crypto Codex Supervisor" "%~dp0pyembed\python.exe" "%~dp0files\local_exec_supervisor.py" --poll-sec 5 --log-level INFO
