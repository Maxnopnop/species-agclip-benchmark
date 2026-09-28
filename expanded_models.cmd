@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe prepare_expanded.py
if errorlevel 1 goto done
.venv\Scripts\python.exe expanded_benchmark.py all
if errorlevel 1 goto done
.venv\Scripts\python.exe report_expanded.py
:done
pause
