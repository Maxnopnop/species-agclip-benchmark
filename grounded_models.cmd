@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe prepare_grounded.py --audit
if errorlevel 1 goto done
.venv\Scripts\python.exe verify_grounded.py smoke
if errorlevel 1 goto done
.venv\Scripts\python.exe grounded_experiment.py all
if errorlevel 1 goto done
.venv\Scripts\python.exe verify_grounded.py deployment
if errorlevel 1 goto done
.venv\Scripts\python.exe report_grounded.py
:done
pause
