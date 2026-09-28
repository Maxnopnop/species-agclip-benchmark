@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
.venv\Scripts\python.exe download_cub.py
if errorlevel 1 goto done
.venv\Scripts\python.exe verify_cub.py
if errorlevel 1 goto done
.venv\Scripts\python.exe audit_cub.py
if errorlevel 1 goto done
.venv\Scripts\python.exe probe_cub.py train
if errorlevel 1 goto done
.venv\Scripts\python.exe cub_experiment.py all
if errorlevel 1 goto done
.venv\Scripts\python.exe probe_cub.py evaluate
if errorlevel 1 goto done
.venv\Scripts\python.exe verify_cub_deployment.py
if errorlevel 1 goto done
.venv\Scripts\python.exe diagnose_cub.py
if errorlevel 1 goto done
.venv\Scripts\python.exe compare_cub_attributes.py
if errorlevel 1 goto done
.venv\Scripts\python.exe report_cub.py
:done
pause
