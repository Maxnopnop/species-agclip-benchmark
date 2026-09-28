@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -u make_pilot.py
if errorlevel 1 goto end
"%~dp0.venv\Scripts\python.exe" -u benchmark.py all
:end
set "projectExit=%ERRORLEVEL%"
pause
exit /b %projectExit%
