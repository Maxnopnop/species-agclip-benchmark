@echo off
echo Downloading official image archives: approximately 50 GB total.
echo Partial downloads will resume when this script is run again.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0project.ps1" -Action download
set "projectExit=%ERRORLEVEL%"
pause
exit /b %projectExit%
