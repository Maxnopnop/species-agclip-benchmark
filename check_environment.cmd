@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0project.ps1" -Action doctor
set "projectExit=%ERRORLEVEL%"
pause
exit /b %projectExit%
