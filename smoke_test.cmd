@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0project.ps1" -Action smoke
set "projectExit=%ERRORLEVEL%"
pause
exit /b %projectExit%
