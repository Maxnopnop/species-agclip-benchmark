@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0project.ps1" -Action predict -ImagePath "%~1"
set "projectExit=%ERRORLEVEL%"
pause
exit /b %projectExit%
