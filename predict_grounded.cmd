@echo off
cd /d "%~dp0"
if "%~1"=="" (
  echo Drag an image onto this file, or run: predict_grounded.cmd "E:\path\photo.jpg"
) else (
  .venv\Scripts\python.exe predict_grounded.py "%~1" --output work\grounded_prediction.json
)
pause
