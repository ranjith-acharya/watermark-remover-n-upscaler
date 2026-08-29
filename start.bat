@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv || goto :fail
  .venv\Scripts\python.exe -m pip install -q --upgrade pip
  .venv\Scripts\python.exe -m pip install -q -r requirements.txt || goto :fail
)
.venv\Scripts\python.exe -m flowclean %*
goto :eof
:fail
echo.
echo Setup failed. Check that Python 3.10+ and ffmpeg are installed and on PATH.
pause
