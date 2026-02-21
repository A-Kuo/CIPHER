@echo off
REM -----------------------------------------------------------------------------
REM Run vision backend with crack detection enabled — PLACEHOLDER (Windows)
REM Edit the variables below; run from repo root.
REM -----------------------------------------------------------------------------

set REPO_ROOT=%~dp0..
cd /d "%REPO_ROOT%"

REM ----- EDIT ME -----
set VENV_PATH=%REPO_ROOT%\.venv
set PORT=8000
REM Optional: set to 0 to disable crack without changing code
set YOLO_CRACK_ENABLED=1
REM -------------------

if exist "%VENV_PATH%\Scripts\activate.bat" call "%VENV_PATH%\Scripts\activate.bat"

echo Starting backend (crack enabled=%YOLO_CRACK_ENABLED%) on port %PORT%...
set PHANTOM_HTTP_ONLY=1
python backend/main.py
