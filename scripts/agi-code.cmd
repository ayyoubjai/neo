@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_PATH=%SCRIPT_DIR%agi_code.py"

if defined PYTHON (
  "%PYTHON%" "%SCRIPT_PATH%" %*
  exit /b %ERRORLEVEL%
)

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
  python "%SCRIPT_PATH%" %*
  exit /b %ERRORLEVEL%
)

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
  py "%SCRIPT_PATH%" %*
  exit /b %ERRORLEVEL%
)

echo [agi-code] Python not found. Set PYTHON or install Python. 1>&2
exit /b 1
