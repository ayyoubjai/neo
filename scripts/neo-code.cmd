@echo off
setlocal
set "SCRIPT_PATH=%~dp0neo_code.py"
if defined PYTHON goto run_configured
where python >nul 2>nul
if not errorlevel 1 goto run_python
where py >nul 2>nul
if not errorlevel 1 goto run_py
echo [neo-code] Python not found. Set PYTHON or install Python. 1>&2
exit /b 1
:run_configured
"%PYTHON%" "%SCRIPT_PATH%" %*
exit /b %ERRORLEVEL%
:run_python
python "%SCRIPT_PATH%" %*
exit /b %ERRORLEVEL%
:run_py
py "%SCRIPT_PATH%" %*
exit /b %ERRORLEVEL%
