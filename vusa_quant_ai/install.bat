@echo off
REM ============================================================================
REM  VUSA AI Quant Terminal - installer (Windows, 64-bit Python 3.10 - 3.14)
REM  Finds a suitable Python, then bootstrap.py does the rest:
REM   - creates the environment at a short path (C:\ProgramData\VUSA-Quant\venv)
REM   - installs essential packages, then optional ones (skipped if unavailable)
REM   - repairs a broken environment from an earlier attempt automatically
REM ============================================================================
setlocal EnableExtensions
cd /d "%~dp0"
title VUSA AI Quant Terminal - installer
echo.
echo  VUSA AI Quant Terminal - installation
echo  =====================================
echo.
call :find_python
if not defined PYEXE goto :no_python
%PYEXE% "%~dp0bootstrap.py" --install
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" echo  Start the application with run.bat
pause
exit /b %RC%

REM ---------------------------------------------------------------------------
:find_python
REM Prefer 64-bit 3.12 (best tested), then 3.13, 3.11, 3.10, 3.14.
set "PYEXE="
set "PYCHECK=import sys,struct; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,14) and struct.calcsize('P') == 8 else 1)"
for %%V in (3.12 3.13 3.11 3.10 3.14) do (
  if not defined PYEXE (
    py -%%V -c "%PYCHECK%" >nul 2>nul && set "PYEXE=py -%%V"
  )
)
if not defined PYEXE (
  python -c "%PYCHECK%" >nul 2>nul && set "PYEXE=python"
)
exit /b 0

:no_python
echo  [ERROR] No 64-bit Python 3.10 - 3.14 was found.
echo          Install Python 3.12 (64-bit) from https://www.python.org/downloads/windows/
echo          and tick "Add python.exe to PATH" during setup. Then run this file again.
echo.
pause
exit /b 1
