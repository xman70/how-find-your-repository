@echo off
REM VUSA AI Quant Terminal - start the application.
REM Installs automatically on first start (or repairs an unfinished installation).
REM Extra arguments are passed on, e.g.:  run.bat --cli --demo
setlocal EnableExtensions
cd /d "%~dp0"
title VUSA AI Quant Terminal
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
if not defined PYEXE (
  echo  [ERROR] No 64-bit Python 3.10 - 3.14 was found.
  echo          Install Python 3.12 64-bit from https://www.python.org/downloads/windows/
  echo          and tick "Add python.exe to PATH". Then double-click run.bat again.
  pause
  exit /b 1
)
%PYEXE% "%~dp0bootstrap.py" --run %*
if errorlevel 1 pause
