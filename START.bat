@echo off
REM ==========================================================================
REM  AI Text Analysis - Windows launcher
REM  1. checks Python  2. creates a virtual environment  3. installs dependencies
REM  4. verifies the installation  5. starts the application (local only)
REM  Usage:  START.bat          (CPU, default "lite" profile)
REM          START.bat full     (also installs optional torch/transformers)
REM ==========================================================================
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
title AI Text Analysis

echo.
echo  AI Text Analysis - local, probabilistic AI-text estimation
echo  Privacy mode: Local processing. Documents never leave this computer.
echo.

REM ---------------------------------------------------------------- 1. Python
set "PY="
for %%V in (3.12 3.11 3.10 3.13) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo [ERROR] Python 3.10 - 3.13 was not found.
    echo         Install Python 3.12 from https://www.python.org/downloads/windows/
    echo         and tick "Add python.exe to PATH" during installation, then run START.bat again.
    start "" "https://www.python.org/downloads/windows/"
    pause
    exit /b 1
)
for /f "delims=" %%i in ('%PY% -c "import sys; print(sys.version.split()[0])"') do set "PYVER=%%i"
echo [1/5] Python found: %PY% (version !PYVER!)

REM ---------------------------------------------------------------- 2. venv
set "VPY=%~dp0.venv\Scripts\python.exe"
if not exist "%VPY%" (
    echo [2/5] Creating virtual environment in .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [2/5] Virtual environment found.
)

REM ---------------------------------------------------------------- 3. dependencies
set "NEED_INSTALL=1"
if exist ".venv\requirements.installed" (
    fc /b requirements.txt ".venv\requirements.installed" >nul 2>&1 && set "NEED_INSTALL=0"
)
if /i "%~1"=="full" set "NEED_INSTALL=1"
if "%NEED_INSTALL%"=="1" (
    echo [3/5] Installing dependencies - this can take several minutes the first time ...
    "%VPY%" -m pip install --upgrade pip >nul
    "%VPY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed. Check your internet connection and try again.
        pause
        exit /b 1
    )
    "%VPY%" -m pip install -e . --no-deps >nul
    if errorlevel 1 (
        echo [ERROR] Could not install the application package.
        pause
        exit /b 1
    )
    if /i "%~1"=="full" (
        echo       Installing optional neural components ^(torch, transformers^) ...
        "%VPY%" -m pip install -r requirements-optional.txt
    )
    copy /y requirements.txt ".venv\requirements.installed" >nul
) else (
    echo [3/5] Dependencies already installed.
)

REM ---------------------------------------------------------------- 4. verify
echo [4/5] Verifying installation ...
"%VPY%" -m aidetect.verify
if errorlevel 1 (
    echo [ERROR] The installation check failed. Deleting the .venv folder and running START.bat again usually fixes it.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------- 5. start
echo [5/5] Starting the application. Your browser will open automatically.
echo       Close this window or press Ctrl+C to stop.
"%VPY%" app.py
if errorlevel 1 pause
endlocal
