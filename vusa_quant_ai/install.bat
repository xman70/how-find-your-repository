@echo off
REM ============================================================================
REM  VUSA AI Quant Terminal - one-time installer (Windows, Python 3.10+)
REM ============================================================================
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (set PY=py -3) else (set PY=python)

%PY% -c "import sys; assert sys.version_info >= (3,10), sys.version" 2>nul
if errorlevel 1 (
  echo Python 3.10 or newer is required. Install it from https://www.python.org/downloads/ and tick "Add to PATH".
  pause
  exit /b 1
)

if not exist .venv (
  echo Creating virtual environment...
  %PY% -m venv .venv || goto :fail
)
call .venv\Scripts\activate.bat || goto :fail

python -m pip install --upgrade pip wheel
echo Installing core requirements...
pip install -r requirements.txt || goto :fail

echo Installing PyTorch (CPU build - deep-learning models; skip-safe)...
pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 echo [WARN] PyTorch could not be installed. Deep-learning models will be reported as unavailable.

if not exist .env copy .env.example .env >nul

echo.
echo Running a quick self-test (synthetic data, no network needed)...
python -m pytest tests\test_data_quality.py tests\test_leakage.py -q
echo.
echo Installation complete. Start the application with run.bat
pause
exit /b 0

:fail
echo Installation failed. See the messages above.
pause
exit /b 1
