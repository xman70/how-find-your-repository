@echo off
REM VUSA AI Quant Terminal - start the GUI (pass --cli / --daemon / --demo for other modes)
cd /d "%~dp0"
if not exist .venv (
  echo First run: installing...
  call install.bat
)
call .venv\Scripts\activate.bat
python main.py %*
if errorlevel 1 pause
