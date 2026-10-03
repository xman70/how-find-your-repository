@echo off
REM Registers a Windows Task Scheduler job that runs the headless analysis every weekday at 18:45.
REM The application itself skips exchange holidays (no fake sessions). Remove with:
REM   schtasks /Delete /TN "VUSA AI Daily Analysis" /F
cd /d "%~dp0\.."
set APPDIR=%cd%
schtasks /Create /TN "VUSA AI Daily Analysis" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 18:45 ^
  /TR "\"%APPDIR%\.venv\Scripts\python.exe\" \"%APPDIR%\main.py\" --cli --mode balanced" /F
if errorlevel 1 (echo Could not create the task. Try running as administrator.) else (echo Task registered.)
pause
