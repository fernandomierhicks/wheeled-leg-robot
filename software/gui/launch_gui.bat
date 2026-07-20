@echo off
cd /d "%~dp0"
"C:\Users\ferna\.platformio\penv\Scripts\python.exe" main.py > launch_log.txt 2>&1
echo Exit code: %errorlevel% >> launch_log.txt
if errorlevel 1 pause
