@echo off
setlocal
title Pickleball Reservation
cd /d "%~dp0"

where node >nul 2>&1
if errorlevel 1 (
  echo.
  echo Node.js is not installed or is not available in PATH.
  echo Install Node.js, then double-click this launcher again.
  echo.
  pause
  exit /b 1
)

if not exist "node_modules\playwright-core\package.json" (
  echo.
  echo First-time setup: installing application dependencies...
  call npm install
  if errorlevel 1 (
    echo.
    echo Dependency installation failed.
    pause
    exit /b 1
  )
)

echo.
echo Starting Pickleball Reservation GUI...
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Pickleball Reservation GUI.ps1"

if errorlevel 1 (
  echo.
  echo The application stopped with an error.
  echo Review the message above before closing this window.
  pause
)

endlocal
