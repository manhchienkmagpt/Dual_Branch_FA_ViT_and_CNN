@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_and_run.ps1" %*
set "TASK_EXIT_CODE=%ERRORLEVEL%"
echo.
pause
exit /b %TASK_EXIT_CODE%
