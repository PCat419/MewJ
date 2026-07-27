@echo off
cd /d "%~dp0"
title MewJ Web

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python and add it to PATH.
    pause
    exit /b 1
)

if not exist ".env" (
    echo First run: creating .env
    copy /y ".env.example" ".env" >nul
    if exist "..\tensoul\.env" (
        echo Found ..\tensoul\.env - you can copy account into MewJ\.env
    )
    notepad ".env"
    echo.
    echo Save .env, then run web.bat again.
    pause
    exit /b 0
)

set "MEWJ_WEB_HOST=127.0.0.1"
if not defined MEWJ_WEB_PORT set "MEWJ_WEB_PORT=8765"

echo Starting MewJ Web on http://%MEWJ_WEB_HOST%:%MEWJ_WEB_PORT%/ ...
echo Close the page or finish a review to exit.
echo.

python "%~dp0web.py"
set ERR=%ERRORLEVEL%
if not %ERR%==0 (
    echo.
    echo [ERROR] exit code %ERR%
    pause
    exit /b %ERR%
)
exit /b 0
