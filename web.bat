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

REM Default: hide the console (server auto-exits when the page closes).
REM Pass --console to keep the terminal visible for debugging.
if /I "%~1"=="--console" goto :run
if /I "%~1"=="--hidden" goto :run

set "VBS=%TEMP%\mewj_web_hide.vbs"
(
    echo Set sh = CreateObject("WScript.Shell"^)
    echo sh.Run "cmd /c """"%~f0"""" --hidden", 0, False
) > "%VBS%"
wscript //nologo "%VBS%"
del "%VBS%" >nul 2>nul
exit /b 0

:run
set "MEWJ_WEB_HOST=127.0.0.1"
if not defined MEWJ_WEB_PORT set "MEWJ_WEB_PORT=8765"

if /I "%~1"=="--console" (
    echo Starting MewJ Web on http://%MEWJ_WEB_HOST%:%MEWJ_WEB_PORT%/ ...
    echo Close the page or finish a review to exit.
    echo.
)

python "%~dp0web.py"
set ERR=%ERRORLEVEL%
if not %ERR%==0 (
    if /I "%~1"=="--console" (
        echo.
        echo [ERROR] exit code %ERR%
        pause
    )
    exit /b %ERR%
)
exit /b 0
