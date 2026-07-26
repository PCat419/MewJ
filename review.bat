@echo off
cd /d "%~dp0"
title MewJ Review

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
    echo Save .env, then run review.bat again.
    pause
    exit /b 0
)

set /p SOURCE=Paipu UUID or Majsoul link: 
if "%SOURCE%"=="" (
    echo Empty input.
    pause
    exit /b 1
)

set /p SEAT=Seat 0-3 [auto from URL]: 
set "SEAT_ARGS="
if not "%SEAT%"=="" set "SEAT_ARGS=--seat %SEAT%"

set "NANIKIRU_PORT=50000"
set "STARTED_NANIKIRU=0"

rem Locate nanikiru.exe: env var > MewJ/engine/ > repo-level mahjong-cpp build
if defined MEWJ_NANIKIRU_EXE (
    set "NANIKIRU=%MEWJ_NANIKIRU_EXE%"
) else if exist "%~dp0engine\nanikiru.exe" (
    set "NANIKIRU=%~dp0engine\nanikiru.exe"
) else (
    set "NANIKIRU=%~dp0..\mahjong-cpp\build\src\server\nanikiru.exe"
)
for %%I in ("%NANIKIRU%") do set "NANIKIRU_DIR=%%~dpI"

echo.
echo Checking nanikiru on 127.0.0.1:%NANIKIRU_PORT% ...
python -c "import sys,socket; s=socket.socket(); s.settimeout(2); r=s.connect_ex(('127.0.0.1',int(sys.argv[1]))); s.close(); sys.exit(0 if r==0 else 1)" %NANIKIRU_PORT%
if not errorlevel 1 (
    echo nanikiru is ready ^(already listening on port %NANIKIRU_PORT%^).
    echo   Tip: a previous review.bat may have left it running in the background.
    goto after_nanikiru
)

if not exist "%NANIKIRU%" (
    echo [ERROR] nanikiru.exe not found:
    echo   %NANIKIRU%
    echo Build mahjong-cpp server first.
    pause
    exit /b 1
)

echo nanikiru not running. Starting ...
rem Ensure MSYS2 runtime DLLs (libspdlog etc.) are visible to nanikiru.exe
if exist "C:\msys64\mingw64\bin" set "PATH=C:\msys64\mingw64\bin;%PATH%"
if exist "C:\msys64\ucrt64\bin" set "PATH=C:\msys64\ucrt64\bin;%PATH%"
start "nanikiru" /min /D "%NANIKIRU_DIR%" "%NANIKIRU%" %NANIKIRU_PORT%
set "STARTED_NANIKIRU=1"

set /a _i=0
:wait_nanikiru
set /a _i+=1
ping -n 2 127.0.0.1 >nul
python -c "import sys,socket; s=socket.socket(); s.settimeout(2); r=s.connect_ex(('127.0.0.1',int(sys.argv[1]))); s.close(); sys.exit(0 if r==0 else 1)" %NANIKIRU_PORT%
if not errorlevel 1 goto nanikiru_ok
if %_i% LSS 15 goto wait_nanikiru

echo [ERROR] nanikiru failed to start within ~30s.
echo Check the minimized nanikiru window for error messages.
pause
exit /b 1

:nanikiru_ok
echo nanikiru is ready.
:after_nanikiru
echo.

echo %SOURCE%| findstr /i /c:"http" /c:"paipu=" >nul
if not errorlevel 1 (
    python "%~dp0cli.py" --link "%SOURCE%" %SEAT_ARGS% --open
) else (
    python "%~dp0cli.py" "%SOURCE%" %SEAT_ARGS% --open
)
set ERR=%ERRORLEVEL%
echo.
if not %ERR%==0 echo [ERROR] exit code %ERR%
if "%STARTED_NANIKIRU%"=="1" (
    echo Stopping nanikiru started by this script ...
    taskkill /IM nanikiru.exe /F >nul 2>nul
)
pause
exit /b %ERR%
