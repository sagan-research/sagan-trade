@echo off
echo ========================================================
echo KCG INFINITE TRADING DAEMON INSTALLER
echo ========================================================
echo Requesting Administrative Privileges to install daily Cron Job...

:: Check for Administrative privileges
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Administrator permissions confirmed.
) else (
    echo Failure: Current permissions inadequate. Please run this .bat as Administrator.
    pause
    exit /b 1
)

set SCRIPT_PATH=C:\Users\91891\.gemini\antigravity-ide\scratch\sagan\infinite_trading_daemon.py
set TASK_NAME=InfiniteTradingDaemon

echo.
echo Installing SchTasks...
echo Task Name: %TASK_NAME%
echo Schedule : Daily at 06:10 AM
echo Action   : python "%SCRIPT_PATH%"

schtasks /Create /TN "%TASK_NAME%" /TR "python \"%SCRIPT_PATH%\"" /SC DAILY /ST 06:10 /RL HIGHEST /F

if %errorLevel% == 0 (
    echo.
    echo [SUCCESS] The Infinite Trading Daemon has been registered successfully!
    echo It will now run completely independently every day from 6:10 AM to 11:00 AM.
) else (
    echo.
    echo [ERROR] Failed to schedule task.
)
