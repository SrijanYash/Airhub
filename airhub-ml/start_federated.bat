@echo off
REM Federated Learning Startup Script for Windows
REM Usage: start_federated.bat [server|client|all] [options]

setlocal enabledelayedexpansion

cd /d "%~dp0"

if "%1"=="" goto :help
if "%1"=="server" goto :start_server
if "%1"=="client" goto :start_client
if "%1"=="all" goto :start_all
goto :help

:start_server
echo ============================================
echo Starting Flower Server (Persistent Mode)
echo ============================================
echo.
python -m federated.server_node --persistent %2 %3 %4
goto :end

:start_client
echo ============================================
echo Starting Flower Client
echo ============================================
echo.
python -m federated.client_node --city %DEFAULT_CITY% --country %DEFAULT_COUNTRY% --persistent --wait-timeout 120 %2 %3 %4
goto :end

:start_all
echo ============================================
echo Starting Federated Learning (Server + Client)
echo ============================================
echo.
echo [1/3] Starting server in background...
start "Flower Server" cmd /k "cd /d %cd% && python -m federated.server_node --persistent"

echo [2/3] Waiting for server to be ready...
timeout /t 10 /nobreak >nul

echo [3/3] Starting client...
python -m federated.client_node --city Delhi --country IN --persistent --wait-timeout 120

echo.
echo ============================================
echo Both server and client are running
echo Close this window to stop the client
echo The server runs in a separate window
echo ============================================
goto :end

:help
echo Usage: start_federated.bat [command] [options]
echo.
echo Commands:
echo   server    - Start only the Flower server (persistent mode)
echo   client    - Start only the Flower client (persistent mode)
echo   all       - Start both server and client
echo.
echo Examples:
echo   start_federated.bat server
echo   start_federated.bat client
echo   start_federated.bat all
echo.
echo For manual control, open TWO terminals:
echo   Terminal 1: python -m federated.server_node --persistent
echo   Terminal 2: python -m federated.client_node --city Delhi --country IN --persistent
goto :end

:end
endlocal
