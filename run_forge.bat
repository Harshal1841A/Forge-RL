@echo off
echo =======================================================
echo     FORGE-MA Forensic Platform - Local Startup
echo =======================================================
echo.

echo [0/2] Clearing existing processes on ports 7860 and 3000...
powershell -Command "Stop-Process -Id (Get-NetTCPConnection -LocalPort 7860 -ErrorAction SilentlyContinue).OwningProcess -Force -ErrorAction SilentlyContinue"
powershell -Command "Stop-Process -Id (Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue).OwningProcess -Force -ErrorAction SilentlyContinue"

echo [1/2] Starting FORGE-MA Backend Server (Port 7860)...
start cmd /k "py -m uvicorn server.main:app --host 0.0.0.0 --port 7860"

echo [2/2] Starting FORGE-MA Frontend (Next.js - Port 3000)...
cd spatial-saas
start cmd /k "npm run dev"


echo.
echo =======================================================
echo     Startup Complete!
echo =======================================================
echo Dashboard will be available at: http://localhost:3000
echo Server API running at:        http://localhost:7860
echo.
pause
