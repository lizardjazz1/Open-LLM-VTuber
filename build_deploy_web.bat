@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Optional flag: /nopause to skip pause at the end when running from an open terminal
if /i "%~1"=="/nopause" set NOPAUSE=1

REM Resolve repo root (folder where the script is located)
set "ROOT=%~dp0"

echo [1/4] Go to web project
pushd "%ROOT%externals\Open-LLM-VTuber-Web" || goto :err

echo [2/4] Install deps (npm ci or npm install)
if exist package-lock.json (
  call npm ci || goto :err
) else (
  call npm install || goto :err
)

echo [3/4] Build web
call npm run build:web || goto :err

echo [4/4] Deploy to frontend
robocopy "%ROOT%externals\Open-LLM-VTuber-Web\dist\web" "%ROOT%frontend" /E /NFL /NDL /NJH /NJS
set RC=%ERRORLEVEL%
if %RC% GEQ 8 goto :err

popd >nul

echo.
echo SUCCESS: Build and deploy completed.
echo Target: "%ROOT%frontend"
if not defined NOPAUSE (
  echo.
  echo Press any key to close...
  pause >nul
)
exit /b 0

:err
popd >nul 2>&1
>&2 echo.
>&2 echo ERROR: Build or deploy failed. See the log above.
if not defined NOPAUSE (
  echo.
  echo Press any key to close...
  pause >nul
)
exit /b 1 