@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Optional flag: /nopause to skip pause at the end when running from an open terminal
if /i "%~1"=="/nopause" set NOPAUSE=1

REM Hardcoded paths (requested)
set "WEB_DIR=F:\Open-LLM-VTuber\Open-LLM-VTuber-Web"
set "FRONTEND_DIR=F:\Open-LLM-VTuber\frontend"

echo Using web project: "%WEB_DIR%"
echo Using frontend target: "%FRONTEND_DIR%"

if not exist "%WEB_DIR%\package.json" (
  >&2 echo.
  >&2 echo ERROR: package.json not found in WEB_DIR: "%WEB_DIR%"
  goto :err
)
if not exist "%FRONTEND_DIR%" (
  >&2 echo.
  >&2 echo ERROR: FRONTEND_DIR not found: "%FRONTEND_DIR%"
  goto :err
)

echo [1/4] Go to web project
pushd "%WEB_DIR%" || goto :err

echo [2/4] Install deps (npm ci or npm install)
if exist package-lock.json (
  call npm ci
  if errorlevel 1 (
    echo npm ci failed, falling back to npm install...
    call npm install || goto :err
  )
) else (
  call npm install || goto :err
)

echo [3/4] Build web (vite --mode web => dist/web)
call npm run build:web || goto :err

set "WEB_DIST=%WEB_DIR%\dist\web"
if not exist "%WEB_DIST%" (
  >&2 echo.
  >&2 echo ERROR: Build output not found at "%WEB_DIST%".
  goto :err
)

echo [4/4] Deploy to frontend
robocopy "%WEB_DIST%" "%FRONTEND_DIR%" /E /NFL /NDL /NJH /NJS
set RC=%ERRORLEVEL%
if %RC% GEQ 8 goto :err

popd >nul

echo.
echo SUCCESS: Build and deploy completed.
echo Target: "%FRONTEND_DIR%"
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