@echo off
setlocal
pushd "%~dp0"
echo Starting Open-LLM-VTuber backend...
uv run run_server.py
echo.
echo Press any key to close this window...
pause >nul
popd 