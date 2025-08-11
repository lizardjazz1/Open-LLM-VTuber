@echo off
chcp 65001 >nul
echo Запуск Open-LLM-VTuber...
echo.

REM Проверяем, что мы в правильной директории
if not exist "conf.yaml" (
    echo Ошибка: conf.yaml не найден
    echo Убедитесь, что вы запускаете скрипт из корневой директории проекта
    pause
    exit /b 1
)

REM Завершаем старые процессы Python (кроме системных)
echo Завершаем старые процессы VTuber...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *VTuber*" >nul 2>&1
taskkill /F /IM python.exe /FI "COMMANDLINE eq *run_server.py*" >nul 2>&1
taskkill /F /IM python.exe /FI "COMMANDLINE eq *mcp_server_time*" >nul 2>&1
taskkill /F /IM python.exe /FI "COMMANDLINE eq *uv*" >nul 2>&1

echo.
echo Проверяем Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Ошибка: Python не найден
    echo Установите Python 3.10+ с https://python.org
    pause
    exit /b 1
)

echo Проверяем uv...
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Ошибка: uv не найден
    echo Установите uv: pip install uv
    pause
    exit /b 1
)

echo Проверяем LM Studio...
curl -s http://127.0.0.1:1234/v1/models >nul 2>&1
if %errorlevel% neq 0 (
    echo LM Studio не запущен на localhost:1234
    echo Запустите LM Studio и нажмите 'Start Server'
    echo.
    echo Нажмите Enter когда LM Studio готов...
    pause
) else (
    echo LM Studio доступен
)

echo.
echo Запускаем VTuber...
echo Веб-интерфейс будет доступен по адресу: http://localhost:12393
echo.

REM Запускаем VTuber
uv run run_server.py

start "" http://localhost:12393

pause 