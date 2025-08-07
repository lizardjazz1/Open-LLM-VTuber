# Open-LLM-VTuber System Checker
# Проверяет все компоненты системы для работы VTuber

param(
    [switch]$Fix,
    [switch]$Verbose
)

# Устанавливаем кодировку
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "🔍 Проверка системы Open-LLM-VTuber..." -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Gray
Write-Host ""

$allGood = $true

# 1. Проверка Python
Write-Host "1️⃣ Проверка Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python 3\.(1[0-9]|[2-9][0-9])") {
        Write-Host "   ✅ Python найден: $pythonVersion" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️ Python найден, но версия может быть устаревшей: $pythonVersion" -ForegroundColor Yellow
        $allGood = $false
    }
} catch {
    Write-Host "   ❌ Python не найден" -ForegroundColor Red
    Write-Host "   💡 Установите Python 3.10+ с https://python.org" -ForegroundColor Cyan
    $allGood = $false
}

# 2. Проверка Node.js
Write-Host "2️⃣ Проверка Node.js..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version 2>&1
    if ($nodeVersion -match "v(1[8-9]|[2-9][0-9])") {
        Write-Host "   ✅ Node.js найден: $nodeVersion" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️ Node.js найден, но версия может быть устаревшей: $nodeVersion" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ❌ Node.js не найден" -ForegroundColor Red
    Write-Host "   💡 Установите Node.js 18+ с https://nodejs.org" -ForegroundColor Cyan
    $allGood = $false
}

# 3. Проверка зависимостей Python
Write-Host "3️⃣ Проверка Python зависимостей..." -ForegroundColor Yellow
$requiredPackages = @("fastapi", "uvicorn", "loguru", "pydantic", "openai")
$missingPackages = @()

foreach ($package in $requiredPackages) {
    try {
        python -c "import $package" 2>$null
        Write-Host "   ✅ $package" -ForegroundColor Green
    } catch {
        Write-Host "   ❌ $package" -ForegroundColor Red
        $missingPackages += $package
        $allGood = $false
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Host "   💡 Установите недостающие пакеты:" -ForegroundColor Cyan
    Write-Host "      pip install -r requirements.txt" -ForegroundColor Gray
}

# 4. Проверка MCP серверов
Write-Host "4️⃣ Проверка MCP серверов..." -ForegroundColor Yellow
try {
    python -c "import mcp_server_time" 2>$null
    Write-Host "   ✅ mcp-server-time" -ForegroundColor Green
} catch {
    Write-Host "   ❌ mcp-server-time" -ForegroundColor Red
    Write-Host "   💡 Установите: pip install mcp-server-time" -ForegroundColor Cyan
    $allGood = $false
}

try {
    npx duckduckgo-mcp-server --help 2>$null
    Write-Host "   ✅ duckduckgo-mcp-server" -ForegroundColor Green
} catch {
    Write-Host "   ❌ duckduckgo-mcp-server" -ForegroundColor Red
    Write-Host "   💡 Установите: npm install -g duckduckgo-mcp-server" -ForegroundColor Cyan
    $allGood = $false
}

# 5. Проверка конфигурации
Write-Host "5️⃣ Проверка конфигурации..." -ForegroundColor Yellow
if (Test-Path "conf.yaml") {
    Write-Host "   ✅ conf.yaml найден" -ForegroundColor Green
} else {
    Write-Host "   ❌ conf.yaml не найден" -ForegroundColor Red
    $allGood = $false
}

if (Test-Path "mcp_servers.json") {
    Write-Host "   ✅ mcp_servers.json найден" -ForegroundColor Green
} else {
    Write-Host "   ❌ mcp_servers.json не найден" -ForegroundColor Red
    $allGood = $false
}

# 6. Проверка LM Studio
Write-Host "6️⃣ Проверка LM Studio..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:1234/v1/models" -Method GET -TimeoutSec 5
    Write-Host "   ✅ LM Studio доступен" -ForegroundColor Green
    Write-Host "   📋 Доступные модели:" -ForegroundColor Cyan
    foreach ($model in $response.data) {
        Write-Host "      - $($model.id)" -ForegroundColor Gray
    }
} catch {
    Write-Host "   ❌ LM Studio недоступен" -ForegroundColor Red
    Write-Host "   💡 Запустите LM Studio и нажмите 'Start Server'" -ForegroundColor Cyan
    $allGood = $false
}

# 7. Проверка портов
Write-Host "7️⃣ Проверка портов..." -ForegroundColor Yellow
$ports = @(1234, 12393)
foreach ($port in $ports) {
    try {
        $connection = Test-NetConnection -ComputerName localhost -Port $port -WarningAction SilentlyContinue
        if ($connection.TcpTestSucceeded) {
            Write-Host "   ✅ Порт $port открыт" -ForegroundColor Green
        } else {
            Write-Host "   ❌ Порт $port закрыт" -ForegroundColor Red
            $allGood = $false
        }
    } catch {
        Write-Host "   ❌ Порт $port недоступен" -ForegroundColor Red
        $allGood = $false
    }
}

# Итоговая оценка
Write-Host ""
Write-Host "================================================" -ForegroundColor Gray
if ($allGood) {
    Write-Host "🎉 Система готова к работе!" -ForegroundColor Green
    Write-Host "💡 Запустите VTuber командой: .\start_vtuber.ps1" -ForegroundColor Cyan
} else {
    Write-Host "⚠️ Обнаружены проблемы" -ForegroundColor Yellow
    Write-Host "💡 Исправьте указанные проблемы и запустите проверку снова" -ForegroundColor Cyan
}

Write-Host "" 