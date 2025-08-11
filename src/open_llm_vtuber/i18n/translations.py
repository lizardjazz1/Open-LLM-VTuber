"""
Centralized translations for Open-LLM-VTuber
This file contains all text strings used in the application.
"""

# Base translations structure
TRANSLATIONS = {
    "en": {
        "app": {
            "name": "Open-LLM-VTuber",
            "description": "Low-latency voice-based LLM interaction tool",
            "version": "Version",
        },
        "server": {
            "starting": "Starting Open-LLM-VTuber server...",
            "started": "Server started successfully",
            "stopping": "Stopping server...",
            "error": "Server error occurred",
            "port_in_use": "Port {port} is already in use",
            "host": "Server host address",
            "port": "Server port number",
            "websocket_connected": "WebSocket client connected",
            "websocket_disconnected": "WebSocket client disconnected",
            "websocket_error": "WebSocket error occurred",
        },
        "config": {
            "loading": "Loading configuration...",
            "loaded": "Configuration loaded successfully",
            "error": "Error loading configuration",
            "not_found": "Configuration file not found",
            "invalid": "Invalid configuration",
            "backup_created": "Configuration backup created",
            "backup_restored": "Configuration backup restored",
            "validation_error": "Configuration validation error",
            "field_required": "Field '{field}' is required",
            "invalid_value": "Invalid value for field '{field}'",
        },
        "llm": {
            "connecting": "Connecting to LLM...",
            "connected": "Connected to LLM",
            "error": "LLM connection error",
            "timeout": "LLM request timeout",
            "generating": "Generating response...",
            "generated": "Response generated",
            "model_loaded": "LLM model loaded successfully",
            "model_error": "Error loading LLM model",
            "context_length": "Context length exceeded",
            "token_limit": "Token limit reached",
        },
        "asr": {
            "initializing": "Initializing speech recognition...",
            "ready": "Speech recognition ready",
            "listening": "Listening...",
            "processing": "Processing speech...",
            "error": "Speech recognition error",
            "no_speech": "No speech detected",
            "model_loaded": "ASR model loaded",
            "model_error": "Error loading ASR model",
            "audio_format": "Unsupported audio format",
            "sample_rate": "Invalid sample rate",
        },
        "tts": {
            "initializing": "Initializing text-to-speech...",
            "ready": "Text-to-speech ready",
            "generating": "Generating speech...",
            "generated": "Speech generated",
            "error": "Text-to-speech error",
            "voice_not_found": "Voice not found",
            "model_loaded": "TTS model loaded",
            "model_error": "Error loading TTS model",
            "audio_quality": "Audio quality issue",
            "rate_limit": "TTS rate limit exceeded",
        },
        "vad": {
            "initializing": "Initializing voice activity detection...",
            "ready": "Voice activity detection ready",
            "speech_detected": "Speech detected",
            "silence_detected": "Silence detected",
            "error": "Voice activity detection error",
            "model_loaded": "VAD model loaded",
            "model_error": "Error loading VAD model",
            "threshold_adjusted": "VAD threshold adjusted",
        },
        "live2d": {
            "loading": "Loading Live2D model...",
            "loaded": "Live2D model loaded",
            "error": "Error loading Live2D model",
            "expression_set": "Expression set to {expression}",
            "motion_played": "Motion played: {motion}",
            "model_not_found": "Live2D model not found",
            "parameter_updated": "Parameter updated: {parameter}",
        },
        "proxy": {
            "enabled": "Proxy mode enabled",
            "disabled": "Proxy mode disabled",
            "client_connected": "Proxy client connected",
            "client_disconnected": "Proxy client disconnected",
            "message_forwarded": "Message forwarded to client",
            "error": "Proxy error occurred",
        },
        "twitch": {
            "connecting": "Connecting to Twitch...",
            "connected": "Connected to Twitch",
            "disconnected": "Disconnected from Twitch",
            "error": "Twitch connection error",
            "chat_message": "Chat message received",
            "donation": "Donation received",
            "subscription": "Subscription event",
            "follow": "New follower",
            "raid": "Raid event",
        },
        "memory": {
            "loading": "Loading memory...",
            "loaded": "Memory loaded successfully",
            "saving": "Saving memory...",
            "saved": "Memory saved successfully",
            "error": "Memory error occurred",
            "context_added": "Context added to memory",
            "context_retrieved": "Context retrieved from memory",
            "memory_full": "Memory is full, cleaning old entries",
        },
        "tools": {
            "executing": "Executing tool: {tool}",
            "executed": "Tool executed successfully",
            "error": "Tool execution error",
            "not_found": "Tool not found: {tool}",
            "timeout": "Tool execution timeout",
            "permission_denied": "Tool permission denied",
        },
        "ui": {
            "loading": "Loading...",
            "error": "An error occurred",
            "success": "Operation completed successfully",
            "warning": "Warning",
            "info": "Information",
            "confirm": "Please confirm",
            "cancel": "Cancel",
            "ok": "OK",
            "yes": "Yes",
            "no": "No",
        },
        "validation": {
            "required": "This field is required",
            "invalid_format": "Invalid format",
            "too_short": "Value is too short",
            "too_long": "Value is too long",
            "invalid_range": "Value is out of range",
            "invalid_email": "Invalid email address",
            "invalid_url": "Invalid URL",
        },
    },
    "zh": {
        "app": {
            "name": "Open-LLM-VTuber",
            "description": "低延迟基于语音的LLM交互工具",
            "version": "版本",
        },
        "server": {
            "starting": "正在启动Open-LLM-VTuber服务器...",
            "started": "服务器启动成功",
            "stopping": "正在停止服务器...",
            "error": "服务器发生错误",
            "port_in_use": "端口{port}已被占用",
            "host": "服务器主机地址",
            "port": "服务器端口号",
            "websocket_connected": "WebSocket客户端已连接",
            "websocket_disconnected": "WebSocket客户端已断开",
            "websocket_error": "WebSocket发生错误",
        },
        "config": {
            "loading": "正在加载配置...",
            "loaded": "配置加载成功",
            "error": "加载配置时出错",
            "not_found": "未找到配置文件",
            "invalid": "配置无效",
            "backup_created": "配置备份已创建",
            "backup_restored": "配置备份已恢复",
            "validation_error": "配置验证错误",
            "field_required": "字段'{field}'是必需的",
            "invalid_value": "字段'{field}'的值无效",
        },
        "llm": {
            "connecting": "正在连接到LLM...",
            "connected": "已连接到LLM",
            "error": "LLM连接错误",
            "timeout": "LLM请求超时",
            "generating": "正在生成响应...",
            "generated": "响应已生成",
            "model_loaded": "LLM模型加载成功",
            "model_error": "加载LLM模型时出错",
            "context_length": "上下文长度超出限制",
            "token_limit": "达到令牌限制",
        },
        "asr": {
            "initializing": "正在初始化语音识别...",
            "ready": "语音识别已就绪",
            "listening": "正在监听...",
            "processing": "正在处理语音...",
            "error": "语音识别错误",
            "no_speech": "未检测到语音",
            "model_loaded": "ASR模型已加载",
            "model_error": "加载ASR模型时出错",
            "audio_format": "不支持的音频格式",
            "sample_rate": "无效的采样率",
        },
        "tts": {
            "initializing": "正在初始化文本转语音...",
            "ready": "文本转语音已就绪",
            "generating": "正在生成语音...",
            "generated": "语音已生成",
            "error": "文本转语音错误",
            "voice_not_found": "未找到语音",
            "model_loaded": "TTS模型已加载",
            "model_error": "加载TTS模型时出错",
            "audio_quality": "音频质量问题",
            "rate_limit": "TTS速率限制超出",
        },
        "vad": {
            "initializing": "正在初始化语音活动检测...",
            "ready": "语音活动检测已就绪",
            "speech_detected": "检测到语音",
            "silence_detected": "检测到静音",
            "error": "语音活动检测错误",
            "model_loaded": "VAD模型已加载",
            "model_error": "加载VAD模型时出错",
            "threshold_adjusted": "VAD阈值已调整",
        },
        "live2d": {
            "loading": "正在加载Live2D模型...",
            "loaded": "Live2D模型已加载",
            "error": "加载Live2D模型时出错",
            "expression_set": "表情设置为{expression}",
            "motion_played": "播放动作：{motion}",
            "model_not_found": "未找到Live2D模型",
            "parameter_updated": "参数已更新：{parameter}",
        },
        "proxy": {
            "enabled": "代理模式已启用",
            "disabled": "代理模式已禁用",
            "client_connected": "代理客户端已连接",
            "client_disconnected": "代理客户端已断开",
            "message_forwarded": "消息已转发给客户端",
            "error": "代理发生错误",
        },
        "twitch": {
            "connecting": "正在连接到Twitch...",
            "connected": "已连接到Twitch",
            "disconnected": "已断开与Twitch的连接",
            "error": "Twitch连接错误",
            "chat_message": "收到聊天消息",
            "donation": "收到捐赠",
            "subscription": "订阅事件",
            "follow": "新关注者",
            "raid": "突袭事件",
        },
        "memory": {
            "loading": "正在加载内存...",
            "loaded": "内存加载成功",
            "saving": "正在保存内存...",
            "saved": "内存保存成功",
            "error": "内存错误",
            "context_added": "上下文已添加到内存",
            "context_retrieved": "从内存中检索到上下文",
            "memory_full": "内存已满，正在清理旧条目",
        },
        "tools": {
            "executing": "正在执行工具：{tool}",
            "executed": "工具执行成功",
            "error": "工具执行错误",
            "not_found": "未找到工具：{tool}",
            "timeout": "工具执行超时",
            "permission_denied": "工具权限被拒绝",
        },
        "ui": {
            "loading": "正在加载...",
            "error": "发生错误",
            "success": "操作成功完成",
            "warning": "警告",
            "info": "信息",
            "confirm": "请确认",
            "cancel": "取消",
            "ok": "确定",
            "yes": "是",
            "no": "否",
        },
        "validation": {
            "required": "此字段是必需的",
            "invalid_format": "格式无效",
            "too_short": "值太短",
            "too_long": "值太长",
            "invalid_range": "值超出范围",
            "invalid_email": "无效的电子邮件地址",
            "invalid_url": "无效的URL",
        },
    },
    "ru": {
        "app": {
            "name": "Open-LLM-VTuber",
            "description": "Инструмент взаимодействия с LLM на основе голоса с низкой задержкой",
            "version": "Версия",
        },
        "server": {
            "starting": "Запуск сервера Open-LLM-VTuber...",
            "started": "Сервер успешно запущен",
            "stopping": "Остановка сервера...",
            "error": "Произошла ошибка сервера",
            "port_in_use": "Порт {port} уже используется",
            "host": "Адрес хоста сервера",
            "port": "Номер порта сервера",
            "websocket_connected": "WebSocket клиент подключен",
            "websocket_disconnected": "WebSocket клиент отключен",
            "websocket_error": "Произошла ошибка WebSocket",
        },
        "config": {
            "loading": "Загрузка конфигурации...",
            "loaded": "Конфигурация успешно загружена",
            "error": "Ошибка загрузки конфигурации",
            "not_found": "Файл конфигурации не найден",
            "invalid": "Недействительная конфигурация",
            "backup_created": "Резервная копия конфигурации создана",
            "backup_restored": "Резервная копия конфигурации восстановлена",
            "validation_error": "Ошибка валидации конфигурации",
            "field_required": "Поле '{field}' обязательно",
            "invalid_value": "Недействительное значение для поля '{field}'",
        },
        "llm": {
            "connecting": "Подключение к LLM...",
            "connected": "Подключено к LLM",
            "error": "Ошибка подключения к LLM",
            "timeout": "Таймаут запроса LLM",
            "generating": "Генерация ответа...",
            "generated": "Ответ сгенерирован",
            "model_loaded": "Модель LLM загружена успешно",
            "model_error": "Ошибка загрузки модели LLM",
            "context_length": "Превышена длина контекста",
            "token_limit": "Достигнут лимит токенов",
        },
        "asr": {
            "initializing": "Инициализация распознавания речи...",
            "ready": "Распознавание речи готово",
            "listening": "Прослушивание...",
            "processing": "Обработка речи...",
            "error": "Ошибка распознавания речи",
            "no_speech": "Речь не обнаружена",
            "model_loaded": "Модель ASR загружена",
            "model_error": "Ошибка загрузки модели ASR",
            "audio_format": "Неподдерживаемый формат аудио",
            "sample_rate": "Недействительная частота дискретизации",
        },
        "tts": {
            "initializing": "Инициализация преобразования текста в речь...",
            "ready": "Преобразование текста в речь готово",
            "generating": "Генерация речи...",
            "generated": "Речь сгенерирована",
            "error": "Ошибка преобразования текста в речь",
            "voice_not_found": "Голос не найден",
            "model_loaded": "Модель TTS загружена",
            "model_error": "Ошибка загрузки модели TTS",
            "audio_quality": "Проблема качества аудио",
            "rate_limit": "Превышен лимит скорости TTS",
        },
        "vad": {
            "initializing": "Инициализация обнаружения речевой активности...",
            "ready": "Обнаружение речевой активности готово",
            "speech_detected": "Обнаружена речь",
            "silence_detected": "Обнаружена тишина",
            "error": "Ошибка обнаружения речевой активности",
            "model_loaded": "Модель VAD загружена",
            "model_error": "Ошибка загрузки модели VAD",
            "threshold_adjusted": "Порог VAD скорректирован",
        },
        "live2d": {
            "loading": "Загрузка модели Live2D...",
            "loaded": "Модель Live2D загружена",
            "error": "Ошибка загрузки модели Live2D",
            "expression_set": "Выражение установлено: {expression}",
            "motion_played": "Воспроизведено движение: {motion}",
            "model_not_found": "Модель Live2D не найдена",
            "parameter_updated": "Параметр обновлен: {parameter}",
        },
        "proxy": {
            "enabled": "Режим прокси включен",
            "disabled": "Режим прокси отключен",
            "client_connected": "Прокси клиент подключен",
            "client_disconnected": "Прокси клиент отключен",
            "message_forwarded": "Сообщение переслано клиенту",
            "error": "Произошла ошибка прокси",
        },
        "twitch": {
            "connecting": "Подключение к Twitch...",
            "connected": "Подключено к Twitch",
            "disconnected": "Отключено от Twitch",
            "error": "Ошибка подключения к Twitch",
            "chat_message": "Получено сообщение чата",
            "donation": "Получено пожертвование",
            "subscription": "Событие подписки",
            "follow": "Новый подписчик",
            "raid": "Событие рейда",
        },
        "memory": {
            "loading": "Загрузка памяти...",
            "loaded": "Память успешно загружена",
            "saving": "Сохранение памяти...",
            "saved": "Память успешно сохранена",
            "error": "Ошибка памяти",
            "context_added": "Контекст добавлен в память",
            "context_retrieved": "Контекст извлечен из памяти",
            "memory_full": "Память заполнена, очистка старых записей",
        },
        "tools": {
            "executing": "Выполнение инструмента: {tool}",
            "executed": "Инструмент выполнен успешно",
            "error": "Ошибка выполнения инструмента",
            "not_found": "Инструмент не найден: {tool}",
            "timeout": "Таймаут выполнения инструмента",
            "permission_denied": "Доступ к инструменту запрещен",
        },
        "ui": {
            "loading": "Загрузка...",
            "error": "Произошла ошибка",
            "success": "Операция успешно завершена",
            "warning": "Предупреждение",
            "info": "Информация",
            "confirm": "Пожалуйста, подтвердите",
            "cancel": "Отмена",
            "ok": "ОК",
            "yes": "Да",
            "no": "Нет",
        },
        "validation": {
            "required": "Это поле обязательно",
            "invalid_format": "Недействительный формат",
            "too_short": "Значение слишком короткое",
            "too_long": "Значение слишком длинное",
            "invalid_range": "Значение вне диапазона",
            "invalid_email": "Недействительный адрес электронной почты",
            "invalid_url": "Недействительный URL",
        },
    },
}


def get_translation(key: str, lang_code: str = "en") -> str:
    """
    Get a translation by key and language code.

    Args:
        key: Translation key (e.g., "server.starting")
        lang_code: Language code (e.g., "en", "zh", "ru")

    Returns:
        Translated text or the key itself if not found
    """
    if lang_code not in TRANSLATIONS:
        lang_code = "en"

    # Split key by dots to navigate nested structure
    keys = key.split(".")
    translation = TRANSLATIONS[lang_code]

    # Navigate through nested structure
    for k in keys:
        if isinstance(translation, dict) and k in translation:
            translation = translation[k]
        else:
            # Key not found, fallback to English
            if lang_code != "en":
                return get_translation(key, "en")
            return key

    # If we found a string, return it
    if isinstance(translation, str):
        return translation

    # Fallback to English
    if lang_code != "en":
        return get_translation(key, "en")

    return key


def get_available_languages() -> list[str]:
    """
    Get list of available language codes.

    Returns:
        List of available language codes
    """
    return list(TRANSLATIONS.keys())


def format_translation(key: str, lang_code: str = "en", **kwargs) -> str:
    """
    Get a formatted translation with placeholders.

    Args:
        key: Translation key
        lang_code: Language code
        **kwargs: Format arguments

    Returns:
        Formatted translated text
    """
    translation = get_translation(key, lang_code)

    try:
        return translation.format(**kwargs)
    except (KeyError, ValueError):
        # If formatting fails, return the translation as is
        return translation
