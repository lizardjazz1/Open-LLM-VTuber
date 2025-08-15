# Open-LLM-VTuber — Архитектура и Обзор (RU)

## 1. Общее описание проекта

Open-LLM-VTuber — это офлайн-ориентированный инструмент для голосового взаимодействия с LLM с очень низкой задержкой, ориентированный на VTuber/стриминг-сценарии. Пользователь говорит — система распознаёт речь (STT), формирует ответ с помощью LLM, озвучивает его (TTS) и управляет Live2D-аватаром. Цель — end-to-end задержка ниже 500 мс.

Основные особенности:
- Полностью асинхронный бэкенд (FastAPI, WebSockets), строгая разделённость фронтенда.
- Офлайн-режим по возможности: локальные STT/TTS/LLM; внешние API — опциональны.
- Расширяемые провайдеры LLM/STT/TTS, интеграция с Twitch-чатом и модуль долговременной памяти на ChromaDB (MemGPT‑like).

## 2. Обзор архитектуры

Высокоуровневая схема потоков:
- Вход: микрофон пользователя → VAD (детектор речи) → ASR (STT) → текст.
- Обработка: сервисный контекст собирает системный промпт (персона + утил‑промпты), подмешивает релевантные воспоминания (Chroma/MemGPT-like), вызывает агента (LLM) → получает сегментированный ответ и команды действий.
- Выход: TTS синтезирует аудио, фронтенд воспроизводит аудио и применяет эмоции/движения Live2D. Twitch‑сообщения обрабатываются тем же конвейером.

Ключевые компоненты:
- WebSocket-сервер (`src/open_llm_vtuber/server.py`) — регистрирует маршруты, статику, Twitch API и админ‑роуты памяти.
- Обработчик WebSocket (`src/open_llm_vtuber/websocket_handler.py`) — сессии, входящие события, буферы аудио, инициализация контекстов.
- Сервисный контекст (`src/open_llm_vtuber/service_context.py`) — фабрики ASR/TTS/VAD/LLM/инструментов, сборка системного промпта, инициализация памяти и Twitch‑клиента.
- Агент с памятью (`src/open_llm_vtuber/agent/agents/basic_memory_agent.py`) — диалог, сегментация, вызовы MCP‑инструментов, фильтрация для TTS и отображения.
- Память: ChromaDB‑слой (`src/open_llm_vtuber/memory/*`) и провайдер MemGPT‑like (`src/open_llm_vtuber/vtuber_memory/providers/memgpt_provider.py`).
- Промпты (`prompts/`) и персонажи (`characters/`).
- Twitch интеграция (`src/open_llm_vtuber/twitch/*`, `src/open_llm_vtuber/routes/twitch_routes.py`).

## 3. Папки и модули (кратко)

- `run_server.py` — точка входа, логирование, загрузка `conf.yaml`, запуск Uvicorn, i18n, проверка фронтенд‑сабмодуля.
- `src/open_llm_vtuber/server.py` — создание FastAPI, маршруты: клиентский WS, webtool, логи (включается флагом `system_config.client_log_ingest_enabled`), Twitch, админ‑память; CORS, rate‑limit (SlowAPI).
- `src/open_llm_vtuber/base_routes.py` — REST/WS: `/client-ws`, `/asr`, `/tts-ws` (пакетная TTS отрезками).
- `src/open_llm_vtuber/websocket_handler.py` — управление подключениями, VAD/сырой аудио‑поток, транскрипция, триггер разговора.
- `src/open_llm_vtuber/service_context.py` — агрегирует конфиг, Live2D, ASR/TTS/VAD/LLM, MCP, память (оба вида), Twitch; собирает системный промпт; планировщик консолидации памяти.
- `src/open_llm_vtuber/agent/*` — фабрики и провайдеры Stateless LLM (OpenAI‑совместимый, LM Studio, Ollama, Claude, Llama.cpp, и т.д.). Агент BasicMemoryAgent добавляет краткосрочную память, вызовы инструментов, настройки первого ответа.
- `src/open_llm_vtuber/asr/*` — STT движки: sherpa_onnx (офлайн), faster‑whisper, whisper‑cpp, Azure, Groq Whisper, FunASR и др.
- `src/open_llm_vtuber/tts/*` — TTS движки: edge‑tts (по умолчанию), OpenAI‑совместимый, pyttsx3, sherpa‑onnx TTS, Azure, SiliconFlow, Coqui, CosyVoice(2), Minimax, Bark и др.
- `src/open_llm_vtuber/vad/*` — VAD (например, Silero).
- `src/open_llm_vtuber/memory/*` — слой ChromaMemory + схемы; `vtuber_memory/*` — провайдер MemGPT‑like, планировщик консолидации.
- `prompts/` — `persona/` и `utils/` (live2d_expression, voice_control, emotion_presets, think_tag, speakable, group_conversation, proactive_speak, mcp_prompt и др.).
- `characters/` — YAML‑конфиги персонажей (в т.ч. `neuri.yaml`).
- `frontend/` — собранный фронтенд (git submodule).

## 4. Технологии и сервисы

- LLM: Stateless интерфейс с провайдерами: LM Studio (`lmstudio_llm`), Ollama, OpenAI‑совместимые, Claude, Llama.cpp, Groq, Gemini, Zhipu, DeepSeek, Mistral и др. Настраиваются в `character_config.agent_config.llm_configs`.
- STT (ASR): По умолчанию `sherpa_onnx_asr` (офлайн), опции — Faster‑Whisper, Whisper‑cpp, Azure Cognitive Services, Groq Whisper, FunASR. Вызовы через `ASRInterface.async_transcribe_np`.
- VAD: Silero VAD для детекции речи и разбиения потока.
- TTS: По умолчанию `edge_tts` с голосом `ru-RU-SvetlanaNeural`. Есть OpenAI‑совместимый endpoint, pyttsx3 (офлайн), sherpa‑onnx TTS, Azure, Coqui, CosyVoice(2), Bark, Minimax, SiliconFlow и др.
- Память: ChromaDB (PersistentClient) для сессии и LTM. Опционально внешние Embeddings (OpenAI) с фоллбеком на `sentence-transformers` локально. Поверх — провайдер MemGPT‑like, API сходен с MemGPT.
- Twitch API: `twitchAPI` для аутентификации и чата; REST‑роуты `/api/twitch/*` для статуса/подключения; сообщения идут в общий конвейер через `ServiceContext._process_twitch_message`.
- Live2D: `Live2dModel` готовит payload эмоций/движений; фронтенд применяет эти команды и воспроизводит аудио.

Замечание: Telegram/Discord интеграции в текущем репозитории не обнаружены (зарезервировано на будущее).

## 5. Зависимости (высокоуровнево)

- FastAPI, Starlette, Uvicorn (сервер, WS, статика), SlowAPI (rate limit), Loguru (логирование), Pydantic v2 (валидация конфигурации).
- ASR/TTS: sherpa-onnx, onnxruntime; edge-tts; альтернативы: openai/groq/azure SDKs, coqui‑TTS и др.
- Память: chromadb, sentence-transformers (опционально), openai SDK (опционально для embeddings).
- Twitch: twitchAPI.
- Прочее: numpy, pydub, etc. Точный список — `pyproject.toml` и `requirements.txt`.

## 6. Системные промпты Витубер «Нейри»

Источник/инициатор и сборка:
- Базовый persona‑промпт: из `character_config.persona_prompt` в `conf.yaml` либо из файла персонажа в `characters/` (например, `characters/neuri.yaml`), а также может подменяться `persona_prompt_name` через `prompts/persona/*.txt`.
- Сервисный контекст собирает System Prompt методом `construct_system_prompt(persona_prompt)`:
  - Подмешивает утил‑промпты из `system_config.tool_prompts` → `prompts/utils/*.txt`.
  - Специальный случай: `live2d_expression_prompt.txt` получает замену плейсхолдера списком эмоций из `Live2dModel.emo_str`.
  - `mcp_prompt.txt` загружается отдельно в `self.mcp_prompt` (не добавляется к system prompt напрямую).
  - Добавляются правила фактичности/стиля и анти‑повторы.
- Готовый `system_prompt` передаётся в агент/LLM через фабрики.

Примеры промптов Нейри:
- Персона (сокр.): «Ты Нейри — русская AI Витуберша… Всегда отвечай на русском… Используй эмоции [joy]/[smile]/… и голосовые команды {rate/volume/pitch} … стиль краткий и эмоциональный…» (см. `conf.yaml` и `characters/neuri.yaml`).
- Утил‑промпты: `live2d_expression_prompt`, `voice_control_prompt`, `emotion_presets_prompt`, `memory_consolidation_prompt`, `speakable_prompt`, `think_tag_prompt`, и др.

Табличная сводка (кратко):
- название: persona_prompt (Нейри) | функция: личность/стиль | источник: `conf.yaml`/`characters/neuri.yaml` | целевой компонент: агент/LLM | сохранение: в `ServiceContext.system_prompt`.
- название: live2d_expression_prompt | функция: разрешённые эмоции/теги | источник: `prompts/utils` | целевой компонент: LLM (генерирует теги) | интеграция: заменяет плейсхолдер эмоциями модели.
- название: voice_control_prompt | функция: команды голоса {rate/volume/pitch} | источник: utils | целевой: LLM → TTS менеджер учитывает.
- название: mcp_prompt | функция: правила инструментария MCP | источник: utils | целевой: MCP клиент/ToolManager | использование: отдельно от system prompt.

## 7. Потоки данных (подробнее)

- ASR: `/asr` REST принимает WAV (клиент может конвертировать в браузере), либо WS‑сырой поток+VAD. В `base_routes.py` данные преобразуются в float32 и передаются в `asr_engine.async_transcribe_np`.
- Диалог: `conversations/*` — `process_single_conversation` вызывает ASR при необходимости, формирует вход для агента, передаёт результаты в `TTSTaskManager`, который упорядочивает озвучку и отправку в фронтенд.
- TTS: `TTSTaskManager` создаёт задачи генерации, отправляет промежуточные/финальные payload’ы. Есть отдельный `/tts-ws`, который разбивает текст на предложения и возвращает аудио пути по мере готовности.
- Память: `VtuberMemoryService` по умолчанию использует MemGPT‑like провайдер поверх Chroma (раздельные коллекции: сессия/долгая). Планировщик регулярно консолидирует и подчищает устаревшие сессии.
- Twitch: `TwitchClient` подключается к чату, события MESSAGE идут в общий обработчик как текстовый ввод.

## 8. Конфигурация

- Основной файл: `conf.yaml`. Важно: любые изменения структуры дублировать в `config_templates/conf.default.yaml` и `conf.ZH.default.yaml` и обновлять pydantic‑модели (`src/open_llm_vtuber/config_manager/*`).
- Персонажи: `characters/*.yaml` переопределяют части `character_config` и связанные блоки (ASR/TTS/LLM/VAD). Переключение — через UI/WS (контекст клонируется).
- Память/Embeddings: пути и коллекции — в `system_config` (`chroma_persist_dir`, `chroma_collection`, `embeddings_model`, `embeddings_api_*`).

## 9. Ограничения и неизвестности

- Telegram/Discord: в текущем коде интеграции отсутствуют — указаны как планы.
- Документация некоторых модулей минимальна; ориентируйтесь по docstrings и именам.
- Производительность зависит от выбранных провайдеров; для офлайн режимов используйте sherpa‑onnx ASR и локальные TTS/LLM.

## 10. Быстрый чек‑лист для новых участников

- Установите зависимости через `uv sync`. Запуск: `uv run run_server.py`.
- Проверьте, что `frontend/` и Live2D‑модели доступны (сабмодуль, `model_dict.json`).
- Настройте `conf.yaml`: язык, порты, персонажа (`config_alt`/`characters`), ASR/TTS/LLM.
- Для Twitch заполните `app_id/app_secret/channel_name` или отключите `twitch_config.enabled`.
- Для памяти офлайн оставьте embeddings пустыми (fallback на sentence‑transformers) или укажите внешние ключи. 

## 11. Дополнение: обзор неизученных директорий
Open-LLM-VTuber-Web/` — отдельный фронтенд‑репозиторий (React/Vite/Electron), содержит исходники и артефакты (`dist/`, `out/`).
- `externals/` — сборник внешних материалов и связанных репозиториев:
  - `externals/Задание.txt` — исходное ТЗ и теперь также итоговый документ.
  - `externals/user_comments/` — логи и заметки пользователей/разработки (аналитика «Нейри», идеи, журналы сессий, записи по фронтенду/памяти).
  - `externals/NeuroMita/` — внешний VTuber‑проект с собственным кодом (src, scripts, prompt_editor, docs, assets); полезен как референс/библиотека идей.
  - `externals/OPEN-LLM-Vtuber-ORIGINAL/` — слепок оригинальных репозиториев (backend/web/docs) для сравнения/миграции.
  - `externals/Vtuber airi/` — монорепо на pnpm/Rust/TS с сервисами/плагинами/крейтами — богатый референс UI/сервисной архитектуры.
  - `externals/Vtuber Neuro-sama/` — референсный проект (socketio, stt/tts, промптер, модули памяти/голоса); полезен для сравнения потоков и код‑сниппетов.
  - `externals/documentation/` — внутренние руководства: быстрый старт, кастомизация персонажей, FAQ, общий обзор, гайд по разработке.
  - Прочее: `TODO.md`, `MEMORY_WS.md`, `open_llm_vtuber_analysis.md`, `llm_memory_architecture.md`, `DxDiag.txt` — заметки, анализы, задачи, системная информация.

- `llm/` — директория‑контейнер для LLM‑ресурсов (подкаталог `hub/`); в текущем состоянии используется как хранилище/заготовка.
- `scripts/` — служебные скрипты: анализ логов, миграция переводов, запуск биллибили‑лайва.
- `docs/` — рабочие заметки (модуль памяти, отладка, отчёт по установке memgpt и пр.).
- `doc/` — помечено как устаревшее; содержит `sample_conf/` и краткий README.

Примечания:
- Директории `externals/*` не участвуют напрямую в рантайме бэкенда, но содержат важные материалы: документацию, UX‑решения, эталонный код и возможные направления развития.
- При ссылке на материалы из `externals/` в официал. документации важно добавлять контекст и отмечать, что это «референс/исследования», а не часть продакшн‑потока. 

## Приложение A. Подробное описание компонентов и потоков

### A.1 Сервер и маршрутизация
- **Точка входа**: `run_server.py`
  - Загружает `.env` (при наличии), устанавливает `HF_HOME`/`MODELSCOPE_CACHE`, инициализирует логирование через Loguru (раздельные JSONL‑синки для app/access/debug).
  - Проверяет и инициализирует фронтенд‑сабмодуль (`check_frontend_submodule`) при необходимости.
  - Загружает конфиг `conf.yaml` через `validate_config(read_yaml(...))` и инициализирует i18n (`src/open_llm_vtuber/i18n`).
  - Создаёт `WebSocketServer` и вызывает `initialize()` (асинхронная загрузка контекста), затем запускает Uvicorn.
- **Приложение**: `src/open_llm_vtuber/server.py`
  - Создаёт `FastAPI`, настраивает CORS и rate‑limiting (SlowAPI).
  - Регистрирует роуты: `init_client_ws_route` (WebSocket `/client-ws`), `init_webtool_routes` (REST `/asr`, WebSocket `/tts-ws`), `init_log_routes`, `init_twitch_routes`.
  - Монтаж статики (включая фронтенд), с CORS для `StaticFiles`.

### A.2 WebSocket‑контур клиента
- **Конечная точка**: `/client-ws` в `src/open_llm_vtuber/base_routes.py`.
- **Обработчик**: `src/open_llm_vtuber/websocket_handler.py`
  - На новое подключение: создаётся сессионный `ServiceContext` на основе «дефолтного» (клон pydantic‑моделей + ссылки на общие движки), биндинг функции отправки для широковещания.
  - Поддерживает буферизацию входящего сырого аудио (`received_data_buffers`) и обработку VAD (`_handle_raw_audio_data`).
  - Реагирует на пользовательские события: аудио‑чанки, триггеры разговора, изменение параметров LLM/агента, прерывание и пр.

### A.3 Сервисный контекст (`ServiceContext`)
- Файл: `src/open_llm_vtuber/service_context.py`.
- Ответственности:
  - Хранит текущие конфиги (`Config/SystemConfig/CharacterConfig`) и движки: `ASRInterface`, `TTSInterface`, `VADInterface`, `AgentInterface`, `TranslateInterface` (опционально), Live2D‑модель.
  - Инициализирует MCP‑контур (реестры серверов, адаптеры инструментов) при `use_mcpp`.
  - Инициализирует память: `MemoryService` (упрощённая на Chroma) и `VtuberMemoryService` (MemGPT‑like поверх Chroma) + планировщик консолидации (`ConsolidationScheduler`).
  - Собирает системный промпт методом `construct_system_prompt(persona_prompt)`: подмешивает утил‑промпты из `system_config.tool_prompts`, добавляет эмоции Live2D, правила фактичности и анти‑повторы, подготавливает `self.mcp_prompt`.
  - Создаёт агента через `AgentFactory.create_agent(...)` с выбранным провайдером LLM и параметрами (включая `faster_first_response`, `segment_method`, `interrupt_method`, `tool_prompts`, лимиты суммаризации/сентимента, `server_label`).

### A.4 Агент и LLM‑провайдеры
- **Агент**: `src/open_llm_vtuber/agent/agents/basic_memory_agent.py`
  - Ведёт краткосрочную память сообщений, подмешивает `system_prompt` и исторические сводки.
  - Сегментирует ответы (по умолчанию `pysbd`), поддерживает ранний вывод первого ответа (снижение задержки).
  - Интегрируется с MCP‑инструментами (если включено), преобразует ответы в безопасный для TTS текст, извлекает действия/эмоции (`transformers.py`: `sentence_divider`, `actions_extractor`, `display_processor`, `tts_filter`).
- **LLM‑обёртки**: `src/open_llm_vtuber/agent/stateless_llm/*`
  - `lmstudio_llm`, `ollama_llm`, `openai_compatible_llm`, `claude_llm`, `llama_cpp_llm`, `stateless_llm_with_template` и др.
  - Конфигурации задаются в `character_config.agent_config.llm_configs` (см. `conf.yaml`), выбор активного провайдера — в `agent_settings.basic_memory_agent.llm_provider`.

### A.5 Распознавание речи (ASR) и VAD
- **ASR интерфейс**: `src/open_llm_vtuber/asr/asr_interface.py` (синхр/асинхр методы, стандартный формат аудио — float32, 16 kHz, моно).
- **Движки**: `sherpa_onnx_asr` (по умолчанию офлайн), `faster_whisper_asr`, `whisper_cpp_asr`, `azure_asr`, `groq_whisper_asr`, `fun_asr` и др.
- **REST `/asr`**: принимает WAV, валидирует заголовок/буфер, конвертирует в float32 (`np.int16 / 32768.0`), отдаёт текст.
- **VAD**: `src/open_llm_vtuber/vad/silero.py` и фабрика VAD; в WebSocket‑потоке «нарезает» фразы, отправляет сигналы `<|PAUSE|>`/`<|RESUME|>`.

### A.6 Синтез речи (TTS)
- **Интерфейс**: `src/open_llm_vtuber/tts/tts_interface.py` (синхр/асинхр генерация, кеширование имён файлов и расширений).
- **Движки**: по умолчанию `edge_tts` (`ru-RU-SvetlanaNeural`), есть `openai_tts` (совместимый endpoint, потоковая запись в файл), `pyttsx3_tts` (офлайн), `sherpa_onnx_tts`, `azure_tts`, `coqui_tts`, `cosyvoice(_2)`, `bark_tts`, `minimax_tts`, `x_tts`, `gpt_sovits_tts`, `siliconflow_tts` и др.
- **Диспетчер задач TTS**: `src/open_llm_vtuber/conversations/tts_manager.py`
  - Упорядочивает выдачу сегментов, имеет анти‑повтор и хранит последние голосовые команды (`{rate}{volume}{pitch}`), формирует полезные нагрузки для фронтенда (через `utils/stream_audio.py`).
- **WS `/tts-ws`**: принимает текст, делит на предложения, для каждого генерирует аудио и отправляет пути к файлам по мере готовности.

### A.7 Память и MemGPT‑like
- **ChromaMemory**: `src/open_llm_vtuber/memory/chroma_memory.py`
  - PersistentClient, коллекция, локальные `sentence-transformers` либо внешние embeddings (если заданы `embeddings_api_key/base`).
  - Методы: `upsert`, `query`, выравнивание типов метаданных.
- **MemGPTProvider**: `src/open_llm_vtuber/vtuber_memory/providers/memgpt_provider.py`
  - Две коллекции: сессионная и LTM; роутинг по `kind/is_session`.
  - Поддержка OpenAI embeddings (при наличии ключа) с фоллбеком на локальные.
  - Методы: `add_memory`, `add_facts(_with_meta)`, `get_relevant_memories`, `adjust_context_for_speaker`, `prune_session_by_ttl_ex` (через сервис).
- **Сервис**: `VtuberMemoryService` — выбирает backend (`letta` либо MemGPT‑like), управляет политиками и планировщиком консолидации (`memory_consolidation_interval_sec`).

### A.8 Twitch интеграция
- **Клиент**: `src/open_llm_vtuber/twitch/twitch_client.py`
  - Инициализация OAuth, подключение к чату, обработчики READY/MESSAGE/SUB (через `ChatEvent` или классические callbacks).
  - Отдаёт статус/историю сообщений; события прокидываются в диалоговый конвейер (`_process_twitch_message` → `process_single_conversation`).
- **REST**: `src/open_llm_vtuber/routes/twitch_routes.py` — `/api/twitch/status`, `/api/twitch/messages`, `/api/twitch/connect`, `/api/twitch/disconnect`.

### A.9 Live2D‑интеграция
- **Модель**: `src/open_llm_vtuber/live2d_model.py` + `model_dict.json`
  - `Live2dModel` подгружает описание модели, строит `emo_map` и строку эмоций `emo_str` для промпта.
  - В `construct_system_prompt` встраиваются разрешённые эмоции в текст `live2d_expression_prompt`.
- **Отображение эмоций**: в `conf.yaml` (`live2d_config.emotion_voice_map`) настраиваются шаблоны голосовых команд для TTS по эмоциям.

### A.10 Промпты
- **Структура**: `prompts/persona/*.txt`, `prompts/utils/*.txt`, загрузчик — `prompts/prompt_loader.py` (устойчив к кодировкам, подменяет плейсхолдер эмоций).
- **Сборка System Prompt**: исключает некоторые утил‑промпты (например, `group_conversation_prompt`, `proactive_speak_prompt`) из общего system prompt; `mcp_prompt` хранится отдельно.
- **Нейри**: персона и команда‑правила вынесены в `conf.yaml` и `characters/neuri.yaml`; системные команды эмоций и голоса стандартизированы.

### A.11 Фронтенд
- **Сабмодуль**: `frontend/` (артефакты из внешнего `Open-LLM-VTuber-Web`).
- **Web‑инструмент**: `web_tool/main.js` демонстрирует работу `/asr` (конверсия аудио в WAV через WebAudio) и статусные UI‑паттерны.
- **Статика**: сервер отдаёт SPA/ресурсы с CORS‑заголовками.

### A.12 Логирование и диагностика
- **Loguru sinks**: app/access/debug JSONL с ротацией и ретенцией.
- **Request ID**: патчер добавляет `request_id` и `component` в логи, маскирует секреты.
- **Подсветка производительности**: в ряде мест логируются этапы и задержки (ASR, на стороне агента, очереди TTS).

### A.13 Производительность и задержки
- **ASR**: офлайн‑движки (sherpa‑onnx) минимизируют сетевые задержки; VAD уменьшает бесполезную обработку.
- **Агент**: `faster_first_response` выдаёт первую часть ответа быстрее; сегментация `pysbd` отдаёт небольшие фрагменты на TTS.
- **TTS**: параллельная генерация сегментов и упорядоченная выдача (`TTSTaskManager`).
- **Память**: консолидация по расписанию, ограничение окна краткосрочной памяти и суммаризация снижают контекст.

### A.14 Конфигурации и шаблоны
- Все изменения схем конфигурации должны дублироваться в `config_templates/conf.default.yaml` и `conf.ZH.default.yaml` и сопровождаться обновлением pydantic‑моделей (`src/open_llm_vtuber/config_manager/*`).
- Ключевые поля `system_config`: `chroma_persist_dir`, `chroma_collection`, `embeddings_model`, `embeddings_api_key/base`, `memory_consolidation_interval_sec`, `server_label`, `tool_prompts`, `twitch_config`.
- `character_config` задаёт персонажа/аватар, `agent_config` и блоки `asr/tts/vad`.

### A.15 Развёртывание и запуск
- Управление зависимостями: `uv` (`uv sync`, `uv run run_server.py`).
- Переменные окружения: поддержка `.env`; `--hf_mirror` устанавливает `HF_ENDPOINT` на зеркало.
- Кроссплатформенность: CPU‑фоллбек для ONNX Runtime; поддержка Windows‑терминалов с гиперссылками в логах.

### A.16 Дополнительные внешние материалы (`externals/`)
- `Open-LLM-VTuber-Web/`: исходники фронтенда (React/Vite/Electron), конфиги `vite`/`electron-builder`, i18n‑сканнер.
- `user_comments/`: аналитика/логи сессий, идеи, заметки по фронтенду и памяти.
- `NeuroMita/`: внешний VTuber‑проект с редактором промптов, сборкой, документацией — как референс для UI/рабочих процессов.
- `OPEN-LLM-Vtuber-ORIGINAL/`: копии оригинальных репозиториев (backend/web/docs) для сравнения.
- `Vtuber airi/`: монорепо с Rust/TS‑сервисами/плагинами — примеры архитектур решений.
- `Vtuber Neuro-sama/`: проект с Socket.IO, собственными STT/TTS/LLM‑обвязками — референс для сравнения конвейеров.
- `documentation/`: быстрый старт, полный мануал, кастомизация персонажей, гайд разработки, FAQ.

### A.17 Риски и рекомендации
- Ясно разделяйте «исследовательские» материалы из `externals/` и продакшн‑код; не импортируйте их напрямую в рабочие модули без ревью.
- Поддерживайте офлайн‑совместимые варианты (ASR/TTS/LLM) как основной путь; внешние API — опциональные.
- Следите за задержкой на каждом этапе (VAD→ASR→LLM→TTS); используйте `faster_first_response`, короткие сегменты и агрессивную консолидацию памяти. 