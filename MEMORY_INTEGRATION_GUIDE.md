# 🧠 Руководство по интеграции системы памяти в Open-LLM-VTuber

## Обзор

Система памяти в Open-LLM-VTuber обеспечивает долгосрочное хранение и поиск информации о взаимодействиях с пользователями, что позволяет Нейри помнить зрителей и адаптировать свое поведение к каждому индивидуально.

## Архитектура системы памяти

### Компоненты

1. **VtuberMemoryService** - основной сервис памяти
2. **ChromaMemory** - векторная база данных для хранения эмбеддингов
3. **RelationshipsDB** - SQLite база данных для отношений с пользователями
4. **ConsolidationScheduler** - планировщик консолидации памяти

### Типы памяти

- **Session Memory** (`vtuber_session`) - временные факты и события
- **Long-term Memory** (`vtuber_memory`) - консолидированные знания
- **Relationships** - отношения с пользователями (affinity, trust, interaction_count)

## Интеграция в агент

### 1. Автоматическое добавление воспоминаний

В `single_conversation.py` реализована автоматическая классификация и сохранение:

```python
# Классификация пользовательского ввода
kind = determine_memory_kind(
    text=input_text,
    speaker="USER",
    conf=context.character_config,
)

# Создание элемента памяти
item = MemoryItemTyped(
    text=input_text,
    kind=kind,
    conf_uid=context.character_config.conf_uid,
    history_uid=context.history_uid,
    importance=calculate_importance(input_text),
    timestamp=float(time.time()),
    tags=extract_tags(input_text),
    emotion=detect_emotion(input_text),
)

# Добавление в память
mem.add_memory(item)
```

### 2. Поиск релевантных воспоминаний

Перед каждым ответом агент получает релевантные воспоминания:

```python
hits = mem.get_relevant_memories(
    query=input_text,
    conf_uid=context.character_config.conf_uid,
    limit=getattr(context, "memory_top_k", 4),
)
```

### 3. Интеграция с системой отношений

Агент получает информацию об отношениях с пользователем:

```python
rel = db.get(effective_from_name)
if rel:
    sys_chunks.append(
        f"[Relations] affinity={rel.affinity}, trust={rel.trust}, interactions={rel.interaction_count}."
    )
```

## Административные API

### Управление памятью

#### Поиск воспоминаний
```bash
GET /admin/memory/search?q=запрос&top_k=5&kind=user&conf_uid=neuro_pro
```

#### Экспорт памяти
```bash
GET /admin/memory/export?conf_uid=neuro_pro&fmt=json&limit=1000
```

#### Импорт памяти
```bash
POST /admin/memory/import?conf_uid=neuro_pro&history_uid=test&default_kind=chat
Content-Type: multipart/form-data
file: memories.json
```

#### Удаление воспоминаний
```bash
DELETE /admin/memory/delete/{memory_id}
```

#### Массовое удаление
```bash
POST /admin/memory/delete_many
{
    "ids": ["memory_id_1", "memory_id_2"]
}
```

#### Редактирование воспоминаний
```bash
POST /admin/memory/edit
{
    "id": "memory_id",
    "new_content": "Новое содержание"
}
```

#### Изменение типа воспоминаний
```bash
POST /admin/memory/retag_many
{
    "ids": ["memory_id_1", "memory_id_2"],
    "new_kind": "user"
}
```

### Управление отношениями

#### Получение отношений пользователя
```bash
GET /admin/relationship/get?user_id=username
```

#### Обновление отношений
```bash
POST /admin/relationship/update
{
    "user_id": "username",
    "affinity_delta": 10,
    "trust_delta": 5,
    "username": "display_name",
    "realname": "real_name"
}
```

#### Список недавних отношений
```bash
GET /admin/relationship/list_recent?limit=20
```

### Управление консолидацией

#### Ручная консолидация
```bash
POST /admin/memory/consolidate
```

#### Глубокая консолидация
```bash
POST /admin/memory/deep_consolidation
```

#### Очистка по TTL
```bash
POST /admin/memory/prune_session_ttl?ttl_sec=604800
```

## WebSocket API для памяти

### Добавление воспоминания
```javascript
websocket.send(JSON.stringify({
    type: "memory-add",
    entry: {
        text: "Текст воспоминания",
        kind: "user",
        importance: 0.8,
        tags: ["игры", "minecraft"],
        emotion: "joy",
        timestamp: Date.now() / 1000
    }
}));
```

### Ручная консолидация
```javascript
websocket.send(JSON.stringify({
    type: "memory-consolidate",
    reason: "manual"
}));
```

## Конфигурация памяти

### В conf.yaml

```yaml
character_config:
  vtuber_memory:
    enabled: true
    provider: "memgpt"  # или "letta"
    memory_consolidation_interval_sec: 1800  # 30 минут
    deep_consolidation_every_n_streams: 5
    summarize_max_tokens: 256
    summarize_timeout_s: 25
    sentiment_max_tokens: 96
    sentiment_timeout_s: 12
    consolidate_recent_messages: 120
```

### В system_config

```yaml
system_config:
  relationships_db_path: "cache/relationships.sqlite3"
  memory_enabled: true
  memory_top_k: 4
```

## Классификация воспоминаний

### Типы (MemoryKind)

- **USER** - информация о пользователе
- **SELF** - информация о Нейри
- **THIRD_PARTY** - информация о других людях
- **CHAT** - общие разговоры
- **EMOTIONS** - эмоциональные состояния
- **OBJECTIVES** - цели и задачи

### Функции классификации

```python
# Определение типа
kind = determine_memory_kind(text, speaker, conf)

# Расчет важности
importance = calculate_importance(text)

# Извлечение тегов
tags = extract_tags(text)

# Определение эмоции
emotion = detect_emotion(text)
```

## Система отношений

### Поля отношений

- **user_id** - уникальный идентификатор пользователя
- **username** - отображаемое имя
- **realname** - реальное имя
- **affinity** - симпатия (-100..+100)
- **trust** - доверие (0..100)
- **interaction_count** - количество взаимодействий
- **last_interaction** - время последнего взаимодействия

### Методы работы с отношениями

```python
# Получение отношений
rel = db.get(user_id)

# Обновление симпатии
new_affinity = db.adjust_affinity(user_id, delta)

# Обновление доверия
new_trust = db.adjust_trust(user_id, delta)

# Установка имени
db.set_username(user_id, username)
```

## Консолидация памяти

### Автоматическая консолидация

- Происходит каждые 30 минут (настраивается)
- Объединяет session memory в long-term memory
- Удаляет устаревшие записи по TTL

### Глубокая консолидация

- Происходит каждые N стримов (по умолчанию 5)
- Более тщательный анализ и объединение
- Создание мета-воспоминаний

### TTL очистка

- Удаляет записи старше 7 дней (настраивается)
- Исключает активную сессию
- Освобождает место в базе данных

## Интеграция с промптом

### Динамические инструкции

Система памяти автоматически добавляет в промпт:

1. **Релевантные воспоминания** - факты о пользователе
2. **Информацию об отношениях** - affinity, trust, interaction_count
3. **Контекстные подсказки** - адаптация тона общения

### Пример интеграции

```
[Memory] Пользователь любит играть в Minecraft
[Memory] Нейри помнит, что пользователь часто донатит
[Relations] affinity=10, trust=5, interactions=15.
```

## Мониторинг и отладка

### Логирование

Все операции с памятью логируются:

```
INFO | memory.chroma_memory:query | {'memgpt_operation': 'query', 'collection': 'vtuber_session', 'top_k': 5, 'hits': 2, 'latency_ms': 45}
```

### Метрики

- Время запросов к памяти
- Количество найденных воспоминаний
- Размер коллекций
- Частота консолидации

### Отладка

```python
# Проверка состояния памяти
print(f"Memory enabled: {context.memory_enabled}")
print(f"Memory service: {context.vtuber_memory_service}")
print(f"Memory top_k: {context.memory_top_k}")
```

## Лучшие практики

### 1. Классификация

- Используйте точные типы для лучшего поиска
- Добавляйте релевантные теги
- Устанавливайте правильную важность

### 2. Консолидация

- Не запускайте консолидацию слишком часто
- Мониторьте размер баз данных
- Регулярно очищайте устаревшие записи

### 3. Отношения

- Обновляйте отношения постепенно
- Учитывайте контекст взаимодействий
- Не злоупотребляйте экстремальными значениями

### 4. Производительность

- Ограничивайте количество запросов к памяти
- Используйте кэширование для частых запросов
- Мониторьте задержки

## Устранение неполадок

### Проблемы с инициализацией

```bash
# Проверка конфигурации
uv run python test_memory_integration.py

# Проверка ChromaDB
ls -la cache/chroma/

# Проверка SQLite
sqlite3 cache/relationships.sqlite3 ".tables"
```

### Проблемы с производительностью

```bash
# Очистка кэша
rm -rf cache/chroma/*

# Пересоздание баз
rm cache/relationships.sqlite3
```

### Проблемы с поиском

```bash
# Проверка эмбеддингов
curl "http://localhost:8000/admin/memory/search?q=test&top_k=5"

# Проверка экспорта
curl "http://localhost:8000/admin/memory/export?fmt=json&limit=10"
```

## Заключение

Система памяти в Open-LLM-VTuber обеспечивает:

1. **Долгосрочное хранение** информации о пользователях
2. **Персонализированное общение** на основе истории отношений
3. **Автоматическую классификацию** и консолидацию
4. **Административные инструменты** для управления
5. **Интеграцию с промптом** для адаптивного поведения

Эта система позволяет Нейри помнить каждого зрителя и создавать более глубокие и персонализированные взаимодействия. 