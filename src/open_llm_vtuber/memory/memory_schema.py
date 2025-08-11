from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional
import json
import re
import time


class MemoryKind(str, Enum):
    SELF = "self"
    USER = "user"
    THIRD_PARTY = "third_party"


@dataclass
class MemoryItemTyped:
    text: str
    kind: MemoryKind
    conf_uid: str
    history_uid: str
    importance: float = 0.5
    timestamp: float = 0.0
    tags: List[str] | None = None
    emotion: str | None = None
    context_window: str | None = None

    def to_metadata(self) -> Dict[str, Any]:
        ts = float(self.timestamp or time.time())
        # tags как JSON-строка для строгой скалярности
        tags_json = json.dumps(
            self.tags or [], ensure_ascii=False, separators=(",", ":")
        )
        return {
            "conf_uid": self.conf_uid,
            "history_uid": self.history_uid,
            "kind": str(
                self.kind.value if isinstance(self.kind, MemoryKind) else self.kind
            ),
            "importance": float(self.importance),
            "timestamp": ts,
            "tags": tags_json,
            **({"emotion": self.emotion} if self.emotion else {}),
            **({"context_window": self.context_window} if self.context_window else {}),
        }


# --- Heuristics & Utilities ---

_PRONOUNS_SELF = ["я", "мне", "меня", "мой", "моя", "мои", "сам", "сама", "себя"]
_PRONOUNS_USER = [
    "вы",
    "вас",
    "вам",
    "ваш",
    "твоя",
    "тебя",
    "тебе",
    "пользователь",
    "зритель",
]


def determine_memory_kind(
    text: str, speaker: str, conf: Optional[object] = None
) -> MemoryKind:
    t = (text or "").lower()
    sp = (speaker or "").upper()
    # 1) Если говорит Нейри и использует "я" — SELF
    if sp == "NEYRI" and any(p in t for p in _PRONOUNS_SELF):
        return MemoryKind.SELF
    # 2) Если говорит пользователь — USER (по умолчанию)
    if sp == "USER":
        # Если явно упоминают сторонние сущности — THIRD_PARTY
        if any(
            k in t
            for k in [
                "друг",
                "подруга",
                "мама",
                "папа",
                "коллег",
                "стример",
                "ведущий",
                "знакомый",
                "создатель",
            ]
        ):
            return MemoryKind.THIRD_PARTY
        return MemoryKind.USER
    # 3) Если упоминания конкретных людей
    if any(
        k in t
        for k in [
            "друг",
            "подруга",
            "мама",
            "папа",
            "коллег",
            "стример",
            "ведущий",
            "знакомый",
            "создатель",
        ]
    ):
        return MemoryKind.THIRD_PARTY
    # 4) Fallback
    return MemoryKind.USER


def calculate_importance(text: str) -> float:
    # Простая эвристика важности: длиннее и с ключевыми словами — выше
    t = (text or "").strip()
    base = min(1.0, max(0.1, len(t) / 400.0))
    if any(
        k in t.lower()
        for k in ["люблю", "ненавижу", "всегда", "никогда", "важно", "запомни"]
    ):
        base = min(1.0, base + 0.2)
    return float(base)


def extract_tags(text: str) -> List[str]:
    # Наивное извлечение тегов: слова длиной >= 4, без пунктуации
    words = re.findall(r"[\w\-]{4,}", (text or "").lower())
    # Ограничим кол-во тегов
    uniq: List[str] = []
    for w in words:
        if w not in uniq:
            uniq.append(w)
        if len(uniq) >= 10:
            break
    return uniq


def detect_emotion(text: str) -> str:
    # Плейсхолдер: можно подключить модель позже
    low = (text or "").lower()
    if any(w in low for w in ["спасибо", "люблю", "нравится", "рад", "рада"]):
        return "joy"
    if any(w in low for w in ["плохо", "ненавижу", "злой", "злая", "грусть"]):
        return "sadness"
    return "neutral"


def _match_context_to_kind(query: str, kind: MemoryKind) -> float:
    t = (query or "").lower()
    kw = {
        MemoryKind.SELF: _PRONOUNS_SELF,
        MemoryKind.USER: _PRONOUNS_USER,
        MemoryKind.THIRD_PARTY: [
            "друг",
            "подруга",
            "мама",
            "папа",
            "коллег",
            "стример",
            "ведущий",
            "знакомый",
            "создатель",
        ],
    }
    score = 0.0
    for k in kw.get(kind, []):
        if k in t:
            score += 0.1
    return min(0.3, score)


def calculate_relevance(query: str, text: str, metadata: Dict[str, Any]) -> float:
    # Базовая близость из Chroma недоступна тут; предполагаем 0.7 в среднем
    base_relevance = 0.7
    try:
        ts = float(metadata.get("timestamp") or time.time())
    except Exception:
        ts = time.time()
    age = max(1.0, time.time() - ts)
    time_factor = 0.3 * (1.0 / age) ** 0.1
    try:
        importance = float(metadata.get("importance") or 0.5)
    except Exception:
        importance = 0.5
    importance_factor = 0.2 * importance
    kind_raw = metadata.get("kind") or "user"
    try:
        kind = MemoryKind(kind_raw)
    except Exception:
        kind = MemoryKind.USER
    context_match = _match_context_to_kind(query, kind)
    return min(1.0, base_relevance + time_factor + importance_factor + context_match)


def adjust_context_for_speaker(
    memory_text: str,
    memory_kind: MemoryKind,
    speaker: str,
    current_user_name: Optional[str] = None,
    character_name: Optional[str] = None,
    character_gender: Optional[str] = None,
) -> str:
    sp = (speaker or "").upper()
    text = memory_text or ""
    if sp == "NEYRI" and memory_kind == MemoryKind.SELF:
        # Имя персонажа для замены (падение на дефолт "Нейри")
        cname = (character_name or "Нейри").strip()
        # Заменяем имя персонажа на "Я" в простых случаях
        name_pattern = re.escape(cname)
        text = re.sub(rf"\b({name_pattern})\b", "Я", text, flags=re.IGNORECASE)
        # Клише вида "<Имя> любит" -> "Я люблю"
        text = re.sub(
            rf"\b({name_pattern})\s+([а-яА-Яa-zA-Z]+)",
            r"Я \2",
            text,
            flags=re.IGNORECASE,
        )
        # Приведение к роду персонажа
        gender = (character_gender or "female").strip().lower()
        if gender == "female":
            fem_map = {
                r"\bрад\b": "рада",
                r"\bготов\b": "готова",
                r"\bуверен\b": "уверена",
                r"\bзанят\b": "занята",
                r"\bустал\b": "устала",
                r"\bсогласен\b": "согласна",
                r"\bсделал\b": "сделала",
                r"\bдумал\b": "думала",
                r"\bхотел\b": "хотела",
                r"\bсказал\b": "сказала",
                r"\bродился\b": "родилась",
                r"\bсам\b": "сама",
            }
            for pat, repl in fem_map.items():
                text = re.sub(pat, repl, text, flags=re.IGNORECASE)
        elif gender == "male":
            masc_map = {
                r"\bрада\b": "рад",
                r"\bготова\b": "готов",
                r"\bуверена\b": "уверен",
                r"\bзанята\b": "занят",
                r"\bустала\b": "устал",
                r"\bсогласна\b": "согласен",
                r"\bсделала\b": "сделал",
                r"\bдумала\b": "думал",
                r"\bхотела\b": "хотел",
                r"\bсказала\b": "сказал",
                r"\bродилась\b": "родился",
                r"\bсама\b": "сам",
            }
            for pat, repl in masc_map.items():
                text = re.sub(pat, repl, text, flags=re.IGNORECASE)
        else:
            # neutral: без изменения рода (сохраняем исходную форму)
            pass
        return text
    if sp == "NEYRI" and memory_kind == MemoryKind.USER:
        if current_user_name:
            return f"Вы, {current_user_name}, {text}"
        return f"Вы: {text}"
    return text
