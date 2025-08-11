from __future__ import annotations

from typing import List, Optional, Dict, Any
from time import time
from loguru import logger
from .chroma_memory import ChromaMemory, MemoryItem
from .memory_schema import (
    MemoryItemTyped,
    MemoryKind,
    calculate_relevance,
    adjust_context_for_speaker as _adjust_ctx,
)


class MemoryService:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.chroma = ChromaMemory()
        if not self.chroma.is_available():
            self.enabled = False
            logger.warning("MemoryService disabled (Chroma not available)")

    def add_facts(
        self, facts: List[str], conf_uid: str, history_uid: str, kind: str = "chat"
    ) -> int:
        if not self.enabled or not facts:
            return 0
        now_ts = int(time())
        items = [
            MemoryItem(
                id=f"{conf_uid}:{history_uid}:{i}:{now_ts}",
                text=txt,
                metadata={
                    "conf_uid": conf_uid,
                    "history_uid": history_uid,
                    "kind": kind,
                    "importance": 0.5,
                    "timestamp": now_ts,
                    # ChromaDB requires scalar metadata; encode lists as strings
                    "tags": [],
                },
            )
            for i, txt in enumerate(facts)
        ]
        added = self.chroma.upsert(items)
        if added:
            logger.info(
                f"Memory upserted: {added} items (kind={kind}) for conf={conf_uid}"
            )
        return added

    # --- Enhanced memory API (compatible with Задание.txt) ---
    def add_memory(self, item: MemoryItemTyped) -> int:
        """Добавляет одно структурированное воспоминание (SELF/USER/THIRD_PARTY).

        Возвращает 1 при успехе, 0 иначе. Не ломает старый API add_facts*.
        """
        if not self.enabled:
            return 0
        try:
            ts = int(item.timestamp or time())
            mem = MemoryItem(
                id=f"{item.conf_uid}:{item.history_uid}:{ts}",
                text=item.text,
                metadata=item.to_metadata(),
            )
            added = self.chroma.upsert([mem])
            if added:
                logger.info(
                    f"Memory upserted: 1 item (kind={mem.metadata.get('kind')}) for conf={mem.metadata.get('conf_uid')}"
                )
            return int(bool(added))
        except Exception as e:
            logger.warning(f"add_memory failed: {e}")
            return 0

    def get_relevant_memories(
        self,
        query: str,
        conf_uid: Optional[str],
        limit: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Ищет релевантные воспоминания с дополнительным скорингом (время/важность/контекст).

        Возвращает список элементов: {id, text, kind, relevance, metadata}.
        """
        if not self.enabled or not query:
            return []

        # Собираем where c учетом conf_uid и внешних фильтров
        filters: List[Dict[str, Any]] = []
        if conf_uid:
            filters.append({"conf_uid": conf_uid})
        if where:
            # Объединяем через $and согласно хелперам chroma
            filters.append(where)
        if not filters:
            final_where = None
        elif len(filters) == 1:
            final_where = filters[0]
        else:
            final_where = {"$and": filters}

        results = self.chroma.query(query, top_k=limit * 2, where=final_where)
        ranked: List[Dict[str, Any]] = []
        for _id, text, _score, meta in results:
            try:
                # relevance комбинирует семантику (базово) + время + importance + контекст под тип
                rel = calculate_relevance(query, text, meta or {})
                kind_raw = (meta or {}).get("kind") or "user"
                try:
                    kind = MemoryKind(kind_raw)
                except Exception:
                    kind = MemoryKind.USER
                ranked.append(
                    {
                        "id": _id,
                        "text": text,
                        "kind": kind,
                        "relevance": rel,
                        "metadata": meta or {},
                    }
                )
            except Exception:
                continue
        ranked.sort(key=lambda x: x.get("relevance", 0.0), reverse=True)
        return ranked[:limit]

    def adjust_context_for_speaker(
        self,
        memory_text: str,
        memory_kind: MemoryKind | str,
        speaker: str,
        current_user_name: Optional[str] = None,
        character_name: Optional[str] = None,
        character_gender: Optional[str] = None,
    ) -> str:
        """Корректирует SELF/USER воспоминания для правильной речи персонажа.

        Пример: "Нейри любит пиццу" -> "Я люблю пиццу" для speaker=NEYRI и kind=SELF.
        """
        try:
            kind = (
                memory_kind
                if isinstance(memory_kind, MemoryKind)
                else MemoryKind(str(memory_kind))
            )
        except Exception:
            kind = MemoryKind.USER
        return _adjust_ctx(
            memory_text,
            kind,
            speaker,
            current_user_name=current_user_name,
            character_name=character_name,
            character_gender=character_gender,
        )

    def add_facts_with_meta(
        self,
        entries: List[Dict[str, Any]],
        conf_uid: str,
        history_uid: str,
        default_kind: str = "chat",
    ) -> int:
        """entries: [{text, kind?, importance?, tags?, timestamp?}]"""
        if not self.enabled or not entries:
            return 0
        now_ts = int(time())
        items: List[MemoryItem] = []
        for i, e in enumerate(entries):
            text = (e.get("text") or "").strip()
            if not text:
                continue
            kind = e.get("kind") or default_kind
            importance = float(e.get("importance") or 0.5)
            tags = list(e.get("tags") or [])
            # Optional richer metadata
            user_id = e.get("user_id")
            category = e.get("category")
            topics = e.get("topics") or e.get("topic")
            emotion_score = e.get("emotion_score")
            stream_id = e.get("stream_id")
            platform = e.get("platform")
            session_id = e.get("session_id")
            ts = int(e.get("timestamp") or now_ts)
            # Merge topics into tags for coarse filtering
            if isinstance(topics, list):
                for t in topics:
                    try:
                        s = str(t).strip()
                        if s:
                            tags.append(f"topic:{s}")
                    except Exception:
                        continue
            items.append(
                MemoryItem(
                    id=f"{conf_uid}:{history_uid}:{i}:{ts}",
                    text=text,
                    metadata={
                        "conf_uid": conf_uid,
                        "history_uid": history_uid,
                        "kind": kind,
                        "importance": importance,
                        "timestamp": ts,
                        # lists/dicts will be sanitized in ChromaMemory.upsert
                        "tags": tags,
                        **({"user_id": user_id} if user_id else {}),
                        **({"category": category} if category else {}),
                        **({"topics": topics} if topics else {}),
                        **(
                            {"emotion_score": float(emotion_score)}
                            if emotion_score is not None
                            else {}
                        ),
                        **({"stream_id": stream_id} if stream_id else {}),
                        **({"platform": platform} if platform else {}),
                        **({"session_id": session_id} if session_id else {}),
                    },
                )
            )
        added = self.chroma.upsert(items)
        if added:
            logger.info(
                f"Memory upserted with meta: {added} items (kind={default_kind}) for conf={conf_uid}"
            )
        return added

    def search(
        self,
        query: str,
        conf_uid: Optional[str] = None,
        top_k: int = 5,
        kind: Optional[str] = None,
        kinds: Optional[List[str]] = None,
        min_importance: Optional[float] = None,
        since_ts: Optional[int] = None,
        until_ts: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Поиск по памяти с поддержкой фильтров.

        Args:
            query: Текст запроса.
            conf_uid: Ограничить памятью конкретного персонажа.
            top_k: Максимум результатов.
            kind: Фильтр по одному типу памяти.
            kinds: Фильтр по нескольким типам памяти (приоритетнее, чем `kind`).
            min_importance: Минимальная важность записи.
            since_ts: Не раньше этого timestamp (epoch сек.).
            until_ts: Не позже этого timestamp (epoch сек.).

        Returns:
            Список элементов [{id, text, score, metadata}].
        """
        if not self.enabled or not query:
            return []
        # Build conditions and wrap with a single top-level operator per Chroma rules
        conditions: List[Dict[str, Any]] = []
        if conf_uid:
            conditions.append({"conf_uid": conf_uid})
        # Multiple kinds has precedence; otherwise single kind
        kinds_list: List[str] = []
        if kinds:
            try:
                kinds_list = [str(k).strip() for k in kinds if str(k).strip()]
            except Exception:
                kinds_list = []
        if kinds_list:
            conditions.append({"$or": [{"kind": k} for k in kinds_list]})
        elif kind:
            conditions.append({"kind": kind})
        if min_importance is not None:
            conditions.append({"importance": {"$gte": float(min_importance)}})
        ts_range: Dict[str, Any] = {}
        if since_ts is not None:
            ts_range["$gte"] = int(since_ts)
        if until_ts is not None:
            ts_range["$lte"] = int(until_ts)
        if ts_range:
            conditions.append({"timestamp": ts_range})

        if not conditions:
            where: Optional[Dict[str, Any]] = None
        elif len(conditions) == 1:
            where = conditions[0]
        else:
            where = {"$and": conditions}

        results = self.chroma.query(query, top_k=top_k, where=where)
        return [
            {
                "id": _id,
                "text": text,
                "score": score,
                "kind": (meta or {}).get("kind"),
                "metadata": meta,
            }
            for (_id, text, score, meta) in results
        ]

    def clear(self, conf_uid: Optional[str] = None) -> int:
        """Delete memories. If conf_uid provided, only for that character; else all."""
        if not self.enabled or not self.chroma or not self.chroma.collection:
            return 0
        try:
            if conf_uid:
                self.chroma.collection.delete(where={"conf_uid": conf_uid})
            else:
                # Danger: drop entire collection
                self.chroma.client.delete_collection(self.chroma.collection_name)  # type: ignore[attr-defined]
                # recreate empty collection
                self.chroma.collection = self.chroma.client.get_or_create_collection(
                    self.chroma.collection_name
                )  # type: ignore[attr-defined]
            return 1
        except Exception as e:
            logger.warning(f"Memory clear failed: {e}")
            return 0

    def prune(
        self,
        conf_uid: Optional[str] = None,
        max_age_ts: Optional[int] = None,
        max_importance: Optional[float] = None,
    ) -> int:
        """Prune low-importance and/or old entries.
        Removes entries with timestamp <= max_age_ts and importance < max_importance.
        If only one of filters provided, applies that filter.
        Returns 1 on success, 0 otherwise.
        """
        if not self.enabled or not self.chroma or not self.chroma.collection:
            return 0
        try:
            conditions: List[Dict[str, Any]] = []
            if conf_uid:
                conditions.append({"conf_uid": conf_uid})
            if max_age_ts is not None:
                conditions.append({"timestamp": {"$lte": int(max_age_ts)}})
            if max_importance is not None:
                conditions.append({"importance": {"$lt": float(max_importance)}})
            if not conditions:
                return 0
            where = conditions[0] if len(conditions) == 1 else {"$and": conditions}
            self.chroma.collection.delete(where=where)
            return 1
        except Exception as e:
            logger.warning(f"Memory prune failed: {e}")
            return 0

    def list(
        self,
        conf_uid: Optional[str] = None,
        limit: int = 50,
        kind: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self.enabled or not self.chroma or not self.chroma.collection:
            return []
        # Chroma .get(where=...) требует один верхнеуровневый оператор.
        # Если фильтров несколько — оборачиваем их в $and.
        conditions: List[Dict[str, Any]] = []
        if conf_uid:
            conditions.append({"conf_uid": conf_uid})
        if kind:
            conditions.append({"kind": kind})

        if not conditions:
            where: Optional[Dict[str, Any]] = None
        elif len(conditions) == 1:
            where = conditions[0]
        else:
            where = {"$and": conditions}

        raw = self.chroma.list(limit=limit, where=where)
        # Normalize to include top-level kind for UI grouping convenience
        out: List[Dict[str, Any]] = []
        for it in raw:
            meta = it.get("metadata") or {}
            out.append(
                {
                    "id": it.get("id"),
                    "text": it.get("text"),
                    "kind": meta.get("kind"),
                    "metadata": meta,
                }
            )
        return out

    def delete(self, ids: List[str]) -> int:
        if not self.enabled or not ids:
            return 0
        return self.chroma.delete_by_ids(ids)
