from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..memory.memory_service import MemoryService
from ..memory.memory_schema import MemoryItemTyped, MemoryKind
from ..config_manager.system import SystemConfig
from ..config_manager.character import CharacterConfig
from .interface import VtuberMemoryInterface
from .providers.memgpt_provider import MemGPTProvider
from .providers.letta_provider import LettaProvider


class VtuberMemoryService(VtuberMemoryInterface):
    """Default vtuber memory service.

    Wraps current MemoryService (Chroma-based) by default and can switch to
    other providers using CharacterConfig.vtuber_memory.provider.
    """

    def __init__(
        self,
        enabled: bool = True,
        *,
        system_config: SystemConfig | None = None,
        character_config: CharacterConfig | None = None,
    ) -> None:
        provider_name = (
            (
                getattr(
                    getattr(character_config, "vtuber_memory", None), "provider", None
                )
                or "default"
            )
            .strip()
            .lower()
        )
        self._backend: VtuberMemoryInterface
        if provider_name == "memgpt":
            # Read paths and embeddings settings from system_config
            chroma_path = (
                getattr(system_config, "chroma_persist_dir", "cache/chroma")
                if system_config
                else "cache/chroma"
            )
            collection = (
                getattr(system_config, "chroma_collection", "vtuber_memory")
                if system_config
                else "vtuber_memory"
            )
            embeddings_model = (
                getattr(
                    system_config,
                    "embeddings_model",
                    "paraphrase-multilingual-MiniLM-L12-v2",
                )
                if system_config
                else "paraphrase-multilingual-MiniLM-L12-v2"
            )
            self._backend = MemGPTProvider(
                chroma_path=chroma_path,
                collection=collection,
                embeddings_model=embeddings_model,
            )
        elif provider_name == "letta":
            self._backend = LettaProvider()
        else:
            self._backend = MemoryService(enabled=enabled)

    @property
    def enabled(self) -> bool:
        return bool(self._backend and getattr(self._backend, "enabled", False))

    def add_memory(self, item: Dict[str, Any]) -> int:
        try:
            if hasattr(self._backend, "add_memory") and not isinstance(
                self._backend, MemoryService
            ):
                return self._backend.add_memory(item)
            typed = MemoryItemTyped(
                text=str(item.get("text", "")),
                kind=MemoryKind(str(item.get("kind", "user"))),
                conf_uid=str(item.get("conf_uid")),
                history_uid=str(item.get("history_uid")),
                importance=float(item.get("importance", 0.5)),
                timestamp=float(item.get("timestamp", 0.0)),
                tags=list(item.get("tags", []))
                if item.get("tags") is not None
                else None,
                emotion=str(item.get("emotion")) if item.get("emotion") else None,
                context_window=str(item.get("context_window"))
                if item.get("context_window")
                else None,
            )
            return (
                self._backend.add_memory(typed)
                if isinstance(self._backend, MemoryService)
                else 0
            )
        except Exception:
            return 0

    def add_facts(
        self, facts: List[str], conf_uid: str, history_uid: str, kind: str = "chat"
    ) -> int:
        return (
            self._backend.add_facts(facts, conf_uid, history_uid, kind)
            if isinstance(self._backend, MemoryService)
            else self._backend.add_facts(facts, conf_uid, history_uid, kind)
        )

    def add_facts_with_meta(
        self,
        entries: List[Dict[str, Any]],
        conf_uid: str,
        history_uid: str,
        default_kind: str = "chat",
    ) -> int:
        return self._backend.add_facts_with_meta(
            entries, conf_uid, history_uid, default_kind
        )

    def get_relevant_memories(
        self,
        query: str,
        conf_uid: Optional[str],
        limit: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return self._backend.get_relevant_memories(query, conf_uid, limit, where)

    def adjust_context_for_speaker(
        self,
        memory_text: str,
        memory_kind: str,
        speaker: str,
        *,
        current_user_name: Optional[str] = None,
        character_name: Optional[str] = None,
        character_gender: Optional[str] = None,
    ) -> str:
        return self._backend.adjust_context_for_speaker(
            memory_text,
            memory_kind,
            speaker,
            current_user_name=current_user_name,
            character_name=character_name,
            character_gender=character_gender,
        )

    def list(
        self,
        conf_uid: Optional[str] = None,
        limit: int = 50,
        kind: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self._backend.list(conf_uid, limit, kind)

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
        return self._backend.search(
            query,
            conf_uid=conf_uid,
            top_k=top_k,
            kind=kind,
            kinds=kinds,
            min_importance=min_importance,
            since_ts=since_ts,
            until_ts=until_ts,
        )

    def clear(self, conf_uid: Optional[str] = None) -> int:
        return self._backend.clear(conf_uid)

    def prune(
        self,
        conf_uid: Optional[str] = None,
        max_age_ts: Optional[int] = None,
        max_importance: Optional[float] = None,
    ) -> int:
        return self._backend.prune(conf_uid, max_age_ts, max_importance)

    def delete(self, ids: List[str]) -> int:
        return self._backend.delete(ids)
