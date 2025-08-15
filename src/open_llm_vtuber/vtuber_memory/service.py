from __future__ import annotations

from typing import Any, Dict, List, Optional

# Legacy MemoryService removed; vtuber_memory is the primary interface now
from ..memory.memory_schema import MemoryItemTyped, MemoryKind
from ..config_manager.system import SystemConfig
from ..config_manager.character import CharacterConfig
from .interface import VtuberMemoryInterface
from .providers.memgpt_provider import MemGPTProvider
from .providers.letta_provider import LettaProvider


class VtuberMemoryService(VtuberMemoryInterface):
    """Vtuber memory service orchestrating backend providers.

    Default provider is MemGPT-like over Chroma unless explicitly set to 'letta'.
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
                or "memgpt"
            )
            .strip()
            .lower()
        )
        self._backend: VtuberMemoryInterface
        if not enabled:
            # Disabled stub
            self._backend = LettaProvider()  # use stub with enabled=False
            self._backend._enabled = False  # type: ignore[attr-defined]
            return
        if provider_name == "letta":
            self._backend = LettaProvider()
        else:
            # MemGPT-like (Chroma) as default
            from .config import CHROMA_PERSIST_DIR

            chroma_path = (
                getattr(system_config, "chroma_persist_dir", CHROMA_PERSIST_DIR)
                if system_config
                else CHROMA_PERSIST_DIR
            )
            collection = (
                getattr(system_config, "chroma_collection", "vtuber_ltm")
                if system_config
                else "vtuber_ltm"
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
                system_config=system_config,
            )

    @property
    def enabled(self) -> bool:
        return bool(self._backend and getattr(self._backend, "enabled", False))

    def add_memory(self, item: Dict[str, Any]) -> int:
        try:
            if hasattr(self._backend, "add_memory"):
                return self._backend.add_memory(item)  # type: ignore[no-any-return]
            # Fallback type normalization (should not happen)
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
            return self._backend.add_memory(typed)  # type: ignore[arg-type]
        except Exception:
            return 0

    def add_facts(
        self, facts: List[str], conf_uid: str, history_uid: str, kind: str = "chat"
    ) -> int:
        return self._backend.add_facts(facts, conf_uid, history_uid, kind)

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
        # If backend supports it, use it
        try:
            out = self._backend.prune(conf_uid, max_age_ts, max_importance)
            if out:
                return out
        except Exception:
            pass
        # Fallback: list and delete
        try:
            items = self._backend.list(conf_uid=conf_uid, limit=5000)
            to_delete: List[str] = []
            for it in items:
                meta = it.get("metadata") or {}
                ts = meta.get("timestamp")
                imp = meta.get("importance", 0.0)
                if (
                    max_age_ts is not None
                    and isinstance(ts, (int, float))
                    and ts <= int(max_age_ts)
                ) or (
                    max_importance is not None and float(imp) < float(max_importance)
                ):
                    if it.get("id"):
                        to_delete.append(str(it["id"]))
            if to_delete:
                return self._backend.delete(to_delete)
        except Exception:
            return 0
        return 0

    def delete(self, ids: List[str]) -> int:
        return self._backend.delete(ids)

    # --- helpers ---
    def prune_session_by_ttl(self, ttl_sec: int) -> int:
        """Prune session entries older than now - ttl_sec based on metadata.timestamp and is_session flag.

        Excludes any entries that belong to active history sessions if needed.
        """
        return self.prune_session_by_ttl_ex(ttl_sec=ttl_sec, exclude_history_uids=None)

    def prune_session_by_ttl_ex(
        self, *, ttl_sec: int, exclude_history_uids: Optional[List[str]]
    ) -> int:
        import time

        now = int(time.time())
        cutoff = now - int(max(60, ttl_sec))
        items = self._backend.list(limit=5000)
        ids: List[str] = []
        exclude_set = set(exclude_history_uids or [])
        for it in items:
            meta = it.get("metadata") or {}
            if exclude_set and str(meta.get("history_uid")) in exclude_set:
                continue
            if (
                bool(meta.get("is_session"))
                and int(meta.get("timestamp") or 0) <= cutoff
            ):
                if it.get("id"):
                    ids.append(str(it["id"]))
        if ids:
            return self._backend.delete(ids)
        return 0

    def prune_ltm(
        self,
        *,
        conf_uid: Optional[str],
        max_age_ts: Optional[int],
        max_importance: Optional[float],
    ) -> int:
        """Prune only LTM entries (is_session != True) with optional age/importance constraints."""
        items = self._backend.list(conf_uid=conf_uid, limit=10000)
        ids: List[str] = []
        for it in items:
            meta = it.get("metadata") or {}
            if bool(meta.get("is_session")):
                continue
            ts = meta.get("timestamp")
            imp = meta.get("importance", 0.0)
            too_old = (
                max_age_ts is not None
                and isinstance(ts, (int, float))
                and ts <= int(max_age_ts)
            )
            low_imp = max_importance is not None and float(imp) < float(max_importance)
            if too_old or low_imp:
                if it.get("id"):
                    ids.append(str(it["id"]))
        if ids:
            return self._backend.delete(ids)
        return 0
