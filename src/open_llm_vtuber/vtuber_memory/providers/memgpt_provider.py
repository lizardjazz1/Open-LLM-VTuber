from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from ..interface import VtuberMemoryInterface


class MemGPTProvider(VtuberMemoryInterface):
    """MemGPT-based provider (stub).

    This is a thin adapter intended to integrate MemGPT when available.
    For now, it logs operations and disables itself if MemGPT deps are missing.
    """

    def __init__(
        self, *, chroma_path: str, collection: str, embeddings_model: str
    ) -> None:
        try:
            import memgpt as _memgpt  # type: ignore

            _ = getattr(_memgpt, "__version__", None)  # no-op usage to satisfy linter
        except Exception as e:
            logger.warning(f"MemGPT not available: {e}")
            self._enabled = False
            return
        self._enabled = True
        # TODO: Wire actual MemGPT init here using provided chroma_path/collection/embeddings_model
        logger.info(
            f"MemGPTProvider initialized (collection={collection}, path={chroma_path}, emb={embeddings_model})"
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    # Below methods mimic current interface but are no-ops until full MemGPT integration
    def add_memory(self, item: Dict[str, Any]) -> int:
        if not self.enabled:
            return 0
        logger.debug(f"[memgpt] add_memory: {item.get('text', '')[:80]}")
        return 1

    def add_facts(
        self, facts: List[str], conf_uid: str, history_uid: str, kind: str = "chat"
    ) -> int:
        if not self.enabled:
            return 0
        logger.debug(f"[memgpt] add_facts: {len(facts)}")
        return len(facts)

    def add_facts_with_meta(
        self,
        entries: List[Dict[str, Any]],
        conf_uid: str,
        history_uid: str,
        default_kind: str = "chat",
    ) -> int:
        if not self.enabled:
            return 0
        logger.debug(f"[memgpt] add_facts_with_meta: {len(entries)}")
        return len(entries)

    def get_relevant_memories(
        self,
        query: str,
        conf_uid: Optional[str],
        limit: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        logger.debug(f"[memgpt] query: {query[:80]}")
        return []

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
        return memory_text

    def list(
        self,
        conf_uid: Optional[str] = None,
        limit: int = 50,
        kind: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return []

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
        return []

    def clear(self, conf_uid: Optional[str] = None) -> int:
        return 0

    def prune(
        self,
        conf_uid: Optional[str] = None,
        max_age_ts: Optional[int] = None,
        max_importance: Optional[float] = None,
    ) -> int:
        return 0

    def delete(self, ids: List[str]) -> int:
        return 0
