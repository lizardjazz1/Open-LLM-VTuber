from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from ..interface import VtuberMemoryInterface


class LettaProvider(VtuberMemoryInterface):
    """Letta-based provider (stub).

    Acts as a placeholder for a Letta-backed memory module.
    """

    def __init__(self, *, host: str = "localhost", port: int = 8283) -> None:
        # Here we could probe Letta server health. For now, mark disabled.
        self._enabled = False
        logger.info(
            f"LettaProvider initialized (host={host}, port={port}) [disabled stub]"
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def add_memory(self, item: Dict[str, Any]) -> int:
        return 0

    def add_facts(
        self, facts: List[str], conf_uid: str, history_uid: str, kind: str = "chat"
    ) -> int:
        return 0

    def add_facts_with_meta(
        self,
        entries: List[Dict[str, Any]],
        conf_uid: str,
        history_uid: str,
        default_kind: str = "chat",
    ) -> int:
        return 0

    def get_relevant_memories(
        self,
        query: str,
        conf_uid: Optional[str],
        limit: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
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
