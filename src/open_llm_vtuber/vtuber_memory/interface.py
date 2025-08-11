from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol


class VtuberMemoryInterface(Protocol):
    """Interface for Vtuber memory providers.

    This interface abstracts long-term memory operations so that different
    backends (e.g., ChromaDB-only, MemGPT/Letta hybrid) can be swapped
    without changing conversation or agent code.
    """

    enabled: bool

    def add_memory(self, item: Dict[str, Any]) -> int:
        """Add a structured memory item.

        Args:
            item: A dict containing at least: text, kind, conf_uid, history_uid.

        Returns:
            int: 1 on success, 0 otherwise.
        """

    def add_facts(
        self, facts: List[str], conf_uid: str, history_uid: str, kind: str = "chat"
    ) -> int:
        """Add multiple simple text facts with same kind."""

    def add_facts_with_meta(
        self,
        entries: List[Dict[str, Any]],
        conf_uid: str,
        history_uid: str,
        default_kind: str = "chat",
    ) -> int:
        """Add entries with rich metadata."""

    def get_relevant_memories(
        self,
        query: str,
        conf_uid: Optional[str],
        limit: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search relevant memories with ranking and return top items."""

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
        """Speaker-aware rendering: SELF→first person, USER→address user, etc."""

    def list(
        self,
        conf_uid: Optional[str] = None,
        limit: int = 50,
        kind: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List stored memories with optional filters."""

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
        """Filtered search API for UI use."""

    def clear(self, conf_uid: Optional[str] = None) -> int:
        """Clear memory entries (dangerous without conf_uid)."""

    def prune(
        self,
        conf_uid: Optional[str] = None,
        max_age_ts: Optional[int] = None,
        max_importance: Optional[float] = None,
    ) -> int:
        """Prune low-importance/old entries."""

    def delete(self, ids: List[str]) -> int:
        """Delete by ids."""
