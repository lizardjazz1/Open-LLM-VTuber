from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from ..interface import VtuberMemoryInterface
from ...memory.chroma_memory import ChromaMemory, MemoryItem
from ...memory.memory_schema import (
    adjust_context_for_speaker as _adjust_ctx,
    MemoryKind,
)
from ...config_manager.system import SystemConfig


class MemGPTProvider(VtuberMemoryInterface):
    """MemGPT-like provider using ChromaDB with pluggable embeddings.

    - Uses OpenAI embeddings if API configured, else falls back to local sentence-transformers via ChromaMemory
    - Provides the same surface as VtuberMemoryInterface
    """

    def __init__(
        self,
        *,
        chroma_path: str,
        collection: str,
        embeddings_model: str,
        system_config: SystemConfig | None = None,
    ) -> None:
        self._enabled = True
        self._openai_client = None
        self._embeddings_model = embeddings_model
        self._use_openai = False

        # Initialize optional OpenAI client
        try:
            from openai import OpenAI  # type: ignore

            api_key = (
                getattr(system_config, "embeddings_api_key", None)
                if system_config
                else None
            )
            api_base = (
                getattr(system_config, "embeddings_api_base", None)
                if system_config
                else None
            )
            if api_key:
                self._openai_client = OpenAI(api_key=api_key, base_url=api_base or None)
                self._use_openai = True
                logger.info("MemGPTProvider: OpenAI embeddings enabled")
        except Exception as e:
            self._openai_client = None
            self._use_openai = False
            logger.debug(f"OpenAI embeddings disabled: {e}")

        def _embedder(texts: List[str]) -> List[List[float]]:
            if not self._use_openai or not self._openai_client:
                # Signal to ChromaMemory to use local embedder
                raise RuntimeError("external embedder not available")
            resp = self._openai_client.embeddings.create(
                model=self._embeddings_model, input=texts
            )  # type: ignore[attr-defined]
            # openai python sdk: resp.data is list with .embedding
            return [list(d.embedding) for d in resp.data]  # type: ignore[attr-defined]

        # Build Chroma index with optional external embedder
        try:
            self._chroma = ChromaMemory(
                persist_dir=chroma_path,
                collection=collection,
                model_name=self._embeddings_model,
                embedder_func=_embedder if self._use_openai else None,
            )
            if not self._chroma.is_available():
                self._enabled = False
                logger.warning("MemGPTProvider disabled (Chroma unavailable)")
        except Exception as e:
            self._enabled = False
            logger.warning(f"MemGPTProvider init failed: {e}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def add_memory(self, item: Dict[str, Any]) -> int:
        if not self.enabled:
            return 0
        try:
            mem = MemoryItem(
                id=str(item.get("id") or item.get("conf_uid") or "mem"),
                text=str(item.get("text", "")),
                metadata=item,
            )
            return self._chroma.upsert([mem])
        except Exception as e:
            logger.debug(f"memgpt.add_memory skipped: {e}")
            return 0

    def add_facts(
        self, facts: List[str], conf_uid: str, history_uid: str, kind: str = "chat"
    ) -> int:
        if not self.enabled or not facts:
            return 0
        items = [
            MemoryItem(
                id=f"{conf_uid}:{history_uid}:{i}",
                text=txt,
                metadata={
                    "conf_uid": conf_uid,
                    "history_uid": history_uid,
                    "kind": kind,
                },
            )
            for i, txt in enumerate(facts)
        ]
        return self._chroma.upsert(items)

    def add_facts_with_meta(
        self,
        entries: List[Dict[str, Any]],
        conf_uid: str,
        history_uid: str,
        default_kind: str = "chat",
    ) -> int:
        if not self.enabled or not entries:
            return 0
        items: List[MemoryItem] = []
        for i, e in enumerate(entries):
            text = str(e.get("text", "")).strip()
            if not text:
                continue
            meta = {
                "conf_uid": conf_uid,
                "history_uid": history_uid,
                "kind": e.get("kind", default_kind),
                "importance": float(e.get("importance", 0.5)),
                "timestamp": e.get("timestamp"),
                "tags": e.get("tags", []),
            }
            items.append(
                MemoryItem(id=f"{conf_uid}:{history_uid}:{i}", text=text, metadata=meta)
            )
        return self._chroma.upsert(items)

    def get_relevant_memories(
        self,
        query: str,
        conf_uid: Optional[str],
        limit: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.enabled or not query:
            return []
        filt: Optional[Dict[str, Any]] = None
        if conf_uid:
            filt = (
                {"conf_uid": conf_uid}
                if not where
                else {"$and": [{"conf_uid": conf_uid}, where]}
            )
        results = self._chroma.query(query, top_k=limit, where=filt)
        out: List[Dict[str, Any]] = []
        for _id, text, score, meta in results:
            out.append(
                {
                    "id": _id,
                    "text": text,
                    "relevance": score,
                    "metadata": meta,
                    "kind": (meta or {}).get("kind"),
                }
            )
        return out

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
        try:
            kind = MemoryKind(memory_kind)
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

    def list(
        self,
        conf_uid: Optional[str] = None,
        limit: int = 50,
        kind: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        where: Optional[Dict[str, Any]] = None
        conds: List[Dict[str, Any]] = []
        if conf_uid:
            conds.append({"conf_uid": conf_uid})
        if kind:
            conds.append({"kind": kind})
        if conds:
            where = conds[0] if len(conds) == 1 else {"$and": conds}
        return self._chroma.list(limit=limit, where=where)

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
        # Reuse get_relevant for now
        return self.get_relevant_memories(query, conf_uid=conf_uid, limit=top_k)

    def clear(self, conf_uid: Optional[str] = None) -> int:
        # Chroma API currently not exposed here for wholesale clear; return 0 to avoid destructive ops
        return 0

    def prune(
        self,
        conf_uid: Optional[str] = None,
        max_age_ts: Optional[int] = None,
        max_importance: Optional[float] = None,
    ) -> int:
        return 0

    def delete(self, ids: List[str]) -> int:
        try:
            return self._chroma.delete_by_ids(ids)
        except Exception:
            return 0
