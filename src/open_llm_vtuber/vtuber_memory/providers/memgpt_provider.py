from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from ..interface import VtuberMemoryInterface
from ...memory.chroma_memory import ChromaMemory, MemoryItem
from ...memory.memory_schema import (
    adjust_context_for_speaker as _adjust_ctx,
    MemoryKind,
    MemoryItemTyped,
)
from ...config_manager.system import SystemConfig
from ..config import (
    CHROMA_PERSIST_DIR,
    CHROMA_SESSION_COLLECTION,
    CHROMA_LTM_COLLECTION,
)


class MemGPTProvider(VtuberMemoryInterface):
    """MemGPT-like provider using ChromaDB with pluggable embeddings.

    - Uses OpenAI embeddings if API configured, else falls back to local sentence-transformers via ChromaMemory
    - Provides the same surface as VtuberMemoryInterface
    - Maintains separate collections for Session and Long-Term memory
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

        # Build Chroma indexes with optional external embedder
        try:
            base_dir = chroma_path or CHROMA_PERSIST_DIR
            # Long-Term collection name (prefer system_config override or provided collection)
            ltm_collection = collection or CHROMA_LTM_COLLECTION
            session_collection = CHROMA_SESSION_COLLECTION

            self._chroma_ltm = ChromaMemory(
                persist_dir=base_dir,
                collection=ltm_collection,
                model_name=self._embeddings_model,
                embedder_func=_embedder if self._use_openai else None,
            )
            self._chroma_session = ChromaMemory(
                persist_dir=base_dir,
                collection=session_collection,
                model_name=self._embeddings_model,
                embedder_func=_embedder if self._use_openai else None,
            )
            if (
                not self._chroma_ltm.is_available()
                or not self._chroma_session.is_available()
            ):
                self._enabled = False
                logger.warning("MemGPTProvider disabled (Chroma unavailable)")
        except Exception as e:
            self._enabled = False
            logger.warning(f"MemGPTProvider init failed: {e}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    # --- Helpers ---
    def _choose_index(self, meta: Dict[str, Any]) -> ChromaMemory:
        """Route to session index for transient kinds, else LTM.

        Heuristic: if metadata.kind in {"chat", "Emotions"} or meta.get("is_session") == True
        then use session collection, else LTM.
        """
        kind = str(meta.get("kind", "chat"))
        is_session = bool(meta.get("is_session", False))
        sessionish = {"chat", "Emotions", "ObjectivesTemp", "SessionEvent"}
        return (
            self._chroma_session
            if is_session or kind in sessionish
            else self._chroma_ltm
        )

    # --- API ---
    def add_memory(self, item: Dict[str, Any] | MemoryItemTyped) -> int:
        if not self.enabled:
            return 0
        try:
            # Handle both Dict and MemoryItemTyped
            if hasattr(item, "text"):  # MemoryItemTyped
                # Convert MemoryItemTyped to dict format
                item_dict = {
                    "id": f"{item.conf_uid}:{item.history_uid}:mem",
                    "text": item.text,
                    "conf_uid": item.conf_uid,
                    "history_uid": item.history_uid,
                    "kind": str(
                        item.kind.value if hasattr(item.kind, "value") else item.kind
                    ),
                    "importance": item.importance,
                    "timestamp": item.timestamp,
                    "tags": item.tags or [],
                    "emotion": item.emotion,
                    "context_window": item.context_window,
                }
            else:  # Dict
                item_dict = item

            index = self._choose_index(item_dict)
            mem = MemoryItem(
                id=str(item_dict.get("id") or item_dict.get("conf_uid") or "mem"),
                text=str(item_dict.get("text", "")),
                metadata=item_dict,
            )
            return index.upsert([mem])
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
                    "is_session": True,
                },
            )
            for i, txt in enumerate(facts)
        ]
        return self._chroma_session.upsert(items)

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
            # Mark as session unless explicitly targeting LTM kinds
            kind = str(meta["kind"]).lower()
            is_session = kind in {"chat", "emotions", "objective_temp", "sessionevent"}
            meta["is_session"] = is_session
            items.append(
                MemoryItem(id=f"{conf_uid}:{history_uid}:{i}", text=text, metadata=meta)
            )
        # Split by target index
        session_items = [m for m in items if m.metadata.get("is_session")]
        ltm_items = [m for m in items if not m.metadata.get("is_session")]
        added = 0
        if session_items:
            added += int(self._chroma_session.upsert(session_items) or 0)
        if ltm_items:
            added += int(self._chroma_ltm.upsert(ltm_items) or 0)
        return added

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
        # Query both indexes and merge
        results: List[Dict[str, Any]] = []
        for idx in (self._chroma_session, self._chroma_ltm):
            try:
                chunk = idx.query(query, top_k=limit, where=filt)
            except Exception:
                chunk = []
            for _id, text, score, meta in chunk:
                results.append(
                    {
                        "id": _id,
                        "text": text,
                        "relevance": score,
                        "metadata": meta,
                        "kind": (meta or {}).get("kind"),
                    }
                )
        # Sort by relevance ascending distance if provided, else leave order
        try:
            results.sort(key=lambda r: float(r.get("relevance", 0.0)))
        except Exception:
            pass
        return results[:limit]

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
        # Merge from both
        out = []
        for idx in (self._chroma_session, self._chroma_ltm):
            out.extend(idx.list(limit=limit, where=where))

        # Sort by timestamp desc, then importance desc if available
        def _key(it: Dict[str, Any]):
            meta = it.get("metadata") or {}
            ts = float(meta.get("timestamp") or 0)
            imp = float(meta.get("importance") or 0)
            return (ts, imp)

        try:
            out.sort(key=_key, reverse=True)
        except Exception:
            pass
        return out[:limit]

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
        # Retrieve more than needed, then apply filters and sort
        limit = max(1, int(top_k or 5))
        try:
            results = self.get_relevant_memories(
                query, conf_uid=conf_uid, limit=limit * 3
            )
        except Exception:
            results = []
        # Normalize filters
        selected_kinds = [
            str(k).strip()
            for k in (kinds or ([] if not kind else [kind]))
            if str(k).strip()
        ]
        has_kind_filter = len(selected_kinds) > 0

        def _pass_filters(it: Dict[str, Any]) -> bool:
            meta = it.get("metadata") or {}
            if has_kind_filter:
                k = str((meta.get("kind") or it.get("kind") or "")).strip()
                if k not in selected_kinds:
                    return False
            if min_importance is not None:
                try:
                    if float(meta.get("importance", 0.0)) < float(min_importance):
                        return False
                except Exception:
                    return False
            if since_ts is not None:
                try:
                    if int(meta.get("timestamp") or 0) < int(since_ts):
                        return False
                except Exception:
                    return False
            if until_ts is not None:
                try:
                    if int(meta.get("timestamp") or 0) > int(until_ts):
                        return False
                except Exception:
                    return False
            return True

        filtered = [r for r in results if _pass_filters(r)]

        # Sort: timestamp desc, then importance desc; fallback to relevance/score if absent
        def _sort_key(it: Dict[str, Any]):
            meta = it.get("metadata") or {}
            ts = float(meta.get("timestamp") or 0)
            imp = float(meta.get("importance") or 0)
            # If no timestamp, keep original order by using negative relevance (some paths use ascending distance)
            rel = -float(it.get("relevance", 0.0) or 0.0)
            return (ts, imp, rel)

        try:
            filtered.sort(key=_sort_key, reverse=True)
        except Exception:
            pass
        return filtered[:limit]

    def clear(self, conf_uid: Optional[str] = None) -> int:
        # Not exposing wholesale clear for safety
        return 0

    def prune(
        self,
        conf_uid: Optional[str] = None,
        max_age_ts: Optional[int] = None,
        max_importance: Optional[float] = None,
    ) -> int:
        # Not implemented for provider-level; can be implemented at service layer
        return 0

    def delete(self, ids: List[str]) -> int:
        try:
            # Attempt delete on both
            deleted = 0
            for idx in (self._chroma_session, self._chroma_ltm):
                deleted += int(idx.delete_by_ids(ids) or 0)
            return deleted
        except Exception:
            return 0
