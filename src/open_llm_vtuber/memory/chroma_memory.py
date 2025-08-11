from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import time
import json

from loguru import logger

try:
    import chromadb  # type: ignore
    from chromadb.config import Settings  # type: ignore
except Exception as e:
    chromadb = None  # type: ignore
    Settings = None  # type: ignore
    logger.warning(f"ChromaDB not available: {e}")

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:
    SentenceTransformer = None  # type: ignore
    logger.warning(f"sentence-transformers not available: {e}")


@dataclass
class MemoryItem:
    id: str
    text: str
    metadata: Dict[str, Any]


class ChromaMemory:
    def __init__(
        self,
        persist_dir: str = "cache/chroma",
        collection: str = "vtuber_memory",
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        embedding_dim: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        # Allow overrides via kwargs for future SystemConfig plumbing
        persist_dir = str(kwargs.get("persist_dir", persist_dir))
        collection = str(kwargs.get("collection", collection))
        model_name = str(kwargs.get("model_name", model_name))

        self.persist_dir = persist_dir
        self.collection_name = collection
        self.model_name = model_name
        self.embedding_dim = embedding_dim

        if chromadb is None or SentenceTransformer is None:
            self.client = None
            self.collection = None
            self.embedder = None
            logger.bind(component="chroma").warning(
                "ChromaMemory disabled (missing deps)"
            )
            return

        self.client = chromadb.PersistentClient(
            path=persist_dir, settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(collection)
        self.embedder = SentenceTransformer(model_name)
        size = -1
        try:
            size = int(getattr(self.collection, "count")())  # type: ignore[operator]
        except Exception:
            pass
        # // DEBUG: [FIXED] Structured init log with collection_size | Ref: 7
        logger.bind(component="chroma").info(
            {
                "memgpt_operation": "init",
                "persist_dir": persist_dir,
                "collection": collection,
                "model": model_name,
                "collection_size": size,
            }
        )

    def is_available(self) -> bool:
        return self.collection is not None and self.embedder is not None

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if not self.embedder:
            logger.bind(component="chroma").warning(
                {
                    "memgpt_operation": "embed",
                    "status": "disabled",
                    "reason": "embedder_not_initialized",
                }
            )
            return []
        return [
            vec.tolist() if hasattr(vec, "tolist") else list(vec)
            for vec in self.embedder.encode(texts, normalize_embeddings=True)
        ]

    def _sanitize_metadata_value(self, value: Any) -> str | int | float | bool:
        """Coerce metadata value to Chroma-supported scalar types.

        Lists/dicts are JSON-encoded strings; other non-scalar types are stringified.
        """
        if isinstance(value, (str, int, float, bool)):
            return value
        try:
            # Prefer compact JSON for complex types
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(value)

    def _sanitize_metadata(
        self, meta: Dict[str, Any]
    ) -> Dict[str, str | int | float | bool]:
        return {k: self._sanitize_metadata_value(v) for k, v in (meta or {}).items()}

    def upsert(self, items: List[MemoryItem]) -> int:
        if not self.is_available() or not items:
            return 0
        t0 = time.monotonic()
        try:
            texts = [it.text for it in items]
            embeddings = self._embed(texts)
            if not embeddings:
                logger.bind(component="chroma").warning(
                    {
                        "memgpt_operation": "upsert",
                        "collection": self.collection_name,
                        "status": "skipped",
                        "reason": "no_embeddings",
                        "items": len(items),
                    }
                )
                return 0
            self.collection.upsert(
                ids=[it.id for it in items],
                embeddings=embeddings,
                metadatas=[self._sanitize_metadata(it.metadata) for it in items],
                documents=texts,
            )
            size = -1
            try:
                size = int(getattr(self.collection, "count")())  # type: ignore[operator]
            except Exception:
                pass
            # // DEBUG: [FIXED] Log memgpt_operation + collection_size | Ref: 1,7
            logger.bind(component="chroma").info(
                {
                    "memgpt_operation": "upsert",
                    "collection": self.collection_name,
                    "added": len(items),
                    "collection_size": size,
                    "latency_ms": int((time.monotonic() - t0) * 1000),
                }
            )
            return len(items)
        except Exception as e:
            logger.bind(component="chroma").error(
                {
                    "memgpt_operation": "upsert",
                    "collection": self.collection_name,
                    "status": "error",
                    "error_details": str(e),
                    "latency_ms": int((time.monotonic() - t0) * 1000),
                }
            )
            return 0

    def query(
        self, query_text: str, top_k: int = 5, where: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        if not self.is_available() or not query_text:
            return []
        t0 = time.monotonic()
        try:
            q = self.collection.query(
                query_embeddings=self._embed([query_text]), n_results=top_k, where=where
            )
            results: List[Tuple[str, str, float, Dict[str, Any]]] = []
            ids = (q.get("ids") or [[]])[0]
            docs = (q.get("documents") or [[]])[0]
            dists = (q.get("distances") or [[]])[0]
            metas = (q.get("metadatas") or [[]])[0]
            for i in range(min(len(ids), len(docs), len(dists))):
                results.append(
                    (
                        ids[i],
                        docs[i],
                        float(dists[i]),
                        metas[i] if metas and i < len(metas) else {},
                    )
                )
            # // DEBUG: [FIXED] Query logging with memgpt_operation | Ref: 7
            logger.bind(component="chroma").info(
                {
                    "memgpt_operation": "query",
                    "collection": self.collection_name,
                    "top_k": top_k,
                    "hits": len(results),
                    "latency_ms": int((time.monotonic() - t0) * 1000),
                }
            )
            return results
        except Exception as e:
            logger.bind(component="chroma").error(
                {
                    "memgpt_operation": "query",
                    "collection": self.collection_name,
                    "status": "error",
                    "error_details": str(e),
                    "latency_ms": int((time.monotonic() - t0) * 1000),
                }
            )
            return []

    def list(
        self, limit: int = 50, where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if not self.is_available():
            return []
        t0 = time.monotonic()
        try:
            # Chroma doesn't expose list-all docs directly via API; emulate via where+limit on get (ids/docs)
            # We use get with where and limit (if supported); if not, return empty for safety.
            res = self.collection.get(where=where, limit=limit)  # type: ignore[attr-defined]
            out: List[Dict[str, Any]] = []
            ids = res.get("ids") or []
            docs = res.get("documents") or []
            metas = res.get("metadatas") or []
            for i in range(min(len(ids), len(docs))):
                out.append(
                    {
                        "id": ids[i],
                        "text": docs[i],
                        "metadata": metas[i] if metas and i < len(metas) else {},
                    }
                )
            logger.bind(component="chroma").info(
                {
                    "memgpt_operation": "list",
                    "collection": self.collection_name,
                    "returned": len(out),
                    "latency_ms": int((time.monotonic() - t0) * 1000),
                }
            )
            return out
        except Exception as e:
            logger.bind(component="chroma").warning(
                {
                    "memgpt_operation": "list",
                    "collection": self.collection_name,
                    "status": "error",
                    "error_details": str(e),
                    "latency_ms": int((time.monotonic() - t0) * 1000),
                }
            )
            return []

    def delete_by_ids(self, ids: List[str]) -> int:
        if not self.is_available() or not ids:
            return 0
        t0 = time.monotonic()
        try:
            self.collection.delete(ids=ids)
            size = -1
            try:
                size = int(getattr(self.collection, "count")())  # type: ignore[operator]
            except Exception:
                pass
            logger.bind(component="chroma").info(
                {
                    "memgpt_operation": "delete",
                    "collection": self.collection_name,
                    "deleted": len(ids),
                    "collection_size": size,
                    "latency_ms": int((time.monotonic() - t0) * 1000),
                }
            )
            return len(ids)
        except Exception as e:
            logger.bind(component="chroma").warning(
                {
                    "memgpt_operation": "delete",
                    "collection": self.collection_name,
                    "status": "error",
                    "error_details": str(e),
                    "latency_ms": int((time.monotonic() - t0) * 1000),
                }
            )
            return 0
