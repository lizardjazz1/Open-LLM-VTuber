from __future__ import annotations

from typing import Any, Dict, Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from pydantic import BaseModel, Field

from ..service_context import ServiceContext

import json
import io
import yaml

router = APIRouter(prefix="/admin", tags=["admin-memory"])

# Global context cache
_global_context: Optional[ServiceContext] = None


def set_global_context(context: ServiceContext) -> None:
    """Set the global context for admin routes."""
    global _global_context
    _global_context = context


def get_context() -> ServiceContext:
    """Get the global context or create a fallback one."""
    global _global_context

    # First try to use the global context
    if _global_context is not None:
        return _global_context

    # Fallback: create a minimal context
    from ..config_manager.utils import validate_config, read_yaml
    from ..service_context import ServiceContext

    try:
        config = validate_config(read_yaml("conf.yaml"))
        context = ServiceContext()
        context.load_from_config(config)
        return context
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Service context not available: {e}"
        )


class MemoryEditRequest(BaseModel):
    id: str = Field(..., description="Memory ID to edit")
    new_content: str = Field(..., description="New content text")


class RelationshipUpdateRequest(BaseModel):
    user_id: str
    affinity_delta: int = 0
    trust_delta: int = 0
    username: Optional[str] = None
    realname: Optional[str] = None


@router.get("/relationship/get")
async def relationship_get(
    user_id: str = Query(...), ctx: ServiceContext = Depends(get_context)
):
    from ..vtuber_memory.relationships import RelationshipsDB

    db_path = getattr(
        ctx.system_config, "relationships_db_path", "cache/relationships.sqlite3"
    )
    db = RelationshipsDB(db_path)
    rec = db.get(user_id)
    return {
        "user_id": rec.user_id if rec else user_id,
        "username": rec.username if rec else None,
        "realname": rec.realname if rec else None,
        "affinity": rec.affinity if rec else 0,
        "trust": rec.trust if rec else 0,
        "interaction_count": rec.interaction_count if rec else 0,
        "last_interaction": rec.last_interaction if rec else None,
    }


@router.post("/relationship/update")
async def relationship_update(
    req: RelationshipUpdateRequest, ctx: ServiceContext = Depends(get_context)
):
    from ..vtuber_memory.relationships import RelationshipsDB

    db_path = getattr(
        ctx.system_config, "relationships_db_path", "cache/relationships.sqlite3"
    )
    db = RelationshipsDB(db_path)

    out: Dict[str, Any] = {}
    if req.username:
        db.set_username(req.user_id, req.username)
        out["username"] = req.username
    if req.realname:
        db.set_realname(req.user_id, req.realname)
        out["realname"] = req.realname
    if req.affinity_delta:
        out["affinity"] = db.adjust_affinity(req.user_id, req.affinity_delta)
    if req.trust_delta:
        out["trust"] = db.adjust_trust(req.user_id, req.trust_delta)
    rec = db.get(req.user_id)
    if rec:
        out.update(
            {
                "user_id": rec.user_id,
                "affinity": rec.affinity,
                "trust": rec.trust,
                "interaction_count": rec.interaction_count,
                "last_interaction": rec.last_interaction,
                "realname": rec.realname,
            }
        )
    return out


@router.get("/relationship/list_recent")
async def relationship_list_recent(
    limit: int = Query(20, ge=1, le=500), ctx: ServiceContext = Depends(get_context)
):
    from ..vtuber_memory.relationships import RelationshipsDB

    db_path = getattr(
        ctx.system_config, "relationships_db_path", "cache/relationships.sqlite3"
    )
    db = RelationshipsDB(db_path)
    rows = db.list_recent(limit=limit)
    return {
        "items": [
            {
                "user_id": r.user_id,
                "username": r.username,
                "realname": r.realname,
                "affinity": r.affinity,
                "trust": r.trust,
                "interaction_count": r.interaction_count,
                "last_interaction": r.last_interaction,
            }
            for r in rows
        ]
    }


@router.get("/memory/export")
async def memory_export(
    conf_uid: Optional[str] = None,
    kind: Optional[str] = None,
    limit: int = Query(1000, ge=1, le=50000),
    fmt: str = Query("json", pattern="^(json|yaml)$"),
    ctx: ServiceContext = Depends(get_context),
):
    svc = ctx.vtuber_memory_service
    if not svc or not getattr(svc, "enabled", False):
        raise HTTPException(status_code=503, detail="Memory not available")
    items = svc.list(conf_uid=conf_uid, limit=limit, kind=kind)
    payload = {"items": items}
    if fmt == "yaml":
        data = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
        return json.loads(json.dumps({"format": "yaml", "data": data}))
    else:
        return payload


@router.post("/memory/import")
async def memory_import(
    conf_uid: str = Query(...),
    history_uid: str = Query(...),
    default_kind: str = Query("chat"),
    file: UploadFile = File(...),
    fmt: str = Query("json", pattern="^(json|yaml)$"),
    ctx: ServiceContext = Depends(get_context),
):
    svc = ctx.vtuber_memory_service
    if not svc or not getattr(svc, "enabled", False):
        raise HTTPException(status_code=503, detail="Memory not available")
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
        if fmt == "yaml":
            payload = yaml.safe_load(io.StringIO(text))
        else:
            payload = json.loads(text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid {fmt} payload: {e}")
    items = payload.get("items") if isinstance(payload, dict) else None
    if not items or not isinstance(items, list):
        raise HTTPException(
            status_code=400, detail="Payload must be an object with list 'items'"
        )

    entries = []
    for it in items:
        txt = (it.get("text") if isinstance(it, dict) else None) or ""
        if not txt.strip():
            continue
        meta = it.get("metadata") or {}
        entries.append(
            {
                "text": txt,
                "kind": meta.get("kind", default_kind),
                "importance": float(meta.get("importance", 0.5)),
                "tags": meta.get("tags") or [],
                "timestamp": meta.get("timestamp"),
            }
        )
    added = svc.add_facts_with_meta(
        entries, conf_uid=conf_uid, history_uid=history_uid, default_kind=default_kind
    )
    return {"imported": int(added or 0)}


@router.post("/memory/prune_session_ttl")
async def memory_prune_session_ttl(
    ttl_sec: int = Query(
        0, description="Override TTL in seconds; 0 uses server default"
    ),
    ctx: ServiceContext = Depends(get_context),
):
    svc = ctx.vtuber_memory_service
    if not svc or not getattr(svc, "enabled", False):
        raise HTTPException(status_code=503, detail="Memory not available")
    try:
        from ..vtuber_memory.config import SESSION_TTL_SEC
    except Exception:
        SESSION_TTL_SEC = 7 * 24 * 3600
    effective = int(ttl_sec or SESSION_TTL_SEC)
    active = [ctx.history_uid] if ctx.history_uid else []
    pruned = svc.prune_session_by_ttl_ex(ttl_sec=effective, exclude_history_uids=active)
    return {"pruned": int(pruned or 0), "ttl_sec": effective}


@router.post("/memory/consolidate")
async def memory_consolidate(ctx: ServiceContext = Depends(get_context)):
    await ctx.trigger_memory_consolidation(reason="manual")
    return {"status": "ok"}


@router.post("/memory/deep_consolidation")
async def memory_deep_consolidation(ctx: ServiceContext = Depends(get_context)):
    # Expose deep consolidation manually for admin use
    try:
        await ctx._deep_consolidation()  # type: ignore[attr-defined]
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/search")
async def memory_search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=50),
    kind: Optional[str] = None,
    kinds: Optional[List[str]] = Query(None),
    conf_uid: Optional[str] = None,
    min_importance: Optional[float] = Query(None),
    since_ts: Optional[int] = Query(None),
    until_ts: Optional[int] = Query(None),
    ctx: ServiceContext = Depends(get_context),
):
    svc = ctx.vtuber_memory_service
    if not svc or not getattr(svc, "enabled", False):
        raise HTTPException(status_code=503, detail="Memory not available")
    items = svc.search(
        q,
        conf_uid=conf_uid,
        top_k=top_k,
        kind=kind,
        kinds=kinds,
        min_importance=min_importance,
        since_ts=since_ts,
        until_ts=until_ts,
    )

    # Ensure metadata fields and sort for UI convenience (timestamp desc, importance desc)
    def _ensure_meta(it: Dict[str, Any]) -> Dict[str, Any]:
        meta = dict(it.get("metadata") or {})
        if "timestamp" not in meta:
            meta["timestamp"] = 0
        if "importance" not in meta:
            meta["importance"] = 0.0
        it["metadata"] = meta
        return it

    items = [_ensure_meta(i) for i in items]
    try:
        items.sort(
            key=lambda i: (
                float((i.get("metadata") or {}).get("timestamp") or 0),
                float((i.get("metadata") or {}).get("importance") or 0.0),
            ),
            reverse=True,
        )
    except Exception:
        pass
    return {"items": items}


class DeleteManyRequest(BaseModel):
    ids: List[str]


@router.post("/memory/delete_many")
async def memory_delete_many(
    req: DeleteManyRequest, ctx: ServiceContext = Depends(get_context)
):
    svc = ctx.vtuber_memory_service
    if not svc or not getattr(svc, "enabled", False):
        raise HTTPException(status_code=503, detail="Memory not available")
    if not req.ids:
        return {"deleted": 0}
    try:
        deleted = svc.delete([str(i) for i in req.ids])
        return {"deleted": int(deleted or 0)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RetagManyRequest(BaseModel):
    ids: List[str]
    new_kind: str


@router.post("/memory/retag_many")
async def memory_retag_many(
    req: RetagManyRequest, ctx: ServiceContext = Depends(get_context)
):
    svc = ctx.vtuber_memory_service
    if not svc or not getattr(svc, "enabled", False):
        raise HTTPException(status_code=503, detail="Memory not available")
    if not req.ids or not req.new_kind:
        return {"updated": 0}
    try:
        # Build index of existing items for metadata reuse
        items = svc.list(limit=10000)
        index = {str(it.get("id")): it for it in items}
        updated = 0
        for mid in req.ids:
            it = index.get(str(mid))
            if not it:
                continue
            meta = it.get("metadata") or {}
            text = str(it.get("text") or "").strip()
            if not text:
                continue
            try:
                svc.delete([str(mid)])
            except Exception:
                pass
            entry = {
                "text": text,
                "kind": req.new_kind,
                "importance": float(meta.get("importance", 0.5)),
                "tags": meta.get("tags") or [],
                "timestamp": meta.get("timestamp"),
            }
            svc.add_facts_with_meta(
                [entry],
                meta.get("conf_uid", ""),
                meta.get("history_uid", ""),
                req.new_kind,
            )
            updated += 1
        return {"updated": updated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/edit")
async def memory_edit(
    req: MemoryEditRequest, ctx: ServiceContext = Depends(get_context)
):
    svc = ctx.vtuber_memory_service
    if not svc or not getattr(svc, "enabled", False):
        raise HTTPException(status_code=503, detail="Memory not available")
    # Our current backend doesn't support in-place edit in Chroma; emulate by delete+add
    items = svc.list(limit=2000)
    target = next((it for it in items if it.get("id") == req.id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Memory not found")
    # Delete then add new
    svc.delete([req.id])
    meta = target.get("metadata") or {}
    entry = {
        "text": req.new_content,
        "kind": meta.get("kind", "chat"),
        "importance": float(meta.get("importance", 0.5)),
        "tags": [],
        "timestamp": meta.get("timestamp"),
    }
    svc.add_facts_with_meta(
        [entry],
        meta.get("conf_uid", ""),
        meta.get("history_uid", ""),
        meta.get("kind", "chat"),
    )
    return {"status": "ok"}


@router.delete("/memory/delete/{memory_id}")
async def memory_delete(memory_id: str, ctx: ServiceContext = Depends(get_context)):
    svc = ctx.vtuber_memory_service
    if not svc or not getattr(svc, "enabled", False):
        raise HTTPException(status_code=503, detail="Memory not available")
    deleted = svc.delete([memory_id])
    return {"deleted": int(deleted or 0)}
