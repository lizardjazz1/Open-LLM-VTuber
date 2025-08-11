from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..service_context import ServiceContext

router = APIRouter(prefix="/admin", tags=["admin-memory"])


def get_context() -> ServiceContext:
    # In the real app, this should pull the current context (singleton or session)
    # Here we import the global context from run_server or provide DI hook.
    from ..websocket_handler import get_global_context

    ctx = get_global_context()
    if not ctx:
        raise HTTPException(status_code=503, detail="Service context not available")
    return ctx


class MemoryEditRequest(BaseModel):
    id: str = Field(..., description="Memory ID to edit")
    new_content: str = Field(..., description="New content text")


class RelationshipUpdateRequest(BaseModel):
    user_id: str
    delta: int = 0
    username: Optional[str] = None


@router.get("/memory/search")
async def memory_search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=50),
    kind: Optional[str] = None,
    conf_uid: Optional[str] = None,
    ctx: ServiceContext = Depends(get_context),
):
    svc = ctx.vtuber_memory_service or ctx.memory_service
    if not svc or not svc.enabled:
        raise HTTPException(status_code=503, detail="Memory not available")
    items = svc.search(q, conf_uid=conf_uid, top_k=top_k, kind=kind)
    return {"items": items}


@router.post("/memory/edit")
async def memory_edit(req: MemoryEditRequest, ctx: ServiceContext = Depends(get_context)):
    svc = ctx.vtuber_memory_service or ctx.memory_service
    if not svc or not svc.enabled:
        raise HTTPException(status_code=503, detail="Memory not available")
    # Our current backend doesn't support in-place edit in Chroma; emulate by delete+add
    items = svc.list(limit=1000)
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
    }
    svc.add_facts_with_meta(
        [entry], meta.get("conf_uid", ""), meta.get("history_uid", ""), meta.get("kind", "chat")
    )
    return {"status": "ok"}


@router.delete("/memory/delete/{memory_id}")
async def memory_delete(memory_id: str, ctx: ServiceContext = Depends(get_context)):
    svc = ctx.vtuber_memory_service or ctx.memory_service
    if not svc or not svc.enabled:
        raise HTTPException(status_code=503, detail="Memory not available")
    deleted = svc.delete([memory_id])
    return {"deleted": int(deleted or 0)}


@router.post("/relationship/update")
async def relationship_update(
    req: RelationshipUpdateRequest, ctx: ServiceContext = Depends(get_context)
):
    from ..vtuber_memory.relationships import RelationshipsDB

    # Choose path from system_config; fallback to local file
    db_path = getattr(ctx.system_config, "relationships_db_path", "cache/relationships.sqlite3")
    db = RelationshipsDB(db_path)

    out: Dict[str, Any] = {}
    if req.username:
        db.set_username(req.user_id, req.username)
        out["username"] = req.username
    if req.delta:
        out["affection"] = db.adjust_affection(req.user_id, req.delta)
    else:
        out["affection"] = db.get_affection(req.user_id)
    return out
