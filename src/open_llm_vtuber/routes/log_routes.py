"""
Log ingestion API routes.

This module exposes endpoints that allow the frontend (or other clients) to
submit lightweight logs to the backend for unified analysis. It is controlled
by configuration flags to remain safe by default in local setups.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from loguru import logger
from typing import Any

from slowapi import Limiter

from ..logging_utils import mask_secrets, set_request_id
from ..config_manager import Config

# // DEBUG: [FIXED] Central client log ingress with rate-limiting | Ref: 4,15


def init_log_routes(config: Config, limiter: Limiter) -> APIRouter:
    """Create routes for client log ingestion with rate limiting.

    Args:
            config (Config): Loaded application configuration (used to read system flags).
            limiter (Limiter): SlowAPI limiter instance used to protect the endpoint.

    Returns:
            APIRouter: Router exposing the POST `/logs` endpoint.
    """
    router = APIRouter()

    @router.post("/logs")
    @limiter.limit("10/second")  # 10 RPS per IP
    async def ingest_logs(request: Request) -> dict[str, Any]:
        """Ingest a single client log record in JSON form.

        Behavior is controlled by `system_config.client_log_ingest_enabled`.

        Args:
                request (Request): Incoming request containing a JSON body produced by the frontend logger.

        Returns:
                dict[str, Any]: `{ "ok": True }` if accepted, or `{ "ok": False }` if disabled.
        """
        # Check if client log ingestion is enabled
        server_cfg = config.system_config
        if not getattr(server_cfg, "client_log_ingest_enabled", False):
            # Soft-deny without error to avoid noise
            return {"ok": False}

        # No token enforcement in this build

        try:
            body = await request.json()
        except Exception:
            body = {}

        # Attach request_id if provided by client, else generate one
        rid = str(body.get("request_id") or "") or None
        set_request_id(rid)

        masked = mask_secrets(body)
        logger.bind(component=str(body.get("component") or "frontend")).info(
            {
                "event": "client_log.ingest",
                "status": "ingest",
                "payload_size": int(request.headers.get("content-length") or 0),
                **(masked if isinstance(masked, dict) else {"payload": masked}),
            }
        )
        return {"ok": True}

    return router
