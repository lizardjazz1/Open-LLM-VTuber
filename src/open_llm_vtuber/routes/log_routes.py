from __future__ import annotations

from fastapi import APIRouter, Request
from loguru import logger
from typing import Any
import secrets

from slowapi import Limiter
from slowapi.util import get_remote_address

from ..logging_utils import mask_secrets, set_request_id
from ..config_manager import Config

# // DEBUG: [FIXED] Central client log ingress with rate-limiting and token check | Ref: 4,15


def init_log_routes(config: Config, limiter: Limiter) -> APIRouter:
    router = APIRouter()

    @router.post("/logs")
    @limiter.limit("10/second")  # 10 RPS per IP
    async def ingest_logs(request: Request) -> dict[str, Any]:
        # Request authentication via token from conf.yaml
        server_cfg = config.system_config
        header_token = request.headers.get("X-Log-Token", "")
        conf_token = getattr(server_cfg, "logging_token", "") or ""
        # Constant-time compare
        if not (conf_token and secrets.compare_digest(header_token, conf_token)):
            logger.bind(component="client_log").warning(
                {
                    "status": "unauthorized",
                    "remote": get_remote_address(request),
                }
            )
            return {"ok": False}

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
                "status": "ingest",
                **(masked if isinstance(masked, dict) else {"payload": masked}),
            }
        )
        return {"ok": True}

    return router
