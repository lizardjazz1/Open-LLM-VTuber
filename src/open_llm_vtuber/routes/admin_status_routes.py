from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..service_context import ServiceContext

router = APIRouter(prefix="/admin", tags=["admin-status"])


def get_context() -> ServiceContext:
    """Obtain a service context, falling back to a lightweight server instance if needed."""
    try:
        from ..server import WebSocketServer  # type: ignore
    except Exception:
        pass
    try:
        from ..server import WebSocketServer
        from ..config_manager.utils import validate_config, read_yaml

        app_server = WebSocketServer(config=validate_config(read_yaml("conf.yaml")))
        return app_server.default_context_cache
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Service context not available: {e}"
        )


@router.get("/status")
async def get_status(ctx: ServiceContext = Depends(get_context)) -> dict[str, Any]:
    """Return current status of core modules and identifiers.

    Args:
            ctx: Injected service context.

    Returns:
            dict: Module status snapshot.
    """
    if not ctx:
        raise HTTPException(status_code=503, detail="Service context not available")

    def _name(obj: Any) -> str | None:
        try:
            return type(obj).__name__ if obj is not None else None
        except Exception:
            return None

    # Twitch status
    twitch_status: dict[str, Any] | None = None
    try:
        if ctx.twitch_client:
            twitch_status = ctx.twitch_client.get_connection_status()
    except Exception:
        pass

    # Basic config identifiers
    conf_uid = None
    conf_name = None
    try:
        conf_uid = getattr(ctx.character_config, "conf_uid", None)
        conf_name = getattr(ctx.character_config, "conf_name", None)
    except Exception:
        pass

    live2d = None
    try:
        live2d = getattr(getattr(ctx, "live2d_model", None), "model_info", None)
    except Exception:
        pass

    return {
        "conf_uid": conf_uid,
        "conf_name": conf_name,
        "language": getattr(getattr(ctx, "system_config", None), "language", None),
        "asr": {
            "engine": _name(ctx.asr_engine),
            "model": getattr(getattr(ctx, "character_config", None), "asr_config", None)
            and getattr(ctx.character_config.asr_config, "asr_model", None),
        },
        "tts": {
            "engine": _name(ctx.tts_engine),
            "model": getattr(getattr(ctx, "character_config", None), "tts_config", None)
            and getattr(ctx.character_config.tts_config, "tts_model", None),
        },
        "vad": {
            "engine": _name(ctx.vad_engine),
            "model": getattr(getattr(ctx, "character_config", None), "vad_config", None)
            and getattr(ctx.character_config.vad_config, "vad_model", None),
        },
        "twitch": twitch_status
        or {"enabled": False, "connected": False, "channel": ""},
        "live2d": live2d,
    }


@router.get("/ws-schema")
async def get_ws_schema() -> dict[str, Any]:
    """Return minimal WS event schema used by frontend for alignment.

    This helps keep frontend/backend in sync without scanning code.
    """
    return {
        "tts_ws": {
            "client->server": {"request": {"text": "string"}},
            "server->client": {
                "segment_start": {
                    "text": "string",
                    "voice_cmds": ["{rate:+10%}", "{volume:+5%}", "{pitch:+15Hz}"],
                    "truncated": "bool",
                },
                "partial": {
                    "audioPath": "string",
                    "text": "string",
                    "latency_ms": "int",
                    "truncated": "bool",
                },
                "complete": {},
                "error": {"message": "string"},
            },
        },
        "client_ws": {
            "note": "UI channel; see frontend docs for detailed message types",
        },
    }
