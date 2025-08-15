from typing import Dict, List, Optional, Callable, TypedDict
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
from enum import Enum
import numpy as np
from loguru import logger

from .service_context import ServiceContext
from .chat_group import (
    ChatGroupManager,
    handle_group_operation,
    handle_client_disconnect,
    broadcast_to_group,
)
from .message_handler import message_handler
from .utils.stream_audio import prepare_audio_payload
from .chat_history_manager import (
    create_new_history,
    get_history,
    delete_history,
    get_history_list,
)
from .config_manager.utils import scan_config_alts_directory, scan_bg_directory
from .conversations.conversation_handler import (
    handle_conversation_trigger,
    handle_group_interrupt,
    handle_individual_interrupt,
)

# from .memory.memory_service import MemoryService  # Legacy import (deprecated in favor of vtuber_memory_service)
from .debug_settings import ensure_log_sinks

# // DEBUG: [FIXED] Request ID utilities | Ref: 5
from .logging_utils import set_request_id
from uuid import uuid4

DEBUG_WS, _DEBUG_LLM_UNUSED = ensure_log_sinks()


class MessageType(Enum):
    """Enum for WebSocket message types"""

    GROUP = ["add-client-to-group", "remove-client-from-group"]
    HISTORY = [
        "fetch-history-list",
        "fetch-and-set-history",
        "create-new-history",
        "delete-history",
    ]
    CONVERSATION = ["mic-audio-end", "text-input", "ai-speak-signal"]
    CONFIG = ["fetch-configs", "switch-config"]
    CONTROL = ["interrupt-signal", "audio-play-start"]
    DATA = ["mic-audio-data"]


class WSMessage(TypedDict, total=False):
    """Type definition for WebSocket messages"""

    type: str
    action: Optional[str]
    text: Optional[str]
    audio: Optional[List[float]]
    images: Optional[List[str]]
    history_uid: Optional[str]
    file: Optional[str]
    display_text: Optional[dict]
    request_id: Optional[str]


class WebSocketHandler:
    """Handles WebSocket connections and message routing"""

    def __init__(self, default_context_cache: ServiceContext):
        """Initialize the WebSocket handler with default context"""
        self.client_connections: Dict[str, WebSocket] = {}
        self.client_contexts: Dict[str, ServiceContext] = {}
        self.chat_group_manager = ChatGroupManager()
        self.current_conversation_tasks: Dict[str, Optional[asyncio.Task]] = {}
        self.default_context_cache = default_context_cache
        self.received_data_buffers: Dict[str, np.ndarray] = {}
        # Per-client locks to guard buffer updates (avoid races between VAD and triggers)
        self._buffer_locks: Dict[str, asyncio.Lock] = {}
        # Track frontend init acks to ensure Live2D/config is delivered
        self._init_ack: Dict[str, bool] = {}

        # Message handlers mapping
        self._message_handlers = self._init_message_handlers()

    def _init_message_handlers(self) -> Dict[str, Callable]:
        """Initialize message type to handler mapping"""
        return {
            "add-client-to-group": self._handle_group_operation,
            "remove-client-from-group": self._handle_group_operation,
            "request-group-info": self._handle_group_info,
            "fetch-history-list": self._handle_history_list_request,
            "history-list-grouped": self._handle_history_list_grouped,
            "fetch-and-set-history": self._handle_fetch_history,
            "create-new-history": self._handle_create_history,
            "delete-history": self._handle_delete_history,
            "interrupt-signal": self._handle_interrupt,
            "mic-audio-data": self._handle_audio_data,
            "mic-audio-end": self._handle_conversation_trigger,
            "raw-audio-data": self._handle_raw_audio_data,
            "text-input": self._handle_conversation_trigger,
            "ai-speak-signal": self._handle_conversation_trigger,
            "fetch-configs": self._handle_fetch_configs,
            "switch-config": self._handle_config_switch,
            "fetch-backgrounds": self._handle_fetch_backgrounds,
            "audio-play-start": self._handle_audio_play_start,
            "request-init-config": self._handle_init_config_request,
            "heartbeat": self._handle_heartbeat,
            # Frontend telemetry (no side effects)
            "frontend-log": self._handle_frontend_log,
            "window-error": self._handle_window_error,
            # Hot-apply LLM params (temperature, top_p, penalties, max_tokens, stop, seed)
            "update-llm-params": self._handle_update_llm_params,
            # Memory controls
            "update-memory-settings": self._handle_update_memory_settings,
            "memory-clear": self._handle_memory_clear,
            "memory-search": self._handle_memory_search,
            "memory-search-grouped": self._handle_memory_search_grouped,
            "memory-prune": self._handle_memory_prune,
            "memory-list": self._handle_memory_list,
            "memory-list-grouped": self._handle_memory_list_grouped,
            "memory-add": self._handle_memory_add,
            "memory-delete": self._handle_memory_delete,
            "memory-consolidate": self._handle_memory_consolidate,
            "memory-consolidate-history": self._handle_memory_consolidate_history,
            "import-history": self._handle_import_history,
            "memory-kinds-info": self._handle_memory_kinds_info,
            # Twitch helpers
            "twitch-fetch": self._handle_twitch_fetch,
            # Mood controls
            "mood-list": self._handle_mood_list,
            "mood-reset": self._handle_mood_reset,
            "mood-set": self._handle_mood_set,
            # Frontend readiness ack
            "frontend-ready": self._handle_frontend_ready,
        }

    async def handle_new_connection(
        self, websocket: WebSocket, client_uid: str
    ) -> None:
        """
        Handle new WebSocket connection setup

        Args:
            websocket: The WebSocket connection
            client_uid: Unique identifier for the client

        Raises:
            Exception: If initialization fails
        """
        try:
            session_service_context = await self._init_service_context(
                websocket.send_text, client_uid
            )

            await self._store_client_data(
                websocket, client_uid, session_service_context
            )

            await self._send_initial_messages(
                websocket, client_uid, session_service_context
            )

            # Mark init as not acked yet and schedule reliable re-sends
            self._init_ack[client_uid] = False
            # Initialize client buffer and lock
            self.received_data_buffers[client_uid] = np.array([])
            self._buffer_locks[client_uid] = asyncio.Lock()
            asyncio.create_task(
                self._resend_init_until_ack(
                    websocket, client_uid, session_service_context
                )
            )

            logger.info(f"Connection established for client {client_uid}")

        except Exception as e:
            logger.error(
                f"Failed to initialize connection for client {client_uid}: {e}"
            )
            await self._cleanup_failed_connection(client_uid)
            raise

    async def _resend_init_until_ack(
        self, websocket: WebSocket, client_uid: str, ctx: ServiceContext
    ) -> None:
        """Reliably deliver model/config to frontend by re-sending until ack or timeout."""
        try:
            for delay in (1.0, 3.0):
                await asyncio.sleep(delay)
                if self._init_ack.get(client_uid):
                    return
                payload = {
                    "type": "set-model-and-conf",
                    "model_info": ctx.live2d_model.model_info
                    if ctx.live2d_model
                    else None,
                    "conf_name": ctx.character_config.conf_name,
                    "conf_uid": ctx.character_config.conf_uid,
                    "client_uid": client_uid,
                    "tts_info": {"model": ctx.character_config.tts_config.tts_model},
                }
                try:
                    await websocket.send_text(json.dumps(payload))
                    logger.debug(
                        f"Resent set-model-and-conf after {int(delay)}s (no frontend ack yet)"
                    )
                except Exception:
                    break
        except Exception:
            pass

    async def _cleanup_failed_connection(self, client_uid: str) -> None:
        """Safely cleanup partially-initialized client state after a failed connect."""
        try:
            # Remove from chat group if mapped
            group = self.chat_group_manager.get_client_group(client_uid)
            if group:
                try:
                    await handle_client_disconnect(
                        client_uid=client_uid,
                        chat_group_manager=self.chat_group_manager,
                        client_connections=self.client_connections,
                        send_group_update=self.send_group_update,
                    )
                except Exception:
                    pass
        except Exception:
            pass
        # Drop maps safely
        try:
            self.client_connections.pop(client_uid, None)
            self.client_contexts.pop(client_uid, None)
            self.received_data_buffers.pop(client_uid, None)
            self._init_ack.pop(client_uid, None)
            if client_uid in self.current_conversation_tasks:
                task = self.current_conversation_tasks.pop(client_uid, None)
                if task and not task.done():
                    try:
                        task.cancel()
                    except Exception:
                        pass
        except Exception:
            pass

    async def _store_client_data(
        self,
        websocket: WebSocket,
        client_uid: str,
        session_service_context: ServiceContext,
    ):
        """Store client data and initialize group status"""
        self.client_connections[client_uid] = websocket
        self.client_contexts[client_uid] = session_service_context
        self.received_data_buffers[client_uid] = np.array([])

        self.chat_group_manager.client_group_map[client_uid] = ""
        await self.send_group_update(websocket, client_uid)

    async def _send_initial_messages(
        self,
        websocket: WebSocket,
        client_uid: str,
        session_service_context: ServiceContext,
    ):
        """Send initial connection messages to the client"""
        await websocket.send_text(
            json.dumps({"type": "full-text", "text": "Connection established"})
        )

        # Логируем информацию о модели перед отправкой
        logger.info(
            f"🔍 Live2D model info: {session_service_context.live2d_model.model_info if session_service_context.live2d_model else 'None'}"
        )
        logger.info(
            f"🔍 Character config: {session_service_context.character_config.conf_name}"
        )

        await websocket.send_text(
            json.dumps(
                {
                    "type": "set-model-and-conf",
                    "model_info": session_service_context.live2d_model.model_info,
                    "conf_name": session_service_context.character_config.conf_name,
                    "conf_uid": session_service_context.character_config.conf_uid,
                    "client_uid": client_uid,
                    "tts_info": {
                        "model": session_service_context.character_config.tts_config.tts_model
                    },
                }
            )
        )

        # Send initial group status
        await self.send_group_update(websocket, client_uid)

        # Start microphone (match original behavior)
        try:
            await websocket.send_text(
                json.dumps({"type": "control", "text": "start-mic"})
            )
        except Exception:
            pass

        # Re-send model/config in case frontend mounted late
        try:
            context = session_service_context
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "set-model-and-conf",
                        "model_info": context.live2d_model.model_info
                        if context.live2d_model
                        else None,
                        "conf_name": context.character_config.conf_name,
                        "conf_uid": context.character_config.conf_uid,
                        "client_uid": client_uid,
                        "tts_info": {
                            "model": context.character_config.tts_config.tts_model
                        },
                    }
                )
            )
        except Exception:
            pass

        # Send current Twitch status and a short backlog of recent messages to this client
        try:
            tc = getattr(self.default_context_cache, "twitch_client", None)
            if tc:
                try:
                    status = tc.get_connection_status()
                    await websocket.send_json({"type": "twitch-status", **status})
                except Exception:
                    pass
                try:
                    recent = tc.get_recent_messages() or []
                    # send last up to 20 messages to populate UI
                    for m in recent[-20:]:
                        try:
                            await websocket.send_json(
                                {
                                    "type": "twitch-message",
                                    "user": getattr(m, "user", None),
                                    "text": getattr(m, "message", None),
                                    "timestamp": getattr(
                                        m, "timestamp", None
                                    ).isoformat()
                                    if getattr(m, "timestamp", None)
                                    else None,
                                    "channel": getattr(m, "channel", None),
                                }
                            )
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

    async def _init_service_context(
        self, send_text: Callable, client_uid: str
    ) -> ServiceContext:
        """Initialize service context for a new session by cloning the default context"""

        # Wrap send_text to broadcast to all connected clients (not just last one)
        async def _broadcast_send_text(payload: str) -> None:
            for ws in list(self.client_connections.values()):
                try:
                    await ws.send_text(payload)
                except Exception:
                    # ignore send failures for individual sockets
                    pass

        session_service_context = ServiceContext()
        await session_service_context.load_cache(
            config=self.default_context_cache.config.model_copy(deep=True),
            system_config=self.default_context_cache.system_config.model_copy(
                deep=True
            ),
            character_config=self.default_context_cache.character_config.model_copy(
                deep=True
            ),
            live2d_model=self.default_context_cache.live2d_model,
            asr_engine=self.default_context_cache.asr_engine,
            tts_engine=self.default_context_cache.tts_engine,
            vad_engine=self.default_context_cache.vad_engine,
            agent_engine=self.default_context_cache.agent_engine,
            translate_engine=self.default_context_cache.translate_engine,
            mcp_server_registery=self.default_context_cache.mcp_server_registery,
            tool_adapter=self.default_context_cache.tool_adapter,
            send_text=_broadcast_send_text,
            client_uid=client_uid,
        )
        # Also wire the default context to this websocket so shared modules (e.g., Twitch)
        # can emit to the active client and process via the default agent engine.
        # Now use broadcasting so all connected clients receive events.
        self.default_context_cache.send_text = _broadcast_send_text
        self.default_context_cache.client_uid = client_uid
        return session_service_context

    async def _handle_update_llm_params(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Hot-apply LLM sampling parameters for current session and default context.

        Accepts any of the following optional fields in `data`:
        - temperature (float)
        - top_p (float)
        - frequency_penalty (float)
        - presence_penalty (float)
        - max_tokens (int)
        - stop (list[str])
        - seed (int)
        """
        try:
            params = {
                k: v
                for k, v in data.items()
                if k
                in (
                    "temperature",
                    "top_p",
                    "frequency_penalty",
                    "presence_penalty",
                    "max_tokens",
                    "stop",
                    "seed",
                )
            }

            def _apply(agent_engine, where: str) -> dict:
                applied = {}
                try:
                    if not agent_engine:
                        return applied
                    llm = getattr(agent_engine, "_llm", None)
                    if not llm:
                        return applied
                    for key, value in params.items():
                        # Basic type guards
                        if key in (
                            "temperature",
                            "top_p",
                            "frequency_penalty",
                            "presence_penalty",
                        ):
                            try:
                                value = float(value)
                            except Exception:
                                continue
                        elif key in ("max_tokens", "seed"):
                            try:
                                value = int(value)
                            except Exception:
                                continue
                        elif key == "stop":
                            if value is None:
                                pass
                            elif isinstance(value, list):
                                # Ensure str list
                                value = [str(x) for x in value]
                            else:
                                # allow comma-separated string
                                value = [
                                    s.strip()
                                    for s in str(value).split(",")
                                    if s.strip()
                                ]
                        if hasattr(llm, key):
                            setattr(llm, key, value)
                            applied[key] = value
                    return applied
                except Exception as e:
                    logger.warning(f"Failed to apply LLM params for {where}: {e}")
                    return applied

            # Apply to current session context
            session_ctx = self.client_contexts.get(client_uid)
            applied_session = (
                _apply(getattr(session_ctx, "agent_engine", None), "session")
                if session_ctx
                else {}
            )

            # Apply to default context (affects new sessions)
            applied_default = _apply(
                getattr(self.default_context_cache, "agent_engine", None), "default"
            )

            await websocket.send_json(
                {
                    "type": "llm-params-updated",
                    "applied_session": applied_session,
                    "applied_default": applied_default,
                }
            )
        except Exception as e:
            logger.error(f"update-llm-params failed: {e}")
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"Failed to update LLM params: {str(e)}",
                }
            )

    async def _handle_update_memory_settings(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Hot-update memory settings (enabled, top_k) for session and default."""
        enabled = data.get("enabled")
        top_k = data.get("top_k")
        min_importance = data.get("min_importance")
        kinds = data.get("kinds")
        try:

            def _apply(ctx: ServiceContext | None) -> dict:
                out = {}
                if not ctx:
                    return out
                if enabled is not None:
                    ctx.memory_enabled = bool(enabled)
                    out["enabled"] = ctx.memory_enabled
                if isinstance(top_k, int) and top_k > 0:
                    ctx.memory_top_k = min(top_k, 20)
                    out["top_k"] = ctx.memory_top_k
                if min_importance is not None:
                    try:
                        ctx.memory_min_importance = float(min_importance)
                        out["min_importance"] = ctx.memory_min_importance
                    except Exception:
                        pass
                if kinds is not None:
                    if isinstance(kinds, list):
                        ctx.memory_kinds = [str(k) for k in kinds]
                    elif isinstance(kinds, str):
                        ctx.memory_kinds = [
                            s.strip() for s in kinds.split(",") if s.strip()
                        ]
                    else:
                        ctx.memory_kinds = None
                    out["kinds"] = ctx.memory_kinds
                # lazy init service if toggled on (legacy)
                # if ctx.memory_enabled and (ctx.memory_service is None):
                #     try:
                #         ctx.memory_service = MemoryService(enabled=True)
                #     except Exception as e:
                #         logger.warning(f"MemoryService init failed on update: {e}")
                return out

            sess = _apply(self.client_contexts.get(client_uid))
            dflt = _apply(self.default_context_cache)
            await websocket.send_json(
                {"type": "memory-settings-updated", "session": sess, "default": dflt}
            )
        except Exception as e:
            logger.error(f"update-memory-settings failed: {e}")
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"Failed to update memory settings: {str(e)}",
                }
            )

    async def _handle_memory_clear(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Clear memory: by conf_uid (default current) or all."""
        try:
            scope = data.get("scope", "current")  # current|all
            ctx = self.client_contexts.get(client_uid)
            conf_uid = (
                ctx.character_config.conf_uid if (ctx and scope != "all") else None
            )
            # Prefer new vtuber memory service, fallback to legacy if needed
            svc = (
                (ctx.vtuber_memory_service if ctx else None)
                or (self.default_context_cache.vtuber_memory_service)
                or (
                    (ctx.memory_service if ctx else None)
                    or self.default_context_cache.memory_service
                )
            )
            if svc and getattr(svc, "enabled", False):
                ok = svc.clear(conf_uid=conf_uid)
            else:
                ok = 0
            await websocket.send_json(
                {"type": "memory-clear-result", "ok": bool(ok), "scope": scope}
            )
        except Exception as e:
            logger.error(f"memory-clear failed: {e}")
            await websocket.send_json(
                {"type": "error", "message": f"Failed to clear memory: {str(e)}"}
            )

    async def _handle_memory_search(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Search memory with filters and return hits."""
        try:
            query = data.get("query", "")
            top_k = int(data.get("top_k", 5))
            kind = data.get("kind")
            kinds = data.get("kinds")  # optional list[str]
            min_importance = data.get("min_importance")
            since_ts = data.get("since_ts")
            until_ts = data.get("until_ts")
            ctx = self.client_contexts.get(client_uid)
            # Prefer new vtuber memory service, fallback to legacy if needed
            svc = (
                (ctx.vtuber_memory_service if ctx else None)
                or (self.default_context_cache.vtuber_memory_service)
                or (
                    (ctx.memory_service if ctx else None)
                    or self.default_context_cache.memory_service
                )
            )
            conf_uid = ctx.character_config.conf_uid if ctx else None
            if not (svc and getattr(svc, "enabled", False) and query):
                await websocket.send_json({"type": "memory-search-result", "hits": []})
                return
            hits = svc.search(
                query=query,
                conf_uid=conf_uid,
                top_k=max(1, min(20, top_k)),
                kind=kind,
                kinds=kinds if isinstance(kinds, list) else None,
                min_importance=float(min_importance)
                if min_importance is not None
                else None,
                since_ts=int(since_ts) if since_ts is not None else None,
                until_ts=int(until_ts) if until_ts is not None else None,
            )
            await websocket.send_json({"type": "memory-search-result", "hits": hits})
        except Exception as e:
            logger.error(f"memory-search failed: {e}")
            await websocket.send_json(
                {"type": "error", "message": f"Failed to search memory: {str(e)}"}
            )

    async def _handle_memory_search_grouped(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Search memory per kind and return grouped results: { kind: hits[] }."""
        try:
            query = data.get("query", "")
            kinds = data.get("kinds")
            top_k = int(data.get("top_k", 5))
            min_importance = data.get("min_importance")
            since_ts = data.get("since_ts")
            until_ts = data.get("until_ts")
            ctx = self.client_contexts.get(client_uid)
            # Prefer new vtuber memory service, fallback to legacy if needed
            svc = (
                (ctx.vtuber_memory_service if ctx else None)
                or (self.default_context_cache.vtuber_memory_service)
                or (
                    (ctx.memory_service if ctx else None)
                    or self.default_context_cache.memory_service
                )
            )
            conf_uid = ctx.character_config.conf_uid if ctx else None
            if not (svc and getattr(svc, "enabled", False) and query):
                await websocket.send_json(
                    {"type": "memory-search-grouped-result", "groups": {}}
                )
                return
            # Determine kinds to search
            if isinstance(kinds, list) and kinds:
                target_kinds = [str(k).strip() for k in kinds if str(k).strip()]
            else:
                # Fallback to common consolidated kinds
                target_kinds = [
                    "FactsAboutUser",
                    "PastEvents",
                    "SelfBeliefs",
                    "Objectives",
                    "KeyFacts",
                    "Emotions",
                    "Mood",
                ]
            groups: Dict[str, List[dict]] = {}
            for k in target_kinds:
                try:
                    hits = svc.search(
                        query=query,
                        conf_uid=conf_uid,
                        top_k=max(1, min(20, top_k)),
                        kind=k,
                        kinds=None,
                        min_importance=float(min_importance)
                        if min_importance is not None
                        else None,
                        since_ts=int(since_ts) if since_ts is not None else None,
                        until_ts=int(until_ts) if until_ts is not None else None,
                    )
                    groups[k] = hits
                except Exception:
                    groups[k] = []
            await websocket.send_json(
                {"type": "memory-search-grouped-result", "groups": groups}
            )
        except Exception as e:
            logger.error(f"memory-search-grouped failed: {e}")
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"Failed to search grouped memory: {str(e)}",
                }
            )

    async def _handle_memory_prune(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Prune memories by age and/or importance."""
        try:
            max_age_ts = data.get("max_age_ts")
            max_importance = data.get("max_importance")
            ctx = self.client_contexts.get(client_uid)
            # Prefer new vtuber memory service, fallback to legacy if needed
            svc = (
                (ctx.vtuber_memory_service if ctx else None)
                or (self.default_context_cache.vtuber_memory_service)
                or (
                    (ctx.memory_service if ctx else None)
                    or self.default_context_cache.memory_service
                )
            )
            conf_uid = ctx.character_config.conf_uid if ctx else None
            if not (svc and getattr(svc, "enabled", False)):
                await websocket.send_json({"type": "memory-prune-result", "ok": False})
                return
            ok = svc.prune(
                conf_uid=conf_uid,
                max_age_ts=int(max_age_ts) if max_age_ts is not None else None,
                max_importance=float(max_importance)
                if max_importance is not None
                else None,
            )
            await websocket.send_json({"type": "memory-prune-result", "ok": bool(ok)})
        except Exception as e:
            logger.error(f"memory-prune failed: {e}")
            await websocket.send_json(
                {"type": "error", "message": f"Failed to prune memory: {str(e)}"}
            )

    async def _handle_memory_list(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        try:
            limit = int(data.get("limit", 50))
            kind = data.get("kind")
            ctx = self.client_contexts.get(client_uid)
            # Prefer new vtuber memory service, fallback to legacy if needed
            svc = (
                (ctx.vtuber_memory_service if ctx else None)
                or (self.default_context_cache.vtuber_memory_service)
                or (
                    (ctx.memory_service if ctx else None)
                    or self.default_context_cache.memory_service
                )
            )
            conf_uid = ctx.character_config.conf_uid if ctx else None
            items = (
                svc.list(conf_uid=conf_uid, limit=max(1, min(200, limit)), kind=kind)
                if (svc and getattr(svc, "enabled", False))
                else []
            )
            await websocket.send_json({"type": "memory-list-result", "items": items})
        except Exception as e:
            logger.error(f"memory-list failed: {e}")
            await websocket.send_json(
                {"type": "error", "message": f"Failed to list memory: {str(e)}"}
            )

    async def _handle_memory_list_grouped(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """List recent items per kind and return groups { kind: items[] }."""
        try:
            kinds = data.get("kinds")
            limit_per_kind = int(data.get("limit_per_kind", 20))
            ctx = self.client_contexts.get(client_uid)
            # Prefer new vtuber memory service, fallback to legacy if needed
            svc = (
                (ctx.vtuber_memory_service if ctx else None)
                or (self.default_context_cache.vtuber_memory_service)
                or (
                    (ctx.memory_service if ctx else None)
                    or self.default_context_cache.memory_service
                )
            )
            conf_uid = ctx.character_config.conf_uid if ctx else None
            if not (svc and getattr(svc, "enabled", False)):
                await websocket.send_json(
                    {"type": "memory-list-grouped-result", "groups": {}}
                )
                return
            # If kinds explicitly provided -> fetch per-kind
            if isinstance(kinds, list) and kinds:
                target_kinds = [str(k).strip() for k in kinds if str(k).strip()]
                groups: Dict[str, List[dict]] = {}
                for k in target_kinds:
                    try:
                        items = svc.list(
                            conf_uid=conf_uid,
                            limit=max(1, min(200, limit_per_kind)),
                            kind=k,
                        )
                        groups[k] = items
                    except Exception:
                        groups[k] = []
            else:
                # No kinds selected: show everything. Fetch a larger mixed list and group by metadata.kind
                fetch_limit = max(1, min(1000, limit_per_kind * 10))
                mixed = svc.list(conf_uid=conf_uid, limit=fetch_limit, kind=None)
                groups = {}
                for it in mixed:
                    meta = it.get("metadata") or {}
                    k = str(meta.get("kind") or "Other")
                    arr = groups.setdefault(k, [])
                    if len(arr) < limit_per_kind:
                        arr.append(it)
            await websocket.send_json(
                {"type": "memory-list-grouped-result", "groups": groups}
            )
        except Exception as e:
            logger.error(f"memory-list-grouped failed: {e}")
            await websocket.send_json(
                {"type": "error", "message": f"Failed to list grouped memory: {str(e)}"}
            )

    async def _handle_twitch_fetch(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Send current Twitch status and a short backlog of recent messages to this client on demand."""
        try:
            tc = getattr(self.default_context_cache, "twitch_client", None)
            if not tc:
                await websocket.send_json(
                    {"type": "twitch-status", "enabled": False, "connected": False}
                )
                return
            try:
                status = tc.get_connection_status()
                await websocket.send_json({"type": "twitch-status", **status})
            except Exception:
                pass
            try:
                recent = tc.get_recent_messages() or []
                for m in recent[-20:]:
                    try:
                        await websocket.send_json(
                            {
                                "type": "twitch-message",
                                "user": getattr(m, "user", None),
                                "text": getattr(m, "message", None),
                                "timestamp": getattr(m, "timestamp", None).isoformat()
                                if getattr(m, "timestamp", None)
                                else None,
                                "channel": getattr(m, "channel", None),
                            }
                        )
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception as e:
            logger.error(f"twitch-fetch failed: {e}")
            await websocket.send_json(
                {"type": "error", "message": f"Failed to fetch twitch state: {str(e)}"}
            )

    async def _handle_memory_add(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Add a single memory entry with optional rich metadata."""
        try:
            entry = data.get("entry") or {}
            if not isinstance(entry, dict) or not entry.get("text"):
                await websocket.send_json(
                    {"type": "memory-add-result", "ok": False, "error": "invalid entry"}
                )
                return
            ctx = self.client_contexts.get(client_uid)
            # Prefer new vtuber memory service, fallback to legacy if needed
            svc = (
                (ctx.vtuber_memory_service if ctx else None)
                or (self.default_context_cache.vtuber_memory_service)
                or (
                    (ctx.memory_service if ctx else None)
                    or self.default_context_cache.memory_service
                )
            )
            if not (svc and getattr(svc, "enabled", False) and ctx):
                await websocket.send_json({"type": "memory-add-result", "ok": False})
                return
            # Normalize numeric fields
            for k in ("importance", "emotion_score"):
                if k in entry and entry[k] is not None:
                    try:
                        entry[k] = float(entry[k])
                    except Exception:
                        entry[k] = None
            # Timestamp: accept ISO or epoch seconds
            ts = entry.get("timestamp")
            if isinstance(ts, str):
                try:
                    from datetime import datetime

                    entry["timestamp"] = int(
                        datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                    )
                except Exception:
                    entry.pop("timestamp", None)
            added = svc.add_facts_with_meta(
                [entry],
                ctx.character_config.conf_uid,
                ctx.history_uid or "manual",
                default_kind=entry.get("kind") or "chat",
            )
            await websocket.send_json(
                {"type": "memory-add-result", "ok": bool(added), "added": int(added)}
            )
        except Exception as e:
            logger.error(f"memory-add failed: {e}")
            await websocket.send_json(
                {"type": "error", "message": f"Failed to add memory: {str(e)}"}
            )

    async def _handle_memory_delete(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        try:
            ids = data.get("ids") or []
            if not isinstance(ids, list):
                await websocket.send_json(
                    {
                        "type": "memory-delete-result",
                        "ok": False,
                        "error": "ids must be list",
                    }
                )
                return
            ctx = self.client_contexts.get(client_uid)
            # Prefer new vtuber memory service, fallback to legacy if needed
            svc = (
                (ctx.vtuber_memory_service if ctx else None)
                or (self.default_context_cache.vtuber_memory_service)
                or (
                    (ctx.memory_service if ctx else None)
                    or self.default_context_cache.memory_service
                )
            )
            if not (svc and getattr(svc, "enabled", False)):
                await websocket.send_json({"type": "memory-delete-result", "ok": False})
                return
            deleted = svc.delete([str(i) for i in ids])
            await websocket.send_json(
                {"type": "memory-delete-result", "ok": True, "deleted": int(deleted)}
            )
        except Exception as e:
            logger.error(f"memory-delete failed: {e}")
            await websocket.send_json(
                {"type": "error", "message": f"Failed to delete memory: {str(e)}"}
            )

    async def _handle_memory_consolidate(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        try:
            ctx = self.client_contexts.get(client_uid)
            if not ctx:
                await websocket.send_json(
                    {"type": "memory-consolidate-result", "ok": False}
                )
                return
            await ctx.trigger_memory_consolidation(reason=data.get("reason", "manual"))
            await websocket.send_json({"type": "memory-consolidate-result", "ok": True})
        except Exception as e:
            logger.error(f"memory-consolidate failed: {e}")
            await websocket.send_json(
                {"type": "error", "message": f"Failed to consolidate memory: {str(e)}"}
            )

    async def _handle_memory_consolidate_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        try:
            history_uid = data.get("history_uid")
            limit = data.get("limit_messages")
            if not history_uid or not isinstance(history_uid, str):
                await websocket.send_json(
                    {
                        "type": "memory-consolidate-history-result",
                        "ok": False,
                        "error": "history_uid must be provided",
                    }
                )
                return
            ctx = self.client_contexts.get(client_uid)
            if not ctx:
                await websocket.send_json(
                    {"type": "memory-consolidate-history-result", "ok": False}
                )
                return
            saved = await ctx.consolidate_history(history_uid, limit_messages=limit)
            await websocket.send_json(
                {
                    "type": "memory-consolidate-history-result",
                    "ok": True,
                    "saved": int(saved),
                    "history_uid": history_uid,
                }
            )
        except Exception as e:
            logger.error(f"memory-consolidate-history failed: {e}")
            await websocket.send_json(
                {"type": "error", "message": f"Failed to consolidate history: {str(e)}"}
            )

    async def _handle_import_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Import chat history from frontend. Expects 'content' list and optional 'history_uid'."""
        try:
            ctx = self.client_contexts.get(client_uid)
            if not ctx:
                await websocket.send_json(
                    {
                        "type": "import-history-result",
                        "ok": False,
                        "error": "no context",
                    }
                )
                return
            content = data.get("content")
            history_uid = data.get("history_uid")
            if not isinstance(content, list) or not content:
                await websocket.send_json(
                    {
                        "type": "import-history-result",
                        "ok": False,
                        "error": "content must be non-empty list",
                    }
                )
                return
            # Validate minimal schema per message
            sanitized: list[dict] = []
            for m in content:
                if not isinstance(m, dict):
                    continue
                role = m.get("role")
                text = (m.get("content") or "").strip()
                if role not in ("human", "ai") or not text:
                    continue
                item = {"role": role, "content": text}
                if m.get("timestamp"):
                    item["timestamp"] = str(m.get("timestamp"))
                if m.get("name"):
                    item["name"] = str(m.get("name"))
                if m.get("avatar"):
                    item["avatar"] = str(m.get("avatar"))
                sanitized.append(item)
            if not sanitized:
                await websocket.send_json(
                    {
                        "type": "import-history-result",
                        "ok": False,
                        "error": "no valid messages",
                    }
                )
                return
            # Create or replace file
            from .chat_history_manager import _get_safe_history_path, create_new_history
            import json

            if not history_uid:
                history_uid = create_new_history(ctx.character_config.conf_uid)
            path = _get_safe_history_path(ctx.character_config.conf_uid, history_uid)
            # Prepend metadata entry if absent
            output = [
                {"role": "metadata", "timestamp": sanitized[0].get("timestamp", "")}
            ]
            output.extend(sanitized)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            await websocket.send_json(
                {
                    "type": "import-history-result",
                    "ok": True,
                    "history_uid": history_uid,
                }
            )
        except Exception as e:
            logger.error(f"import-history failed: {e}")
            await websocket.send_json(
                {"type": "error", "message": f"Failed to import history: {str(e)}"}
            )

    async def _handle_memory_kinds_info(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Return known memory kinds with localized labels and descriptions.

        Payload: { lang?: 'ru'|'en'|'zh' }
        Response: { type: 'memory-kinds-info', kinds: string[], labels: {kind: str}, descriptions: {kind: str} }
        """
        try:
            lang = str(data.get("lang") or "ru").lower()
            # Stable order of consolidated kinds used in UI
            kinds = [
                "FactsAboutUser",
                "PastEvents",
                "SelfBeliefs",
                "Objectives",
                "KeyFacts",
                "Emotions",
                "Mood",
            ]
            # Minimal built-in localization (ru/en). Fallback to kind name.
            labels_ru = {
                "FactsAboutUser": "Факты о зрителях",
                "PastEvents": "События на стриме",
                "SelfBeliefs": "Убеждения Нейри",
                "Objectives": "Цели Нейри",
                "KeyFacts": "Важные факты",
                "Emotions": "Эмоции",
                "Mood": "Настроение",
            }
            desc_ru = {
                "FactsAboutUser": "Факты о зрителях (ник, интересы, привычки, предпочтения)",
                "PastEvents": "События, произошедшие на стриме (челленджи, шутки, конфликты)",
                "SelfBeliefs": "Убеждения и установки самой Нейри (о себе, своей роли)",
                "Objectives": "Цели, которые Нейри поставила в ходе стрима",
                "KeyFacts": "Любая важная информация для будущих стримов",
                "Emotions": "Эмоции, которые испытывала Нейри или в целом",
                "Mood": "Агрегированное настроение пользователей",
            }
            labels_en = {
                "FactsAboutUser": "Facts about viewers",
                "PastEvents": "Past events",
                "SelfBeliefs": "Self beliefs",
                "Objectives": "Objectives",
                "KeyFacts": "Key facts",
                "Emotions": "Emotions",
                "Mood": "Mood",
            }
            desc_en = {
                "FactsAboutUser": "Viewer facts (nick, interests, habits, preferences)",
                "PastEvents": "Events that happened on stream",
                "SelfBeliefs": "The VTuber's beliefs and assumptions",
                "Objectives": "Goals set during the stream",
                "KeyFacts": "Any important info for future streams",
                "Emotions": "Emotional snapshots",
                "Mood": "Aggregated user mood",
            }
            labels = labels_ru if lang.startswith("ru") else labels_en
            desc = desc_ru if lang.startswith("ru") else desc_en
            await websocket.send_json(
                {
                    "type": "memory-kinds-info",
                    "kinds": kinds,
                    "labels": labels,
                    "descriptions": desc,
                }
            )
        except Exception as e:
            logger.error(f"memory-kinds-info failed: {e}")
            await websocket.send_json(
                {"type": "error", "message": f"Failed to get kinds info: {str(e)}"}
            )

    async def _handle_mood_list(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Return current aggregated user moods as an array of {user, score, label}."""
        try:
            ctx = self.client_contexts.get(client_uid) or self.default_context_cache
            moods = getattr(ctx, "user_mood", {}) or {}
            items = []
            for user, score in moods.items():
                label = "нейтрально"
                try:
                    s = float(score)
                    label = (
                        "позитив"
                        if s > 0.2
                        else ("негатив" if s < -0.2 else "нейтрально")
                    )
                except Exception:
                    s = 0.0
                items.append({"user": str(user), "score": float(s), "label": label})
            # sort by absolute score desc
            items.sort(key=lambda x: abs(x.get("score", 0.0)), reverse=True)
            await websocket.send_json({"type": "mood-list-result", "items": items})
        except Exception as e:
            logger.error(f"mood-list failed: {e}")
            await websocket.send_json(
                {"type": "error", "message": f"Failed to list moods: {str(e)}"}
            )

    async def _handle_mood_reset(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Reset mood for a specific user or all if no user provided."""
        try:
            ctx = self.client_contexts.get(client_uid) or self.default_context_cache
            user = data.get("user")
            if user:
                try:
                    if user in ctx.user_mood:
                        del ctx.user_mood[user]
                except Exception:
                    pass
                await websocket.send_json(
                    {"type": "mood-reset-result", "ok": True, "user": user}
                )
            else:
                ctx.user_mood = {}
                await websocket.send_json(
                    {"type": "mood-reset-result", "ok": True, "all": True}
                )
        except Exception as e:
            logger.error(f"mood-reset failed: {e}")
            await websocket.send_json(
                {"type": "error", "message": f"Failed to reset mood: {str(e)}"}
            )

    async def _handle_mood_set(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Manually set mood score for a user (overrides smoothing once)."""
        try:
            ctx = self.client_contexts.get(client_uid) or self.default_context_cache
            user = data.get("user")
            score = data.get("score")
            if not user:
                await websocket.send_json(
                    {"type": "mood-updated", "ok": False, "error": "user required"}
                )
                return
            try:
                s = float(score)
                s = max(-1.0, min(1.0, s))
            except Exception:
                await websocket.send_json(
                    {"type": "mood-updated", "ok": False, "error": "invalid score"}
                )
                return
            ctx.user_mood[user] = s
            await websocket.send_json(
                {"type": "mood-updated", "ok": True, "user": user, "score": s}
            )
        except Exception as e:
            logger.error(f"mood-set failed: {e}")
            await websocket.send_json(
                {"type": "error", "message": f"Failed to set mood: {str(e)}"}
            )

    async def handle_websocket_communication(
        self, websocket: WebSocket, client_uid: str
    ) -> None:
        """
        Handle ongoing WebSocket communication

        Args:
            websocket: The WebSocket connection
            client_uid: Unique identifier for the client
        """
        try:
            while True:
                try:
                    data = await websocket.receive_json()
                    # Log incoming client message (FROM frontend)
                    if DEBUG_WS:
                        try:
                            logger.bind(src="frontend", uid=client_uid).log(
                                "DEBUG", f"WS IN: {data}"
                            )
                            logger.opt(depth=0).log(
                                "DEBUG", f"WS_IN {client_uid}: {data}"
                            )
                        except Exception:
                            pass
                    message_handler.handle_message(client_uid, data)
                    await self._route_message(websocket, client_uid, data)
                except WebSocketDisconnect:
                    raise
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": str(e)})
                    )
                    continue

        except WebSocketDisconnect:
            logger.info(f"Client {client_uid} disconnected")
            raise
        except Exception as e:
            logger.error(f"Fatal error in WebSocket communication: {e}")
            raise

    async def _route_message(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """
        Route incoming message to appropriate handler

        Args:
            websocket: The WebSocket connection
            client_uid: Client identifier
            data: Message data
        """
        msg_type = data.get("type")
        if not msg_type:
            logger.warning("Message received without type")
            # Heuristic fallback: if audio present, treat as chunk
            if isinstance(data.get("audio"), list) and data["audio"]:
                await self._handle_audio_data(websocket, client_uid, data)
            return

        # // DEBUG: [FIXED] Assign or propagate request_id for correlation | Ref: 5
        rid = data.get("request_id") or str(uuid4())
        data["request_id"] = rid
        set_request_id(rid)

        handlers = {
            "memory-consolidate-history": self._handle_memory_consolidate_history,
        }
        handler = handlers.get(msg_type, self._message_handlers.get(msg_type))
        if handler:
            await handler(websocket, client_uid, data)
        else:
            if msg_type != "frontend-playback-complete":
                logger.warning(f"Unknown message type: {msg_type}")

    async def _handle_group_operation(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Handle group-related operations"""
        operation = data.get("type")
        target_uid = data.get(
            "invitee_uid" if operation == "add-client-to-group" else "target_uid"
        )

        await handle_group_operation(
            operation=operation,
            client_uid=client_uid,
            target_uid=target_uid,
            chat_group_manager=self.chat_group_manager,
            client_connections=self.client_connections,
            send_group_update=self.send_group_update,
        )

    async def handle_disconnect(self, client_uid: str) -> None:
        """Handle client disconnection"""
        group = self.chat_group_manager.get_client_group(client_uid)
        if group:
            await handle_group_interrupt(
                group_id=group.group_id,
                heard_response="",
                current_conversation_tasks=self.current_conversation_tasks,
                chat_group_manager=self.chat_group_manager,
                client_contexts=self.client_contexts,
                broadcast_to_group=self.broadcast_to_group,
            )

        await handle_client_disconnect(
            client_uid=client_uid,
            chat_group_manager=self.chat_group_manager,
            client_connections=self.client_connections,
            send_group_update=self.send_group_update,
        )

        # Clean up other client data
        self.client_connections.pop(client_uid, None)
        ctx = self.client_contexts.pop(client_uid, None)
        self.received_data_buffers.pop(client_uid, None)
        self._init_ack.pop(client_uid, None)
        if client_uid in self.current_conversation_tasks:
            task = self.current_conversation_tasks[client_uid]
            if task and not task.done():
                task.cancel()
            self.current_conversation_tasks.pop(client_uid, None)

        # Call context close to clean up resources (e.g., MCPClient)
        if ctx:
            try:
                await ctx.close()
            except Exception:
                pass
        # Drop shared send_text to avoid sends after close
        if self.default_context_cache:
            self.default_context_cache.send_text = None

        logger.info(f"Client {client_uid} disconnected")
        message_handler.cleanup_client(client_uid)

    async def broadcast_to_group(
        self, group_members: list[str], message: dict, exclude_uid: str = None
    ) -> None:
        """Broadcasts a message to group members"""
        await broadcast_to_group(
            group_members=group_members,
            message=message,
            client_connections=self.client_connections,
            exclude_uid=exclude_uid,
        )

    async def send_group_update(self, websocket: WebSocket, client_uid: str):
        """Sends group information to a client"""
        group = self.chat_group_manager.get_client_group(client_uid)
        if group:
            current_members = self.chat_group_manager.get_group_members(client_uid)
            payload = {
                "type": "group-update",
                "members": current_members,
                "is_owner": group.owner_uid == client_uid,
            }
            if DEBUG_WS:
                try:
                    logger.bind(dst="frontend", uid=client_uid).log(
                        "DEBUG", f"WS OUT: {payload}"
                    )
                except Exception:
                    pass
            await websocket.send_text(json.dumps(payload))
        else:
            payload = {"type": "group-update", "members": [], "is_owner": False}
            if DEBUG_WS:
                try:
                    logger.bind(dst="frontend", uid=client_uid).log(
                        "DEBUG", f"WS OUT: {payload}"
                    )
                except Exception:
                    pass
            await websocket.send_text(json.dumps(payload))

    async def _handle_interrupt(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle conversation interruption"""
        heard_response = data.get("text", "")
        context = self.client_contexts[client_uid]
        group = self.chat_group_manager.get_client_group(client_uid)

        if group and len(group.members) > 1:
            await handle_group_interrupt(
                group_id=group.group_id,
                heard_response=heard_response,
                current_conversation_tasks=self.current_conversation_tasks,
                chat_group_manager=self.chat_group_manager,
                client_contexts=self.client_contexts,
                broadcast_to_group=self.broadcast_to_group,
            )
        else:
            await handle_individual_interrupt(
                client_uid=client_uid,
                current_conversation_tasks=self.current_conversation_tasks,
                context=context,
                heard_response=heard_response,
            )

    async def _handle_history_list_request(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for chat history list"""
        context = self.client_contexts[client_uid]
        histories = get_history_list(context.character_config.conf_uid)
        await websocket.send_text(
            json.dumps({"type": "history-list", "histories": histories})
        )

    async def _handle_history_list_grouped(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Return histories grouped by calendar date with friendly labels.

        Response: { type: 'history-list-grouped', groups: [{ date: 'YYYY-MM-DD', label: string, items: [...] }] }
        """
        try:
            context = self.client_contexts[client_uid]
            items = get_history_list(context.character_config.conf_uid)
            groups: Dict[str, dict] = {}
            from datetime import datetime

            for it in items:
                ts = it.get("timestamp") or ""
                try:
                    d = datetime.fromisoformat(str(ts))
                    date_key = d.strftime("%Y-%m-%d")
                    label = d.strftime("stream-%d.%m.%y")
                except Exception:
                    date_key = "unknown"
                    label = "stream-unknown"
                g = groups.setdefault(
                    date_key, {"date": date_key, "label": label, "items": []}
                )
                g["items"].append(it)
            out = sorted(groups.values(), key=lambda g: g.get("date", ""), reverse=True)
            await websocket.send_text(
                json.dumps({"type": "history-list-grouped", "groups": out})
            )
        except Exception as e:
            logger.error(f"history-list-grouped failed: {e}")
            await websocket.send_json(
                {"type": "error", "message": f"Failed to group histories: {str(e)}"}
            )

    async def _handle_fetch_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle fetching and setting specific chat history"""
        history_uid = data.get("history_uid")
        if not history_uid:
            return

        context = self.client_contexts[client_uid]
        # Update history_uid in service context
        context.history_uid = history_uid
        context.agent_engine.set_memory_from_history(
            conf_uid=context.character_config.conf_uid,
            history_uid=history_uid,
        )

        messages = [
            msg
            for msg in get_history(
                context.character_config.conf_uid,
                history_uid,
            )
            if msg["role"] != "system"
        ]
        await websocket.send_text(
            json.dumps({"type": "history-data", "messages": messages})
        )
        if history_uid == context.history_uid:
            context.history_uid = None

    async def _handle_create_history(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle creation of new chat history"""
        context = self.client_contexts[client_uid]
        history_uid = create_new_history(context.character_config.conf_uid)
        if history_uid:
            context.history_uid = history_uid
            context.agent_engine.set_memory_from_history(
                conf_uid=context.character_config.conf_uid,
                history_uid=history_uid,
            )
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "new-history-created",
                        "history_uid": history_uid,
                    }
                )
            )

    async def _handle_delete_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Handle deletion of chat history"""
        history_uid = data.get("history_uid")
        if not history_uid:
            return

        context = self.client_contexts[client_uid]
        success = delete_history(
            context.character_config.conf_uid,
            history_uid,
        )
        await websocket.send_text(
            json.dumps(
                {
                    "type": "history-deleted",
                    "success": success,
                    "history_uid": history_uid,
                }
            )
        )
        if history_uid == context.history_uid:
            context.history_uid = None

    async def _handle_audio_data(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle incoming audio data"""
        audio_data = data.get("audio", [])
        if audio_data:
            t0 = asyncio.get_running_loop().time()
            lock = self._buffer_locks.get(client_uid)
            if not lock:
                self._buffer_locks[client_uid] = asyncio.Lock()
                lock = self._buffer_locks[client_uid]
            async with lock:
                before = len(self.received_data_buffers.get(client_uid, np.array([])))
                self.received_data_buffers[client_uid] = np.append(
                    self.received_data_buffers[client_uid],
                    np.array(audio_data, dtype=np.float32),
                )
                after = len(self.received_data_buffers[client_uid])
            dt_ms = int((asyncio.get_running_loop().time() - t0) * 1000)
            logger.bind(component="vad").debug(
                {
                    "event": "buffer.append",
                    "client_uid": client_uid,
                    "added": after - before,
                    "total": after,
                    "latency_ms": dt_ms,
                }
            )

    async def _handle_raw_audio_data(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle incoming raw audio data for VAD processing"""
        context = self.client_contexts[client_uid]
        chunk = data.get("audio", [])
        if chunk:
            t0 = asyncio.get_running_loop().time()
            for audio_bytes in context.vad_engine.detect_speech(chunk):
                if audio_bytes == b"<|PAUSE|>":
                    logger.bind(component="vad").info(
                        {
                            "event": "vad.pause",
                            "client_uid": client_uid,
                        }
                    )
                    await websocket.send_text(
                        json.dumps({"type": "control", "text": "interrupt"})
                    )
                elif audio_bytes == b"<|RESUME|>":
                    logger.bind(component="vad").info(
                        {
                            "event": "vad.resume",
                            "client_uid": client_uid,
                        }
                    )
                    pass
                elif len(audio_bytes) > 1024:
                    # Detected audio activity (voice)
                    logger.bind(component="vad").info(
                        {
                            "event": "vad.voice",
                            "client_uid": client_uid,
                            "payload_size": len(audio_bytes),
                        }
                    )
                    lock = self._buffer_locks.get(client_uid)
                    if not lock:
                        self._buffer_locks[client_uid] = asyncio.Lock()
                        lock = self._buffer_locks[client_uid]
                    async with lock:
                        self.received_data_buffers[client_uid] = np.append(
                            self.received_data_buffers[client_uid],
                            np.frombuffer(audio_bytes, dtype=np.int16).astype(
                                np.float32
                            ),
                        )
                    await websocket.send_text(
                        json.dumps({"type": "control", "text": "mic-audio-end"})
                    )
            logger.bind(component="vad").debug(
                {
                    "event": "vad.chunk_processed",
                    "client_uid": client_uid,
                    "latency_ms": int((asyncio.get_running_loop().time() - t0) * 1000),
                }
            )

    async def _handle_conversation_trigger(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle triggers that start a conversation"""
        # Snapshot buffer length under lock to avoid race with VAD appends
        lock = self._buffer_locks.get(client_uid)
        if not lock:
            self._buffer_locks[client_uid] = asyncio.Lock()
            lock = self._buffer_locks[client_uid]
        async with lock:
            total_samples = len(
                self.received_data_buffers.get(client_uid, np.array([]))
            )
        logger.info(
            f"Conversation trigger received ({data.get('type')}). Buffered audio samples: {total_samples}"
        )
        await handle_conversation_trigger(
            msg_type=data.get("type", ""),
            data=data,
            client_uid=client_uid,
            context=self.client_contexts[client_uid],
            websocket=websocket,
            client_contexts=self.client_contexts,
            client_connections=self.client_connections,
            chat_group_manager=self.chat_group_manager,
            received_data_buffers=self.received_data_buffers,
            current_conversation_tasks=self.current_conversation_tasks,
            broadcast_to_group=self.broadcast_to_group,
        )

    async def _handle_fetch_configs(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle fetching available configurations"""
        # Mark init delivered successfully (frontend requested configs)
        self._init_ack[client_uid] = True
        context = self.client_contexts[client_uid]
        config_files = scan_config_alts_directory(context.system_config.config_alts_dir)
        await websocket.send_text(
            json.dumps({"type": "config-files", "configs": config_files})
        )

        # Also send current model/config to ensure frontend initializes Live2D reliably
        try:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "set-model-and-conf",
                        "model_info": context.live2d_model.model_info
                        if context.live2d_model
                        else None,
                        "conf_name": context.character_config.conf_name,
                        "conf_uid": context.character_config.conf_uid,
                        "client_uid": client_uid,
                        "tts_info": {
                            "model": context.character_config.tts_config.tts_model
                        },
                    }
                )
            )
        except Exception as e:
            logger.warning(f"Failed to send set-model-and-conf on fetch-configs: {e}")

    async def _handle_config_switch(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle switching to a different configuration"""
        config_file_name = data.get("file")
        if config_file_name:
            context = self.client_contexts[client_uid]
            await context.handle_config_switch(websocket, config_file_name)

    async def _handle_fetch_backgrounds(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle fetching available background images"""
        bg_files = scan_bg_directory()
        await websocket.send_text(
            json.dumps({"type": "background-files", "files": bg_files})
        )

    async def _handle_audio_play_start(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """
        Handle audio playback start notification
        """
        group_members = self.chat_group_manager.get_group_members(client_uid)
        if len(group_members) > 1:
            display_text = data.get("display_text")
            if display_text:
                silent_payload = prepare_audio_payload(
                    audio_path=None,
                    display_text=display_text,
                    actions=None,
                    forwarded=True,
                )
                await self.broadcast_to_group(
                    group_members, silent_payload, exclude_uid=client_uid
                )

    async def _handle_group_info(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle group info request"""
        await self.send_group_update(websocket, client_uid)

    async def _handle_init_config_request(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for initialization configuration"""
        # Frontend explicitly asked for init config -> ack
        self._init_ack[client_uid] = True
        context = self.client_contexts.get(client_uid)
        if not context:
            context = self.default_context_cache

        await websocket.send_text(
            json.dumps(
                {
                    "type": "set-model-and-conf",
                    "model_info": context.live2d_model.model_info,
                    "conf_name": context.character_config.conf_name,
                    "conf_uid": context.character_config.conf_uid,
                    "client_uid": client_uid,
                    "tts_info": {
                        "model": context.character_config.tts_config.tts_model
                    },
                }
            )
        )

    async def _handle_heartbeat(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle heartbeat messages from clients"""
        try:
            await websocket.send_json({"type": "heartbeat-ack"})
        except Exception as e:
            logger.error(f"Error sending heartbeat acknowledgment: {e}")

    async def _handle_frontend_ready(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Frontend reports it is ready to receive state. Mark ack and resend once."""
        self._init_ack[client_uid] = True
        try:
            ctx = self.client_contexts.get(client_uid) or self.default_context_cache
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "set-model-and-conf",
                        "model_info": ctx.live2d_model.model_info
                        if ctx.live2d_model
                        else None,
                        "conf_name": ctx.character_config.conf_name,
                        "conf_uid": ctx.character_config.conf_uid,
                        "client_uid": client_uid,
                        "tts_info": {
                            "model": ctx.character_config.tts_config.tts_model
                        },
                    }
                )
            )
        except Exception:
            pass

    async def _handle_frontend_log(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Ingest lightweight frontend logs over WS without side effects."""
        try:
            level = str(data.get("level") or "info").lower()
            payload = {k: v for k, v in data.items() if k != "type"}
            bound = logger.bind(component="frontend", client_uid=client_uid)
            if level == "error":
                bound.error(payload)
            elif level in ("warn", "warning"):
                bound.warning(payload)
            else:
                bound.info(payload)
        except Exception:
            pass

    async def _handle_window_error(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Record window error reports from frontend."""
        try:
            payload = {k: v for k, v in data.items() if k != "type"}
            logger.bind(component="frontend", client_uid=client_uid).error(payload)
        except Exception:
            pass
