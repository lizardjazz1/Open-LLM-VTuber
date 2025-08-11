import asyncio
import json
import os
from typing import Any, Callable, Dict, List, Optional

from fastapi import WebSocket
from loguru import logger

from .agent.agent_factory import AgentFactory
from .agent.agents.agent_interface import AgentInterface
from .asr.asr_factory import ASRFactory
from .asr.asr_interface import ASRInterface
from .config_manager import (
    AgentConfig,
    ASRConfig,
    CharacterConfig,
    Config,
    SystemConfig,
    TTSConfig,
    TranslatorConfig,
    VADConfig,
    read_yaml,
    validate_config,
)
from .live2d_model import Live2dModel
from .mcpp.server_registry import ServerRegistry
from .mcpp.tool_adapter import ToolAdapter
from .mcpp.tool_executor import ToolExecutor
from .mcpp.tool_manager import ToolManager
from .mcpp.mcp_client import MCPClient
from .translate.translate_factory import TranslateFactory
from .translate.translate_interface import TranslateInterface
from .tts.tts_factory import TTSFactory
from .tts.tts_interface import TTSInterface
from .vad.vad_factory import VADFactory
from .vad.vad_interface import VADInterface
from .twitch import TwitchClient, TwitchMessage
from .memory.memory_service import MemoryService
from .vtuber_memory import VtuberMemoryService, VtuberMemoryInterface
from .vtuber_memory.scheduler import ConsolidationScheduler

# Import i18n system
from .i18n import t

from prompts import prompt_loader

# // DEBUG: [FIXED] Request ID propagation utilities | Ref: 5
from .logging_utils import set_request_id
from uuid import uuid4


def _prune_unknown_keys(
    template: Dict[str, Any], data: Dict[str, Any]
) -> Dict[str, Any]:
    """Recursively keep only keys that exist in template, pruning unknown fields.

    - template: dictionary representing the expected schema (e.g., model_dump of CharacterConfig)
    - data: incoming dictionary to be pruned
    """
    if not isinstance(template, dict) or not isinstance(data, dict):
        return {}
    pruned: Dict[str, Any] = {}
    for key, tmpl_val in template.items():
        if key not in data:
            continue
        val = data[key]
        if isinstance(tmpl_val, dict) and isinstance(val, dict):
            nested = _prune_unknown_keys(tmpl_val, val)
            pruned[key] = nested
        else:
            pruned[key] = val
    return pruned


class ServiceContext:
    """Initializes, stores, and updates the asr, tts, and llm instances and other
    configurations for a connected client."""

    # Shared Twitch client across all contexts to avoid duplicate event handlers
    _global_twitch_client = None
    _global_twitch_owner: "ServiceContext | None" = None

    def __init__(self):
        self.config: Config = None
        self.system_config: SystemConfig = None
        self.character_config: CharacterConfig = None

        self.live2d_model: Live2dModel = None
        self.asr_engine: ASRInterface = None
        self.tts_engine: TTSInterface = None
        self.agent_engine: AgentInterface = None
        # translate_engine can be none if translation is disabled
        self.vad_engine: VADInterface | None = None
        self.translate_engine: TranslateInterface | None = None

        self.mcp_server_registery: ServerRegistry | None = None
        self.tool_adapter: ToolAdapter | None = None
        self.tool_manager: ToolManager | None = None
        self.mcp_client: MCPClient | None = None
        self.tool_executor: ToolExecutor | None = None

        # Long-term memory services
        self.memory_service: MemoryService | None = None
        self.vtuber_memory_service: VtuberMemoryInterface | None = None
        self.memory_enabled: bool = True
        self.memory_top_k: int = 4
        self.memory_min_importance: float | None = None
        self.memory_kinds: list[str] | None = None
        self._consolidation_scheduler: ConsolidationScheduler | None = None

        # Twitch integration
        self.twitch_client: TwitchClient | None = None

        # the system prompt is a combination of the persona prompt and live2d expression prompt
        self.system_prompt: str = None

        # Store the generated MCP prompt string (if MCP enabled)
        self.mcp_prompt: str = ""

        self.history_uid: str = ""  # Add history_uid field

        self.send_text: Callable = None
        self.client_uid: str = None

        # Incoming message priority queue
        # Lower number = higher priority
        self._incoming_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._queue_worker_task: Optional[asyncio.Task] = None
        self._seq_counter: int = 0  # tie-breaker for PriorityQueue
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._daily_summary_task: Optional[asyncio.Task] = None
        self._queue_loop: Optional[asyncio.AbstractEventLoop] = None
        # Aggregated mood per external user (e.g., Twitch), exponential smoothing
        self.user_mood: Dict[str, float] = {}

    def __str__(self):
        return (
            f"ServiceContext:\n"
            f"  System Config: {'Loaded' if self.system_config else 'Not Loaded'}\n"
            f"    Details: {json.dumps(self.system_config.model_dump(), indent=6) if self.system_config else 'None'}\n"
            f"  Live2D Model: {self.live2d_model.model_info if self.live2d_model else 'Not Loaded'}\n"
            f"  ASR Engine: {type(self.asr_engine).__name__ if self.asr_engine else 'Not Loaded'}\n"
            f"    Config: {json.dumps(self.character_config.asr_config.model_dump(), indent=6) if self.character_config.asr_config else 'None'}\n"
            f"  TTS Engine: {type(self.tts_engine).__name__ if self.tts_engine else 'Not Loaded'}\n"
            f"    Config: {json.dumps(self.character_config.tts_config.model_dump(), indent=6) if self.character_config.tts_config else 'None'}\n"
            f"  LLM Engine: {type(self.agent_engine).__name__ if self.agent_engine else 'Not Loaded'}\n"
            f"    Agent Config: {json.dumps(self.character_config.agent_config.model_dump(), indent=6) if self.character_config.agent_config else 'None'}\n"
            f"  VAD Engine: {type(self.vad_engine).__name__ if self.vad_engine else 'Not Loaded'}\n"
            f"    Agent Config: {json.dumps(self.character_config.vad_config.model_dump(), indent=6) if self.character_config.vad_config else 'None'}\n"
            f"  System Prompt: {self.system_prompt or 'Not Set'}\n"
            f"  MCP Enabled: {'Yes' if self.mcp_client else 'No'}"
        )

    # ==== Queue helpers ====

    def _compute_priority(
        self, source: str, twitch_flags: Optional[Dict[str, Any]] = None
    ) -> int:
        """Compute message priority (lower is more important).
        - Creator/local UI and mic input => 0 (highest)
        - Paid Twitch (bits>0 or subscription) => 0 (same as creator)
        - Moderator/Broadcaster => 1
        - Everyone else => 2
        """
        if source in ("local-ui", "local-voice", "system"):
            return 0
        if source == "twitch" and twitch_flags:
            if twitch_flags.get("bits", 0) > 0 or twitch_flags.get(
                "is_subscriber", False
            ):
                return 0
            if twitch_flags.get("is_broadcaster", False) or twitch_flags.get(
                "is_moderator", False
            ):
                return 1
        return 2

    async def enqueue_message(
        self,
        *,
        content: str,
        from_name: str,
        source: str,
        text_source_enum: Any,
        images: Optional[List[Dict[str, Any]]] = None,
        twitch_flags: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Put a message into the priority queue for sequential processing."""
        priority = self._compute_priority(source, twitch_flags)
        self._seq_counter += 1
        item = {
            "content": content,
            "from_name": from_name,
            "source": source,
            "text_source_enum": text_source_enum,
            "images": images,
            "metadata": metadata or {},
        }
        logger.debug(
            f"Queued message (priority={priority}) from {source}:{from_name}: {content[:120]}"
        )

        # Ensure the worker is running (in bound loop)
        await self._ensure_worker()

        # Put item respecting the queue's loop affinity
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None  # called from non-async context

        if self._loop and current_loop is not self._loop:
            # Schedule put in the bound loop
            async def _put():
                await self._incoming_queue.put((priority, self._seq_counter, item))

            import asyncio as _asyncio

            _asyncio.run_coroutine_threadsafe(_put(), self._loop)
            return
        else:
            await self._incoming_queue.put((priority, self._seq_counter, item))

    async def _queue_worker(self) -> None:
        """Continuously process incoming messages one-by-one in priority order."""
        logger.info("Message queue worker started")
        while True:
            try:
                priority, _seq, item = await self._incoming_queue.get()
                # Lazy import to avoid circular
                from .conversations.single_conversation import (
                    process_single_conversation,
                )

                # Merge metadata
                md = dict(item.get("metadata") or {})
                md.setdefault("from_name", item["from_name"])
                md.setdefault("text_source", item["text_source_enum"])  # Enum instance

                logger.info(
                    f"Dequeued (p={priority}) [{item['source']}:{item['from_name']}] {item['content'][:200]}"
                )
                try:
                    await process_single_conversation(
                        context=self,
                        websocket_send=self.send_text,
                        client_uid=self.client_uid,
                        user_input=item["content"],
                        images=item.get("images"),
                        metadata=md,
                    )
                except Exception as e:
                    logger.error(f"Queue item processing failed: {e}")
                finally:
                    self._incoming_queue.task_done()
            except asyncio.CancelledError:
                logger.info("Message queue worker cancelled")
                break
            except Exception as e:
                logger.error(f"Queue worker error: {e}")

    async def _ensure_worker(self) -> None:
        """Ensure the queue worker is running and the queue is bound to the current loop."""
        # Bind to current running loop if not set or closed
        try:
            loop_closed = (self._loop is not None) and getattr(
                self._loop, "is_closed", lambda: False
            )()
        except Exception:
            loop_closed = True
        if (not self._loop) or loop_closed:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                # If no running loop, try default
                self._loop = asyncio.get_event_loop()

        # Recreate queue if it was created on a different loop
        try:
            queue_loop_closed = (self._queue_loop is not None) and getattr(
                self._queue_loop, "is_closed", lambda: False
            )()
        except Exception:
            queue_loop_closed = True
        if (
            (self._queue_loop is None)
            or queue_loop_closed
            or (self._queue_loop is not self._loop)
        ):
            # Cancel old worker if any
            if self._queue_worker_task and not self._queue_worker_task.done():
                try:
                    self._queue_worker_task.cancel()
                except Exception:
                    pass
            # Create a fresh queue for this loop
            self._incoming_queue = asyncio.PriorityQueue()
            self._queue_loop = self._loop

        if (not self._queue_worker_task) or self._queue_worker_task.done():
            # Start worker in bound loop
            if asyncio.get_running_loop() is self._loop:
                self._queue_worker_task = asyncio.create_task(self._queue_worker())
            else:
                import asyncio as _asyncio

                def _start():
                    self._queue_worker_task = _asyncio.create_task(self._queue_worker())

                try:
                    self._loop.call_soon_threadsafe(_start)
                except RuntimeError:
                    # Loop might have been closed between checks; retry by rebinding
                    try:
                        self._loop = asyncio.get_running_loop()
                        # Ensure queue also matches
                        self._incoming_queue = asyncio.PriorityQueue()
                        self._queue_loop = self._loop
                        self._queue_worker_task = asyncio.create_task(
                            self._queue_worker()
                        )
                    except Exception:
                        pass
        # Nightly consolidation is disabled by design; frontend will trigger consolidation explicitly
        if self._daily_summary_task and not self._daily_summary_task.done():
            try:
                self._daily_summary_task.cancel()
            except Exception:
                pass

    # ==== Initializers

    async def _init_mcp_components(self, use_mcpp, enabled_servers):
        """Initializes MCP components based on configuration, dynamically fetching tool info."""
        logger.debug(
            f"Initializing MCP components: use_mcpp={use_mcpp}, enabled_servers={enabled_servers}"
        )

        # Reset MCP components first
        self.mcp_server_registery = None
        self.tool_manager = None
        self.mcp_client = None
        self.tool_executor = None
        self.json_detector = None
        self.mcp_prompt = ""

        if use_mcpp and enabled_servers:
            # 1. Initialize ServerRegistry
            self.mcp_server_registery = ServerRegistry()
            logger.info(t("service.server_registry_initialized"))

            # 2. Use ToolAdapter to get the MCP prompt and tools
            if not self.tool_adapter:
                logger.error(
                    "ToolAdapter not initialized before calling _init_mcp_components."
                )
                self.mcp_prompt = "[Error: ToolAdapter not initialized]"
                return  # Exit if ToolAdapter is mandatory and not initialized

            try:
                (
                    mcp_prompt_string,
                    openai_tools,
                    claude_tools,
                ) = await self.tool_adapter.get_tools(enabled_servers)
                # Store the generated prompt string
                self.mcp_prompt = mcp_prompt_string
                logger.info(
                    t(
                        "service.dynamically_generated_mcp_prompt",
                        length=len(self.mcp_prompt),
                    )
                )
                logger.info(
                    t(
                        "service.dynamically_formatted_tools",
                        openai=len(openai_tools),
                        claude=len(claude_tools),
                    )
                )

                # 3. Initialize ToolManager with the fetched formatted tools
                _, raw_tools_dict = await self.tool_adapter.get_server_and_tool_info(
                    enabled_servers
                )
                self.tool_manager = ToolManager(
                    formatted_tools_openai=openai_tools,
                    formatted_tools_claude=claude_tools,
                    initial_tools_dict=raw_tools_dict,
                )
                logger.info(t("service.tool_manager_initialized"))

            except Exception as e:
                logger.error(
                    t("service.failed_dynamic_mcp_tool_construction", error=str(e)),
                    exc_info=True,
                )
                # Ensure dependent components are not created if construction fails
                self.tool_manager = None
                self.mcp_prompt = "[Error constructing MCP tools/prompt]"

            # 4. Initialize MCPClient
            if self.mcp_server_registery:
                self.mcp_client = MCPClient(
                    self.mcp_server_registery, self.send_text, self.client_uid
                )
                logger.info(t("service.mcp_client_initialized"))
            else:
                logger.error(t("service.mcp_enabled_but_server_registry_not_available"))
                self.mcp_client = None  # Ensure it's None

            # 5. Initialize ToolExecutor
            if self.mcp_client and self.tool_manager:
                self.tool_executor = ToolExecutor(self.mcp_client, self.tool_manager)
                logger.info(t("service.tool_executor_initialized"))
            else:
                logger.warning(t("service.mcp_client_or_tool_manager_not_available"))
                self.tool_executor = None  # Ensure it's None

            logger.info(t("service.stream_json_detector_initialized"))

        elif use_mcpp and not enabled_servers:
            logger.warning(
                t("service.use_mcpp_is_true_but_mcp_enabled_servers_list_is_empty")
            )
        else:
            logger.debug(t("service.mcp_components_not_initialized"))

    async def close(self):
        """Clean up resources, especially the MCPClient."""
        logger.info(t("service.closing_service_context_resources"))
        if self.mcp_client:
            logger.info(
                t(
                    "service.closing_mcp_client_for_context_instance",
                    context_id=id(self),
                )
            )
            await self.mcp_client.aclose()
            self.mcp_client = None
        if self.agent_engine and hasattr(self.agent_engine, "close"):
            await self.agent_engine.close()  # Ensure agent resources are also closed
        # Only the global owner should close the Twitch client
        if ServiceContext._global_twitch_owner is self and self.twitch_client:
            try:
                await self.twitch_client.disconnect()
            except Exception:
                pass
            ServiceContext._global_twitch_client = None
            ServiceContext._global_twitch_owner = None
        # Cancel queue worker if running
        if self._queue_worker_task and not self._queue_worker_task.done():
            self._queue_worker_task.cancel()
        if self._daily_summary_task and not self._daily_summary_task.done():
            self._daily_summary_task.cancel()
        logger.info(t("service.service_context_closed"))

    async def _nightly_summary_loop(self) -> None:
        """Background task: once per ~24h, summarize recent conversation into long-term memory."""
        try:
            while True:
                await asyncio.sleep(60 * 60 * 24)  # 24h
                try:
                    await self.trigger_memory_consolidation(reason="nightly")
                except Exception as e:
                    logger.debug(f"nightly summary skipped: {e}")
        except asyncio.CancelledError:
            return

    async def trigger_memory_consolidation(self, reason: str = "manual") -> None:
        """Summarize last N messages and write categorized memory entries (facts/events/beliefs/objectives)."""
        try:
            mem = self.vtuber_memory_service or self.memory_service
            if not (self.agent_engine and mem and getattr(mem, "enabled", False)):
                return
            # ensure history exists
            if not self.history_uid:
                try:
                    from .chat_history_manager import create_new_history

                    self.history_uid = create_new_history(
                        self.character_config.conf_uid
                    )
                    logger.info(
                        f"Created new history for consolidation: {self.history_uid}"
                    )
                except Exception as e:
                    logger.debug(f"Failed to create history before consolidation: {e}")
            # collect last N messages (configurable)
            from .chat_history_manager import get_history

            msgs = []
            try:
                msgs = (
                    get_history(self.character_config.conf_uid, self.history_uid)
                    if self.history_uid
                    else []
                )
            except Exception:
                msgs = []
            if not msgs:
                return
            recent_texts: list[str] = []
            try:
                # read consolidation window from config
                window = (
                    getattr(
                        self.character_config.agent_config.agent_settings.basic_memory_agent,
                        "consolidate_recent_messages",
                        120,
                    )
                    if self.character_config
                    and self.character_config.agent_config
                    and self.character_config.agent_config.agent_settings
                    and self.character_config.agent_config.agent_settings.basic_memory_agent
                    else 120
                )
                window = int(max(20, min(500, window)))
            except Exception:
                window = 120
            for m in msgs[-window:]:
                content = (m.get("content") or "").strip()
                if content:
                    recent_texts.append(content)
            # ask agent to summarize and categorize
            try:
                summary = await self.agent_engine.summarize_texts(recent_texts)  # type: ignore[attr-defined]
            except Exception:
                summary = {}
            if not summary:
                return
            # build entries with importance and tags
            entries: list[dict] = []

            def push_all(arr, kind):
                for x in arr or []:
                    text = (x if isinstance(x, str) else x.get("text")) if x else None
                    if not text:
                        continue
                    importance = (
                        float(x.get("importance", 0.6)) if isinstance(x, dict) else 0.6
                    )
                    tags = list(x.get("tags", [])) if isinstance(x, dict) else []
                    entries.append(
                        {
                            "text": text,
                            "kind": kind,
                            "importance": importance,
                            "tags": tags,
                        }
                    )

            push_all(summary.get("facts_about_user"), "FactsAboutUser")
            push_all(summary.get("past_events"), "PastEvents")
            push_all(summary.get("self_beliefs"), "SelfBeliefs")
            push_all(summary.get("objectives"), "Objectives")
            # emotions
            for e in summary.get("emotions") or []:
                name = (e.get("name") if isinstance(e, dict) else None) or "emotion"
                score = float(e.get("score", 0.5)) if isinstance(e, dict) else 0.5
                entries.append(
                    {
                        "text": f"[emotion] {name}={score}",
                        "kind": "Emotions",
                        "importance": min(1.0, 0.4 + score / 2),
                    }
                )
            # key_facts
            push_all(summary.get("key_facts"), "KeyFacts")
            if entries:
                added = mem.add_facts_with_meta(
                    entries,
                    self.character_config.conf_uid,
                    self.history_uid,
                    default_kind="chat",
                )
                if added:
                    logger.info(
                        f"Memory consolidation ({reason}): saved {added} entries"
                    )
                    # Mark history as consolidated
                    try:
                        from .chat_history_manager import update_metadate
                        from datetime import datetime

                        update_metadate(
                            self.character_config.conf_uid,
                            self.history_uid,
                            {
                                "memory_consolidated": True,
                                "memory_consolidated_ts": datetime.now().isoformat(
                                    timespec="seconds"
                                ),
                            },
                        )
                    except Exception as e:
                        logger.debug(f"Failed to mark history as consolidated: {e}")
                else:
                    logger.warning(
                        "Memory consolidation produced entries but nothing was written to Chroma (added=0)"
                    )
        except Exception as e:
            logger.debug(f"trigger_memory_consolidation failed: {e}")

    async def consolidate_history(
        self, history_uid: str, limit_messages: int | None = None
    ) -> int:
        """Консолидация выбранной истории: читает историю целиком (или ограничение),
        зовёт summarize_texts() и раскладывает данные по типам, эмоции в раздел 'Mood'.

        Возвращает число сохранённых элементов в память.
        """
        try:
            mem = self.vtuber_memory_service or self.memory_service
            if not (self.agent_engine and mem and getattr(mem, "enabled", False)):
                return 0
            if not history_uid:
                return 0
            from .chat_history_manager import get_history

            msgs = get_history(self.character_config.conf_uid, history_uid) or []
            if not msgs:
                return 0
            texts: list[str] = []
            it = msgs if not limit_messages else msgs[-int(max(1, limit_messages)) :]
            for m in it:
                c = (m.get("content") or "").strip()
                if c:
                    texts.append(c)
            if not texts:
                return 0
            try:
                summary = await self.agent_engine.summarize_texts(texts)  # type: ignore[attr-defined]
            except Exception:
                summary = {}
            if not summary:
                return 0
            entries: list[dict] = []

            def push_all(arr, kind):
                for x in arr or []:
                    text = (x if isinstance(x, str) else x.get("text")) if x else None
                    if not text:
                        continue
                    importance = (
                        float(x.get("importance", 0.6)) if isinstance(x, dict) else 0.6
                    )
                    tags = list(x.get("tags", [])) if isinstance(x, dict) else []
                    entries.append(
                        {
                            "text": text,
                            "kind": kind,
                            "importance": importance,
                            "tags": tags,
                        }
                    )

            push_all(summary.get("facts_about_user"), "FactsAboutUser")
            push_all(summary.get("past_events"), "PastEvents")
            push_all(summary.get("self_beliefs"), "SelfBeliefs")
            push_all(summary.get("objectives"), "Objectives")
            # emotions -> Mood
            for e in summary.get("emotions") or []:
                try:
                    name = (e.get("name") if isinstance(e, dict) else None) or "emotion"
                    score = float(e.get("score", 0.0)) if isinstance(e, dict) else 0.0
                except Exception:
                    name, score = "emotion", 0.0
                entries.append(
                    {
                        "text": f"[mood] {name}={score}",
                        "kind": "Mood",
                        "importance": min(1.0, 0.4 + abs(score) / 2),
                        "tags": ["mood"],
                    }
                )
            push_all(summary.get("key_facts"), "KeyFacts")

            if not entries:
                return 0
            added = mem.add_facts_with_meta(
                entries,
                self.character_config.conf_uid,
                history_uid,
                default_kind="chat",
            )
            if added:
                logger.info(
                    f"Consolidated history {history_uid}: saved {added} entries"
                )
                # Mark selected history as consolidated
                try:
                    from .chat_history_manager import update_metadate
                    from datetime import datetime

                    update_metadate(
                        self.character_config.conf_uid,
                        history_uid,
                        {
                            "memory_consolidated": True,
                            "memory_consolidated_ts": datetime.now().isoformat(
                                timespec="seconds"
                            ),
                        },
                    )
                except Exception as e:
                    logger.debug(f"Failed to mark history as consolidated: {e}")
            return int(added or 0)
        except Exception as e:
            logger.debug(f"consolidate_history failed: {e}")
            return 0

    async def load_cache(
        self,
        config: Config,
        system_config: SystemConfig,
        character_config: CharacterConfig,
        live2d_model: Live2dModel,
        asr_engine: ASRInterface,
        tts_engine: TTSInterface,
        vad_engine: VADInterface,
        agent_engine: AgentInterface,
        translate_engine: TranslateInterface | None,
        mcp_server_registery: ServerRegistry | None = None,
        tool_adapter: ToolAdapter | None = None,
        send_text: Callable = None,
        client_uid: str = None,
    ) -> None:
        """
        Load the ServiceContext with the reference of the provided instances.
        Pass by reference so no reinitialization will be done.
        """
        if not character_config:
            raise ValueError("character_config cannot be None")
        if not system_config:
            raise ValueError("system_config cannot be None")

        self.config = config
        self.system_config = system_config
        self.character_config = character_config
        self.live2d_model = live2d_model
        self.asr_engine = asr_engine
        self.tts_engine = tts_engine
        self.vad_engine = vad_engine
        self.agent_engine = agent_engine
        self.translate_engine = translate_engine
        # Load potentially shared components by reference
        self.mcp_server_registery = mcp_server_registery
        self.tool_adapter = tool_adapter
        self.send_text = send_text
        self.client_uid = client_uid

        # Initialize session-specific MCP components
        await self._init_mcp_components(
            self.character_config.agent_config.agent_settings.basic_memory_agent.use_mcpp,
            self.character_config.agent_config.agent_settings.basic_memory_agent.mcp_enabled_servers,
        )

        # Ensure queue worker is running in this loop
        await self._ensure_worker()

        logger.debug(
            t(
                "service.loaded_service_context_with_cache",
                character_config=character_config,
            )
        )

    async def load_from_config(self, config: Config) -> None:
        """
        Load the ServiceContext with the config.
        Reinitialize the instances if the config is different.

        Parameters:
        - config (Dict): The configuration dictionary.
        """
        if not self.config:
            self.config = config

        if not self.system_config:
            self.system_config = config.system_config

        # Merge character alt config if specified
        merged_character_config = config.character_config
        try:
            alt_name = config.system_config.config_alt
            alts_dir = config.system_config.config_alts_dir
            if alt_name:
                # Build candidate paths (support .yaml/.yml or raw name)
                from os.path import join, exists

                candidates = [
                    join(alts_dir, alt_name),
                    join(alts_dir, f"{alt_name}.yaml"),
                    join(alts_dir, f"{alt_name}.yml"),
                ]
                alt_path = next((p for p in candidates if exists(p)), None)
                if alt_path:
                    alt_cfg = read_yaml(alt_path).get("character_config")
                    if alt_cfg:
                        # Map legacy keys
                        if "persona" in alt_cfg and "persona_prompt" not in alt_cfg:
                            alt_cfg["persona_prompt"] = alt_cfg.pop("persona")
                        # Build template from current CharacterConfig to prune unknown keys (deep)
                        template = config.character_config.model_dump()
                        pruned_alt = _prune_unknown_keys(template, alt_cfg)
                        merged_dict = deep_merge(
                            config.character_config.model_dump(), pruned_alt
                        )
                        from .config_manager import CharacterConfig as _CharacterConfig

                        merged_character_config = _CharacterConfig(**merged_dict)
                        logger.info(
                            t(
                                "service.loaded_alt_character_config",
                                config_file=alt_path,
                            )
                        )
                else:
                    logger.warning(
                        t("service.alt_config_not_found", config_file=alt_name)
                    )
        except Exception as e:
            logger.warning(f"Failed to merge alt character config: {e}")

        if not self.character_config:
            self.character_config = merged_character_config

        # update all sub-configs

        # init live2d from character config
        self.init_live2d(self.character_config.live2d_model_name)

        # init asr from character config
        self.init_asr(self.character_config.asr_config)

        # init tts from character config
        self.init_tts(self.character_config.tts_config)

        # init vad from character config
        self.init_vad(self.character_config.vad_config)

        # Initialize shared ToolAdapter if it doesn't exist yet
        if (
            not self.tool_adapter
            and self.character_config.agent_config.agent_settings.basic_memory_agent.use_mcpp
        ):
            if not self.mcp_server_registery:
                logger.info(
                    t(
                        "service.initializing_shared_server_registry_within_load_from_config"
                    )
                )
                self.mcp_server_registery = ServerRegistry()
            logger.info(
                t("service.initializing_shared_tool_adapter_within_load_from_config")
            )
            self.tool_adapter = ToolAdapter(server_registery=self.mcp_server_registery)

        # Initialize MCP Components before initializing Agent
        await self._init_mcp_components(
            self.character_config.agent_config.agent_settings.basic_memory_agent.use_mcpp,
            self.character_config.agent_config.agent_settings.basic_memory_agent.mcp_enabled_servers,
        )

        # init agent from character config
        await self.init_agent(
            self.character_config.agent_config,
            self.character_config.persona_prompt,
        )

        self.init_translate(
            self.character_config.tts_preprocessor_config.translator_config
        )

        # init twitch from character config
        await self.init_twitch(self.character_config.twitch_config)

        # Ensure queue worker is running in this loop
        await self._ensure_worker()

        # store typed config references
        self.config = config
        self.system_config = config.system_config or self.system_config
        self.character_config = self.character_config

    def init_live2d(self, live2d_model_name: str) -> None:
        logger.info(t("service.initializing_live2d", model=live2d_model_name))
        try:
            self.live2d_model = Live2dModel(live2d_model_name)
            self.character_config.live2d_model_name = live2d_model_name
            logger.info(
                t("service.live2d_initialized", info=self.live2d_model.model_info)
            )
        except Exception as e:
            logger.critical(t("service.live2d_error", error=str(e)))
            logger.critical(t("service.live2d_proceed_without"))
            self.live2d_model = None

    def init_asr(self, asr_config: ASRConfig) -> None:
        if not self.asr_engine or (self.character_config.asr_config != asr_config):
            logger.info(t("service.initializing_asr", model=asr_config.asr_model))
            self.asr_engine = ASRFactory.get_asr_system(
                asr_config.asr_model,
                **getattr(asr_config, asr_config.asr_model).model_dump(),
            )
            # saving config should be done after successful initialization
            self.character_config.asr_config = asr_config
        else:
            logger.info(t("service.asr_already_initialized"))

    def init_tts(self, tts_config: TTSConfig) -> None:
        if not self.tts_engine or (self.character_config.tts_config != tts_config):
            logger.info(t("service.initializing_tts", model=tts_config.tts_model))
            self.tts_engine = TTSFactory.get_tts_engine(
                tts_config.tts_model,
                **getattr(tts_config, tts_config.tts_model.lower()).model_dump(),
            )
            # saving config should be done after successful initialization
            self.character_config.tts_config = tts_config
        else:
            logger.info(t("service.tts_already_initialized"))

    def init_vad(self, vad_config: VADConfig) -> None:
        if vad_config.vad_model is None:
            logger.info(t("service.vad_disabled"))
            self.vad_engine = None
            return

        if not self.vad_engine or (self.character_config.vad_config != vad_config):
            logger.info(t("service.initializing_vad", model=vad_config.vad_model))
            self.vad_engine = VADFactory.get_vad_engine(
                vad_config.vad_model,
                **getattr(vad_config, vad_config.vad_model.lower()).model_dump(),
            )
            # saving config should be done after successful initialization
            self.character_config.vad_config = vad_config
        else:
            logger.info(t("service.vad_already_initialized"))

    async def init_agent(self, agent_config: AgentConfig, persona_prompt: str) -> None:
        """Initialize or update the LLM engine based on agent configuration."""
        logger.info(
            t(
                "service.initializing_agent",
                agent=agent_config.conversation_agent_choice,
            )
        )

        # Initialize memory service once
        if self.memory_service is None:
            try:
                self.memory_service = MemoryService(enabled=True)
            except Exception as e:
                logger.warning(f"MemoryService init failed: {e}")

        # Initialize vtuber memory wrapper according to config (non-breaking)
        if self.vtuber_memory_service is None:
            try:
                use_vm = True
                if hasattr(self, "character_config") and getattr(
                    self.character_config, "vtuber_memory", None
                ):
                    use_vm = bool(
                        getattr(self.character_config.vtuber_memory, "enabled", True)
                    )
                self.vtuber_memory_service = VtuberMemoryService(
                    enabled=use_vm,
                    system_config=self.system_config,
                    character_config=self.character_config,
                )
            except Exception as e:
                logger.warning(f"VtuberMemoryService init failed: {e}")
                self.vtuber_memory_service = VtuberMemoryService(enabled=False)

        # Start consolidation scheduler using system config interval
        try:
            interval = int(
                getattr(self.system_config, "memory_consolidation_interval_sec", 900)
                or 900
            )
            if interval >= 60:
                if self._consolidation_scheduler is None:
                    self._consolidation_scheduler = ConsolidationScheduler(
                        interval_sec=interval, trigger=self.trigger_memory_consolidation
                    )
                await self._consolidation_scheduler.start()
        except Exception:
            pass

        if (
            self.agent_engine is not None
            and agent_config == self.character_config.agent_config
            and persona_prompt == self.character_config.persona_prompt
        ):
            logger.debug(t("service.agent_already_initialized"))
            return

        system_prompt = await self.construct_system_prompt(persona_prompt)

        # Pass avatar to agent factory
        avatar = self.character_config.avatar or ""  # Get avatar from config

        try:
            self.agent_engine = AgentFactory.create_agent(
                conversation_agent_choice=agent_config.conversation_agent_choice,
                agent_settings=agent_config.agent_settings.model_dump(),
                llm_configs=agent_config.llm_configs.model_dump(),
                system_prompt=system_prompt,
                live2d_model=self.live2d_model,
                tts_preprocessor_config=self.character_config.tts_preprocessor_config,
                character_avatar=avatar,
                system_config=self.system_config.model_dump(),
                tool_manager=self.tool_manager,
                tool_executor=self.tool_executor,
                mcp_prompt_string=self.mcp_prompt,
            )

            logger.debug(f"Agent choice: {agent_config.conversation_agent_choice}")
            logger.debug(f"System prompt: {system_prompt}")

            # Save the current configuration
            self.character_config.agent_config = agent_config
            self.system_prompt = system_prompt

        except Exception as e:
            logger.error(t("service.agent_init_failed", error=str(e)))
            raise

    def init_translate(self, translator_config: TranslatorConfig) -> None:
        """Initialize or update the translation engine based on the configuration."""

        if not translator_config.translate_audio:
            logger.debug("Translation is disabled.")
            return

        if (
            not self.translate_engine
            or self.character_config.tts_preprocessor_config.translator_config
            != translator_config
        ):
            logger.info(
                t(
                    "service.initializing_translator",
                    provider=translator_config.translate_provider,
                )
            )
            self.translate_engine = TranslateFactory.get_translator(
                translator_config.translate_provider,
                getattr(
                    translator_config, translator_config.translate_provider
                ).model_dump(),
            )
            self.character_config.tts_preprocessor_config.translator_config = (
                translator_config
            )
        else:
            logger.info(t("service.translation_already_initialized"))

    async def init_twitch(self, twitch_config: Dict[str, Any]) -> None:
        """Initialize or update the Twitch client based on the configuration."""

        # Normalize config to dict if a Pydantic model (TwitchConfig) was passed
        try:
            if hasattr(twitch_config, "model_dump"):
                twitch_cfg: Dict[str, Any] = twitch_config.model_dump()
            elif isinstance(twitch_config, dict):
                twitch_cfg = twitch_config
            else:
                # Fallback: best-effort attribute extraction
                twitch_cfg = {
                    "enabled": getattr(twitch_config, "enabled", False),
                    "channel_name": getattr(twitch_config, "channel_name", ""),
                    "app_id": getattr(twitch_config, "app_id", ""),
                    "app_secret": getattr(twitch_config, "app_secret", ""),
                    "max_message_length": getattr(
                        twitch_config, "max_message_length", 300
                    ),
                    "max_recent_messages": getattr(
                        twitch_config, "max_recent_messages", 10
                    ),
                }
        except Exception:
            twitch_cfg = {}

        if not twitch_cfg.get("enabled", False):
            logger.info(t("twitch.disabled"))
            # Notify frontend that twitch is disabled
            if self.send_text:
                payload = {
                    "type": "twitch-status",
                    "enabled": False,
                    "connected": False,
                    "channel": twitch_cfg.get("channel_name", ""),
                }
                await self.send_text(json.dumps(payload))
            return

        try:
            logger.info(t("service.initializing_twitch"))

            # Reuse global Twitch client if already initialized to prevent duplicates
            if ServiceContext._global_twitch_client is not None:
                self.twitch_client = ServiceContext._global_twitch_client
                logger.info(
                    "Using shared Twitch client (already connected or connecting)"
                )
                # Send current status to this UI
                if self.twitch_client and self.send_text:
                    await self._broadcast_twitch_status(
                        self.twitch_client.get_connection_status()
                    )
                return

            # Create new Twitch client as global singleton
            self.twitch_client = TwitchClient(twitch_cfg)

            # Initialize and connect
            if await self.twitch_client.initialize():
                if await self.twitch_client.connect():
                    # Set message callback
                    self.twitch_client.set_message_callback(self._handle_twitch_message)
                    # Set status callback to broadcast status to frontend
                    self.twitch_client.set_status_callback(self._handle_twitch_status)
                    # Send initial status
                    await self._broadcast_twitch_status(
                        self.twitch_client.get_connection_status()
                    )
                    # Mark ownership
                    ServiceContext._global_twitch_client = self.twitch_client
                    ServiceContext._global_twitch_owner = self
                    logger.info(t("service.twitch_initialized"))
                else:
                    logger.error(t("service.twitch_connection_failed"))
            else:
                logger.error(t("service.twitch_init_failed"))

        except Exception as e:
            logger.error(t("service.twitch_error", error=str(e)))
            self.twitch_client = None

    async def _broadcast_twitch_status(self, status: Dict[str, Any]) -> None:
        """Broadcast twitch connection status to frontend."""
        try:
            if self.send_text:
                payload = {"type": "twitch-status", **status}
                await self.send_text(json.dumps(payload))
        except Exception:
            # Silently ignore if websocket is closed
            pass

    def _handle_twitch_status(self, status: Dict[str, Any]) -> None:
        """Handle status callback from TwitchClient and forward to websocket asynchronously."""
        if self.send_text:
            try:
                asyncio.create_task(self._broadcast_twitch_status(status))
            except Exception:
                pass

    async def _analyze_and_store_mood(self, user: str, text: str) -> None:
        """Analyze sentiment for a user's message, update aggregated mood, and persist to memory."""
        try:
            if not (
                self.agent_engine and hasattr(self.agent_engine, "analyze_sentiment")
            ):
                return
            mem = self.vtuber_memory_service or self.memory_service
            if not (
                self.memory_service and getattr(self.memory_service, "enabled", False)
            ):
                pass  # we still can track in RAM
            # Call agent's analyzer
            result = await self.agent_engine.analyze_sentiment(text)  # type: ignore[attr-defined]
            if not isinstance(result, dict):
                return
            score = float(result.get("score", 0.0))
            label = str(result.get("label", "нейтрально"))
            # Smooth mood per user
            prev = float(self.user_mood.get(user, 0.0))
            new_score = 0.7 * prev + 0.3 * max(-1.0, min(1.0, score))
            self.user_mood[user] = new_score
            # Broadcast mood update to frontend
            try:
                if self.send_text:
                    payload = {
                        "type": "mood-updated",
                        "user": user,
                        "score": new_score,
                        "label": label,
                    }
                    await self.send_text(json.dumps(payload))
            except Exception:
                pass
            # Persist as emotional memory
            mem = self.vtuber_memory_service or self.memory_service
            if mem and getattr(mem, "enabled", False):
                entry = {
                    "text": f"[emotion:user:{user}] {label}={new_score:.3f}",
                    "kind": "Emotions",
                    "importance": min(1.0, 0.5 + abs(new_score) / 2),
                    "tags": ["twitch", f"user:{user}"],
                }
                try:
                    mem.add_facts_with_meta(
                        [entry],
                        self.character_config.conf_uid,
                        self.history_uid,
                        default_kind="emotions",
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"mood analysis skipped: {e}")

    def _handle_twitch_message(self, message: TwitchMessage) -> None:
        """
        Handle incoming Twitch message.

        Args:
            message: TwitchMessage object
        """
        try:
            # // DEBUG: [FIXED] Assign request_id for Twitch flow | Ref: 5
            rid = str(uuid4())
            set_request_id(rid)
            # Standardized inbound log for Twitch
            chan = f"#{message.channel}" if message.channel else ""
            logger.info(
                f"[twitch:{message.user}]{' ' + chan if chan else ''} {message.message[:200]}"
            )

            # Broadcast Twitch message to frontend for visual separation
            if self.send_text:
                try:
                    payload = {
                        "type": "twitch-message",
                        "user": message.user,
                        "text": message.message,
                        "is_subscriber": message.is_subscriber,
                        "is_moderator": message.is_moderator,
                        "is_broadcaster": message.is_broadcaster,
                        "channel": message.channel,
                        "timestamp": message.timestamp.isoformat(),
                        "request_id": rid,
                    }

                    async def _safe_send(data: dict):
                        try:
                            await self.send_text(json.dumps(data))
                        except Exception:
                            pass

                    asyncio.create_task(_safe_send(payload))
                except Exception:
                    pass

            logger.info(
                t(
                    "twitch.message_processed",
                    user=message.user,
                    message=message.message,
                )
            )
            # Concise processed log
            logger.debug(f"[twitch:{message.user}] processed")

            # Kick off mood analysis and store emotional memory (non-blocking)
            asyncio.create_task(
                self._analyze_and_store_mood(user=message.user, text=message.message)
            )

            # Enqueue to agent processing with priority
            if self.agent_engine and self.send_text:
                from .agent.input_types import TextSource

                twitch_flags = {
                    "is_subscriber": message.is_subscriber,
                    "is_moderator": message.is_moderator,
                    "is_broadcaster": message.is_broadcaster,
                    "bits": getattr(message, "bits", 0) or 0,
                }
                asyncio.create_task(
                    self.enqueue_message(
                        content=message.message,
                        from_name=message.user,
                        source="twitch",
                        text_source_enum=TextSource.TWITCH,
                        images=None,
                        twitch_flags=twitch_flags,
                        metadata={"channel": message.channel, "request_id": rid},
                    )
                )

        except Exception as e:
            logger.error(t("twitch.message_processing_error", error=str(e)))

    async def _process_twitch_message(self, message: TwitchMessage) -> None:
        """
        Process Twitch message through the agent.

        Args:
            message: TwitchMessage
        """
        try:
            # Lazy import to avoid circular dependencies
            from .conversations.single_conversation import process_single_conversation
            from .agent.input_types import TextSource

            metadata = {
                "from_name": message.user,
                "text_source": TextSource.TWITCH,
                # Future flags: e.g., 'skip_memory': False
            }
            await process_single_conversation(
                context=self,
                websocket_send=self.send_text,
                client_uid=self.client_uid,
                user_input=message.message,
                images=None,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(t("twitch.agent_processing_error", error=str(e)))

    # ==== utils

    async def construct_system_prompt(self, persona_prompt: str) -> str:
        """
        Append tool prompts to persona prompt.

        Parameters:
        - persona_prompt (str): The persona prompt.

        Returns:
        - str: The system prompt with all tool prompts appended.
        """
        # If config specifies a persona prompt file name, load it
        try:
            persona_name = str(
                getattr(self.character_config, "persona_prompt_name", "") or ""
            )
        except Exception:
            persona_name = ""

        if persona_name:
            try:
                file_persona = prompt_loader.load_persona(persona_name)
                if file_persona:
                    persona_prompt = file_persona
            except Exception as e:
                logger.warning(f"Failed to load persona prompt '{persona_name}': {e}")

        logger.debug(f"constructing persona_prompt: '''{persona_prompt}'''")

        for prompt_name, prompt_file in self.system_config.tool_prompts.items():
            if (
                prompt_name == "group_conversation_prompt"
                or prompt_name == "proactive_speak_prompt"
            ):
                continue

            prompt_content = prompt_loader.load_util(prompt_file)

            if prompt_name == "live2d_expression_prompt":
                prompt_content = prompt_content.replace(
                    "[<insert_emomap_keys>]", self.live2d_model.emo_str
                )

            if prompt_name == "mcp_prompt":
                continue

            persona_prompt += prompt_content

        # Add grounding/verification guardrails for tool usage to avoid personal attributions
        persona_prompt += (
            "\n\nFACTUALITY AND TOOL-USAGE RULES\n"
            "- When answering questions that rely on external information (e.g., web search), base factual statements strictly on tool outputs you received in this session.\n"
            "- Cite at least the domain names of sources you used (e.g., [youtube.com], [fandom.com]) when presenting facts.\n"
            "- If the available tool outputs do not confirm a detail, say you don't know rather than inferring.\n"
            "- Do not invent or assume personal details about the user, the streamer, or creators. Never attribute content to the current user unless the tool output explicitly states it.\n"
            "- Prefer concise, neutral summaries grounded in the provided tool content.\n"
            "- Never fabricate events about named people (e.g., Ирина, Lizard). If you have no memory confirming an event, do not claim it happened.\n"
            "- Avoid repeating yourself. If you have already said a sentence this turn, express a new thought or end the message.\n"
            "- Keep replies short (<= 3 sentences).\n"
        )

        logger.debug("\n === System Prompt ===")
        logger.debug(persona_prompt)

        return persona_prompt

    async def handle_config_switch(
        self,
        websocket: WebSocket,
        config_file_name: str,
    ) -> None:
        """
        Handle the configuration switch request.
        Change the configuration to a new config and notify the client.

        Parameters:
        - websocket (WebSocket): The WebSocket connection.
        - config_file_name (str): The name of the configuration file.
        """
        try:
            new_character_config_data = None

            if config_file_name == "conf.yaml":
                # Load base config
                new_character_config_data = read_yaml("conf.yaml").get(
                    "character_config"
                )
            else:
                # Load alternative config and merge with base config
                characters_dir = self.system_config.config_alts_dir
                file_path = os.path.normpath(
                    os.path.join(characters_dir, config_file_name)
                )
                if not file_path.startswith(characters_dir):
                    raise ValueError("Invalid configuration file path")

                alt_config_data = read_yaml(file_path).get("character_config")

                # Start with original config data and perform a deep merge
                new_character_config_data = deep_merge(
                    self.config.character_config.model_dump(), alt_config_data
                )

            if new_character_config_data:
                # Prune unknown keys in new character config using current template
                try:
                    template = self.character_config.model_dump()
                except Exception:
                    template = (
                        self.config.character_config.model_dump() if self.config else {}
                    )
                pruned_character = _prune_unknown_keys(
                    template, new_character_config_data
                )

                new_config = {
                    "system_config": self.system_config.model_dump(),
                    "character_config": pruned_character,
                }
                new_config = validate_config(new_config)
                await self.load_from_config(new_config)  # Await the async load
                logger.debug(f"New config: {self}")
                logger.debug(
                    f"New character config: {self.character_config.model_dump()}"
                )

                # Send responses to client
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "set-model-and-conf",
                            "model_info": self.live2d_model.model_info,
                            "conf_name": self.character_config.conf_name,
                            "conf_uid": self.character_config.conf_uid,
                        }
                    )
                )

                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "config-switched",
                            "message": f"Switched to config: {config_file_name}",
                        }
                    )
                )

                logger.info(
                    t(
                        "service.configuration_switched",
                        config_file_name=config_file_name,
                    )
                )
            else:
                raise ValueError(
                    t(
                        "service.failed_to_load_configuration_from",
                        config_file_name=config_file_name,
                    )
                )

        except Exception as e:
            logger.error(t("service.error_switching_configuration", error=str(e)))
            logger.debug(self)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"Error switching configuration: {str(e)}",
                    }
                )
            )
            raise e

    async def on_stream_end(self) -> None:
        """Hook to call when a stream/session ends to consolidate memory immediately."""
        try:
            await self.trigger_memory_consolidation(reason="stream_end")
        except Exception as e:
            logger.debug(f"stream_end consolidation skipped: {e}")


def deep_merge(dict1, dict2):
    """
    Recursively merges dict2 into dict1, prioritizing values from dict2.
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
