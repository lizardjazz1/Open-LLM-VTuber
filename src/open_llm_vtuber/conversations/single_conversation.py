from typing import Union, List, Dict, Any, Optional
import asyncio
import json
from loguru import logger
import numpy as np
import time

from .conversation_utils import (
    create_batch_input,
    process_agent_output,
    send_conversation_start_signals,
    process_user_input,
    finalize_conversation_turn,
    cleanup_conversation,
    EMOJI_LIST,
)
from .types import WebSocketSend
from .tts_manager import TTSTaskManager
from ..chat_history_manager import store_message
from ..service_context import ServiceContext
from ..agent.input_types import TextSource
from ..memory.memory_schema import (
    MemoryItemTyped,
    MemoryKind,
    determine_memory_kind,
    calculate_importance,
    extract_tags,
    detect_emotion,
)

# Import necessary types from agent outputs
from ..agent.output_types import SentenceOutput, AudioOutput

# // DEBUG: [FIXED] Sampling helper for logs | Ref: 6
from ..logging_utils import truncate_and_hash


async def process_single_conversation(
    context: ServiceContext,
    websocket_send: WebSocketSend,
    client_uid: str,
    user_input: Union[str, np.ndarray],
    images: Optional[List[Dict[str, Any]]] = None,
    session_emoji: str = np.random.choice(EMOJI_LIST),
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Process a single-user conversation turn

    Args:
        context: Service context containing all configurations and engines
        websocket_send: WebSocket send function
        client_uid: Client unique identifier
        user_input: Text or audio input from user
        images: Optional list of image data
        session_emoji: Emoji identifier for the conversation
        metadata: Optional metadata for special processing flags

    Returns:
        str: Complete response text
    """
    # Create TTSTaskManager for this conversation
    tts_manager = TTSTaskManager()
    full_response = ""  # Initialize full_response here

    try:
        # Per-conversation counters
        ws_send_count = 0
        outputs_processed = 0
        sentence_outputs = 0
        audio_outputs = 0
        # NEW: message counter for consolidation triggers
        try:
            context._msg_counter = (
                int(getattr(context, "_msg_counter", 0)) + 1
            )  # SAFETY: session-scoped
        except Exception:
            context._msg_counter = 1

        # High-level WS logging wrapper
        async def ws_send_logged(msg: str):
            try:
                data = json.loads(msg)
                p_type = data.get("type", "unknown")
                disp = data.get("display_text")
                text_val = (
                    disp.get("text") if isinstance(disp, dict) else None
                ) or data.get("text")
                has_audio = bool(data.get("audio"))
                seq = data.get("seq") or data.get("sequence")
                logger.debug(
                    f"WS[logged] send type={p_type} has_audio={'yes' if has_audio else 'no'} text_len={len(text_val) if isinstance(text_val, str) else 0} seq={seq}"
                )
            except Exception:
                logger.debug(
                    f"WS[logged] send raw_len={len(msg) if isinstance(msg, str) else 'n/a'}"
                )
            nonlocal ws_send_count
            ws_send_count += 1
            await websocket_send(msg)

        # Send initial signals
        await send_conversation_start_signals(ws_send_logged)
        logger.info(f"New Conversation Chain {session_emoji} started!")

        # Process user input
        input_text = await process_user_input(
            user_input, context.asr_engine, ws_send_logged
        )
        # // DEBUG: [FIXED] Structured log for inbound text with sampling | Ref: 6
        logger.bind(component="conversation").info(
            {"event": "user_input", **truncate_and_hash(input_text)}
        )

        # NEW: periodic consolidation trigger every N messages
        try:
            n = int(
                getattr(context.system_config, "consolidate_every_n_messages", 30) or 30
            )
            if n > 0 and (context._msg_counter % n == 0):
                try:
                    # Use the unified consolidation flow in ServiceContext
                    await context.trigger_memory_consolidation(
                        reason="periodic_n_messages"
                    )
                    logger.debug(
                        f"Triggered periodic consolidation at {context._msg_counter} messages"
                    )
                except Exception as _:
                    pass
        except Exception:
            pass

        # Determine overrides
        from_name_override = (metadata or {}).get("from_name") if metadata else None
        text_source_override = (
            (metadata or {}).get("text_source") if metadata else TextSource.INPUT
        )

        effective_from_name = from_name_override or context.character_config.human_name
        effective_text_source = text_source_override or TextSource.INPUT

        # Create batch input
        batch_input = create_batch_input(
            input_text=input_text,
            images=images,
            from_name=effective_from_name,
            metadata=metadata,
            text_source=effective_text_source,
        )

        # If Twitch message, inject current mood towards this user
        try:
            if effective_text_source == TextSource.TWITCH:
                current_mood = None
                try:
                    current_mood = context.user_mood.get(effective_from_name)  # type: ignore[attr-defined]
                except Exception:
                    current_mood = None
                if isinstance(current_mood, (int, float)):
                    from ..agent.input_types import TextData

                    mood_text = f"[Внутреннее состояние] Текущее отношение к пользователю {effective_from_name}: {current_mood:.2f} (−1…1)."
                    batch_input.texts.insert(
                        0,
                        TextData(
                            source=TextSource.INPUT,
                            content=mood_text,
                            from_name=None,
                        ),
                    )
        except Exception:
            pass

        # Retrieve relevant long-term memory snippets (if available)
        try:
            mem = context.vtuber_memory_service
            if (
                context.memory_enabled
                and mem
                and getattr(mem, "enabled", False)
                and input_text
            ):
                hits = mem.get_relevant_memories(
                    query=input_text,
                    conf_uid=context.character_config.conf_uid,
                    limit=getattr(context, "memory_top_k", 4),
                )
                if hits:
                    # prepend memory snippets into system-only context (not displayed)
                    from ..agent.input_types import TextData

                    def _looks_first_person(txt: str) -> bool:
                        low = (txt or "").lower()
                        first_person = any(
                            p in low
                            for p in [
                                "я ",
                                "мне ",
                                "меня ",
                                "мой ",
                                "моя ",
                                "моё ",
                                "мои ",
                            ]
                        )
                        return first_person

                    corrected: List[str] = []
                    for h in hits:
                        kind = h.get("kind") or MemoryKind.USER
                        text = str(h.get("text") or "")
                        try:
                            self_ref_mode = getattr(
                                context.character_config,
                                "self_reference_mode",
                                "backend",
                            )
                            apply_adjust = False
                            if self_ref_mode == "backend":
                                apply_adjust = True
                            elif self_ref_mode == "prompt":
                                apply_adjust = False
                            else:  # hybrid
                                apply_adjust = _looks_first_person(text)
                            if apply_adjust:
                                adj = mem.adjust_context_for_speaker(
                                    text,
                                    kind=str(kind),
                                    speaker=context.character_config.character_name,
                                    current_user_name=effective_from_name,
                                    character_name=context.character_config.character_name,
                                    character_gender="female",
                                )
                                text = adj or text
                        except Exception:
                            pass
                        corrected.append(text)

                    # Inject relationships tone hint and memory as system guidance only
                    sys_chunks: List[str] = []
                    try:
                        from ..vtuber_memory.relationships import RelationshipsDB

                        db_path = getattr(
                            context.system_config,
                            "relationships_db_path",
                            "cache/relationships.sqlite3",
                        )
                        db = RelationshipsDB(db_path)
                        rel = db.get(effective_from_name)
                        if rel:
                            sys_chunks.append(
                                f"[Relations] affinity={rel.affinity}, trust={rel.trust}, interactions={rel.interaction_count}."
                            )
                    except Exception:
                        pass
                    if corrected:
                        for t in corrected[: context.memory_top_k]:
                            sys_chunks.append(f"[Memory] {t}")
                    if sys_chunks:
                        # Use TextSource.SYSTEM to indicate non-display context
                        batch_input.texts.insert(
                            0,
                            TextData(
                                source=TextSource.SYSTEM,
                                content="\n".join(sys_chunks),
                                from_name=None,
                            ),
                        )
        except Exception:
            pass

        # Simple anti-greeting guard: avoid repeated greetings if same user within short window
        try:
            last_hello = getattr(context, "_last_greet_ts", 0.0)
            last_user = getattr(context, "_last_user_name", None)
            now_ts = time.time()
            if effective_from_name == last_user and (now_ts - float(last_hello)) < 90:
                # Mark in metadata for the agent to avoid greeting again
                batch_input.metadata = batch_input.metadata or {}
                batch_input.metadata["avoid_greeting"] = True
            setattr(context, "_last_greet_ts", now_ts)
            setattr(context, "_last_user_name", effective_from_name)
        except Exception:
            pass

        # Store user message (check if we should skip storing to history)
        skip_history = metadata and metadata.get("skip_history", False)
        if context.history_uid and not skip_history:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="human",
                content=input_text,
                name=effective_from_name,
            )

            # Классификация и сохранение пользовательского ввода в долгосрочную память
            try:
                mem = context.vtuber_memory_service
                if mem and getattr(mem, "enabled", False):
                    kind = determine_memory_kind(
                        text=input_text,
                        speaker="USER",
                        conf=context.character_config,
                    )
                    item = MemoryItemTyped(
                        text=input_text,
                        kind=kind,
                        conf_uid=context.character_config.conf_uid,
                        history_uid=context.history_uid,
                        importance=calculate_importance(input_text),
                        timestamp=float(time.time()),
                        tags=extract_tags(input_text),
                        emotion=detect_emotion(input_text),
                    )
                    mem.add_memory(item)
            except Exception as e:
                logger.debug(f"User memory add skipped: {e}")

        if skip_history:
            logger.debug("Skipping storing user input to history (proactive speak)")

        logger.info(f"User input: {input_text}")
        if images:
            logger.info(f"With {len(images)} images")

        # Ensure history_uid exists before any LLM call or storage when not skipping history
        try:
            if not context.history_uid and not (
                metadata and metadata.get("skip_history", False)
            ):
                from ..chat_history_manager import create_new_history

                context.history_uid = create_new_history(
                    context.character_config.conf_uid
                )
                logger.info(
                    f"Initialized history_uid before processing: {context.history_uid}"
                )
        except Exception as e:
            logger.debug(f"Failed to ensure history_uid: {e}")

        try:
            # Load short-term memory bounded by recent minutes
            from ..chat_history_manager import get_history
            from datetime import datetime, timedelta

            stm_minutes = getattr(context.character_config, "stm_window_minutes", None)
            if not isinstance(stm_minutes, int) or stm_minutes <= 0:
                try:
                    from ..vtuber_memory.config import DEFAULT_STM_WINDOW_MINUTES

                    stm_minutes = DEFAULT_STM_WINDOW_MINUTES
                except Exception:
                    stm_minutes = 20
            msgs = get_history(context.character_config.conf_uid, context.history_uid)
            if msgs:
                cutoff = datetime.utcnow() - timedelta(minutes=int(stm_minutes))
                trimmed: list = []
                for m in msgs:
                    try:
                        ts = datetime.fromisoformat(m.get("timestamp"))
                    except Exception:
                        ts = cutoff  # keep if parsing fails
                    if ts >= cutoff:
                        trimmed.append(m)
                # rehydrate into agent memory interface if available
                if hasattr(context.agent_engine, "_memory") and hasattr(
                    context.agent_engine, "_max_memory_messages"
                ):
                    # Temporarily monkey-patch get_history to return trimmed
                    # Safer: provide a direct setter
                    context.agent_engine._memory = []
                    for msg in trimmed[-context.agent_engine._max_memory_messages :]:
                        role = "user" if msg.get("role") == "human" else "assistant"
                        content = msg.get("content", "")
                        if isinstance(content, str) and content.strip():
                            context.agent_engine._memory.append(
                                {"role": role, "content": content}
                            )

            # agent.chat yields Union[SentenceOutput, Dict[str, Any]]
            agent_output_stream = context.agent_engine.chat(batch_input)
            logger.info("agent.chat stream started")

            async for output_item in agent_output_stream:
                outputs_processed += 1
                if (
                    isinstance(output_item, dict)
                    and output_item.get("type") == "partial-text"
                ):
                    # Forward partial text updates for realtime UI streaming
                    try:
                        await ws_send_logged(json.dumps(output_item))
                    except Exception:
                        logger.debug(f"Failed to send partial-text: {output_item}")
                    continue
                if (
                    isinstance(output_item, dict)
                    and output_item.get("type") == "tool_call_status"
                ):
                    # Handle tool status event: send WebSocket message
                    output_item["name"] = context.character_config.character_name
                    logger.debug(f"Sending tool status update: {output_item}")

                    await ws_send_logged(json.dumps(output_item))

                elif isinstance(output_item, (SentenceOutput, AudioOutput)):
                    # Handle SentenceOutput or AudioOutput
                    try:
                        from ..agent.output_types import (
                            SentenceOutput as _SO,
                            AudioOutput as _AO,
                        )
                    except Exception:
                        _SO = SentenceOutput
                        _AO = AudioOutput
                    if isinstance(output_item, _SO):
                        sentence_outputs += 1
                    elif isinstance(output_item, _AO):
                        audio_outputs += 1
                    response_part = await process_agent_output(
                        output=output_item,
                        character_config=context.character_config,
                        live2d_model=context.live2d_model,
                        tts_engine=context.tts_engine,
                        websocket_send=ws_send_logged,  # Pass wrapper for audio/tts messages
                        tts_manager=tts_manager,
                        translate_engine=context.translate_engine,
                    )
                    # Ensure response_part is treated as a string before concatenation
                    response_part_str = (
                        str(response_part) if response_part is not None else ""
                    )
                    full_response += response_part_str  # Accumulate text response
                else:
                    logger.warning(
                        f"Received unexpected item type from agent chat stream: {type(output_item)}"
                    )
                    logger.debug(f"Unexpected item content: {output_item}")

        except Exception as e:
            logger.exception(
                f"Error processing agent response stream: {e}"
            )  # Log with stack trace
            await websocket_send(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"Error processing agent response: {str(e)}",
                    }
                )
            )
            # full_response will contain partial response before error
        # --- End processing agent response ---

        # Wait for any pending TTS tasks
        if tts_manager.task_list:
            await asyncio.gather(*tts_manager.task_list)
            await ws_send_logged(json.dumps({"type": "backend-synth-complete"}))

        await finalize_conversation_turn(
            tts_manager=tts_manager,
            websocket_send=ws_send_logged,
            client_uid=client_uid,
        )

        if context.history_uid and full_response:  # Check full_response before storing
            # Suppress storing duplicates to history: compare against last few AI messages
            try:
                from .conversation_utils import clean_voice_commands_from_text
                from .conversation_utils import _re_json_guard as _json_guard
                import difflib as _df
                from .chat_history_manager import get_history
                import re

                def _norm(txt: str) -> str:
                    try:
                        s = clean_voice_commands_from_text(txt)
                        s = (
                            _json_guard.sub(r"", s)
                            if hasattr(_json_guard, "sub")
                            else s
                        )
                        s = re.sub(r"[\s.,!?，。！？'»«“”‘’\-]+", " ", s)
                        return s.strip().lower()
                    except Exception:
                        return str(txt or "").strip().lower()

                history = get_history(
                    context.character_config.conf_uid, context.history_uid
                )
                last_ai = [
                    h.get("content", "") for h in history[::-1] if h.get("role") == "ai"
                ][:5]
                cur = _norm(full_response)
                is_dup = False
                for prev in last_ai:
                    p = _norm(prev)
                    if not p:
                        continue
                    if (
                        cur == p
                        or (len(cur) < 64 and (cur in p or p in cur))
                        or _df.SequenceMatcher(a=p, b=cur).ratio() > 0.92
                    ):
                        is_dup = True
                        break
                if is_dup:
                    logger.info("🗃️ Skip storing duplicate AI response in history")
                else:
                    store_message(
                        conf_uid=context.character_config.conf_uid,
                        history_uid=context.history_uid,
                        role="ai",
                        content=full_response,
                        name=context.character_config.character_name,
                        avatar=context.character_config.avatar,
                    )
            except Exception:
                # Fallback to storing if anything goes wrong in de-dup
                store_message(
                    conf_uid=context.character_config.conf_uid,
                    history_uid=context.history_uid,
                    role="ai",
                    content=full_response,
                    name=context.character_config.character_name,
                    avatar=context.character_config.avatar,
                )
            # // DEBUG: [FIXED] Structured log for AI response with sampling | Ref: 6
            logger.bind(component="conversation").info(
                {"event": "ai_response", **truncate_and_hash(full_response)}
            )

            # Update rolling summary for memory priming if agent supports it
            try:
                if hasattr(context.agent_engine, "update_history_summary"):
                    context.agent_engine.update_history_summary(
                        context.character_config.conf_uid, context.history_uid
                    )
            except Exception:
                pass

            # Upsert key facts to long-term memory (very lightweight rule)
            try:
                mem = context.vtuber_memory_service
                if mem and getattr(mem, "enabled", False):
                    # naive fact extraction: split by sentences; take short lines
                    import re

                    def sanitize_for_memory(text: str) -> str:
                        # Remove emotion tags like [joy], [thinking], [confused]
                        text = re.sub(r"\[[^\]]+\]", " ", text)
                        # Remove TTS voice commands {rate:..}{volume:..}{pitch:..}
                        text = re.sub(
                            r"\{\s*(rate|volume|pitch)\s*:[^}]*\}",
                            " ",
                            text,
                            flags=re.IGNORECASE,
                        )
                        # Collapse repeated whitespace
                        text = re.sub(r"\s+", " ", text)
                        return text.strip()

                    sentences = []
                    for raw in full_response.split("."):
                        clean = sanitize_for_memory(raw)
                        if 6 <= len(clean) <= 220:
                            sentences.append(clean)
                    top = sentences[:5]
                    if top:
                        mem.add_facts(
                            top,
                            context.character_config.conf_uid,
                            context.history_uid,
                            kind="ai",
                        )
            except Exception as e:
                logger.debug(f"Memory upsert skipped: {e}")

        logger.info(
            f"conv_summary: ws_sends={ws_send_count} outputs={outputs_processed} sentence_outputs={sentence_outputs} audio_outputs={audio_outputs}"
        )
        return full_response  # Return accumulated full_response

    except asyncio.CancelledError:
        logger.info(f"🤡👍 Conversation {session_emoji} cancelled because interrupted.")
        raise
    except Exception as e:
        logger.error(f"Error in conversation chain: {e}")
        await ws_send_logged(
            json.dumps({"type": "error", "message": f"Conversation error: {str(e)}"})
        )
        raise
    finally:
        cleanup_conversation(tts_manager, session_emoji)
