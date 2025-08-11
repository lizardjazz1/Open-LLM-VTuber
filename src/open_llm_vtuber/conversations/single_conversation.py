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
from ..agent.agents.basic_memory_agent import BasicMemoryAgent
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
        # Send initial signals
        await send_conversation_start_signals(websocket_send)
        logger.info(f"New Conversation Chain {session_emoji} started!")

        # Process user input
        input_text = await process_user_input(
            user_input, context.asr_engine, websocket_send
        )
        # // DEBUG: [FIXED] Structured log for inbound text with sampling | Ref: 6
        logger.bind(component="conversation").info(
            {"event": "user_input", **truncate_and_hash(input_text)}
        )

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
            mem = context.vtuber_memory_service or context.memory_service
            if (
                context.memory_enabled
                and mem
                and getattr(mem, "enabled", False)
                and input_text
            ):
                # Новый поиск релевантных воспоминаний с коррекцией самоссылки
                hits = mem.get_relevant_memories(
                    query=input_text,
                    conf_uid=context.character_config.conf_uid,
                    limit=getattr(context, "memory_top_k", 4),
                )
                if hits:
                    # prepend memory snippets into user prompt as context
                    from ..agent.input_types import TextData

                    # Режим самоссылки: backend|prompt|hybrid (по умолчанию backend)
                    self_ref_mode = getattr(
                        context.character_config, "self_reference_mode", "backend"
                    )

                    def _looks_first_person(txt: str) -> bool:
                        low = (txt or "").lower()
                        first_person = any(
                            p in low
                            for p in [
                                "я ",
                                "мне ",
                                "меня ",
                                "моя ",
                                "мои ",
                                "сам",
                                "сама",
                            ]
                        ) or low.startswith("я")
                        gender = str(
                            getattr(
                                context.character_config, "character_gender", "female"
                            )
                            or "female"
                        ).lower()
                        if gender == "female":
                            gender_words = [
                                "рада",
                                "готова",
                                "согласна",
                                "устала",
                                "сама",
                                "думала",
                                "сказала",
                                "хотела",
                            ]
                        elif gender == "male":
                            gender_words = [
                                "рад",
                                "готов",
                                "согласен",
                                "устал",
                                "сам",
                                "думал",
                                "сказал",
                                "хотел",
                            ]
                        else:
                            gender_words = []  # neutral: не требуем слов по роду
                        gender_ok = (
                            True
                            if not gender_words
                            else any(w in low for w in gender_words)
                        )
                        import re as _re

                        char_name = str(
                            getattr(context.character_config, "character_name", "Нейри")
                            or "Нейри"
                        ).strip()
                        # Поиск имени как отдельного слова (учёт любых символов и пробелов в имени)
                        name_pat = _re.escape(char_name)
                        name_third = (
                            bool(
                                _re.search(
                                    rf"\b{name_pat}\b", low, flags=_re.IGNORECASE
                                )
                            )
                            if char_name
                            else False
                        )
                        return first_person and gender_ok and not name_third

                    corrected: List[str] = []
                    for h in hits:
                        kind = h.get("kind") or MemoryKind.USER
                        text = str(h.get("text") or "")
                        try:
                            apply_adjust = False
                            if self_ref_mode == "backend":
                                apply_adjust = True
                            elif self_ref_mode == "prompt":
                                apply_adjust = False
                            elif self_ref_mode == "hybrid":
                                # Если уже ок в первом лице и соответствует выбранному роду — не трогаем
                                apply_adjust = (
                                    not _looks_first_person(text)
                                    if kind == MemoryKind.SELF
                                    else True
                                )
                            else:
                                apply_adjust = True

                            if apply_adjust:
                                text = mem.adjust_context_for_speaker(
                                    memory_text=text,
                                    memory_kind=kind,
                                    speaker="NEYRI",
                                    current_user_name=effective_from_name,
                                    character_name=str(
                                        getattr(
                                            context.character_config,
                                            "character_name",
                                            "Нейри",
                                        )
                                        or "Нейри"
                                    ),
                                    character_gender=str(
                                        getattr(
                                            context.character_config,
                                            "character_gender",
                                            "female",
                                        )
                                        or "female"
                                    ),
                                )
                        except Exception:
                            pass
                        corrected.append(f"[Memory] {text}")
                    batch_input.texts.insert(
                        0,
                        TextData(
                            source=TextSource.INPUT,
                            content="\n".join(corrected),
                            from_name=None,
                        ),
                    )
        except Exception as e:
            logger.debug(f"Memory search skipped: {e}")

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
                mem = context.vtuber_memory_service or context.memory_service
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

        try:
            # agent.chat yields Union[SentenceOutput, Dict[str, Any]]
            agent_output_stream = context.agent_engine.chat(batch_input)

            async for output_item in agent_output_stream:
                if (
                    isinstance(output_item, dict)
                    and output_item.get("type") == "tool_call_status"
                ):
                    # Handle tool status event: send WebSocket message
                    output_item["name"] = context.character_config.character_name
                    logger.debug(f"Sending tool status update: {output_item}")

                    await websocket_send(json.dumps(output_item))

                elif isinstance(output_item, (SentenceOutput, AudioOutput)):
                    # Handle SentenceOutput or AudioOutput
                    response_part = await process_agent_output(
                        output=output_item,
                        character_config=context.character_config,
                        live2d_model=context.live2d_model,
                        tts_engine=context.tts_engine,
                        websocket_send=websocket_send,  # Pass websocket_send for audio/tts messages
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
            await websocket_send(json.dumps({"type": "backend-synth-complete"}))

        await finalize_conversation_turn(
            tts_manager=tts_manager,
            websocket_send=websocket_send,
            client_uid=client_uid,
        )

        if context.history_uid and full_response:  # Check full_response before storing
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
                if isinstance(context.agent_engine, BasicMemoryAgent):
                    context.agent_engine.update_history_summary(
                        context.character_config.conf_uid, context.history_uid
                    )
            except Exception:
                pass

            # Upsert key facts to long-term memory (very lightweight rule)
            try:
                mem = context.vtuber_memory_service or context.memory_service
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

        return full_response  # Return accumulated full_response

    except asyncio.CancelledError:
        logger.info(f"🤡👍 Conversation {session_emoji} cancelled because interrupted.")
        raise
    except Exception as e:
        logger.error(f"Error in conversation chain: {e}")
        await websocket_send(
            json.dumps({"type": "error", "message": f"Conversation error: {str(e)}"})
        )
        raise
    finally:
        cleanup_conversation(tts_manager, session_emoji)
