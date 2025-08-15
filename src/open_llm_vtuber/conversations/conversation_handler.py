import asyncio
import json
from typing import Dict, Optional, Callable, Any

import numpy as np
from fastapi import WebSocket
from loguru import logger

from ..chat_group import ChatGroupManager
from ..chat_history_manager import store_message
from ..service_context import ServiceContext
from .group_conversation import process_group_conversation
from .single_conversation import process_single_conversation
from .conversation_utils import EMOJI_LIST
from .types import GroupConversationState
from prompts import prompt_loader
from ..agent.input_types import TextSource


async def handle_conversation_trigger(
    msg_type: str,
    data: dict,
    client_uid: str,
    context: ServiceContext,
    websocket: WebSocket,
    client_contexts: Dict[str, ServiceContext],
    client_connections: Dict[str, WebSocket],
    chat_group_manager: ChatGroupManager,
    received_data_buffers: Dict[str, np.ndarray],
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    broadcast_to_group: Callable,
) -> None:
    """Handle triggers that start a conversation"""
    metadata = None
    enqueue_local = False
    enqueue_payload: Optional[Dict[str, Any]] = None

    if msg_type == "ai-speak-signal":
        # Anti-spam: cooldown proactive speaks
        try:
            import time as _time

            now = int(_time.time())
            cooldown_until = int(getattr(context, "_proactive_cooldown_until", 0))
            if now < cooldown_until:
                logger.info(
                    f"Proactive speak throttled (cooldown until {cooldown_until}, now={now})"
                )
                return
            # Set next cooldown (45s)
            context._proactive_cooldown_until = now + 45
        except Exception:
            pass
        try:
            # Get proactive speak prompt from config
            prompt_name = "proactive_speak_prompt"
            prompt_file = context.system_config.tool_prompts.get(prompt_name)
            if prompt_file:
                user_input = prompt_loader.load_util(prompt_file)
            else:
                logger.warning("Proactive speak prompt not configured, using default")
                user_input = "Please say something."
        except Exception as e:
            logger.error(f"Error loading proactive speak prompt: {e}")
            user_input = "Please say something."

        # Add metadata to indicate this is a proactive speak request
        # that should be skipped in both memory and history
        metadata = {
            "proactive_speak": True,
            "skip_memory": True,  # Skip storing in AI's internal memory
            "skip_history": True,  # Skip storing in local conversation history
        }

        # Standardized inbound log
        logger.info(f"[server:AI] {user_input[:200]}")

        await websocket.send_text(
            json.dumps(
                {
                    "type": "full-text",
                    "text": "AI wants to speak something...",
                }
            )
        )
        # Enqueue as system message to avoid interrupting current speech
        enqueue_local = True
        enqueue_payload = {
            "content": user_input,
            "from_name": context.character_config.human_name,
            "source": "system",
            "text_source_enum": TextSource.INPUT,
            "images": data.get("images"),
            "metadata": metadata,
        }
    elif msg_type == "text-input":
        user_input = data.get("text", "")
        # Standardized inbound log
        logger.info(
            f"[local:{context.character_config.human_name}] {str(user_input)[:200]}"
        )
        # Enqueue local UI message for queued processing
        enqueue_local = True
        enqueue_payload = {
            "content": user_input,
            "from_name": context.character_config.human_name,
            "source": "local-ui",
            "text_source_enum": TextSource.INPUT,
            "images": data.get("images"),
            "metadata": metadata,
        }
    else:  # mic-audio-end
        user_input = received_data_buffers[client_uid]
        received_data_buffers[client_uid] = np.array([])
        # Standardized inbound log
        try:
            audio_len = (
                int(user_input.shape[0])
                if hasattr(user_input, "shape")
                else len(user_input)
            )
        except Exception:
            audio_len = 0
        logger.info(
            f"[local:{context.character_config.human_name}] audio_samples={audio_len}"
        )
        # Guard: if no audio captured (e.g., rapid mic toggle), skip starting ASR
        if audio_len == 0:
            logger.info(
                "Skipping conversation start: no audio samples captured (mic toggled without voice)"
            )
            return
        # Min/max duration guard to avoid micro-segments or overlong buffers
        try:
            sr = int(
                getattr(getattr(context, "asr_engine", None), "SAMPLE_RATE", 16000)
            )
        except Exception:
            sr = 16000
        # Read from config if available (milliseconds -> samples)
        try:
            vad_cfg = getattr(
                getattr(context.character_config, "vad_config", None),
                "silero_vad",
                None,
            )
            min_ms = int(getattr(vad_cfg, "min_speech_ms", 250) or 250)
            max_ms = int(getattr(vad_cfg, "max_speech_ms", 12000) or 12000)
        except Exception:
            min_ms, max_ms = 250, 12000
        min_samples = int((min_ms / 1000.0) * sr)
        max_samples = int((max_ms / 1000.0) * sr)
        if audio_len < min_samples:
            logger.info(
                f"Skipping ASR: too short segment ({audio_len} < {min_samples} samples)"
            )
            return
        if audio_len > max_samples:
            # Trim hard to max duration to keep latency predictable
            try:
                user_input = user_input[:max_samples]
                logger.info(
                    f"Trimming overlong audio segment to {max_samples} samples (from {audio_len})"
                )
            except Exception:
                pass

    images = data.get("images")
    session_emoji = np.random.choice(EMOJI_LIST)

    # If using queue for local/system text, do not start a direct task
    if enqueue_local and enqueue_payload:
        await context.enqueue_message(**enqueue_payload)  # type: ignore[arg-type]
        return

    group = chat_group_manager.get_client_group(client_uid)
    if group and len(group.members) > 1:
        # Use group_id as task key for group conversations
        task_key = group.group_id
        if (
            task_key not in current_conversation_tasks
            or current_conversation_tasks[task_key].done()
        ):
            logger.info(f"Starting new group conversation for {task_key}")

            current_conversation_tasks[task_key] = asyncio.create_task(
                process_group_conversation(
                    client_contexts=client_contexts,
                    client_connections=client_connections,
                    broadcast_func=broadcast_to_group,
                    group_members=group.members,
                    initiator_client_uid=client_uid,
                    user_input=user_input,
                    images=images,
                    session_emoji=session_emoji,
                    metadata=metadata,
                )
            )
    else:
        # Use client_uid as task key for individual conversations
        current_conversation_tasks[client_uid] = asyncio.create_task(
            process_single_conversation(
                context=context,
                websocket_send=websocket.send_text,
                client_uid=client_uid,
                user_input=user_input,
                images=images,
                session_emoji=session_emoji,
                metadata=metadata,
            )
        )


async def handle_individual_interrupt(
    client_uid: str,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    context: ServiceContext,
    heard_response: str,
):
    if client_uid in current_conversation_tasks:
        task = current_conversation_tasks[client_uid]
        if task and not task.done():
            task.cancel()
            logger.info("🛑 Conversation task was successfully interrupted")

        try:
            context.agent_engine.handle_interrupt(heard_response)
        except Exception as e:
            logger.error(f"Error handling interrupt: {e}")

        if context.history_uid:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="ai",
                content=heard_response,
                name=context.character_config.character_name,
                avatar=context.character_config.avatar,
            )
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="system",
                content="[Interrupted by user]",
            )


async def handle_group_interrupt(
    group_id: str,
    heard_response: str,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    chat_group_manager: ChatGroupManager,
    client_contexts: Dict[str, ServiceContext],
    broadcast_to_group: Callable,
) -> None:
    """Handles interruption for a group conversation"""
    task = current_conversation_tasks.get(group_id)
    if not task or task.done():
        return

    # Get state and speaker info before cancellation
    state = GroupConversationState.get_state(group_id)
    current_speaker_uid = state.current_speaker_uid if state else None

    # Get context from current speaker
    context = None
    group = chat_group_manager.get_group_by_id(group_id)
    if current_speaker_uid:
        context = client_contexts.get(current_speaker_uid)
        logger.info(f"Found current speaker context for {current_speaker_uid}")
    if not context and group and group.members:
        logger.warning(f"No context found for group {group_id}, using first member")
        context = client_contexts.get(next(iter(group.members)))

    # Now cancel the task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        logger.info(f"🛑 Group conversation {group_id} cancelled successfully.")

    current_conversation_tasks.pop(group_id, None)
    GroupConversationState.remove_state(group_id)  # Clean up state after we've used it

    # Store messages with speaker info
    if context and group:
        for member_uid in group.members:
            if member_uid in client_contexts:
                try:
                    member_ctx = client_contexts[member_uid]
                    member_ctx.agent_engine.handle_interrupt(heard_response)
                    store_message(
                        conf_uid=member_ctx.character_config.conf_uid,
                        history_uid=member_ctx.history_uid,
                        role="ai",
                        content=heard_response,
                        name=context.character_config.character_name,
                        avatar=context.character_config.avatar,
                    )
                    store_message(
                        conf_uid=member_ctx.character_config.conf_uid,
                        history_uid=member_ctx.history_uid,
                        role="system",
                        content="[Interrupted by user]",
                    )
                except Exception as e:
                    logger.error(f"Error handling interrupt for {member_uid}: {e}")

    await broadcast_to_group(
        list(group.members),
        {
            "type": "interrupt-signal",
            "text": "conversation-interrupted",
        },
    )
