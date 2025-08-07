import asyncio
import re
from typing import Optional, Union, Any, List, Dict
import numpy as np
import json
from loguru import logger

from ..message_handler import message_handler
from .types import WebSocketSend, BroadcastContext
from .tts_manager import TTSTaskManager
from ..agent.output_types import SentenceOutput, AudioOutput
from ..agent.input_types import BatchInput, TextData, ImageData, TextSource, ImageSource
from ..asr.asr_interface import ASRInterface
from ..live2d_model import Live2dModel
from ..tts.tts_interface import TTSInterface
from ..utils.stream_audio import prepare_audio_payload


def clean_voice_commands_from_text(text: str) -> str:
    """
    Remove voice commands and emotion commands from text for frontend display.
    Commands like {rate:+10%}, {volume:-5%}, {pitch:+15Hz} and [neutral], [joy] will be removed.
    """
    import re
    # Remove voice commands in curly braces
    voice_pattern = r'\{rate:(?:\+|\-)?\d+%\}|\{volume:(?:\+|\-)?\d+%\}|\{pitch:(?:\+|\-)?\d+Hz\}|\{neutral\}'
    clean_text = re.sub(voice_pattern, '', text)
    
    # Remove emotion commands in square brackets
    emotion_pattern = r'\[(?:neutral|joy|smile|laugh|anger|disgust|fear|sadness|surprise|confused|thinking|excited|shy|wink)\]'
    clean_text = re.sub(emotion_pattern, '', clean_text)
    
    # Remove extra spaces
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text


# Convert class methods to standalone functions
def create_batch_input(
    input_text: str,
    images: Optional[List[Dict[str, Any]]],
    from_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> BatchInput:
    """Create batch input for agent processing"""
    return BatchInput(
        texts=[
            TextData(source=TextSource.INPUT, content=input_text, from_name=from_name)
        ],
        images=[
            ImageData(
                source=ImageSource(img["source"]),
                data=img["data"],
                mime_type=img["mime_type"],
            )
            for img in (images or [])
        ]
        if images
        else None,
        metadata=metadata,
    )


async def process_agent_output(
    output: Union[AudioOutput, SentenceOutput],
    character_config: Any,
    live2d_model: Live2dModel,
    tts_engine: TTSInterface,
    websocket_send: WebSocketSend,
    tts_manager: TTSTaskManager,
    translate_engine: Optional[Any] = None,
) -> str:
    """Process agent output with character information and optional translation"""
    output.display_text.name = character_config.character_name
    output.display_text.avatar = character_config.avatar

    full_response = ""
    try:
        if isinstance(output, SentenceOutput):
            full_response = await handle_sentence_output(
                output,
                live2d_model,
                tts_engine,
                websocket_send,
                tts_manager,
                translate_engine,
            )
        elif isinstance(output, AudioOutput):
            full_response = await handle_audio_output(output, websocket_send)
        else:
            logger.warning(f"Unknown output type: {type(output)}")
    except Exception as e:
        logger.error(f"Error processing agent output: {e}")
        await websocket_send(
            json.dumps(
                {"type": "error", "message": f"Error processing response: {str(e)}"}
            )
        )

    return full_response


async def handle_sentence_output(
    output: SentenceOutput,
    live2d_model: Live2dModel,
    tts_engine: TTSInterface,
    websocket_send: WebSocketSend,
    tts_manager: TTSTaskManager,
    translate_engine: Optional[Any] = None,
) -> str:
    """Handle sentence output type with optional translation support"""
    full_response = ""
    async for display_text, tts_text, actions in output:
        logger.debug(f"🏃 Processing output: '''{tts_text}'''...")
        
        # Add detailed logging for debugging voice commands
        if '{rate:' in tts_text or '{volume:' in tts_text or '{pitch:' in tts_text:
            logger.info(f"🎵 Voice commands detected in tts_text: {tts_text}")
            # Show how the text will look on the frontend
            clean_display = clean_voice_commands_from_text(display_text.text)
            logger.info(f"📱 Frontend will see: {clean_display}")
        elif '[neutral]' in tts_text or '[joy]' in tts_text or '[smile]' in tts_text or '[laugh]' in tts_text or '[anger]' in tts_text or '[disgust]' in tts_text or '[fear]' in tts_text or '[sadness]' in tts_text or '[surprise]' in tts_text or '[confused]' in tts_text or '[thinking]' in tts_text or '[excited]' in tts_text or '[shy]' in tts_text or '[wink]' in tts_text:
            logger.info(f"😊 Emotion commands detected in tts_text: {tts_text}")
            # Show how the text will look on the frontend
            clean_display = clean_voice_commands_from_text(display_text.text)
            logger.info(f"📱 Frontend will see: {clean_display}")
        else:
            logger.debug(f"🔍 No voice or emotion commands in tts_text: {tts_text}")

        if translate_engine:
            if len(re.sub(r'[\s.,!?，。！？\'"』」）】\s]+', "", tts_text)):
                tts_text = translate_engine.translate(tts_text)
            logger.info(f"🏃 Text after translation: '''{tts_text}'''...")
        else:
            logger.debug("🚫 No translation engine available. Skipping translation.")

        # Extract voice commands and apply them to the entire text
        import re
        voice_pattern = r'\{rate:(?:\+|\-)?\d+%\}|\{volume:(?:\+|\-)?\d+%\}|\{pitch:(?:\+|\-)?\d+Hz\}|\{neutral\}'
        voice_commands = re.findall(voice_pattern, tts_text)
        if voice_commands:
            logger.info(f"🎵 Extracted voice commands: {voice_commands}")
            # Remove voice commands from tts_text completely
            clean_tts_text = re.sub(voice_pattern, '', tts_text).strip()
            if not clean_tts_text:
                clean_tts_text = "."  # Ensure we have some text for TTS
            logger.info(f"🎵 Clean tts_text for TTS: '{clean_tts_text}'")
            
            # Pass commands to TTS by prepending them to the text
            # edge_tts.py will parse and apply the commands, then remove them
            tts_text_with_commands = "".join(voice_commands) + clean_tts_text
            logger.info(f"🎵 TTS text with commands: '{tts_text_with_commands}'")
            tts_text = tts_text_with_commands

        # Handle emotion commands - if text contains only emotions, skip TTS
        emotion_pattern = r'\[(?:neutral|joy|smile|laugh|anger|disgust|fear|sadness|surprise|confused|thinking|excited|shy|wink)\]'
        # Also handle {neutral} as emotion command
        neutral_pattern = r'\{neutral\}'
        clean_tts_text = re.sub(emotion_pattern, '', tts_text).strip()
        clean_tts_text = re.sub(neutral_pattern, '', clean_tts_text).strip()
        if not clean_tts_text and (re.search(emotion_pattern, tts_text) or re.search(neutral_pattern, tts_text)):
            logger.info(f"😊 Emotion-only text detected, skipping TTS: {tts_text}")
            # Skip TTS for emotion-only text, but still send to frontend
            full_response += display_text.text
            display_text.text = clean_voice_commands_from_text(display_text.text)
            continue

        # Note: Voice commands are cleaned in edge_tts.py, not here
        # The TTS engine will parse and apply voice commands automatically

        full_response += display_text.text
        
        # Clean display_text from voice commands for frontend
        display_text.text = clean_voice_commands_from_text(display_text.text)
        
        await tts_manager.speak(
            tts_text=tts_text,  # Отправляем текст с командами голоса в TTS
            display_text=display_text,
            actions=actions,
            live2d_model=live2d_model,
            tts_engine=tts_engine,
            websocket_send=websocket_send,
        )
    return full_response


async def handle_audio_output(
    output: AudioOutput,
    websocket_send: WebSocketSend,
) -> str:
    """Process and send AudioOutput directly to the client"""
    full_response = ""
    async for audio_path, display_text, transcript, actions in output:
        full_response += transcript
        
        # Clean display_text from voice commands for frontend
        display_text.text = clean_voice_commands_from_text(display_text.text)
        
        audio_payload = prepare_audio_payload(
            audio_path=audio_path,
            display_text=display_text,
            actions=actions.to_dict() if actions else None,
        )
        await websocket_send(json.dumps(audio_payload))
    return full_response


async def send_conversation_start_signals(websocket_send: WebSocketSend) -> None:
    """Send initial conversation signals"""
    await websocket_send(
        json.dumps(
            {
                "type": "control",
                "text": "conversation-chain-start",
            }
        )
    )
    await websocket_send(json.dumps({"type": "full-text", "text": "Thinking..."}))


async def process_user_input(
    user_input: Union[str, np.ndarray],
    asr_engine: ASRInterface,
    websocket_send: WebSocketSend,
) -> str:
    """Process user input, converting audio to text if needed"""
    if isinstance(user_input, np.ndarray):
        logger.info("Transcribing audio input...")
        input_text = await asr_engine.async_transcribe_np(user_input)
        await websocket_send(
            json.dumps({"type": "user-input-transcription", "text": input_text})
        )
        return input_text
    return user_input


async def finalize_conversation_turn(
    tts_manager: TTSTaskManager,
    websocket_send: WebSocketSend,
    client_uid: str,
    broadcast_ctx: Optional[BroadcastContext] = None,
) -> None:
    """Finalize a conversation turn"""
    if tts_manager.task_list:
        await asyncio.gather(*tts_manager.task_list)
        await websocket_send(json.dumps({"type": "backend-synth-complete"}))

        response = await message_handler.wait_for_response(
            client_uid, "frontend-playback-complete"
        )

        if not response:
            logger.warning(f"No playback completion response from {client_uid}")
            return

    await websocket_send(json.dumps({"type": "force-new-message"}))

    if broadcast_ctx and broadcast_ctx.broadcast_func:
        await broadcast_ctx.broadcast_func(
            broadcast_ctx.group_members,
            {"type": "force-new-message"},
            broadcast_ctx.current_client_uid,
        )

    await send_conversation_end_signal(websocket_send, broadcast_ctx)


async def send_conversation_end_signal(
    websocket_send: WebSocketSend,
    broadcast_ctx: Optional[BroadcastContext],
    session_emoji: str = "😊",
) -> None:
    """Send conversation chain end signal"""
    chain_end_msg = {
        "type": "control",
        "text": "conversation-chain-end",
    }

    await websocket_send(json.dumps(chain_end_msg))

    if broadcast_ctx and broadcast_ctx.broadcast_func and broadcast_ctx.group_members:
        await broadcast_ctx.broadcast_func(
            broadcast_ctx.group_members,
            chain_end_msg,
        )

    logger.info(f"😎👍✅ Conversation Chain {session_emoji} completed!")


def cleanup_conversation(tts_manager: TTSTaskManager, session_emoji: str) -> None:
    """Clean up conversation resources"""
    tts_manager.clear()
    logger.debug(f"🧹 Clearing up conversation {session_emoji}.")


EMOJI_LIST = [
    "🐶",
    "🐱",
    "🐭",
    "🐹",
    "🐰",
    "🦊",
    "🐻",
    "🐼",
    "🐨",
    "🐯",
    "🦁",
    "🐮",
    "🐷",
    "🐸",
    "🐵",
    "🐔",
    "🐧",
    "🐦",
    "🐤",
    "🐣",
    "🐥",
    "🦆",
    "🦅",
    "🦉",
    "🦇",
    "🐺",
    "🐗",
    "🐴",
    "🦄",
    "🐝",
    "🌵",
    "🎄",
    "🌲",
    "🌳",
    "🌴",
    "🌱",
    "🌿",
    "☘️",
    "🍀",
    "🍂",
    "🍁",
    "🍄",
    "🌾",
    "💐",
    "🌹",
    "🌸",
    "🌛",
    "🌍",
    "⭐️",
    "🔥",
    "🌈",
    "🌩",
    "⛄️",
    "🎃",
    "🎄",
    "🎉",
    "🎏",
    "🎗",
    "🀄️",
    "🎭",
    "🎨",
    "🧵",
    "🪡",
    "🧶",
    "🥽",
    "🥼",
    "🦺",
    "👔",
    "👕",
    "👜",
    "👑",
]
