import asyncio
import json
import re
import uuid
from datetime import datetime
from typing import List, Optional, Dict
from loguru import logger

from ..agent.output_types import DisplayText, Actions
from ..live2d_model import Live2dModel
from ..tts.tts_interface import TTSInterface
from ..utils.stream_audio import prepare_audio_payload
from .types import WebSocketSend


class TTSTaskManager:
    """Manages TTS tasks and ensures ordered delivery to frontend while allowing parallel TTS generation"""

    def __init__(self) -> None:
        self.task_list: List[asyncio.Task] = []
        self._lock = asyncio.Lock()
        # Queue to store ordered payloads
        self._payload_queue: asyncio.Queue[Dict] = asyncio.Queue()
        # Task to handle sending payloads in order
        self._sender_task: Optional[asyncio.Task] = None
        # Counter for maintaining order
        self._sequence_counter = 0
        self._next_sequence_to_send = 0
        # Persist last seen voice commands so they carry over between segments until overridden
        self._last_voice_cmds: str = ""  # e.g., "{rate:+15%}{volume:+5%}{pitch:+18Hz}"

    async def speak(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        live2d_model: Live2dModel,
        tts_engine: TTSInterface,
        websocket_send: WebSocketSend,
    ) -> None:
        """
        Queue a TTS task while maintaining order of delivery.

        Args:
            tts_text: Text to synthesize
            display_text: Text to display in UI
            actions: Live2D model actions
            live2d_model: Live2D model instance
            tts_engine: TTS engine instance
            websocket_send: WebSocket send function
        """
        if len(re.sub(r'[\s.,!?，。！？\'"』」）】\s]+', "", tts_text)) == 0:
            logger.debug("Empty TTS text, sending silent display payload")
            # Get current sequence number for silent payload
            current_sequence = self._sequence_counter
            self._sequence_counter += 1

            # Start sender task if not running
            if not self._sender_task or self._sender_task.done():
                self._sender_task = asyncio.create_task(
                    self._process_payload_queue(websocket_send)
                )

            await self._send_silent_payload(display_text, actions, current_sequence)
            return

        logger.info(
            f"Enqueuing TTS task with sequence number {self._sequence_counter}:\n"
            f"display_text='''{display_text.text}'''\n"
            f"tts_text='''{tts_text}''' (by {display_text.name})"
        )

        # Log voice commands for debugging
        if "{rate:" in tts_text or "{volume:" in tts_text or "{pitch:" in tts_text:
            logger.info(f"🎵 Voice commands detected in tts_manager: {tts_text}")
        elif (
            "[neutral]" in tts_text
            or "[joy]" in tts_text
            or "[smile]" in tts_text
            or "[laugh]" in tts_text
            or "[anger]" in tts_text
            or "[disgust]" in tts_text
            or "[fear]" in tts_text
            or "[sadness]" in tts_text
            or "[surprise]" in tts_text
            or "[confused]" in tts_text
            or "[thinking]" in tts_text
            or "[excited]" in tts_text
            or "[shy]" in tts_text
            or "[wink]" in tts_text
        ):
            logger.info(f"😊 Emotion commands detected in tts_manager: {tts_text}")
        else:
            logger.debug(f"🔍 No voice or emotion commands in tts_manager: {tts_text}")

        # Carry-over voice commands across segments: if current text has none, prepend last seen
        voice_pattern = r"\{rate:(?:\+|\-)?\d+%\}|\{volume:(?:\+|\-)?\d+%\}|\{pitch:(?:\+|\-)?\d+Hz\}"
        has_current_cmds = re.search(voice_pattern, tts_text) is not None
        if has_current_cmds:
            # Update last seen command string to the concatenation of commands in order
            cmds = re.findall(voice_pattern, tts_text)
            if cmds:
                self._last_voice_cmds = "".join(cmds)
                logger.info(
                    f"🎛️ Updated carry-over voice cmds: '{self._last_voice_cmds}'"
                )
        elif self._last_voice_cmds:
            # Prepend previously used commands to keep consistent voice until overridden
            tts_text = f"{self._last_voice_cmds}{tts_text}"
            logger.info(
                f"🔁 Applied carry-over voice cmds to current segment → '{self._last_voice_cmds}'"
            )

        # Get current sequence number
        current_sequence = self._sequence_counter
        self._sequence_counter += 1

        # Start sender task if not running
        if not self._sender_task or self._sender_task.done():
            self._sender_task = asyncio.create_task(
                self._process_payload_queue(websocket_send)
            )

        # Create and queue the TTS task
        task = asyncio.create_task(
            self._process_tts(
                tts_text=tts_text,
                display_text=display_text,
                actions=actions,
                live2d_model=live2d_model,
                tts_engine=tts_engine,
                sequence_number=current_sequence,
            )
        )
        self.task_list.append(task)

    async def _process_payload_queue(self, websocket_send: WebSocketSend) -> None:
        """
        Process and send payloads in correct order.
        Runs continuously until all payloads are processed.
        """
        buffered_payloads: Dict[int, Dict] = {}

        while True:
            try:
                # Get payload from queue
                payload, sequence_number = await self._payload_queue.get()
                buffered_payloads[sequence_number] = payload

                # Send payloads in order
                while self._next_sequence_to_send in buffered_payloads:
                    next_payload = buffered_payloads.pop(self._next_sequence_to_send)
                    await websocket_send(json.dumps(next_payload))
                    self._next_sequence_to_send += 1

                self._payload_queue.task_done()

            except asyncio.CancelledError:
                break

    async def _send_silent_payload(
        self,
        display_text: DisplayText,
        actions: Optional[Actions],
        sequence_number: int,
    ) -> None:
        """Queue a silent audio payload"""
        audio_payload = prepare_audio_payload(
            audio_path=None,
            display_text=display_text,
            actions=actions,
        )
        await self._payload_queue.put((audio_payload, sequence_number))

    async def _process_tts(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        live2d_model: Live2dModel,
        tts_engine: TTSInterface,
        sequence_number: int,
    ) -> None:
        """Process TTS generation and queue the result for ordered delivery"""
        audio_file_path = None
        try:
            audio_file_path = await self._generate_audio(tts_engine, tts_text)
            payload = prepare_audio_payload(
                audio_path=audio_file_path,
                display_text=display_text,
                actions=actions,
            )
            # Queue the payload with its sequence number
            await self._payload_queue.put((payload, sequence_number))

        except Exception as e:
            logger.error(f"Error preparing audio payload: {e}")
            # Queue silent payload for error case
            payload = prepare_audio_payload(
                audio_path=None,
                display_text=display_text,
                actions=actions,
            )
            await self._payload_queue.put((payload, sequence_number))

        finally:
            if audio_file_path:
                tts_engine.remove_file(audio_file_path)
                logger.debug("Audio cache file cleaned.")

    async def _generate_audio(self, tts_engine: TTSInterface, text: str) -> str:
        """Generate audio file from text"""
        logger.debug(f"🏃Generating audio for '''{text}'''...")

        # Add logging for voice commands
        if "{rate:" in text or "{volume:" in text or "{pitch:" in text:
            logger.info(f"🎵 Voice commands detected in _generate_audio: {text}")
        elif (
            "[neutral]" in text
            or "[joy]" in text
            or "[smile]" in text
            or "[laugh]" in text
            or "[anger]" in text
            or "[disgust]" in text
            or "[fear]" in text
            or "[sadness]" in text
            or "[surprise]" in text
            or "[confused]" in text
            or "[thinking]" in text
            or "[excited]" in text
            or "[shy]" in text
            or "[wink]" in text
        ):
            logger.info(f"😊 Emotion commands detected in _generate_audio: {text}")
        else:
            logger.debug(f"🔍 No voice or emotion commands in _generate_audio: {text}")

        return await tts_engine.async_generate_audio(
            text=text,
            file_name_no_ext=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}",
        )

    def clear(self) -> None:
        """Clear all pending tasks and reset state"""
        self.task_list.clear()
        if self._sender_task:
            self._sender_task.cancel()
        self._sequence_counter = 0
        self._next_sequence_to_send = 0
        # Create a new queue to clear any pending items
        self._payload_queue = asyncio.Queue()
        # Reset carry-over voice commands
        self._last_voice_cmds = ""
