import asyncio
import base64
from typing import AsyncIterator, Optional
import json
import websockets
from loguru import logger
from pathlib import Path

from .agent_interface import AgentInterface
from ..output_types import AudioOutput, Actions, DisplayText
from ..input_types import BatchInput
from ...chat_history_manager import get_metadata, update_metadate


class HumeAIAgent(AgentInterface):
    """
    Hume AI Agent that handles text input and audio output.
    Uses AudioOutput type to provide audio responses with transcripts.
    """

    AGENT_TYPE = "hume_ai_agent"

    def __init__(
        self,
        api_key: str,
        host: str = "api.hume.ai",
        config_id: Optional[str] = None,
        idle_timeout: int = 15,
    ):
        """
        Initialize Hume AI agent

        Args:
            api_key: Hume AI API key
            host: API host
            config_id: Optional configuration ID
            idle_timeout: Connection idle timeout in seconds
        """
        self.api_key = api_key
        self.host = host
        self.config_id = config_id
        self.idle_timeout = idle_timeout
        self._ws = None
        self._current_text = None
        self._current_id = None
        self._connected = False
        self._chat_group_id = None
        self._idle_timer = None
        self._current_conf_uid = None
        self._current_history_uid = None

        # Create cache directory if it doesn't exist
        self.cache_dir = Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)

    async def connect(self, resume_chat_group_id: Optional[str] = None):
        """
        Establish WebSocket connection with optional chat group resumption

        Args:
            resume_chat_group_id: Optional chat group ID to resume
        """
        if self._ws:
            await self._ws.close()
            self._ws = None
            self._connected = False

        # Build URL with query parameters
        socket_url = f"wss://{self.host}/v0/evi/chat?api_key={self.api_key}"

        if self.config_id:
            socket_url += f"&config_id={self.config_id}"

        if resume_chat_group_id:
            logger.info(f"Resuming chat group: {resume_chat_group_id}")
            socket_url += f"&resumed_chat_group_id={resume_chat_group_id}"
            self._chat_group_id = resume_chat_group_id

        logger.info(f"Connecting to EVI with config_id: {self.config_id}")

        self._ws = await websockets.connect(socket_url)
        self._connected = True

        async for message in self._ws:
            data = json.loads(message)
            if data.get("type") == "chat_metadata":
                new_chat_group_id = data.get("chat_group_id")

                if not resume_chat_group_id and self._current_history_uid:
                    update_metadate(
                        self._current_conf_uid,
                        self._current_history_uid,
                        {"resume_id": new_chat_group_id, "agent_type": self.AGENT_TYPE},
                    )

                self._chat_group_id = new_chat_group_id
                logger.info(
                    f"{'Resumed' if resume_chat_group_id else 'Created new'} "
                    f"chat group: {self._chat_group_id}"
                )

    def _reset_idle_timer(self):
        """Reset the idle timer to prevent disconnection."""
        if self._idle_timer:
            self._idle_timer.cancel()

        async def disconnect_after_timeout():
            await asyncio.sleep(self.idle_timeout)
            if self._ws:
                await self._ws.close()
                self._connected = False
                logger.info("Disconnected due to idle timeout")

        self._idle_timer = asyncio.create_task(disconnect_after_timeout())

    async def _ensure_connection(self):
        """Ensure WebSocket connection is active."""
        if not self._connected or not self._ws:
            await self.connect()

    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        """Set memory from chat history."""
        self._current_conf_uid = conf_uid
        self._current_history_uid = history_uid

        metadata = get_metadata(conf_uid, history_uid)
        if metadata and metadata.get("agent_type") == self.AGENT_TYPE:
            resume_id = metadata.get("resume_id")
            if resume_id:
                asyncio.create_task(self.connect(resume_id))

    async def chat(self, batch_input: BatchInput) -> AsyncIterator[AudioOutput]:
        """
        Process chat input and yield audio responses.

        Args:
            batch_input: Input data containing text and metadata

        Yields:
            AudioOutput: Audio responses with transcripts
        """
        await self._ensure_connection()

        # Extract text from input
        text = ""
        for item in batch_input.inputs:
            if hasattr(item, "text"):
                text += item.text + " "

        if not text.strip():
            return

        # Send message to Hume AI
        message = {
            "type": "user_message",
            "message": {
                "content": text.strip(),
                "role": "user"
            }
        }

        await self._ws.send(json.dumps(message))
        self._reset_idle_timer()

        # Process responses
        async for message in self._ws:
            data = json.loads(message)
            
            if data.get("type") == "assistant_message":
                content = data.get("message", {}).get("content", "")
                if content:
                    # Create audio output
                    audio_output = AudioOutput(
                        text=content,
                        audio_data=b"",  # Hume AI doesn't provide audio directly
                        audio_format="wav"
                    )
                    yield audio_output

            elif data.get("type") == "error":
                logger.error(f"Hume AI error: {data}")
                break

    def handle_interrupt(self, heard_response: str) -> None:
        """Handle interruption of the agent."""
        logger.info(f"Hume AI agent interrupted: {heard_response}")

    def __del__(self):
        """Cleanup on deletion."""
        if self._ws:
            asyncio.create_task(self._ws.close())
