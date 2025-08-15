import asyncio
import json
import re as _re
import uuid
from datetime import datetime
from typing import List, Optional, Dict
from loguru import logger
from collections import deque
import difflib
import os
from contextlib import suppress

from ..agent.output_types import DisplayText, Actions
from ..live2d_model import Live2dModel
from ..tts.tts_interface import TTSInterface
from ..utils.stream_audio import prepare_audio_payload
from .types import WebSocketSend

# ===== Precompiled regex patterns for performance =====
_VOICE_CMD_RE = _re.compile(
    r"\{(?:rate|volume):(?:\+|\-)?\d+%\}|\{pitch:(?:\+|\-)?\d+Hz\}|\{neutral\}"
)
_EMO_CMD_RE = _re.compile(
    r"\[(?:neutral|joy|smile|laugh|anger|disgust|fear|sadness|surprise|confused|thinking|excited|shy|wink)\]"
)
_PUNCT_WS_RE = _re.compile(r"[\s.,!?，。！？'»«“”‘’\-]+")
_EMPTY_TEXT_RE = _re.compile(r"[\s.,!?，。！？'\"』」）】\s]+")
_CARRY_VOICE_RE = _re.compile(
    r"\{rate:(?:\+|\-)?\d+%\}|\{volume:(?:\+|\-)?\d+%\}|\{pitch:(?:\+|\-)?\d+Hz\}"
)
_GREETING_RES = [
    _re.compile(r"^privet"),
    _re.compile(r"^привет"),
    _re.compile(r"^всем\s+привет"),
    _re.compile(r"^хэй|^эй"),
    _re.compile(r"^здравств"),
    _re.compile(r"звездочки"),
    _re.compile(r"сегодня\s+у\s+нас\s+особенный\s+день"),
]


class TTSTaskManager:
    """Manages TTS tasks and ensures ordered delivery to frontend while allowing parallel TTS generation

    Also performs light anti-repetition filtering to improve UX:
    - Suppresses near-duplicate segments within a short recent window
    - Rate-limits obvious greetings to avoid spam at stream start
    """

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
        # Buffer for incomplete trailing voice-command fragments like "{volume:" or "{"
        self._pending_cmd_fragment: str = ""
        # Anti-repetition state
        self._recent_norm_texts: deque[str] = deque(maxlen=8)
        self._last_greeting_ts_ms: int = 0
        # Lightweight LRU cache for repeated TTS segments (text with voice commands)
        self._audio_cache: Dict[str, str] = {}
        self._audio_cache_order: deque[str] = deque()
        self._audio_cache_max: int = 32

    def _normalize_for_repeat_check(self, text: str) -> str:
        """Normalize text by removing voice/emotion tags and collapsing whitespace.

        Args:
            text: Raw or command-tagged TTS/display text.
        Returns:
            A lowercase, tag-free, compact string for similarity checking.
        """
        try:
            cleaned = _VOICE_CMD_RE.sub(" ", text)
            cleaned = _EMO_CMD_RE.sub(" ", cleaned)
            cleaned = _PUNCT_WS_RE.sub(" ", cleaned)
            return cleaned.strip().lower()
        except Exception:
            return text.strip().lower()

    def _merge_and_strip_incomplete_cmds(self, tts_text: str) -> str:
        """Merge pending fragment and strip any trailing incomplete voice-command braces.

        Keeps only complete voice-command tokens; buffers incomplete tail for next call.

        Args:
            tts_text: candidate text possibly containing partial command tokens
        Returns:
            Sanitized text without dangling "{" fragments at the end.
        """
        try:
            if self._pending_cmd_fragment:
                tts_text = f"{self._pending_cmd_fragment}{tts_text}"
                self._pending_cmd_fragment = ""
            # If the text ends with an unmatched "{...", buffer it
            m = _re.search(r"\{[^}]*$", tts_text)
            if m:
                self._pending_cmd_fragment = tts_text[m.start() :]
                tts_text = tts_text[: m.start()]
            return tts_text
        except Exception:
            return tts_text

    def _has_speakable_text(self, tts_text: str) -> bool:
        """Detect if text contains speakable characters after removing command tags.

        Returns True only if there are letters/digits or meaningful punctuation beyond tags.
        """
        try:
            cleaned = _VOICE_CMD_RE.sub(" ", tts_text)
            # Remove any leftover braces just in case
            cleaned = cleaned.replace("{", " ").replace("}", " ")
            cleaned = cleaned.strip()
            return _re.search(r"[A-Za-zА-Яа-яЁё0-9]", cleaned) is not None
        except Exception:
            return True

    def _looks_like_greeting(self, norm_text: str) -> bool:
        """Heuristic to detect greetings in RU/EN to rate-limit at startup.

        Returns:
            True if the text appears to be a greeting opener.
        """
        if not norm_text:
            return False
        # Common Russian and playful greetings observed in logs
        for pat in _GREETING_RES:
            if pat.search(norm_text):
                return True
        return False

    def _is_near_duplicate(self, norm_text: str) -> bool:
        """Check similarity against recent normalized texts.

        Uses difflib ratio; considers duplicate only if extremely similar (> 0.99).
        """
        for prev in self._recent_norm_texts:
            try:
                if not prev or not norm_text:
                    continue
                # Exact or contained
                if norm_text == prev:
                    return True
                # Fuzzy similarity
                if difflib.SequenceMatcher(a=prev, b=norm_text).ratio() > 0.99:
                    return True
            except Exception:
                continue
        return False

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
        # Merge previously buffered partial command fragments and strip dangling ones
        tts_text = self._merge_and_strip_incomplete_cmds(tts_text)

        # Anti-repetition guard: skip near-duplicate TTS segments
        norm_disp = self._normalize_for_repeat_check(display_text.text)
        now_ms = int(datetime.utcnow().timestamp() * 1000)
        # Rate-limit greetings: at most one every 45 seconds
        if self._looks_like_greeting(norm_disp):
            if (
                self._last_greeting_ts_ms
                and (now_ms - self._last_greeting_ts_ms) < 45_000
            ):
                logger.info("🛑 Suppressing repeated greeting (rate-limited)")
                # Ensure UI still sees the display text for continuity
                current_sequence = self._sequence_counter
                self._sequence_counter += 1
                if not self._sender_task or self._sender_task.done():
                    self._sender_task = asyncio.create_task(
                        self._process_payload_queue(websocket_send)
                    )
                # Enqueue 'full-text' first for compatibility with original frontend
                await self._enqueue_full_text(display_text, current_sequence)
                # Then enqueue silent audio payload with next sequence
                next_sequence = self._sequence_counter
                self._sequence_counter += 1
                await self._send_silent_payload(display_text, actions, next_sequence)
                return
            else:
                self._last_greeting_ts_ms = now_ms
        # General near-duplicate suppression
        if self._is_near_duplicate(norm_disp):
            logger.info("🛑 Suppressing near-duplicate segment to avoid repetition")
            current_sequence = self._sequence_counter
            self._sequence_counter += 1
            if not self._sender_task or self._sender_task.done():
                self._sender_task = asyncio.create_task(
                    self._process_payload_queue(websocket_send)
                )
            await self._enqueue_full_text(display_text, current_sequence)
            next_sequence = self._sequence_counter
            self._sequence_counter += 1
            await self._send_silent_payload(display_text, actions, next_sequence)
            return
        # Track this normalized text
        self._recent_norm_texts.append(norm_disp)

        # If nothing speakable (after removing tags), only send display + silent audio
        if len(_EMPTY_TEXT_RE.sub("", tts_text)) == 0 or not self._has_speakable_text(
            tts_text
        ):
            logger.debug("Empty TTS text, sending silent display payload")
            # Get current sequence number for silent payload
            current_sequence = self._sequence_counter
            self._sequence_counter += 1

            # Start sender task if not running
            if not self._sender_task or self._sender_task.done():
                self._sender_task = asyncio.create_task(
                    self._process_payload_queue(websocket_send)
                )
            # Enqueue 'full-text' first
            await self._enqueue_full_text(display_text, current_sequence)
            # Then enqueue silent audio with next sequence
            next_sequence = self._sequence_counter
            self._sequence_counter += 1
            await self._send_silent_payload(display_text, actions, next_sequence)
            return

        logger.info(
            f"Queueing display-first flow starting at seq={self._sequence_counter}:\n"
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
        has_current_cmds = _CARRY_VOICE_RE.search(tts_text) is not None
        if has_current_cmds:
            # Update last seen command string to the concatenation of commands in order
            cmds = _CARRY_VOICE_RE.findall(tts_text)
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

        # Ensure sender loop is active
        if not self._sender_task or self._sender_task.done():
            self._sender_task = asyncio.create_task(
                self._process_payload_queue(websocket_send)
            )

        # 1) Immediately enqueue a silent payload with display text so UI shows text before audio
        display_text_sequence = self._sequence_counter
        self._sequence_counter += 1
        # Enqueue 'full-text' first to ensure compatibility
        await self._enqueue_full_text(display_text, display_text_sequence)
        # Then enqueue silent audio right after
        silent_audio_sequence = self._sequence_counter
        self._sequence_counter += 1
        await self._send_silent_payload(display_text, actions, silent_audio_sequence)
        logger.debug(
            f"Enqueued display-first pair: full-text seq={display_text_sequence}, silent-audio seq={silent_audio_sequence}"
        )

        # 2) Generate TTS and enqueue audio payload next to preserve order
        audio_sequence = self._sequence_counter
        self._sequence_counter += 1

        try:
            logger.info(
                "Enqueuing TTS task with sequence number {}:\n"
                "display_text='''{}'''\n"
                "tts_text='''{}''' (by {})".format(
                    audio_sequence,
                    getattr(display_text, "text", ""),
                    tts_text,
                    getattr(display_text, "name", "AI"),
                )
            )
        except Exception:
            pass

        task = asyncio.create_task(
            self._process_tts(
                tts_text=tts_text,
                display_text=display_text,
                actions=actions,
                live2d_model=live2d_model,
                tts_engine=tts_engine,
                sequence_number=audio_sequence,
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
                    try:
                        disp_obj = next_payload.get("display_text")
                        disp_text = (
                            disp_obj.get("text") if isinstance(disp_obj, dict) else None
                        )
                    except Exception:
                        disp_text = None
                    try:
                        has_audio = bool(next_payload.get("audio"))
                    except Exception:
                        has_audio = False
                    try:
                        p_type = next_payload.get("type") or "unknown"
                    except Exception:
                        p_type = "unknown"
                    logger.debug(
                        f"WS send seq={self._next_sequence_to_send} type={p_type} has_audio={'yes' if has_audio else 'no'} text={repr(disp_text)}"
                    )
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
        try:
            logger.debug(
                f"Enqueue silent payload seq={sequence_number} text={repr(getattr(display_text, 'text', None))}"
            )
        except Exception:
            pass
        await self._payload_queue.put((audio_payload, sequence_number))

    async def _enqueue_full_text(
        self, display_text: DisplayText, sequence_number: int
    ) -> None:
        """Queue a 'full-text' message for frontend compatibility (text-before-audio)."""
        try:
            disp = display_text.to_dict() if hasattr(display_text, "to_dict") else None
            text_val = (
                disp.get("text")
                if isinstance(disp, dict)
                else getattr(display_text, "text", None)
            )
        except Exception:
            disp = None
            text_val = getattr(display_text, "text", None)
        payload = {
            "type": "full-text",
            "text": text_val,
            # Optionally include speaker info to help UI, but keep it backward compatible
            "display_text": disp,
        }
        logger.debug(
            f"Enqueue full-text payload seq={sequence_number} text={repr(text_val)}"
        )
        await self._payload_queue.put((payload, sequence_number))

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
        cached = False
        try:
            audio_file_path, cached = await self._generate_audio(tts_engine, tts_text)
            # Do not include display_text in final audio payload to avoid duplicate appends on frontend.
            # The UI receives the text via 'full-text' and silent payloads prior to audio.
            payload = prepare_audio_payload(
                audio_path=audio_file_path,
                display_text=None,
                actions=actions,
            )
            # Queue the payload with its sequence number
            try:
                logger.debug(
                    f"Enqueue audio payload seq={sequence_number} path={'cached' if cached else (audio_file_path or 'None')} text=None"
                )
            except Exception:
                pass
            await self._payload_queue.put((payload, sequence_number))

        except Exception as e:
            logger.error(f"Error preparing audio payload: {e}")
            # Queue silent payload for error case
            try:
                await self._send_silent_payload(
                    display_text=display_text,
                    actions=actions,
                    sequence_number=sequence_number,
                )
            except Exception:
                pass

        finally:
            # Remove temp file only if it is not cached
            if audio_file_path and not cached:
                try:
                    tts_engine.remove_file(audio_file_path)
                    logger.debug("Audio cache file cleaned.")
                except Exception:
                    pass

    async def _generate_audio(
        self, tts_engine: TTSInterface, text: str
    ) -> tuple[str, bool]:
        """Generate audio file from text with small LRU cache.

        Returns:
            tuple[str, bool]: (audio_file_path, cached_flag)
        """
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

        # Try cache
        key = text
        try:
            if key in self._audio_cache:
                path = self._audio_cache[key]
                if path and os.path.exists(path):
                    logger.debug("TTS cache hit")
                    return path, True
                # stale entry
                try:
                    self._audio_cache.pop(key, None)
                    with suppress(Exception):
                        self._audio_cache_order.remove(key)
                except Exception:
                    pass
        except Exception:
            pass

        # Generate fresh audio
        path = await tts_engine.async_generate_audio(
            text=text,
            file_name_no_ext=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}",
        )
        # Insert into LRU
        try:
            self._audio_cache[key] = path
            self._audio_cache_order.append(key)
            if len(self._audio_cache_order) > self._audio_cache_max:
                evict_key = self._audio_cache_order.popleft()
                evict_path = self._audio_cache.pop(evict_key, None)
                if evict_path:
                    with suppress(Exception):
                        tts_engine.remove_file(evict_path)
        except Exception:
            pass
        return path, False

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
        # Clear anti-repetition state
        self._recent_norm_texts.clear()
        self._last_greeting_ts_ms = 0
        # Cleanup cache
        try:
            for k in list(self._audio_cache_order):
                path = self._audio_cache.get(k)
                if path:
                    with suppress(Exception):
                        if os.path.exists(path):
                            os.remove(path)
            self._audio_cache.clear()
            self._audio_cache_order.clear()
        except Exception:
            pass
