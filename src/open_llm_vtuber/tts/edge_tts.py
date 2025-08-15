import sys
import os
import re
import asyncio
import shlex
import subprocess
from typing import Optional

import edge_tts
from loguru import logger
from .tts_interface import TTSInterface

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


class VoiceCommandParser:
    """Parser for voice control commands in text"""

    def __init__(self):
        # Regular expressions for finding voice commands in curly braces
        # Format: {rate:+10%}, {volume:-5%}, {pitch:+15Hz}
        self.rate_pattern = r"\{rate:([+-]?\d+%)\}"
        self.volume_pattern = r"\{volume:([+-]?\d+%)\}"
        self.pitch_pattern = r"\{pitch:([+-]?\d+Hz)\}"

    def parse_commands(self, text):
        """Parses commands from text and returns cleaned text and parameters"""
        # Search for rate commands
        rate_matches = re.findall(self.rate_pattern, text)
        rate_adjustment = rate_matches[-1] if rate_matches else "+0%"

        # Search for volume commands
        volume_matches = re.findall(self.volume_pattern, text)
        volume_adjustment = volume_matches[-1] if volume_matches else "+0%"

        # Search for pitch commands
        pitch_matches = re.findall(self.pitch_pattern, text)
        pitch_adjustment = pitch_matches[-1] if pitch_matches else "+0Hz"

        # Log found commands
        if rate_matches or volume_matches or pitch_matches:
            logger.debug(
                f"🔍 Found commands in text: rate={rate_matches}, volume={volume_matches}, pitch={pitch_matches}"
            )
            logger.debug(
                f"🔍 Selected commands: rate={rate_adjustment}, volume={volume_adjustment}, pitch={pitch_adjustment}"
            )

        # Check and fix commands without signs
        if rate_adjustment != "+0%":
            if not rate_adjustment.startswith("+") and not rate_adjustment.startswith(
                "-"
            ):
                # If no sign, add +
                rate_adjustment = "+" + rate_adjustment
                logger.debug(f"⚠️ Fixed rate command: {rate_adjustment}")
        elif rate_adjustment == "0%":
            # Handle zero values without sign
            rate_adjustment = "+0%"
            logger.debug(f"⚠️ Fixed zero rate command: {rate_adjustment}")

        if volume_adjustment != "+0%":
            if not volume_adjustment.startswith(
                "+"
            ) and not volume_adjustment.startswith("-"):
                # If no sign, add +
                volume_adjustment = "+" + volume_adjustment
                logger.debug(f"⚠️ Fixed volume command: {volume_adjustment}")
        elif volume_adjustment == "0%":
            # Handle zero values without sign
            volume_adjustment = "+0%"
            logger.debug(f"⚠️ Fixed zero volume command: {volume_adjustment}")

        if pitch_adjustment != "+0Hz":
            if not pitch_adjustment.startswith("+") and not pitch_adjustment.startswith(
                "-"
            ):
                # If no sign, add +
                pitch_adjustment = "+" + pitch_adjustment
                logger.debug(f"⚠️ Fixed pitch command: {pitch_adjustment}")
        elif pitch_adjustment == "0Hz":
            # Handle zero values without sign
            pitch_adjustment = "+0Hz"
            logger.debug(f"⚠️ Fixed zero pitch command: {pitch_adjustment}")

        # Limit rate from -100% to +100%
        try:
            rate_value = int(rate_adjustment.replace("%", ""))
            if rate_value > 100:
                logger.debug(f"⚠️ Rate {rate_adjustment} limited to +100%")
                rate_adjustment = "+100%"
            elif rate_value < -100:
                logger.debug(f"⚠️ Rate {rate_adjustment} limited to -100%")
                rate_adjustment = "-100%"
        except ValueError:
            logger.warning(f"❌ Invalid rate format: {rate_adjustment}, using +0%")
            rate_adjustment = "+0%"

        # Limit pitch from +15Hz to +30Hz (more natural limits)
        try:
            pitch_value = int(pitch_adjustment.replace("Hz", ""))
            if pitch_value > 30:
                logger.debug(f"⚠️ Pitch {pitch_adjustment} limited to +30Hz")
                pitch_adjustment = "+30Hz"
            elif pitch_value < 15:
                logger.debug(f"⚠️ Pitch {pitch_adjustment} limited to +15Hz")
                pitch_adjustment = "+15Hz"
        except ValueError:
            logger.warning(f"❌ Invalid pitch format: {pitch_adjustment}, using +15Hz")
            pitch_adjustment = "+15Hz"

        # Log final parameters
        if (
            rate_adjustment != "+0%"
            or volume_adjustment != "+0%"
            or pitch_adjustment != "+0Hz"
        ):
            logger.debug(
                f"🎯 Final parsed parameters: rate={rate_adjustment}, volume={volume_adjustment}, pitch={pitch_adjustment}"
            )

        # Remove all commands from text
        clean_text = re.sub(self.rate_pattern, "", text)
        clean_text = re.sub(self.volume_pattern, "", clean_text)
        clean_text = re.sub(self.pitch_pattern, "", clean_text)

        # Remove extra spaces
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        return clean_text, rate_adjustment, volume_adjustment, pitch_adjustment


def _looks_like_memory_json(text: str) -> bool:
    """Heuristic: detect raw memory/JSON blocks to avoid voicing them."""
    if not text:
        return False
    if re.search(
        r"\{\s*\"(facts_about_user|past_events|self_beliefs|objectives|emotions|key_facts)\"",
        text,
    ):
        return True
    if text.lstrip().startswith("{") or text.lstrip().startswith("["):
        # Large JSON-ish structures
        return True
    return False


# Check out doc at https://github.com/rany2/edge-tts
# Use `edge-tts --list-voices` to list all available voices


class TTSEngine(TTSInterface):
    def __init__(
        self,
        voice="en-US-AvaMultilingualNeural",
        rate="+0%",
        volume="+0%",
        pitch="+0Hz",
        *,
        timeout_ms: int | None = None,
        max_retries: int | None = None,
        enable_fallback: bool | None = None,
        piper_model_path: str | None = None,
    ):
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        self.command_parser = VoiceCommandParser()

        self.temp_audio_file = "temp"
        self.file_extension = "mp3"
        self.new_audio_dir = "cache"

        if not os.path.exists(self.new_audio_dir):
            os.makedirs(self.new_audio_dir)

        # Tunables (env-overridable)
        self.timeout_ms: int = int(
            timeout_ms or os.getenv("EDGE_TTS_TIMEOUT_MS", "15000")
        )
        self.max_retries: int = max(
            0, int(max_retries or os.getenv("EDGE_TTS_RETRIES", "1"))
        )
        self.enable_fallback: bool = bool(
            int(
                (
                    enable_fallback
                    if enable_fallback is not None
                    else os.getenv("EDGE_TTS_FALLBACK", "1")
                )
            )
        )
        self.piper_model_path: str = piper_model_path or os.getenv("PIPER_MODEL", "")

    async def async_generate_audio(
        self, text: str, file_name_no_ext=None
    ) -> Optional[str]:
        """Async TTS with timeout/retries and optional Piper CLI fallback."""
        file_name = self.generate_cache_file_name(file_name_no_ext, self.file_extension)

        # Parse voice commands and sanitize
        clean_text, rate_adj, vol_adj, pitch_adj = self.command_parser.parse_commands(
            text
        )
        if not clean_text:
            clean_text = "."
        if _looks_like_memory_json(clean_text):
            logger.debug(
                "🧹 JSON-like content detected for TTS; replacing with placeholder to avoid leakage"
            )
            clean_text = "."

        # Try edge-tts asynchronously with timeout/retries
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                communicate = edge_tts.Communicate(
                    clean_text,
                    self.voice,
                    rate=rate_adj,
                    volume=vol_adj,
                    pitch=pitch_adj,
                )
                await asyncio.wait_for(
                    communicate.save(file_name), timeout=self.timeout_ms / 1000
                )
                if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
                    return file_name
                last_err = RuntimeError("edge-tts produced empty file")
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < self.max_retries:
                    await asyncio.sleep(0.2 * (attempt + 1))
                continue

        logger.critical(f"\nError: edge-tts unable to generate audio: {last_err}")
        logger.critical("It's possible that edge-tts is blocked in your region.")

        # Fallback
        if self.enable_fallback:
            fb = await asyncio.to_thread(
                self._fallback_generate_audio, clean_text, file_name
            )
            if fb:
                return fb
        return None

    def generate_audio(self, text, file_name_no_ext=None):
        """
        Generate speech audio file using TTS.
        text: str
            the text to speak
        file_name_no_ext: str
            name of the file without extension

        Returns:
        str: the path to the generated audio file

        """
        file_name = self.generate_cache_file_name(file_name_no_ext, self.file_extension)

        try:
            # Parses commands from text
            clean_text, rate_adjustment, volume_adjustment, pitch_adjustment = (
                self.command_parser.parse_commands(text)
            )

            # Guard against raw JSON/memory dumps in TTS
            if not clean_text.strip():
                clean_text = "."
            if _looks_like_memory_json(clean_text):
                logger.debug(
                    "🧹 JSON-like content detected for TTS (sync); replacing with placeholder"
                )
                clean_text = "."

            # Log for debugging
            if (
                rate_adjustment != "+0%"
                or volume_adjustment != "+0%"
                or pitch_adjustment != "+0Hz"
            ):
                logger.debug(
                    f"🎵 Voice commands detected: rate={rate_adjustment}, volume={volume_adjustment}, pitch={pitch_adjustment}"
                )
                logger.debug(f"📝 Original text: {text}")
                logger.debug(f"🧹 Clean text: {clean_text}")

            # Check if clean_text is not empty
            if not clean_text.strip():
                logger.warning("⚠️ Clean text is empty, but voice commands detected")
                logger.info(
                    "🎵 Applying voice commands to previous text or using default"
                )
                # If we have voice commands but no text, we'll still apply the commands
                # The TTS will use the voice parameters even with empty text
                clean_text = "."  # Use a dot to ensure TTS works

            # If both original text and clean text are empty, skip audio generation
            if (
                not clean_text.strip()
                and rate_adjustment == "+0%"
                and volume_adjustment == "+0%"
                and pitch_adjustment == "+0Hz"
            ):
                logger.warning(
                    "⚠️ Both clean text and original text are empty, skipping audio generation"
                )
                return None

            # If we have voice commands but no text, still generate audio with a space
            if not clean_text.strip() and (
                rate_adjustment != "+0%"
                or volume_adjustment != "+0%"
                or pitch_adjustment != "+0Hz"
            ):
                logger.debug(
                    "🎵 Generating audio with voice commands but no text - using dot"
                )
                clean_text = "."

            # Log final parameters for diagnosis
            logger.debug(
                f"🎯 Final TTS parameters: rate={rate_adjustment}, volume={volume_adjustment}, pitch={pitch_adjustment}"
            )

            # Log file path
            logger.debug(f"📁 Generating audio file: {file_name}")

            # Create TTS with dynamic parameters
            communicate = edge_tts.Communicate(
                clean_text,
                self.voice,
                rate=rate_adjustment,
                volume=volume_adjustment,
                pitch=pitch_adjustment,
            )

            logger.debug(
                f"🎤 Edge TTS communicate object created with voice: {self.voice}"
            )
            logger.debug(f"📝 Text to synthesize: '{clean_text}'")

            # Use the correct async approach
            communicate.save_sync(file_name)

            logger.debug(f"💾 Audio file saved: {file_name}")

            # Check if the file was created and is not empty
            if not os.path.exists(file_name):
                logger.error(f"❌ Generated audio file does not exist: {file_name}")
                return None
            elif os.path.getsize(file_name) == 0:
                logger.error(f"❌ Generated audio file is empty: {file_name}")
                return None
            else:
                logger.debug(
                    f"✅ Audio file created successfully: {file_name} ({os.path.getsize(file_name)} bytes)"
                )

        except Exception as e:
            logger.critical(f"\nError: edge-tts unable to generate audio: {e}")
            logger.critical("It's possible that edge-tts is blocked in your region.")

            # Try fallback synchronously as a last resort (when caller uses sync API)
            if self.enable_fallback:
                fb = self._fallback_generate_audio(clean_text, file_name)
                if fb:
                    return fb
            return None

        return file_name

    def _fallback_generate_audio(
        self, clean_text: str, file_name: str
    ) -> Optional[str]:
        """Attempt offline Piper CLI synthesis if available.

        Expects env PIPER_MODEL to point to a *.onnx model. Command example:
        piper --model <model.onnx> --output_file <file> --text "..."
        """
        try:
            if not self.piper_model_path or not os.path.exists(self.piper_model_path):
                logger.debug(
                    "Piper fallback not configured or model not found; skipping"
                )
                return None
            cmd = f"piper --model {shlex.quote(self.piper_model_path)} --output_file {shlex.quote(file_name)} --text {shlex.quote(clean_text)}"
            logger.info(f"🛟 Using Piper fallback: {cmd}")
            res = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            if res.returncode != 0:
                logger.warning(
                    f"Piper fallback failed: rc={res.returncode}, stderr={res.stderr.decode(errors='ignore')}"
                )
                return None
            if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
                return file_name
            logger.warning("Piper produced no audio file or empty file")
            return None
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Piper fallback error: {e}")
            return None


# en-US-AvaMultilingualNeural
# en-US-EmmaMultilingualNeural
# en-US-JennyNeural
