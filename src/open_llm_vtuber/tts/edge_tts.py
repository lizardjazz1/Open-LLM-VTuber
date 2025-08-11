import sys
import os
import re

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


# Check out doc at https://github.com/rany2/edge-tts
# Use `edge-tts --list-voices` to list all available voices


class TTSEngine(TTSInterface):
    def __init__(
        self,
        voice="en-US-AvaMultilingualNeural",
        rate="+0%",
        volume="+0%",
        pitch="+0Hz",
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
            return None

        return file_name


# en-US-AvaMultilingualNeural
# en-US-EmmaMultilingualNeural
# en-US-JennyNeural
