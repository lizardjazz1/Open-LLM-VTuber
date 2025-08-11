# config_manager/character.py
from pydantic import Field, field_validator, BaseModel
from typing import Dict, ClassVar
from .i18n import I18nMixin, Description
from .asr import ASRConfig
from .tts import TTSConfig
from .vad import VADConfig
from .tts_preprocessor import TTSPreprocessorConfig
from .system import TwitchConfig

from .agent import AgentConfig


class Live2DConfig(BaseModel):
    """Live2D configuration settings."""

    enable_expressions: bool = Field(True, description="Enable Live2D expressions")
    enable_motions: bool = Field(True, description="Enable Live2D motions")
    idle_motion_interval: int = Field(8, description="Idle motion interval in seconds")
    expression_sensitivity: float = Field(0.8, description="Expression sensitivity")
    emotion_voice_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Map emotion keywords (e.g., 'joy') to TTS voice command presets like '{rate:+20%}{volume:+10%}{pitch:+20Hz}'",
    )


class VtuberMemoryConfig(BaseModel):
    """Vtuber memory module settings."""

    enabled: bool = Field(True, description="Enable vtuber memory module")
    provider: str = Field(
        "default",
        description="Memory provider implementation: 'default' (Chroma wrapper), 'letta', 'memgpt', etc.",
    )


class CharacterConfig(I18nMixin):
    """Character configuration settings."""

    conf_name: str = Field(..., alias="conf_name")
    conf_uid: str = Field(..., alias="conf_uid")
    live2d_model_name: str = Field(..., alias="live2d_model_name")
    character_name: str = Field(default="", alias="character_name")
    human_name: str = Field(default="Human", alias="human_name")
    avatar: str = Field(default="", alias="avatar")
    persona_prompt: str = Field(..., alias="persona_prompt")
    persona_prompt_name: str = Field(default="", alias="persona_prompt_name")
    agent_config: AgentConfig = Field(..., alias="agent_config")
    asr_config: ASRConfig = Field(..., alias="asr_config")
    tts_config: TTSConfig = Field(..., alias="tts_config")
    vad_config: VADConfig = Field(..., alias="vad_config")
    live2d_config: Live2DConfig = Field(
        default_factory=Live2DConfig, alias="live2d_config"
    )
    tts_preprocessor_config: TTSPreprocessorConfig = Field(
        ..., alias="tts_preprocessor_config"
    )
    twitch_config: TwitchConfig = Field(
        default_factory=TwitchConfig, alias="twitch_config"
    )
    vtuber_memory: VtuberMemoryConfig = Field(
        default_factory=VtuberMemoryConfig, alias="vtuber_memory"
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "conf_name": Description(i18n_key="config.character.conf_name"),
        "conf_uid": Description(i18n_key="config.character.conf_uid"),
        "live2d_model_name": Description(i18n_key="config.character.live2d_model_name"),
        "character_name": Description(i18n_key="config.character.character_name"),
        "persona_prompt": Description(i18n_key="config.character.persona_prompt"),
        "persona_prompt_name": Description(
            i18n_key="config.character.persona_prompt_name"
        ),
        "agent_config": Description(i18n_key="config.character.agent_config"),
        "asr_config": Description(i18n_key="config.character.asr_config"),
        "tts_config": Description(i18n_key="config.character.tts_config"),
        "vad_config": Description(i18n_key="config.character.vad_config"),
        "live2d_config": Description(i18n_key="config.character.live2d_config"),
        "tts_preprocessor_config": Description(
            i18n_key="config.character.tts_preprocessor_config"
        ),
        "human_name": Description(i18n_key="config.character.human_name"),
        "avatar": Description(i18n_key="config.character.avatar"),
        "twitch_config": Description(i18n_key="config.character.twitch_config"),
        "vtuber_memory": Description(i18n_key="config.character.vtuber_memory"),
    }

    @field_validator("persona_prompt")
    def check_default_persona_prompt(cls, v):
        if not v:
            raise ValueError(
                "Persona_prompt cannot be empty. Please provide a persona prompt."
            )
        return v

    @field_validator("character_name")
    def set_default_character_name(cls, v, values):
        if not v and "conf_name" in values:
            return values["conf_name"]
        return v
