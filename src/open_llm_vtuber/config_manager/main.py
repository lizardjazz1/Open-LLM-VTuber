# config_manager/main.py
from pydantic import BaseModel, Field
from typing import Dict, ClassVar

from .system import SystemConfig
from .character import CharacterConfig
from .live import LiveConfig
from .i18n import I18nMixin, Description


class Config(I18nMixin, BaseModel):
    """
    Main configuration for the application.
    """

    system_config: SystemConfig = Field(default=None, alias="system_config")
    character_config: CharacterConfig = Field(..., alias="character_config")
    live_config: LiveConfig = Field(default=LiveConfig(), alias="live_config")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "system_config": Description(i18n_key="system_configuration_settings"),
        "character_config": Description(i18n_key="character_configuration_settings"),
        "live_config": Description(i18n_key="live_streaming_platform_integration_settings"),
    }
