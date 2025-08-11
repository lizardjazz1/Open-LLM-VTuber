# config_manager/system.py
from pydantic import Field, model_validator, BaseModel
from typing import Dict, ClassVar, Optional
from .i18n import I18nMixin, Description


class TwitchConfig(BaseModel):
    """Twitch integration configuration."""

    enabled: bool = Field(False, description="Enable Twitch integration")
    channel_name: str = Field("your_channel_name", description="Twitch channel name")
    app_id: str = Field("", description="Twitch application ID")
    app_secret: str = Field("", description="Twitch application secret")
    max_message_length: int = Field(300, description="Maximum message length")
    max_recent_messages: int = Field(
        10, description="Maximum number of recent messages"
    )


class SystemConfig(I18nMixin):
    """System configuration settings."""

    conf_version: str = Field(..., alias="conf_version")
    host: str = Field(..., alias="host")
    port: int = Field(..., alias="port")
    language: str = Field("en", alias="language")
    config_alts_dir: str = Field(..., alias="config_alts_dir")
    config_alt: Optional[str] = Field(None, alias="config_alt")
    enable_proxy: bool = Field(False, alias="enable_proxy")
    twitch_config: TwitchConfig = Field(
        default_factory=TwitchConfig, alias="twitch_config"
    )
    tool_prompts: Dict[str, str] = Field(..., alias="tool_prompts")
    # // DEBUG: [FIXED] Token for /logs auth | Ref: 4,15
    logging_token: Optional[str] = Field("", alias="logging_token")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "conf_version": Description(i18n_key="config.conf_version"),
        "host": Description(i18n_key="config.host"),
        "port": Description(i18n_key="config.port"),
        "language": Description(i18n_key="config.language"),
        "config_alts_dir": Description(i18n_key="config.config_alts_dir"),
        "config_alt": Description(i18n_key="config.config_alt"),
        "enable_proxy": Description(i18n_key="config.enable_proxy"),
        "twitch_config": Description(i18n_key="config.twitch_config"),
        "tool_prompts": Description(i18n_key="config.tool_prompts"),
        "logging_token": Description(i18n_key="config.logging_token"),
    }

    @model_validator(mode="after")
    def check_port(cls, values):
        port = values.port
        if port < 0 or port > 65535:
            raise ValueError("Port must be between 0 and 65535")
        return values
