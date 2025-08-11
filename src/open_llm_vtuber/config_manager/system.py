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

    # Memory module settings
    relationships_db_path: str = Field(
        default="cache/relationships.sqlite3",
        alias="relationships_db_path",
        description="Filesystem path to SQLite DB storing user relationships.",
    )
    memory_consolidation_interval_sec: int = Field(
        default=900,
        alias="memory_consolidation_interval_sec",
        description="Periodic consolidation interval in seconds (>=60).",
    )
    chroma_persist_dir: str = Field(
        default="cache/chroma",
        alias="chroma_persist_dir",
        description="ChromaDB persistent directory.",
    )
    chroma_collection: str = Field(
        default="vtuber_memory",
        alias="chroma_collection",
        description="ChromaDB collection name.",
    )
    embeddings_model: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2",
        alias="embeddings_model",
        description="Embeddings model used for memory (for default backend and MemGPT).",
    )

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
        "relationships_db_path": Description(i18n_key="config.relationships_db_path"),
        "memory_consolidation_interval_sec": Description(
            i18n_key="config.memory_consolidation_interval_sec"
        ),
        "chroma_persist_dir": Description(i18n_key="config.chroma_persist_dir"),
        "chroma_collection": Description(i18n_key="config.chroma_collection"),
        "embeddings_model": Description(i18n_key="config.embeddings_model"),
    }

    @model_validator(mode="after")
    def check_port(cls, values):
        port = values.port
        if port < 0 or port > 65535:
            raise ValueError("Port must be between 0 and 65535")
        # Clamp consolidation interval
        try:
            ival = int(values.memory_consolidation_interval_sec)
            if ival < 60:
                values.memory_consolidation_interval_sec = 60
        except Exception:
            values.memory_consolidation_interval_sec = 900
        return values
