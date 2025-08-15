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

    # Memory module settings
    relationships_db_path: str = Field(
        default="cache/relationships.sqlite3",
        alias="relationships_db_path",
        description="Filesystem path to SQLite DB storing user relationships.",
    )
    memory_consolidation_interval_sec: int = Field(
        default=1800,
        alias="memory_consolidation_interval_sec",
        description="Periodic consolidation interval in seconds (>=60). Default 1800 (30 minutes).",
    )
    # NEW: trigger deep consolidation every N messages (session-level)
    consolidate_every_n_messages: int = Field(
        default=30,
        alias="consolidate_every_n_messages",
        description="Trigger memory consolidation every N inbound messages (default 30).",
    )
    chroma_persist_dir: str = Field(
        default="cache/chroma",
        alias="chroma_persist_dir",
        description="ChromaDB persistent directory.",
    )
    chroma_collection: str = Field(
        default="vtuber_ltm",
        alias="chroma_collection",
        description="ChromaDB collection name for Long-Term Memory.",
    )
    embeddings_model: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2",
        alias="embeddings_model",
        description="Embeddings model used for memory (for default backend and MemGPT).",
    )
    embeddings_api_key: Optional[str] = Field(
        default=None,
        alias="embeddings_api_key",
        description="API key for external embeddings (optional)",
    )
    # Label used in prompts to denote server/system-origin messages, e.g. [server:User] or [server]
    server_label: str = Field(
        default="server",
        alias="server_label",
        description="Label prefix for server-origin messages in prompts (e.g., 'server', 'home').",
    )
    embeddings_api_base: Optional[str] = Field(
        default=None,
        alias="embeddings_api_base",
        description="Base URL for external embeddings provider (optional)",
    )
    deep_consolidation_every_n_streams: int = Field(
        default=5,
        alias="deep_consolidation_every_n_streams",
        description="Perform deep consolidation every N streams (prune LTM and re-extract key facts).",
    )

    # Client log ingestion settings
    client_log_ingest_enabled: bool = Field(
        default=False,
        alias="client_log_ingest_enabled",
        description="Enable /logs endpoint to ingest logs from frontend (default: disabled).",
    )
    client_log_ingest_require_token: bool = Field(
        default=False,
        alias="client_log_ingest_require_token",
        description="Require X-Log-Token for /logs ingestion (recommended for production).",
    )
    # Note: token-based auth for client log ingestion is disabled in this build

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
        "relationships_db_path": Description(i18n_key="config.relationships_db_path"),
        "memory_consolidation_interval_sec": Description(
            i18n_key="config.memory_consolidation_interval_sec"
        ),
        "consolidate_every_n_messages": Description(
            i18n_key="config.consolidate_every_n_messages"
        ),
        "chroma_persist_dir": Description(i18n_key="config.chroma_persist_dir"),
        "chroma_collection": Description(i18n_key="config.chroma_collection"),
        "embeddings_model": Description(i18n_key="config.embeddings_model"),
        "embeddings_api_key": Description(i18n_key="config.embeddings_api_key"),
        "embeddings_api_base": Description(i18n_key="config.embeddings_api_base"),
        "deep_consolidation_every_n_streams": Description(
            i18n_key="config.deep_consolidation_every_n_streams"
        ),
        "client_log_ingest_enabled": Description(
            i18n_key="config.client_log_ingest_enabled"
        ),
        "client_log_ingest_require_token": Description(
            i18n_key="config.client_log_ingest_require_token"
        ),
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
            values.memory_consolidation_interval_sec = 1800
        # Sanitize deep consolidation N
        try:
            n = int(values.deep_consolidation_every_n_streams)
            if n < 1:
                values.deep_consolidation_every_n_streams = 5
        except Exception:
            values.deep_consolidation_every_n_streams = 5
        # Sanitize consolidate_every_n_messages
        try:
            m = int(values.consolidate_every_n_messages)
            if m < 1:
                values.consolidate_every_n_messages = 30
        except Exception:
            values.consolidate_every_n_messages = 30
        return values
