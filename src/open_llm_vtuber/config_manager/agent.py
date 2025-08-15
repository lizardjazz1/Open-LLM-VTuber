"""
This module contains the pydantic model for the configurations of
different types of agents.
"""

from pydantic import BaseModel, Field
from typing import Dict, ClassVar, Optional, Literal, List
from .i18n import I18nMixin, Description
from .stateless_llm import StatelessLLMConfigs

# ======== Configurations for different Agents ========


class BasicMemoryAgentConfig(I18nMixin, BaseModel):
    """Configuration for the basic memory agent."""

    llm_provider: Literal[
        "stateless_llm_with_template",
        "openai_compatible_llm",
        "claude_llm",
        "llama_cpp_llm",
        "ollama_llm",
        "lmstudio_llm",
        "openai_llm",
        "gemini_llm",
        "zhipu_llm",
        "deepseek_llm",
        "groq_llm",
        "mistral_llm",
    ] = Field(..., alias="llm_provider")

    # NEW: allow separate LLMs for chat and memory without breaking existing configs
    chat_llm_provider: Optional[
        Literal[
            "stateless_llm_with_template",
            "openai_compatible_llm",
            "claude_llm",
            "llama_cpp_llm",
            "ollama_llm",
            "lmstudio_llm",
            "openai_llm",
            "gemini_llm",
            "zhipu_llm",
            "deepseek_llm",
            "groq_llm",
            "mistral_llm",
        ]
    ] = Field(None, alias="chat_llm_provider")  # WHY: split chat vs memory LLM
    chat_llm_key: Optional[str] = Field(
        None, alias="chat_llm_key"
    )  # key in llm_configs

    memory_llm_provider: Optional[
        Literal[
            "stateless_llm_with_template",
            "openai_compatible_llm",
            "claude_llm",
            "llama_cpp_llm",
            "ollama_llm",
            "lmstudio_llm",
            "openai_llm",
            "gemini_llm",
            "zhipu_llm",
            "deepseek_llm",
            "groq_llm",
            "mistral_llm",
        ]
    ] = Field(None, alias="memory_llm_provider")
    memory_llm_key: Optional[str] = Field(None, alias="memory_llm_key")

    faster_first_response: Optional[bool] = Field(True, alias="faster_first_response")
    segment_method: Literal["regex", "pysbd"] = Field("pysbd", alias="segment_method")
    use_mcpp: Optional[bool] = Field(False, alias="use_mcpp")
    mcp_enabled_servers: Optional[List[str]] = Field([], alias="mcp_enabled_servers")

    # New: configurable LLM limits/timeouts for memory ops
    summarize_max_tokens: int = Field(256, alias="summarize_max_tokens")
    summarize_timeout_s: int = Field(25, alias="summarize_timeout_s")
    sentiment_max_tokens: int = Field(96, alias="sentiment_max_tokens")
    sentiment_timeout_s: int = Field(12, alias="sentiment_timeout_s")
    consolidate_recent_messages: int = Field(120, alias="consolidate_recent_messages")

    # New: personality and behavior settings for VTuber characters
    stream_mode: bool = Field(True, alias="stream_mode")
    spicy_mode: bool = Field(False, alias="spicy_mode")
    personality_consistency: float = Field(
        0.8, alias="personality_consistency", ge=0.0, le=1.0
    )
    creativity_level: float = Field(0.7, alias="creativity_level", ge=0.0, le=1.0)
    emotional_adaptability: float = Field(
        0.9, alias="emotional_adaptability", ge=0.0, le=1.0
    )

    # NEW: Disable reasoning-style outputs in models that support inner thoughts
    no_think_mode: bool = Field(False, alias="no_think_mode")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "llm_provider": Description(i18n_key="llm_provider_to_use_for_this_agent"),
        "chat_llm_provider": Description(
            i18n_key="chat_llm_provider_for_realtime_dialogue"
        ),
        "chat_llm_key": Description(i18n_key="llm_config_key_used_for_chat"),
        "memory_llm_provider": Description(
            i18n_key="memory_llm_provider_for_background_analysis"
        ),
        "memory_llm_key": Description(i18n_key="llm_config_key_used_for_memory_tasks"),
        "faster_first_response": Description(
            i18n_key="whether_to_respond_as_soon_as_encountering_a_comma_in_the_first_sentence_to_reduce_latency_default_true"
        ),
        "segment_method": Description(
            i18n_key="method_for_segmenting_sentences_regex_or_pysbd_default_pysbd"
        ),
        "use_mcpp": Description(
            i18n_key="whether_to_use_mcp_model_context_protocol_for_the_agent_default_true"
        ),
        "mcp_enabled_servers": Description(
            i18n_key="list_of_mcp_servers_to_enable_for_the_agent"
        ),
        "summarize_max_tokens": Description(
            i18n_key="memory_agent_summarize_max_tokens"
        ),
        "summarize_timeout_s": Description(
            i18n_key="memory_agent_summarize_timeout_seconds"
        ),
        "sentiment_max_tokens": Description(
            i18n_key="memory_agent_sentiment_max_tokens"
        ),
        "sentiment_timeout_s": Description(
            i18n_key="memory_agent_sentiment_timeout_seconds"
        ),
        "consolidate_recent_messages": Description(
            i18n_key="memory_agent_consolidation_recent_messages_window"
        ),
        "stream_mode": Description(
            i18n_key="enable_stream_mode_for_vtuber_behavior_default_true"
        ),
        "spicy_mode": Description(
            i18n_key="enable_spicy_mode_for_more_sarcastic_responses_default_false"
        ),
        "personality_consistency": Description(
            i18n_key="personality_consistency_level_0_0_to_1_0_default_0_8"
        ),
        "creativity_level": Description(
            i18n_key="creativity_level_for_response_generation_0_0_to_1_0_default_0_7"
        ),
        "emotional_adaptability": Description(
            i18n_key="emotional_adaptability_level_0_0_to_1_0_default_0_9"
        ),
    }


class Mem0VectorStoreConfig(I18nMixin, BaseModel):
    """Configuration for Mem0 vector store."""

    provider: str = Field(..., alias="provider")
    config: Dict = Field(..., alias="config")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "provider": Description(i18n_key="vector_store_provider_eg_qdrant"),
        "config": Description(i18n_key="provider_specific_configuration"),
    }


class Mem0LLMConfig(I18nMixin, BaseModel):
    """Configuration for Mem0 LLM."""

    provider: str = Field(..., alias="provider")
    config: Dict = Field(..., alias="config")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "provider": Description(i18n_key="llm_provider_name"),
        "config": Description(i18n_key="provider_specific_configuration"),
    }


class Mem0EmbedderConfig(I18nMixin, BaseModel):
    """Configuration for Mem0 embedder."""

    provider: str = Field(..., alias="provider")
    config: Dict = Field(..., alias="config")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "provider": Description(i18n_key="embedder_provider_name"),
        "config": Description(i18n_key="provider_specific_configuration"),
    }


class Mem0Config(I18nMixin, BaseModel):
    """Configuration for Mem0."""

    vector_store: Mem0VectorStoreConfig = Field(..., alias="vector_store")
    llm: Mem0LLMConfig = Field(..., alias="llm")
    embedder: Mem0EmbedderConfig = Field(..., alias="embedder")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "vector_store": Description(i18n_key="vector_store_configuration"),
        "llm": Description(i18n_key="llm_configuration"),
        "embedder": Description(i18n_key="embedder_configuration"),
    }


# =================================


class HumeAIConfig(I18nMixin, BaseModel):
    """Configuration for the Hume AI agent."""

    api_key: str = Field(..., alias="api_key")
    host: str = Field("api.hume.ai", alias="host")
    config_id: Optional[str] = Field(None, alias="config_id")
    idle_timeout: int = Field(15, alias="idle_timeout")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "api_key": Description(i18n_key="api_key_for_hume_ai_service"),
        "host": Description(i18n_key="host_url_for_hume_ai_service_default_apihumeai"),
        "config_id": Description(i18n_key="configuration_id_for_evi_settings"),
        "idle_timeout": Description(
            i18n_key="idle_timeout_in_seconds_before_disconnecting_default_15"
        ),
    }


# =================================


class LettaConfig(I18nMixin, BaseModel):
    """Configuration for the Letta agent."""

    host: str = Field("localhost", alias="host")
    port: int = Field(8283, alias="port")
    id: str = Field(..., alias="id")
    faster_first_response: Optional[bool] = Field(True, alias="faster_first_response")
    segment_method: Literal["regex", "pysbd"] = Field("pysbd", alias="segment_method")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "host": Description(i18n_key="host_address_for_the_letta_server"),
        "port": Description(i18n_key="port_number_for_the_letta_server_default_8283"),
        "id": Description(i18n_key="agent_instance_id_running_on_the_letta_server"),
    }


class AgentSettings(I18nMixin, BaseModel):
    """Settings for different types of agents."""

    basic_memory_agent: Optional[BasicMemoryAgentConfig] = Field(
        None, alias="basic_memory_agent"
    )
    mem0_agent: Optional[Mem0Config] = Field(None, alias="mem0_agent")
    hume_ai_agent: Optional[HumeAIConfig] = Field(None, alias="hume_ai_agent")
    letta_agent: Optional[LettaConfig] = Field(None, alias="letta_agent")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "basic_memory_agent": Description(
            i18n_key="configuration_for_basic_memory_agent"
        ),
        "mem0_agent": Description(i18n_key="configuration_for_mem0_agent"),
        "hume_ai_agent": Description(i18n_key="configuration_for_hume_ai_agent"),
        "letta_agent": Description(i18n_key="configuration_for_letta_agent"),
    }


class AgentConfig(I18nMixin, BaseModel):
    """This class contains all of the configurations related to agent."""

    conversation_agent_choice: Literal[
        "basic_memory_agent", "mem0_agent", "hume_ai_agent", "letta_agent"
    ] = Field(..., alias="conversation_agent_choice")
    agent_settings: AgentSettings = Field(..., alias="agent_settings")
    llm_configs: StatelessLLMConfigs = Field(..., alias="llm_configs")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "conversation_agent_choice": Description(
            i18n_key="type_of_conversation_agent_to_use"
        ),
        "agent_settings": Description(i18n_key="settings_for_different_agent_types"),
        "llm_configs": Description(i18n_key="pool_of_llm_provider_configurations"),
        "faster_first_response": Description(
            i18n_key="whether_to_respond_as_soon_as_encountering_a_comma_in_the_first_sentence_to_reduce_latency_default_true"
        ),
        "segment_method": Description(
            i18n_key="method_for_segmenting_sentences_regex_or_pysbd_default_pysbd"
        ),
    }
