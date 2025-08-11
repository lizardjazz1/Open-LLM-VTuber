# config_manager/stateless_llm.py
from pydantic import BaseModel, Field
from typing import Literal, ClassVar
from .i18n import I18nMixin, Description


class StatelessLLMBaseConfig(I18nMixin):
    """Base configuration for StatelessLLM."""

    # interrupt_method. If the provider supports inserting system prompt anywhere in the chat memory, use "system". Otherwise, use "user".
    interrupt_method: Literal["system", "user"] = Field(
        "user", alias="interrupt_method"
    )
    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "interrupt_method": Description(i18n_key="config.llm.interrupt_method"),
    }


class StatelessLLMWithTemplate(StatelessLLMBaseConfig):
    """Configuration for OpenAI-compatible LLM providers."""

    base_url: str = Field(..., alias="base_url")
    llm_api_key: str = Field(..., alias="llm_api_key")
    model: str = Field(..., alias="model")
    organization_id: str | None = Field(None, alias="organization_id")
    project_id: str | None = Field(None, alias="project_id")
    template: str | None = Field(None, alias="template")
    temperature: float = Field(1.0, alias="temperature")

    _OPENAI_COMPATIBLE_DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "base_url": Description(i18n_key="config.llm.base_url"),
        "llm_api_key": Description(i18n_key="config.llm.llm_api_key"),
        "organization_id": Description(i18n_key="config.llm.organization_id"),
        "project_id": Description(i18n_key="config.llm.project_id"),
        "model": Description(i18n_key="config.llm.model"),
        "temperature": Description(i18n_key="config.llm.temperature"),
    }

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        **StatelessLLMBaseConfig.DESCRIPTIONS,
        **_OPENAI_COMPATIBLE_DESCRIPTIONS,
    }


class OpenAICompatibleConfig(StatelessLLMBaseConfig):
    """Configuration for OpenAI-compatible LLM providers."""

    base_url: str = Field(..., alias="base_url")
    llm_api_key: str = Field(..., alias="llm_api_key")
    model: str = Field(..., alias="model")
    organization_id: str | None = Field(None, alias="organization_id")
    project_id: str | None = Field(None, alias="project_id")
    temperature: float = Field(1.0, alias="temperature")
    top_p: float = Field(1.0, alias="top_p")
    frequency_penalty: float = Field(0.0, alias="frequency_penalty")
    presence_penalty: float = Field(0.0, alias="presence_penalty")
    stop: list[str] | None = Field(None, alias="stop")
    seed: int | None = Field(None, alias="seed")

    _OPENAI_COMPATIBLE_DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "base_url": Description(i18n_key="config.llm.base_url"),
        "llm_api_key": Description(i18n_key="config.llm.llm_api_key"),
        "organization_id": Description(i18n_key="config.llm.organization_id"),
        "project_id": Description(i18n_key="config.llm.project_id"),
        "model": Description(i18n_key="config.llm.model"),
        "temperature": Description(i18n_key="config.llm.temperature"),
        "top_p": Description(i18n_key="config.llm.top_p"),
        "frequency_penalty": Description(i18n_key="config.llm.frequency_penalty"),
        "presence_penalty": Description(i18n_key="config.llm.presence_penalty"),
        "stop": Description(i18n_key="config.llm.stop"),
        "seed": Description(i18n_key="config.llm.seed"),
    }

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        **StatelessLLMBaseConfig.DESCRIPTIONS,
        **_OPENAI_COMPATIBLE_DESCRIPTIONS,
    }


# Ollama config is completely the same as OpenAICompatibleConfig


class OllamaConfig(OpenAICompatibleConfig):
    """Configuration for Ollama API."""

    llm_api_key: str = Field("default_api_key", alias="llm_api_key")
    keep_alive: float = Field(-1, alias="keep_alive")
    unload_at_exit: bool = Field(True, alias="unload_at_exit")
    use_harmony: bool = Field(False, alias="use_harmony")
    top_p: float = Field(1.0, alias="top_p")
    max_tokens: int = Field(150, alias="max_tokens")
    interrupt_method: Literal["system", "user"] = Field(
        "system", alias="interrupt_method"
    )

    _OLLAMA_DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "llm_api_key": Description(i18n_key="config.llm.ollama.llm_api_key"),
        "keep_alive": Description(i18n_key="config.llm.ollama.keep_alive"),
        "unload_at_exit": Description(i18n_key="config.llm.ollama.unload_at_exit"),
        "use_harmony": Description(i18n_key="config.llm.ollama.use_harmony"),
        "top_p": Description(i18n_key="config.llm.ollama.top_p"),
        "max_tokens": Description(i18n_key="config.llm.ollama.max_tokens"),
    }

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        **OpenAICompatibleConfig.DESCRIPTIONS,
        **_OLLAMA_DESCRIPTIONS,
    }


class LmStudioConfig(OpenAICompatibleConfig):
    """Configuration for LM Studio."""

    llm_api_key: str = Field("default_api_key", alias="llm_api_key")
    base_url: str = Field("http://localhost:1234/v1", alias="base_url")
    use_harmony: bool = Field(False, alias="use_harmony")
    max_tokens: int = Field(150, alias="max_tokens")
    stream: bool = Field(False, alias="stream")
    interrupt_method: Literal["system", "user"] = Field(
        "system", alias="interrupt_method"
    )

    _LMSTUDIO_DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "llm_api_key": Description(i18n_key="config.llm.lmstudio.llm_api_key"),
        "base_url": Description(i18n_key="config.llm.lmstudio.base_url"),
        "use_harmony": Description(i18n_key="config.llm.lmstudio.use_harmony"),
        "max_tokens": Description(i18n_key="config.llm.lmstudio.max_tokens"),
        "stream": Description(i18n_key="config.llm.lmstudio.stream"),
    }

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        **OpenAICompatibleConfig.DESCRIPTIONS,
        **_LMSTUDIO_DESCRIPTIONS,
    }


class OpenAIConfig(OpenAICompatibleConfig):
    """Configuration for Official OpenAI API."""

    base_url: str = Field("https://api.openai.com/v1", alias="base_url")
    interrupt_method: Literal["system", "user"] = Field(
        "user", alias="interrupt_method"
    )


class GeminiConfig(OpenAICompatibleConfig):
    """Configuration for Gemini API."""

    base_url: str = Field(
        "https://generativelanguage.googleapis.com/v1beta/openai/", alias="base_url"
    )
    interrupt_method: Literal["system", "user"] = Field(
        "user", alias="interrupt_method"
    )


class MistralConfig(OpenAICompatibleConfig):
    """Configuration for Mistral API."""

    base_url: str = Field("https://api.mistral.ai/v1", alias="base_url")
    interrupt_method: Literal["system", "user"] = Field(
        "user", alias="interrupt_method"
    )


class ZhipuConfig(OpenAICompatibleConfig):
    """Configuration for Zhipu API."""

    base_url: str = Field("https://open.bigmodel.cn/api/paas/v4/", alias="base_url")


class DeepseekConfig(OpenAICompatibleConfig):
    """Configuration for Deepseek API."""

    base_url: str = Field("https://api.deepseek.com/v1", alias="base_url")


class GroqConfig(OpenAICompatibleConfig):
    """Configuration for Groq API."""

    base_url: str = Field("https://api.groq.com/openai/v1", alias="base_url")
    interrupt_method: Literal["system", "user"] = Field(
        "user", alias="interrupt_method"
    )


class ClaudeConfig(StatelessLLMBaseConfig):
    """Configuration for OpenAI Official API."""

    base_url: str = Field("https://api.anthropic.com", alias="base_url")
    llm_api_key: str = Field(..., alias="llm_api_key")
    model: str = Field(..., alias="model")
    interrupt_method: Literal["system", "user"] = Field(
        "user", alias="interrupt_method"
    )

    _CLAUDE_DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "base_url": Description(i18n_key="config.llm.claude.base_url"),
        "llm_api_key": Description(i18n_key="config.llm.claude.llm_api_key"),
        "model": Description(i18n_key="config.llm.claude.model"),
    }

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        **StatelessLLMBaseConfig.DESCRIPTIONS,
        **_CLAUDE_DESCRIPTIONS,
    }


class LlamaCppConfig(StatelessLLMBaseConfig):
    """Configuration for LlamaCpp."""

    model_path: str = Field(..., alias="model_path")
    verbose: bool = Field(False, alias="verbose")
    interrupt_method: Literal["system", "user"] = Field(
        "system", alias="interrupt_method"
    )

    _LLAMA_DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "model_path": Description(i18n_key="config.llm.llama.model_path"),
        "verbose": Description(i18n_key="config.llm.llama.verbose"),
    }

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        **StatelessLLMBaseConfig.DESCRIPTIONS,
        **_LLAMA_DESCRIPTIONS,
    }


class StatelessLLMConfigs(I18nMixin, BaseModel):
    """Pool of LLM provider configurations.
    This class contains configurations for different LLM providers."""

    stateless_llm_with_template: StatelessLLMWithTemplate | None = Field(
        None, alias="stateless_llm_with_template"
    )
    openai_compatible_llm: OpenAICompatibleConfig | None = Field(
        None, alias="openai_compatible_llm"
    )
    ollama_llm: OllamaConfig | None = Field(None, alias="ollama_llm")
    lmstudio_llm: LmStudioConfig | None = Field(None, alias="lmstudio_llm")
    openai_llm: OpenAIConfig | None = Field(None, alias="openai_llm")
    gemini_llm: GeminiConfig | None = Field(None, alias="gemini_llm")
    zhipu_llm: ZhipuConfig | None = Field(None, alias="zhipu_llm")
    deepseek_llm: DeepseekConfig | None = Field(None, alias="deepseek_llm")
    groq_llm: GroqConfig | None = Field(None, alias="groq_llm")
    claude_llm: ClaudeConfig | None = Field(None, alias="claude_llm")
    llama_cpp_llm: LlamaCppConfig | None = Field(None, alias="llama_cpp_llm")
    mistral_llm: MistralConfig | None = Field(None, alias="mistral_llm")

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "stateless_llm_with_template": Description(
            i18n_key="config.llm.providers.stateless_llm_with_template"
        ),
        "openai_compatible_llm": Description(
            i18n_key="config.llm.providers.openai_compatible_llm"
        ),
        "ollama_llm": Description(i18n_key="config.llm.providers.ollama_llm"),
        "lmstudio_llm": Description(i18n_key="config.llm.providers.lmstudio_llm"),
        "openai_llm": Description(i18n_key="config.llm.providers.openai_llm"),
        "gemini_llm": Description(i18n_key="config.llm.providers.gemini_llm"),
        "mistral_llm": Description(i18n_key="config.llm.providers.mistral_llm"),
        "zhipu_llm": Description(i18n_key="config.llm.providers.zhipu_llm"),
        "deepseek_llm": Description(i18n_key="config.llm.providers.deepseek_llm"),
        "groq_llm": Description(i18n_key="config.llm.providers.groq_llm"),
        "claude_llm": Description(i18n_key="config.llm.providers.claude_llm"),
        "llama_cpp_llm": Description(i18n_key="config.llm.providers.llama_cpp_llm"),
    }
