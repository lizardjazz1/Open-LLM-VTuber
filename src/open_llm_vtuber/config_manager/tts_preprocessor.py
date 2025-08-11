# config_manager/translate.py
from typing import Literal, Optional, Dict, ClassVar
from pydantic import ValidationInfo, Field, model_validator
from .i18n import I18nMixin, Description

# --- Sub-models for specific Translator providers ---


class DeepLXConfig(I18nMixin):
    """Configuration for DeepLX translation service."""

    deeplx_target_lang: str = Field(..., alias="deeplx_target_lang")
    deeplx_api_endpoint: str = Field(..., alias="deeplx_api_endpoint")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "deeplx_target_lang": Description(i18n_key="target_language_code_for_deeplx_translation"),
        "deeplx_api_endpoint": Description(i18n_key="api_endpoint_url_for_deeplx_service"),
    }


class TencentConfig(I18nMixin):
    """Configuration for tencent translation service."""

    secret_id: str = Field(..., description="Tencent Secret ID")
    secret_key: str = Field(..., description="Tencent Secret Key")
    region: str = Field(..., description="Region for Tencent Service")
    source_lang: str = Field(
        ..., description="Source language code for tencent translation"
    )
    target_lang: str = Field(
        ..., description="Target language code for tencent translation"
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "secret_id": Description(i18n_key="tencent_secret_id"),
        "secret_key": Description(i18n_key="tencent_secret_key"),
        "region": Description(i18n_key="region_for_tencent_service"),
        "source_lang": Description(i18n_key="source_language_code_for_tencent_translation"),
        "target_lang": Description(i18n_key="target_language_code_for_tencent_translation"),
    }


# --- Main TranslatorConfig model ---


class TranslatorConfig(I18nMixin):
    """Configuration for translation services."""

    translate_audio: bool = Field(..., alias="translate_audio")
    translate_provider: Literal["deeplx", "tencent"] = Field(
        ..., alias="translate_provider"
    )
    deeplx: Optional[DeepLXConfig] = Field(None, alias="deeplx")
    tencent: Optional[TencentConfig] = Field(None, alias="tencent")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "translate_audio": Description(i18n_key="enable_audio_translation_requires_deeplx_deployment"),
        "translate_provider": Description(i18n_key="translation_service_provider_to_use"),
        "deeplx": Description(i18n_key="configuration_for_deeplx_translation_service"),
        "tencent": Description(i18n_key="configuration_for_tencent_translation_service"),
    }

    @model_validator(mode="after")
    def check_translator_config(cls, values: "TranslatorConfig", info: ValidationInfo):
        translate_audio = values.translate_audio
        translate_provider = values.translate_provider

        if translate_audio:
            if translate_provider == "deeplx" and values.deeplx is None:
                raise ValueError(
                    "DeepLX configuration must be provided when translate_audio is True and translate_provider is 'deeplx'"
                )
            elif translate_provider == "tencent" and values.tencent is None:
                raise ValueError(
                    "Tencent configuration must be provided when translate_audio is True and translate_provider is 'tencent'"
                )

        return values


class TTSPreprocessorConfig(I18nMixin):
    """Configuration for TTS preprocessor."""

    remove_special_char: bool = Field(..., alias="remove_special_char")
    ignore_brackets: bool = Field(default=True, alias="ignore_brackets")
    ignore_parentheses: bool = Field(default=True, alias="ignore_parentheses")
    ignore_asterisks: bool = Field(default=True, alias="ignore_asterisks")
    ignore_angle_brackets: bool = Field(default=True, alias="ignore_angle_brackets")
    translator_config: TranslatorConfig = Field(..., alias="translator_config")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "remove_special_char": Description(i18n_key="remove_special_characters_from_the_input_text"),
        "ignore_brackets": Description(i18n_key="ignore_everything_inside_brackets"),
        "ignore_parentheses": Description(i18n_key="ignore_everything_inside_parentheses"),
        "ignore_asterisks": Description(i18n_key="ignore_everything_wrapped_inside_asterisks"),
        "ignore_angle_brackets": Description(i18n_key="ignore_everything_wrapped_inside_text"),
        "translator_config": Description(i18n_key="configuration_for_translation_services"),
    }
