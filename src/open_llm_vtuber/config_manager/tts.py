# config_manager/tts.py
from pydantic import ValidationInfo, Field, model_validator
from typing import Literal, Optional, Dict, ClassVar
from .i18n import I18nMixin, Description


class AzureTTSConfig(I18nMixin):
    """Configuration for Azure TTS service."""

    api_key: str = Field(..., alias="api_key")
    region: str = Field(..., alias="region")
    voice: str = Field(..., alias="voice")
    pitch: str = Field(..., alias="pitch")
    rate: str = Field(..., alias="rate")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "api_key": Description(i18n_key="api_key_for_azure_tts_service"),
        "region": Description(i18n_key="azure_region_eg_eastus"),
        "voice": Description(i18n_key="voice_name_to_use_for_azure_tts"),
        "pitch": Description(i18n_key="pitch_adjustment_percentage"),
        "rate": Description(i18n_key="speaking_rate_adjustment"),
    }


class BarkTTSConfig(I18nMixin):
    """Configuration for Bark TTS."""

    voice: str = Field(..., alias="voice")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "voice": Description(i18n_key="voice_name_to_use_for_bark_tts"),
    }


class EdgeTTSConfig(I18nMixin):
    """Configuration for Edge TTS."""

    voice: str = Field(..., alias="voice")
    rate: str = Field("+0%", alias="rate")
    volume: str = Field("+0%", alias="volume")
    pitch: str = Field("+0Hz", alias="pitch")
    # New tunables for reliability/offline fallback
    timeout_ms: int = Field(15000, alias="timeout_ms")
    max_retries: int = Field(1, alias="max_retries")
    enable_fallback: bool = Field(True, alias="enable_fallback")
    piper_model_path: Optional[str] = Field(None, alias="piper_model_path")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "voice": Description(
            i18n_key="voice_name_to_use_for_edge_tts_use_edge_tts_list_voices_to_list_available_voices"
        ),
        "rate": Description(
            i18n_key="speech_rate_adjustment_eg_10_10_0_limited_to_100_to_100"
        ),
        "volume": Description(i18n_key="volume_adjustment_eg_10_10_0"),
        "pitch": Description(
            i18n_key="pitch_adjustment_in_hz_eg_10hz_10hz_0hz_limited_to_50hz_to_50hz"
        ),
        # New fields
        "timeout_ms": Description(
            i18n_key="edge_tts_request_timeout_in_milliseconds_default_15000"
        ),
        "max_retries": Description(
            i18n_key="edge_tts_max_retries_on_failure_default_1"
        ),
        "enable_fallback": Description(
            i18n_key="enable_offline_piper_fallback_if_edge_tts_fails"
        ),
        "piper_model_path": Description(
            i18n_key="filesystem_path_to_piper_onnx_model_for_offline_tts"
        ),
    }


class CosyvoiceTTSConfig(I18nMixin):
    """Configuration for Cosyvoice TTS."""

    client_url: str = Field(..., alias="client_url")
    mode_checkbox_group: str = Field(..., alias="mode_checkbox_group")
    sft_dropdown: str = Field(..., alias="sft_dropdown")
    prompt_text: str = Field(..., alias="prompt_text")
    prompt_wav_upload_url: str = Field(..., alias="prompt_wav_upload_url")
    prompt_wav_record_url: str = Field(..., alias="prompt_wav_record_url")
    instruct_text: str = Field(..., alias="instruct_text")
    seed: int = Field(..., alias="seed")
    api_name: str = Field(..., alias="api_name")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "client_url": Description(i18n_key="url_of_the_cosyvoice_gradio_web_ui"),
        "mode_checkbox_group": Description(i18n_key="mode_checkbox_group_value"),
        "sft_dropdown": Description(i18n_key="sft_dropdown_value"),
        "prompt_text": Description(i18n_key="prompt_text"),
        "prompt_wav_upload_url": Description(i18n_key="url_for_prompt_wav_file_upload"),
        "prompt_wav_record_url": Description(
            i18n_key="url_for_prompt_wav_file_recording"
        ),
        "instruct_text": Description(i18n_key="instruction_text"),
        "seed": Description(i18n_key="random_seed"),
        "api_name": Description(i18n_key="api_endpoint_name"),
    }


class Cosyvoice2TTSConfig(I18nMixin):
    """Configuration for Cosyvoice2 TTS."""

    client_url: str = Field(..., alias="client_url")
    mode_checkbox_group: str = Field(..., alias="mode_checkbox_group")
    sft_dropdown: str = Field(..., alias="sft_dropdown")
    prompt_text: str = Field(..., alias="prompt_text")
    prompt_wav_upload_url: str = Field(..., alias="prompt_wav_upload_url")
    prompt_wav_record_url: str = Field(..., alias="prompt_wav_record_url")
    instruct_text: str = Field(..., alias="instruct_text")
    stream: bool = Field(..., alias="stream")
    seed: int = Field(..., alias="seed")
    speed: float = Field(..., alias="speed")
    api_name: str = Field(..., alias="api_name")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "client_url": Description(i18n_key="url_of_the_cosyvoice_gradio_web_ui"),
        "mode_checkbox_group": Description(i18n_key="mode_checkbox_group_value"),
        "sft_dropdown": Description(i18n_key="sft_dropdown_value"),
        "prompt_text": Description(i18n_key="prompt_text"),
        "prompt_wav_upload_url": Description(i18n_key="url_for_prompt_wav_file_upload"),
        "prompt_wav_record_url": Description(
            i18n_key="url_for_prompt_wav_file_recording"
        ),
        "instruct_text": Description(i18n_key="instruction_text"),
        "stream": Description(i18n_key="streaming_inference"),
        "seed": Description(i18n_key="random_seed"),
        "speed": Description(i18n_key="speech_speed_multiplier"),
        "api_name": Description(i18n_key="api_endpoint_name"),
    }


class MeloTTSConfig(I18nMixin):
    """Configuration for Melo TTS."""

    speaker: str = Field(..., alias="speaker")
    language: str = Field(..., alias="language")
    device: str = Field("auto", alias="device")
    speed: float = Field(1.0, alias="speed")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "speaker": Description(i18n_key="speaker_name_eg_en_default_zh"),
        "language": Description(i18n_key="language_code_eg_en_zh"),
        "device": Description(
            i18n_key="device_to_use_for_inference_auto_cpu_cuda_or_mps"
        ),
        "speed": Description(i18n_key="speech_speed_10_is_normal_speed"),
    }


class XTTSConfig(I18nMixin):
    """Configuration for XTTS."""

    api_url: str = Field(..., alias="api_url")
    speaker_wav: str = Field(..., alias="speaker_wav")
    language: str = Field(..., alias="language")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "api_url": Description(i18n_key="url_of_the_xtts_api_endpoint"),
        "speaker_wav": Description(i18n_key="speaker_reference_wav_file"),
        "language": Description(i18n_key="language_code_eg_en_zh"),
    }


class GPTSoVITSConfig(I18nMixin):
    """Configuration for GPT-SoVITS."""

    api_url: str = Field(..., alias="api_url")
    text_lang: str = Field(..., alias="text_lang")
    ref_audio_path: str = Field(..., alias="ref_audio_path")
    prompt_lang: str = Field(..., alias="prompt_lang")
    prompt_text: str = Field(..., alias="prompt_text")
    text_split_method: str = Field(..., alias="text_split_method")
    batch_size: str = Field(..., alias="batch_size")
    media_type: str = Field(..., alias="media_type")
    streaming_mode: str = Field(..., alias="streaming_mode")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "api_url": Description(i18n_key="url_of_the_gpt_sovits_api_endpoint"),
        "text_lang": Description(i18n_key="language_of_the_input_text"),
        "ref_audio_path": Description(i18n_key="path_to_reference_audio_file"),
        "prompt_lang": Description(i18n_key="language_of_the_prompt"),
        "prompt_text": Description(i18n_key="prompt_text"),
        "text_split_method": Description(i18n_key="method_for_splitting_text"),
        "batch_size": Description(i18n_key="batch_size_for_processing"),
        "media_type": Description(i18n_key="output_media_type"),
        "streaming_mode": Description(i18n_key="enable_streaming_mode"),
    }


class FishAPITTSConfig(I18nMixin):
    """Configuration for Fish API TTS."""

    api_key: str = Field(..., alias="api_key")
    reference_id: str = Field(..., alias="reference_id")
    latency: Literal["normal", "balanced"] = Field(..., alias="latency")
    base_url: str = Field(..., alias="base_url")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "api_key": Description(i18n_key="api_key_for_fish_tts_api"),
        "reference_id": Description(
            i18n_key="voice_reference_id_from_fish_audio_website"
        ),
        "latency": Description(i18n_key="latency_mode_normal_or_balanced"),
        "base_url": Description(i18n_key="base_url_for_fish_tts_api"),
    }


class CoquiTTSConfig(I18nMixin):
    """Configuration for Coqui TTS."""

    model_name: str = Field(..., alias="model_name")
    speaker_wav: str = Field("", alias="speaker_wav")
    language: str = Field(..., alias="language")
    device: str = Field("", alias="device")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "model_name": Description(i18n_key="name_of_the_coqui_tts_model_to_use"),
        "speaker_wav": Description(
            i18n_key="path_to_speaker_wav_file_for_voice_cloning"
        ),
        "language": Description(i18n_key="language_code_eg_en_zh"),
        "device": Description(i18n_key="device_to_use_for_inference_cpu_cuda_or_auto"),
    }


class SherpaOnnxTTSConfig(I18nMixin):
    """Configuration for Sherpa Onnx TTS."""

    vits_model: str = Field(..., alias="vits_model")
    vits_lexicon: Optional[str] = Field(None, alias="vits_lexicon")
    vits_tokens: str = Field(..., alias="vits_tokens")
    vits_data_dir: Optional[str] = Field(None, alias="vits_data_dir")
    vits_dict_dir: Optional[str] = Field(None, alias="vits_dict_dir")
    tts_rule_fsts: Optional[str] = Field(None, alias="tts_rule_fsts")
    max_num_sentences: int = Field(2, alias="max_num_sentences")
    sid: int = Field(1, alias="sid")
    provider: Literal["cpu", "cuda", "coreml"] = Field("cpu", alias="provider")
    num_threads: int = Field(1, alias="num_threads")
    speed: float = Field(1.0, alias="speed")
    debug: bool = Field(False, alias="debug")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "vits_model": Description(i18n_key="path_to_vits_model_file"),
        "vits_lexicon": Description(i18n_key="path_to_lexicon_file_optional"),
        "vits_tokens": Description(i18n_key="path_to_tokens_file"),
        "vits_data_dir": Description(
            i18n_key="path_to_espeak_ng_data_directory_optional"
        ),
        "vits_dict_dir": Description(
            i18n_key="path_to_jieba_dictionary_directory_optional_for_chinese"
        ),
        "tts_rule_fsts": Description(i18n_key="path_to_rule_fsts_file_optional"),
        "max_num_sentences": Description(
            i18n_key="maximum_number_of_sentences_per_batch"
        ),
        "sid": Description(i18n_key="speaker_id_for_multi_speaker_models"),
        "provider": Description(i18n_key="provider_for_inference_cpu_cuda_or_coreml"),
        "num_threads": Description(i18n_key="number_of_computation_threads"),
        "speed": Description(i18n_key="speech_speed_10_is_normal_speed"),
        "debug": Description(i18n_key="enable_debug_mode"),
    }


class SiliconFlowTTSConfig(I18nMixin):
    """Configuration for SiliconFlow TTS."""

    api_url: str = Field("https://api.siliconflow.cn/v1/audio/speech", alias="api_url")
    api_key: str = Field(..., alias="api_key")
    default_model: str = Field("FunAudioLLM/CosyVoice2-0.5B", alias="default_model")
    default_voice: str = Field(
        "speech:Dreamflowers:5bdstvc39i:xkqldnpasqmoqbakubom", alias="default_voice"
    )
    sample_rate: int = Field(32000, alias="sample_rate")
    response_format: str = Field("mp3", alias="response_format")
    stream: bool = Field(True, alias="stream")
    speed: float = Field(1, alias="speed")
    gain: int = Field(0, alias="gain")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "api_key": Description(i18n_key="api_key_for_siliconflow_tts_service"),
        "url": Description(i18n_key="api_endpoint_url_for_siliconflow_tts"),
        "model": Description(i18n_key="model_to_use_for_siliconflow_tts"),
        "voice": Description(i18n_key="voice_name_to_use_for_siliconflow_tts"),
        "sample_rate": Description(i18n_key="sample_rate_of_the_output_audio"),
        "stream": Description(i18n_key="enable_streaming_mode"),
        "speed": Description(i18n_key="speaking_speed_multiplier"),
        "gain": Description(i18n_key="audio_gain_adjustment"),
    }


class OpenAITTSConfig(I18nMixin):
    """Configuration for OpenAI-compatible TTS client."""

    model: Optional[str] = Field(None, alias="model")
    voice: Optional[str] = Field(None, alias="voice")
    api_key: Optional[str] = Field(None, alias="api_key")
    base_url: Optional[str] = Field(None, alias="base_url")
    file_extension: Literal["mp3", "wav"] = Field("mp3", alias="file_extension")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "model": Description(
            i18n_key="model_name_for_the_tts_server_overrides_default"
        ),
        "voice": Description(
            i18n_key="voice_names_for_the_tts_server_overrides_default"
        ),
        "api_key": Description(
            i18n_key="api_key_if_required_by_the_tts_server_overrides_default"
        ),
        "base_url": Description(
            i18n_key="base_url_of_the_tts_server_overrides_default"
        ),
        "file_extension": Description(
            i18n_key="audio_file_format_mp3_or_wav_defaults_to_mp3"
        ),
    }


class SparkTTSConfig(I18nMixin):
    """Configuration for Spark TTS."""

    api_url: str = Field(..., alias="api_url")
    prompt_wav_upload: str = Field(..., alias="prompt_wav_upload")
    api_name: str = Field(..., alias="api_name")
    gender: str = Field(..., alias="gender")
    pitch: int = Field(..., alias="pitch")
    speed: int = Field(..., alias="speed")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "prompt_wav_upload": Description(
            i18n_key="reference_audio_used_when_using_voice_cloning"
        ),
        "api_url": Description(
            i18n_key="api_address_of_the_spark_tts_gradio_web_frontend_for_example_http1270017860voice_clone"
        ),
        "api_name": Description(
            i18n_key="the_api_endpoint_name_for_example_voice_clonevoice_creation"
        ),
        "gender": Description(i18n_key="gender_of_the_voice_male_or_female"),
        "pitch": Description(i18n_key="pitch_shift_in_semitones_default_3range_1_5"),
        "speed": Description(
            i18n_key="speed_of_the_voice_in_percent_default_3range_1_5"
        ),
    }


class MinimaxTTSConfig(I18nMixin):
    """Configuration for Minimax TTS."""

    group_id: str = Field(..., alias="group_id")
    api_key: str = Field(..., alias="api_key")
    model: str = Field("speech-02-turbo", alias="model")
    voice_id: str = Field("male-qn-qingse", alias="voice_id")
    pronunciation_dict: str = Field("", alias="pronunciation_dict")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "group_id": Description(i18n_key="minimax_group_id"),
        "api_key": Description(i18n_key="minimax_api_key"),
        "model": Description(i18n_key="minimax_model_name"),
        "voice_id": Description(i18n_key="minimax_voice_id"),
        "pronunciation_dict": Description(
            i18n_key="custom_pronunciation_dictionary_string"
        ),
    }


class TTSConfig(I18nMixin):
    """Configuration for Text-to-Speech."""

    tts_model: Literal[
        "azure_tts",
        "bark_tts",
        "edge_tts",
        "cosyvoice_tts",
        "cosyvoice2_tts",
        "melo_tts",
        "coqui_tts",
        "x_tts",
        "gpt_sovits_tts",
        "fish_api_tts",
        "sherpa_onnx_tts",
        "siliconflow_tts",
        "openai_tts",  # Add openai_tts here
        "spark_tts",
        "minimax_tts",
    ] = Field(..., alias="tts_model")

    azure_tts: Optional[AzureTTSConfig] = Field(None, alias="azure_tts")
    bark_tts: Optional[BarkTTSConfig] = Field(None, alias="bark_tts")
    edge_tts: Optional[EdgeTTSConfig] = Field(None, alias="edge_tts")
    cosyvoice_tts: Optional[CosyvoiceTTSConfig] = Field(None, alias="cosyvoice_tts")
    cosyvoice2_tts: Optional[Cosyvoice2TTSConfig] = Field(None, alias="cosyvoice2_tts")
    melo_tts: Optional[MeloTTSConfig] = Field(None, alias="melo_tts")
    coqui_tts: Optional[CoquiTTSConfig] = Field(None, alias="coqui_tts")
    x_tts: Optional[XTTSConfig] = Field(None, alias="x_tts")
    gpt_sovits_tts: Optional[GPTSoVITSConfig] = Field(None, alias="gpt_sovits_tts")
    fish_api_tts: Optional[FishAPITTSConfig] = Field(None, alias="fish_api_tts")
    sherpa_onnx_tts: Optional[SherpaOnnxTTSConfig] = Field(
        None, alias="sherpa_onnx_tts"
    )
    siliconflow_tts: Optional[SiliconFlowTTSConfig] = Field(
        None, alias="siliconflow_tts"
    )
    openai_tts: Optional[OpenAITTSConfig] = Field(None, alias="openai_tts")
    spark_tts: Optional[SparkTTSConfig] = Field(None, alias="spark_tts")
    minimax_tts: Optional[MinimaxTTSConfig] = Field(None, alias="minimax_tts")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "tts_model": Description(i18n_key="text_to_speech_model_to_use"),
        "azure_tts": Description(i18n_key="configuration_for_azure_tts"),
        "bark_tts": Description(i18n_key="configuration_for_bark_tts"),
        "edge_tts": Description(i18n_key="configuration_for_edge_tts"),
        "cosyvoice_tts": Description(i18n_key="configuration_for_cosyvoice_tts"),
        "cosyvoice2_tts": Description(i18n_key="configuration_for_cosyvoice2_tts"),
        "melo_tts": Description(i18n_key="configuration_for_melo_tts"),
        "coqui_tts": Description(i18n_key="configuration_for_coqui_tts"),
        "x_tts": Description(i18n_key="configuration_for_xtts"),
        "gpt_sovits_tts": Description(i18n_key="configuration_for_gpt_sovits"),
        "fish_api_tts": Description(i18n_key="configuration_for_fish_api_tts"),
        "sherpa_onnx_tts": Description(i18n_key="configuration_for_sherpa_onnx_tts"),
        "siliconflow_tts": Description(i18n_key="configuration_for_siliconflow_tts"),
        "openai_tts": Description(i18n_key="configuration_for_openai_compatible_tts"),
        "spark_tts": Description(i18n_key="configuration_for_spark_tts"),
        "minimax_tts": Description(i18n_key="configuration_for_minimax_tts"),
    }

    @model_validator(mode="after")
    def check_tts_config(cls, values: "TTSConfig", info: ValidationInfo):
        tts_model = values.tts_model

        # Only validate the selected TTS model
        if tts_model == "azure_tts" and values.azure_tts is not None:
            values.azure_tts.model_validate(values.azure_tts.model_dump())
        elif tts_model == "bark_tts" and values.bark_tts is not None:
            values.bark_tts.model_validate(values.bark_tts.model_dump())
        elif tts_model == "edge_tts" and values.edge_tts is not None:
            values.edge_tts.model_validate(values.edge_tts.model_dump())
        elif tts_model == "cosyvoice_tts" and values.cosyvoice_tts is not None:
            values.cosyvoice_tts.model_validate(values.cosyvoice_tts.model_dump())
        elif tts_model == "cosyvoice2_tts" and values.cosyvoice2_tts is not None:
            values.cosyvoice2_tts.model_validate(values.cosyvoice2_tts.model_dump())
        elif tts_model == "melo_tts" and values.melo_tts is not None:
            values.melo_tts.model_validate(values.melo_tts.model_dump())
        elif tts_model == "coqui_tts" and values.coqui_tts is not None:
            values.coqui_tts.model_validate(values.coqui_tts.model_dump())
        elif tts_model == "x_tts" and values.x_tts is not None:
            values.x_tts.model_validate(values.x_tts.model_dump())
        elif tts_model == "gpt_sovits_tts" and values.gpt_sovits_tts is not None:
            values.gpt_sovits_tts.model_validate(values.gpt_sovits_tts.model_dump())
        elif tts_model == "fish_api_tts" and values.fish_api_tts is not None:
            values.fish_api_tts.model_validate(values.fish_api_tts.model_dump())
        elif tts_model == "sherpa_onnx_tts" and values.sherpa_onnx_tts is not None:
            values.sherpa_onnx_tts.model_validate(values.sherpa_onnx_tts.model_dump())
        elif tts_model == "siliconflow_tts" and values.siliconflow_tts is not None:
            values.siliconflow_tts.model_validate(values.siliconflow_tts.model_dump())
        elif tts_model == "openai_tts" and values.openai_tts is not None:
            values.openai_tts.model_validate(values.openai_tts.model_dump())
        elif tts_model == "spark_tts" and values.spark_tts is not None:
            values.spark_tts.model_validate(values.spark_tts.model_dump())
        elif tts_model == "minimax_tts" and values.minimax_tts is not None:
            values.minimax_tts.model_validate(values.minimax_tts.model_dump())

        return values
