# config_manager/asr.py
from pydantic import ValidationInfo, Field, model_validator
from typing import Literal, Optional, Dict, ClassVar
from .i18n import I18nMixin, Description


class AzureASRConfig(I18nMixin):
    """Configuration for Azure ASR service."""

    api_key: str = Field(..., alias="api_key")
    region: str = Field(..., alias="region")
    languages: list[str] = Field(["en-US", "zh-CN"], alias="languages")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "api_key": Description(i18n_key="config.asr.azure.api_key"),
        "region": Description(i18n_key="config.asr.azure.region"),
        "languages": Description(i18n_key="config.asr.azure.languages"),
    }


class FasterWhisperConfig(I18nMixin):
    """Configuration for Faster Whisper ASR."""

    model_path: str = Field(..., alias="model_path")
    download_root: str = Field(..., alias="download_root")
    language: Optional[str] = Field(None, alias="language")
    device: str = Field("auto", alias="device")
    compute_type: Literal["int8", "float16", "float32"] = Field(
        "int8", alias="compute_type"
    )
    prompt: str = Field("", alias="prompt")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "model_path": Description(i18n_key="config.asr.faster_whisper.model_path"),
        "download_root": Description(i18n_key="config.asr.faster_whisper.download_root"),
        "language": Description(i18n_key="config.asr.faster_whisper.language"),
        "device": Description(i18n_key="config.asr.faster_whisper.device"),
        "compute_type": Description(i18n_key="config.asr.faster_whisper.compute_type"),
        "prompt": Description(i18n_key="config.asr.faster_whisper.prompt"),
    }


class WhisperCPPConfig(I18nMixin):
    """Configuration for WhisperCPP ASR."""

    model_name: str = Field(..., alias="model_name")
    model_dir: str = Field(..., alias="model_dir")
    print_realtime: bool = Field(False, alias="print_realtime")
    print_progress: bool = Field(False, alias="print_progress")
    language: str = Field("auto", alias="language")
    prompt: str = Field("", alias="prompt")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "model_name": Description(i18n_key="config.asr.whisper_cpp.model_name"),
        "model_dir": Description(i18n_key="config.asr.whisper_cpp.model_dir"),
        "print_realtime": Description(i18n_key="config.asr.whisper_cpp.print_realtime"),
        "print_progress": Description(i18n_key="config.asr.whisper_cpp.print_progress"),
        "language": Description(i18n_key="config.asr.whisper_cpp.language"),
        "prompt": Description(i18n_key="config.asr.whisper_cpp.prompt"),
    }


class WhisperConfig(I18nMixin):
    """Configuration for OpenAI Whisper ASR."""

    name: str = Field(..., alias="name")
    download_root: str = Field(..., alias="download_root")
    device: Literal["cpu", "cuda"] = Field("cpu", alias="device")
    prompt: str = Field("", alias="prompt")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "name": Description(i18n_key="config.asr.whisper.name"),
        "download_root": Description(i18n_key="config.asr.whisper.download_root"),
        "device": Description(i18n_key="config.asr.whisper.device"),
        "prompt": Description(i18n_key="config.asr.whisper.prompt"),
    }


class FunASRConfig(I18nMixin):
    """Configuration for FunASR."""

    model_name: str = Field("iic/SenseVoiceSmall", alias="model_name")
    vad_model: str = Field("fsmn-vad", alias="vad_model")
    punc_model: str = Field("ct-punc", alias="punc_model")
    device: Literal["cpu", "cuda"] = Field("cpu", alias="device")
    disable_update: bool = Field(True, alias="disable_update")
    ncpu: int = Field(4, alias="ncpu")
    hub: Literal["ms", "hf"] = Field("ms", alias="hub")
    use_itn: bool = Field(False, alias="use_itn")
    language: str = Field("auto", alias="language")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "model_name": Description(i18n_key="config.asr.funasr.model_name"),
        "vad_model": Description(i18n_key="config.asr.funasr.vad_model"),
        "punc_model": Description(i18n_key="config.asr.funasr.punc_model"),
        "device": Description(i18n_key="config.asr.funasr.device"),
        "disable_update": Description(i18n_key="config.asr.funasr.disable_update"),
        "ncpu": Description(i18n_key="config.asr.funasr.ncpu"),
        "hub": Description(i18n_key="config.asr.funasr.hub"),
        "use_itn": Description(i18n_key="config.asr.funasr.use_itn"),
        "language": Description(i18n_key="config.asr.funasr.language"),
    }


class GroqWhisperASRConfig(I18nMixin):
    """Configuration for Groq Whisper ASR."""

    api_key: str = Field(..., alias="api_key")
    model: str = Field("whisper-large-v3-turbo", alias="model")
    lang: Optional[str] = Field(None, alias="lang")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "api_key": Description(i18n_key="config.asr.groq.api_key"),
        "model": Description(i18n_key="config.asr.groq.model"),
        "lang": Description(i18n_key="config.asr.groq.lang"),
    }


class SherpaOnnxASRConfig(I18nMixin):
    """Configuration for Sherpa Onnx ASR."""

    model_type: Literal[
        "transducer",
        "paraformer",
        "nemo_ctc",
        "wenet_ctc",
        "whisper",
        "tdnn_ctc",
        "sense_voice",
    ] = Field(..., alias="model_type")
    encoder: Optional[str] = Field(None, alias="encoder")
    decoder: Optional[str] = Field(None, alias="decoder")
    joiner: Optional[str] = Field(None, alias="joiner")
    paraformer: Optional[str] = Field(None, alias="paraformer")
    nemo_ctc: Optional[str] = Field(None, alias="nemo_ctc")
    wenet_ctc: Optional[str] = Field(None, alias="wenet_ctc")
    tdnn_model: Optional[str] = Field(None, alias="tdnn_model")
    whisper_encoder: Optional[str] = Field(None, alias="whisper_encoder")
    whisper_decoder: Optional[str] = Field(None, alias="whisper_decoder")
    sense_voice: Optional[str] = Field(None, alias="sense_voice")
    tokens: str = Field(..., alias="tokens")
    num_threads: int = Field(4, alias="num_threads")
    use_itn: bool = Field(True, alias="use_itn")
    provider: Literal["cpu", "cuda", "rocm"] = Field("cpu", alias="provider")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "model_type": Description(i18n_key="config.asr.sherpa_onnx.model_type"),
        "encoder": Description(i18n_key="config.asr.sherpa_onnx.encoder"),
        "decoder": Description(i18n_key="config.asr.sherpa_onnx.decoder"),
        "joiner": Description(i18n_key="config.asr.sherpa_onnx.joiner"),
        "paraformer": Description(i18n_key="config.asr.sherpa_onnx.paraformer"),
        "nemo_ctc": Description(i18n_key="config.asr.sherpa_onnx.nemo_ctc"),
        "wenet_ctc": Description(i18n_key="config.asr.sherpa_onnx.wenet_ctc"),
        "tdnn_model": Description(i18n_key="config.asr.sherpa_onnx.tdnn_model"),
        "whisper_encoder": Description(i18n_key="config.asr.sherpa_onnx.whisper_encoder"),
        "whisper_decoder": Description(i18n_key="config.asr.sherpa_onnx.whisper_decoder"),
        "sense_voice": Description(i18n_key="config.asr.sherpa_onnx.sense_voice"),
        "tokens": Description(i18n_key="config.asr.sherpa_onnx.tokens"),
        "num_threads": Description(i18n_key="config.asr.sherpa_onnx.num_threads"),
        "use_itn": Description(i18n_key="config.asr.sherpa_onnx.use_itn"),
        "provider": Description(i18n_key="config.asr.sherpa_onnx.provider"),
    }

    @model_validator(mode="after")
    def check_model_paths(cls, values: "SherpaOnnxASRConfig", info: ValidationInfo):
        model_type = values.model_type
        if model_type == "transducer":
            if not values.encoder or not values.decoder or not values.joiner:
                raise ValueError(
                    "For transducer model type, encoder, decoder, and joiner paths are required"
                )
        elif model_type == "paraformer":
            if not values.paraformer:
                raise ValueError("For paraformer model type, paraformer path is required")
        elif model_type == "nemo_ctc":
            if not values.nemo_ctc:
                raise ValueError("For nemo_ctc model type, nemo_ctc path is required")
        elif model_type == "wenet_ctc":
            if not values.wenet_ctc:
                raise ValueError("For wenet_ctc model type, wenet_ctc path is required")
        elif model_type == "whisper":
            if not values.whisper_encoder or not values.whisper_decoder:
                raise ValueError(
                    "For whisper model type, whisper_encoder and whisper_decoder paths are required"
                )
        elif model_type == "tdnn_ctc":
            if not values.tdnn_model:
                raise ValueError("For tdnn_ctc model type, tdnn_model path is required")
        elif model_type == "sense_voice":
            if not values.sense_voice:
                raise ValueError("For sense_voice model type, sense_voice path is required")
        return values


class ASRConfig(I18nMixin):
    """Configuration for Automatic Speech Recognition."""

    asr_model: Literal[
        "faster_whisper",
        "whisper_cpp",
        "whisper",
        "azure_asr",
        "fun_asr",
        "groq_whisper_asr",
        "sherpa_onnx_asr",
    ] = Field(..., alias="asr_model")
    azure_asr: Optional[AzureASRConfig] = Field(None, alias="azure_asr")
    faster_whisper: Optional[FasterWhisperConfig] = Field(None, alias="faster_whisper")
    whisper_cpp: Optional[WhisperCPPConfig] = Field(None, alias="whisper_cpp")
    whisper: Optional[WhisperConfig] = Field(None, alias="whisper")
    fun_asr: Optional[FunASRConfig] = Field(None, alias="fun_asr")
    groq_whisper_asr: Optional[GroqWhisperASRConfig] = Field(
        None, alias="groq_whisper_asr"
    )
    sherpa_onnx_asr: Optional[SherpaOnnxASRConfig] = Field(
        None, alias="sherpa_onnx_asr"
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "asr_model": Description(i18n_key="config.asr.asr_model"),
        "azure_asr": Description(i18n_key="config.asr.azure_asr"),
        "faster_whisper": Description(i18n_key="config.asr.faster_whisper"),
        "whisper_cpp": Description(i18n_key="config.asr.whisper_cpp"),
        "whisper": Description(i18n_key="config.asr.whisper"),
        "fun_asr": Description(i18n_key="config.asr.fun_asr"),
        "groq_whisper_asr": Description(i18n_key="config.asr.groq_whisper_asr"),
        "sherpa_onnx_asr": Description(i18n_key="config.asr.sherpa_onnx_asr"),
    }

    @model_validator(mode="after")
    def check_asr_config(cls, values: "ASRConfig", info: ValidationInfo):
        asr_model = values.asr_model
        if asr_model == "azure_asr" and not values.azure_asr:
            raise ValueError("Azure ASR configuration is required when using azure_asr")
        elif asr_model == "faster_whisper" and not values.faster_whisper:
            raise ValueError("Faster Whisper configuration is required when using faster_whisper")
        elif asr_model == "whisper_cpp" and not values.whisper_cpp:
            raise ValueError("WhisperCPP configuration is required when using whisper_cpp")
        elif asr_model == "whisper" and not values.whisper:
            raise ValueError("Whisper configuration is required when using whisper")
        elif asr_model == "fun_asr" and not values.fun_asr:
            raise ValueError("FunASR configuration is required when using fun_asr")
        elif asr_model == "groq_whisper_asr" and not values.groq_whisper_asr:
            raise ValueError("Groq Whisper ASR configuration is required when using groq_whisper_asr")
        elif asr_model == "sherpa_onnx_asr" and not values.sherpa_onnx_asr:
            raise ValueError("Sherpa Onnx ASR configuration is required when using sherpa_onnx_asr")
        return values
