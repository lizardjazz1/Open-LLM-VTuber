from __future__ import annotations

import io
import time
import wave
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from ..service_context import ServiceContext

router = APIRouter(prefix="/admin", tags=["admin-benchmark"])


def get_context() -> ServiceContext:
    """Obtain a service context, falling back to a lightweight server instance if needed."""
    try:
        from ..server import WebSocketServer  # type: ignore
    except Exception:
        pass
    try:
        from ..server import WebSocketServer
        from ..config_manager.utils import validate_config, read_yaml

        app_server = WebSocketServer(config=validate_config(read_yaml("conf.yaml")))
        return app_server.default_context_cache
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Service context not available: {e}"
        )


@router.post("/asr/benchmark")
async def asr_benchmark(
    file: UploadFile = File(...),
    ctx: ServiceContext = Depends(get_context),
    repeats: int = Query(1, ge=1, le=5),
) -> dict[str, Any]:
    """Benchmark ASR latency on an uploaded mono 16k PCM16 WAV using current engine.

    Args:
            file: WAV file to transcribe (mono, 16 kHz, PCM16).
            ctx: Injected service context with current ASR engine.
            repeats: Number of times to run for averaging (1-5).

    Returns:
            dict: { engine, samples, avg_latency_ms, runs: [{latency_ms, text}] }
    """
    if not ctx or not getattr(ctx, "asr_engine", None):
        raise HTTPException(status_code=503, detail="ASR engine not available")

    raw = await file.read()
    try:
        if len(raw) < 44:
            raise ValueError("Invalid WAV file: File too small")
        with wave.open(io.BytesIO(raw), "rb") as wf:
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            comptype = wf.getcomptype()
            nframes = wf.getnframes()
            if comptype != "NONE":
                raise ValueError("Unsupported WAV: compressed format not allowed")
            if sampwidth != 2:
                raise ValueError("Unsupported WAV: 16-bit PCM required")
            if nchannels != 1:
                raise ValueError("Unsupported WAV: mono channel required")
            if framerate != 16000:
                raise ValueError("Unsupported WAV: sample rate must be 16000 Hz")
            raw_audio = wf.readframes(nframes)
    except wave.Error as e:
        raise HTTPException(status_code=400, detail=f"Invalid WAV file: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    audio_array = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
    results: list[dict[str, Any]] = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        text = await ctx.asr_engine.async_transcribe_np(audio_array)
        lat_ms = int((time.perf_counter() - t0) * 1000)
        results.append({"latency_ms": lat_ms, "text": text})

    avg = int(sum(r["latency_ms"] for r in results) / len(results))
    return {
        "engine": type(ctx.asr_engine).__name__,
        "samples": int(audio_array.shape[0]),
        "avg_latency_ms": avg,
        "runs": results,
    }


@router.get("/tts/benchmark")
async def tts_benchmark(
    text: str,
    ctx: ServiceContext = Depends(get_context),
    repeats: int = Query(1, ge=1, le=5),
) -> dict[str, Any]:
    """Benchmark TTS latency on provided text using current engine.

    Args:
        text: Text to synthesize
        ctx: Injected service context with current TTS engine
        repeats: Number of times to run (1-5)

    Returns:
        dict: { engine, avg_latency_ms, runs: [{latency_ms, audio_path}] }
    """
    if not ctx or not getattr(ctx, "tts_engine", None):
        raise HTTPException(status_code=503, detail="TTS engine not available")

    results: list[dict[str, Any]] = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        file_name = f"bench_{int(time.time() * 1000)}"
        path = await ctx.tts_engine.async_generate_audio(
            text=text, file_name_no_ext=file_name
        )
        lat_ms = int((time.perf_counter() - t0) * 1000)
        results.append({"latency_ms": lat_ms, "audio_path": path})
        # Cleanup after measuring
        try:
            ctx.tts_engine.remove_file(path)
        except Exception:
            pass

    avg = int(sum(r["latency_ms"] for r in results) / len(results))
    return {
        "engine": type(ctx.tts_engine).__name__,
        "avg_latency_ms": avg,
        "runs": results,
    }
