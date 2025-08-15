import os
import json
from uuid import uuid4
import numpy as np
from datetime import datetime
import time
import io
import wave
import contextlib
import re as _re
import time as _time
from collections import deque
from fastapi import APIRouter, WebSocket, UploadFile, File, Response, Request
from starlette.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect
from loguru import logger
from .service_context import ServiceContext
from .websocket_handler import WebSocketHandler
from .proxy_handler import ProxyHandler

# // DEBUG: [FIXED] Add masking helper | Ref: 4,15
from .logging_utils import set_request_id, truncate_and_hash

# ===== TTS WS performance constants (module-level to avoid reallocation) =====
_SENTENCE_SPLIT_RE = _re.compile(r"(?<=[\.!\?])\s+")
_VOICE_CMD_RE = _re.compile(
    r"\{rate:(?:\+|\-)?\d+%\}|\{volume:(?:\+|\-)?\d+%\}|\{pitch:(?:\+|\-)?\d+Hz\}"
)
_MAX_TTS_TEXT_LEN = 2000  # characters
_MAX_SENTENCES = 8
_RATE_LIMIT_MAX = 5  # max messages/2s per client
_RATE_LIMIT_WINDOW_SEC = 2.0


def _split_sentences_fast(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s and s.strip()]


def _extract_voice_cmds_fast(sentence: str) -> list[str]:
    # Avoid regex unless commands marker is present
    return _VOICE_CMD_RE.findall(sentence) if "{" in sentence else []


def init_client_ws_route(default_context_cache: ServiceContext) -> APIRouter:
    """Create WebSocket routes for client connections.

    Args:
        default_context_cache (ServiceContext): Default pre-initialized service context used as a template for new sessions.

    Returns:
        APIRouter: Router exposing the `/client-ws` endpoint that accepts WebSocket connections from the frontend client.
    """

    router = APIRouter()
    ws_handler = WebSocketHandler(default_context_cache)

    @router.websocket("/client-ws")
    async def websocket_endpoint(websocket: WebSocket):
        """Handle lifecycle and messaging for a single client WebSocket.

        Args:
            websocket (WebSocket): The incoming WebSocket connection from the client UI.

        Returns:
            None: Keeps the connection alive until the client disconnects; logs lifecycle events and errors.
        """
        await websocket.accept()
        client_uid = str(uuid4())

        logger.bind(component="ws").info(
            {
                "event": "ws.open",
                "client_uid": client_uid,
                "route": "/client-ws",
                "status": "accepted",
            }
        )

        try:
            await ws_handler.handle_new_connection(websocket, client_uid)
            await ws_handler.handle_websocket_communication(websocket, client_uid)
        except WebSocketDisconnect:
            logger.bind(component="ws").info(
                {
                    "event": "ws.disconnect",
                    "client_uid": client_uid,
                    "route": "/client-ws",
                    "status": "closed",
                }
            )
            await ws_handler.handle_disconnect(client_uid)
        except Exception as e:
            logger.bind(component="ws").error(
                {
                    "event": "ws.error",
                    "client_uid": client_uid,
                    "route": "/client-ws",
                    "status": "error",
                    "error_details": str(e),
                }
            )
            await ws_handler.handle_disconnect(client_uid)
            raise

    return router


def init_proxy_route(server_url: str) -> APIRouter:
    """Create WebSocket routes for a reverse-proxy to another backend.

    Args:
        server_url (str): Target server WebSocket URL to proxy to.

    Returns:
        APIRouter: Router exposing the `/proxy-ws` endpoint that forwards messages to the target server.
    """
    router = APIRouter()
    proxy_handler = ProxyHandler(server_url)

    @router.websocket("/proxy-ws")
    async def proxy_endpoint(websocket: WebSocket):
        """Forward a WebSocket session to the configured upstream server.

        Args:
            websocket (WebSocket): The client WebSocket to be proxied.

        Returns:
            None: Runs until the session ends; errors are logged and re-raised.
        """
        try:
            await proxy_handler.handle_client_connection(websocket)
        except Exception as e:
            logger.error(f"Error in proxy connection: {e}")
            raise

    return router


def init_webtool_routes(default_context_cache: ServiceContext) -> APIRouter:
    """Create HTTP and WS routes for web tools and Live2D helper endpoints.

    Args:
        default_context_cache (ServiceContext): Default service context used to serve model/config info and ASR/TTS services.

    Returns:
        APIRouter: Router exposing utility HTTP endpoints and TTS/ASR WebSockets.
    """

    router = APIRouter()
    # Simple in-process rate limiter for /tts-ws keyed by client IP
    _tts_ws_rate: dict[str, deque[float]] = {}

    @router.get("/canvas-only")
    async def canvas_only_page(_: Request) -> Response:
        """Serve a Live2D-only page (no UI) with opaque background and floating subtitles."""
        try:
            model_info = (
                getattr(default_context_cache.live2d_model, "model_info", {}) or {}
            )
            model_json_url = str(model_info.get("url") or "")
            if not model_json_url:
                name = (
                    getattr(
                        default_context_cache.character_config, "live2d_model_name", ""
                    )
                    or ""
                )
                if name:
                    model_json_url = f"/live2d-models/{name}/runtime/{name}.model3.json"
        except Exception:
            model_json_url = ""

        html = f"""<!doctype html>
<html lang=\"ru\">\n<head>\n<meta charset=\"utf-8\"/>\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>\n<title>VTuber Canvas Only</title>\n<style>
  html, body, #stage {{ margin:0; padding:0; width:100%; height:100%; overflow:hidden; }}
  body {{ background:#000; color:#fff; font:14px system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif; }}
  #bg {{ position: absolute; inset: 0; background-position: center; background-size: cover; background-repeat: no-repeat; }}
  #stage {{ position: relative; display:block; z-index: 1; }}
  #subtitle {{
    position: absolute;
    left: 50%;
    bottom: 6vh;
    transform: translateX(-50%);
    max-width: min(90vw, 1200px);
    padding: 12px 16px;
    border-radius: 12px;
    background: rgba(0,0,0,0.55);
    color: #fff;
    box-shadow: 0 6px 24px rgba(0,0,0,0.35);
    backdrop-filter: blur(2px);
    line-height: 1.35;
    text-align: center;
    pointer-events: auto;
    user-select: none;
    cursor: move;
    z-index: 2;
  }}
  #subtitle.hidden {{ display:none; }}
  #subtitle .label {{ opacity: 0.65; font-size: 12px; margin-right: 8px; }}
  #subtitle .text {{ font-size: clamp(14px, 2.4vw, 22px); white-space: pre-wrap; }}
</style>\n</head>\n<body>\n<div id=\"bg\"></div>\n<canvas id=\"stage\"></canvas>\n<div id=\"subtitle\" class=\"hidden\"><span class=\"label\">Нейри:</span><span class=\"text\"></span></div>\n<script>
  async function ensureLibs(){{
    function load(src){{ return new Promise((res, rej) => {{ const s=document.createElement('script'); s.src=src; s.crossOrigin='anonymous'; s.onload=()=>res(); s.onerror=()=>rej(new Error('load failed: '+src)); document.head.appendChild(s); }}); }}
    if (!window.PIXI) {{
      try {{ await load('https://cdn.jsdelivr.net/npm/pixi.js@7/dist/pixi.min.js'); }} catch (e) {{ await load('https://unpkg.com/pixi.js@7/dist/pixi.min.js'); }}
    }}
    if (!(window.PIXI && (window.PIXI).live2d)) {{
      try {{ await load('https://cdn.jsdelivr.net/npm/pixi-live2d-display@0.5.4/dist/index.min.js'); }} catch (e) {{ await load('https://unpkg.com/pixi-live2d-display@0.5.4/dist/index.min.js'); }}
    }}
  }}

  (async function() {{
    const modelUrl = {json.dumps(model_json_url)};

    // Resolve background image: query param ?bg=, then localStorage, then default
    const params = new URLSearchParams(location.search);
    const defaultBg = '/bg/ceiling-window-room-night.jpeg';
    let bgUrl = params.get('bg');
    if (!bgUrl) {{ try {{ bgUrl = localStorage.getItem('backgroundUrl') || ''; }} catch (_) {{ bgUrl=''; }} }}
    if (!bgUrl) bgUrl = defaultBg;
    try {{ document.getElementById('bg').style.backgroundImage = 'url(\'' + bgUrl + '\')'; }} catch(_) {{}}

    await ensureLibs();

    const app = new PIXI.Application({{ view: document.getElementById('stage'), resizeTo: window, transparent: true, antialias: true }});
    const sub = document.getElementById('subtitle');
    const subText = sub.querySelector('.text');

    // Remove voice commands and emotion tags for display (e.g., rate/volume/pitch and [joy], [thinking])
    function cleanText(input){{
      try {{
        let s = String(input || '');
        // Strip voice command tokens
        s = s.replace(/\{{\s*(?:rate|volume):[^}}]*\}}/gi, ' ');
        s = s.replace(/\{{\s*pitch:[^}}]*\}}/gi, ' ');
        // Strip known emotion tags like [joy], [thinking], etc.
        s = s.replace(/\[(neutral|joy|smile|laugh|anger|disgust|fear|sadness|surprise|confused|thinking|excited|shy|wink|indignation)\]/gi, ' ');
        // Collapse whitespace
        s = s.replace(/\s+/g, ' ').trim();
        return s;
      }} catch(_) {{ return String(input || ''); }}
    }}

    (function enableDrag(el){{
      let dragging = false, sx = 0, sy = 0, lx = 0, ly = 0;
      const onDown = (e) => {{ dragging = true; sx = (e.touches?e.touches[0].clientX:e.clientX); sy = (e.touches?e.touches[0].clientY:e.clientY); const rect = el.getBoundingClientRect(); lx = rect.left; ly = rect.top; document.body.style.userSelect = 'none'; }};
      const onMove = (e) => {{ if(!dragging) return; const cx = (e.touches?e.touches[0].clientX:e.clientX); const cy = (e.touches?e.touches[0].clientY:e.clientY); const nx = lx + (cx - sx); const ny = ly + (cy - sy); el.style.left = nx + 'px'; el.style.top = ny + 'px'; el.style.bottom = 'auto'; el.style.transform = 'translateX(0)'; }};
      const onUp = () => {{ dragging = false; document.body.style.userSelect = ''; }};
      el.addEventListener('mousedown', onDown); el.addEventListener('touchstart', onDown, {{passive:true}});
      window.addEventListener('mousemove', onMove); window.addEventListener('touchmove', onMove, {{passive:true}});
      window.addEventListener('mouseup', onUp); window.addEventListener('touchend', onUp);
    }})(sub);

    function setSubtitle(text){{
      const t = cleanText(text);
      if (t.trim()) {{ subText.textContent = t; sub.classList.remove('hidden'); }}
      else {{ sub.classList.add('hidden'); subText.textContent = ''; }}
    }}

    try {{
      if (!modelUrl) {{ throw new Error('Model URL is empty'); }}
      const {{ Live2DModel }} = PIXI.live2d || {{}};
      if (!Live2DModel) throw new Error('PIXI.live2d is undefined');
      const model = await Live2DModel.from(modelUrl);
      app.stage.addChild(model);
      const fit = () => {{
        const w = app.renderer.width, h = app.renderer.height;
        const scale = Math.min(w / (model.width || 1), h / (model.height || 1)) * 0.9;
        model.scale.set(scale);
        model.x = (w - model.width * scale) * 0.5;
        model.y = (h - model.height * scale) * 0.5;
      }};
      window.addEventListener('resize', fit);
      fit();
    }} catch (e) {{
      console.error('Failed to load Live2D model:', e);
      const el = document.createElement('div');
      el.style.cssText = 'position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#f55;font:14px system-ui;z-index:3;';
      el.textContent = 'Failed to load Live2D model. Check server logs';
      document.body.appendChild(el);
    }}

    try {{
      const proto = (location.protocol === 'https:') ? 'wss' : 'ws';
      const ws = new WebSocket(proto + '://' + location.host + '/client-ws');
      ws.onmessage = (ev) => {{
        try {{
          const data = JSON.parse(ev.data);
          if (!data) return;
          if (data.type === 'partial-text' && typeof data.text === 'string') {{
            setSubtitle(data.text);
          }} else if (data.type === 'full-text' && typeof data.text === 'string') {{
            setSubtitle(data.text);
          }} else if (data.type === 'audio' && data.display_text && typeof data.display_text.text === 'string') {{
            setSubtitle(data.display_text.text);
          }} else if (data.type === 'control' && data.text === 'conversation-chain-end') {{
            // Keep last line
          }}
        }} catch (_) {{}}
      }};
      ws.onerror = () => {{}};
    }} catch (_) {{}}
  }})();
</script>\n</body>\n</html>"""
        return Response(content=html, media_type="text/html")

    @router.get("/web-tool")
    async def web_tool_redirect():
        """Redirect `/web-tool` to static index page.

        Returns:
            Response: 302 redirect to `/web-tool/index.html`.
        """
        return Response(status_code=302, headers={"Location": "/web-tool/index.html"})

    @router.get("/web_tool")
    async def web_tool_redirect_alt():
        """Redirect legacy `/web_tool` path to static index page.

        Returns:
            Response: 302 redirect to `/web-tool/index.html`.
        """
        return Response(status_code=302, headers={"Location": "/web-tool/index.html"})

    @router.get("/live2d-models/info")
    async def get_live2d_folder_info():
        """List available Live2D models on the server.

        Returns:
            JSONResponse: JSON with fields `type`, `count`, and an array of `characters` with `name`, `avatar`, and `model_path`.
        """
        live2d_dir = "live2d-models"
        if not os.path.exists(live2d_dir):
            return JSONResponse(
                {"error": "Live2D models directory not found"}, status_code=404
            )

        valid_characters = []
        supported_extensions = [".png", ".jpg", ".jpeg"]

        for entry in os.scandir(live2d_dir):
            if entry.is_dir():
                folder_name = entry.name.replace("\\", "/")
                model3_file = os.path.join(
                    live2d_dir, folder_name, f"{folder_name}.model3.json"
                ).replace("\\", "/")

                if os.path.isfile(model3_file):
                    # Find avatar file if it exists
                    avatar_file = None
                    for ext in supported_extensions:
                        avatar_path = os.path.join(
                            live2d_dir, folder_name, f"{folder_name}{ext}"
                        )
                        if os.path.isfile(avatar_path):
                            avatar_file = avatar_path.replace("\\", "/")
                            break

                    valid_characters.append(
                        {
                            "name": folder_name,
                            "avatar": avatar_file,
                            "model_path": model3_file,
                        }
                    )
        return JSONResponse(
            {
                "type": "live2d-models/info",
                "count": len(valid_characters),
                "characters": valid_characters,
            }
        )

    @router.post("/asr")
    async def transcribe_audio(request: Request, file: UploadFile = File(...)):
        """Transcribe a mono 16 kHz PCM16 WAV file to text.

        Args:
            request (Request): Incoming HTTP request, used only for logging and metrics.
            file (UploadFile): Uploaded WAV file. Must be uncompressed PCM16, mono, 16 kHz.

        Returns:
            dict | Response: On success returns `{ "text": str, "request_id": str }`. On validation error returns 400 JSON.

        Raises:
            ValueError: If the WAV file is invalid or not PCM16/mono/16kHz.
        """
        # // DEBUG: [FIXED] Generate request_id for HTTP route | Ref: 5,11
        rid = str(uuid4())
        set_request_id(rid)
        start_ns = time.perf_counter_ns()
        logger.bind(component="api").info(
            {
                "event": "asr.request",
                "route": "/asr",
                "status": "start",
                "request_id": rid,
            }
        )

        try:
            contents = await file.read()

            # Enforce maximum payload size (10 MB)
            max_bytes = 10 * 1024 * 1024
            if len(contents) > max_bytes:
                latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
                logger.bind(component="api").error(
                    {
                        "event": "asr.error",
                        "route": "/asr",
                        "status": "too_large",
                        "payload_size": len(contents),
                        "latency_ms": round(latency_ms, 2),
                    }
                )
                return Response(
                    content=json.dumps({"error": "Payload too large (>10 MB)"}),
                    status_code=413,
                    media_type="application/json",
                )

            # Validate minimum file size
            if len(contents) < 44:  # Minimum WAV header size
                raise ValueError("Invalid WAV file: File too small")

            # Robust WAV parsing and validation
            try:
                with contextlib.closing(wave.open(io.BytesIO(contents), "rb")) as wf:
                    nchannels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    framerate = wf.getframerate()
                    comptype = wf.getcomptype()
                    nframes = wf.getnframes()
                    if comptype != "NONE":
                        raise ValueError(
                            "Unsupported WAV: compressed format not allowed"
                        )
                    if sampwidth != 2:
                        raise ValueError("Unsupported WAV: 16-bit PCM required")
                    if nchannels != 1:
                        raise ValueError("Unsupported WAV: mono channel required")
                    if framerate != 16000:
                        raise ValueError(
                            "Unsupported WAV: sample rate must be 16000 Hz"
                        )
                    raw_audio = wf.readframes(nframes)
            except wave.Error as e:
                raise ValueError(f"Invalid WAV file: {str(e)}")

            # Convert to float32
            audio_array = (
                np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
            )

            # Validate audio data
            if len(audio_array) == 0:
                raise ValueError("Empty audio data")

            text = await default_context_cache.asr_engine.async_transcribe_np(
                audio_array
            )
            sampled = truncate_and_hash(text)
            latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            logger.bind(component="api").info(
                {
                    "event": "asr.success",
                    "route": "/asr",
                    "status": "success",
                    "payload_size": len(contents),
                    "latency_ms": round(latency_ms, 2),
                    **sampled,
                }
            )
            return {"text": text, "request_id": rid}

        except ValueError as e:
            latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            logger.bind(component="api").error(
                {
                    "event": "asr.error",
                    "route": "/asr",
                    "status": "error",
                    "error_details": str(e),
                    "latency_ms": round(latency_ms, 2),
                }
            )
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=400,
                media_type="application/json",
            )
        except Exception as e:
            latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            logger.bind(component="api").error(
                {
                    "event": "asr.error",
                    "route": "/asr",
                    "status": "error",
                    "error_details": str(e),
                    "latency_ms": round(latency_ms, 2),
                }
            )
            return Response(
                content=json.dumps(
                    {"error": "Internal server error during transcription"}
                ),
                status_code=500,
                media_type="application/json",
            )

    @router.websocket("/tts-ws")
    async def tts_endpoint(websocket: WebSocket):
        """Stream TTS audio for incoming text over WebSocket.

        Args:
            websocket (WebSocket): Client connection sending JSON with `text` to synthesize.

        Returns:
            None: Sends partial audio paths and a completion message to the client.
        """
        await websocket.accept()
        # Resolve client key for rate limiting
        try:
            client_ip = getattr(getattr(websocket, "client", None), "host", "unknown")
        except Exception:
            client_ip = "unknown"
        logger.bind(component="tts").info(
            {
                "event": "tts.ws.open",
                "route": "/tts-ws",
                "status": "accepted",
                "client_ip": client_ip,
            }
        )

        try:
            while True:
                data = await websocket.receive_json()
                text = data.get("text")
                if not text:
                    continue

                # TTS engine availability (fast check)
                if not getattr(default_context_cache, "tts_engine", None):
                    await websocket.send_json(
                        {"status": "error", "message": "TTS engine not available"}
                    )
                    continue

                # Max input length safeguard
                text_str = text if isinstance(text, str) else str(text)
                if len(text_str) > _MAX_TTS_TEXT_LEN:
                    await websocket.send_json(
                        {
                            "status": "error",
                            "message": f"Text too long (> {_MAX_TTS_TEXT_LEN} chars)",
                        }
                    )
                    continue

                # Rate limiting per client
                try:
                    now = _time.monotonic()
                    buf = _tts_ws_rate.get(client_ip)
                    if buf is None:
                        buf = deque(maxlen=_RATE_LIMIT_MAX)
                        _tts_ws_rate[client_ip] = buf
                    # Drop old entries from left
                    while buf and (now - buf[0]) > _RATE_LIMIT_WINDOW_SEC:
                        buf.popleft()
                    if len(buf) >= _RATE_LIMIT_MAX:
                        await websocket.send_json(
                            {
                                "status": "error",
                                "message": "Rate limited. Try again shortly.",
                            }
                        )
                        continue
                    buf.append(now)
                except Exception:
                    pass

                logger.bind(component="tts").info(
                    {
                        "event": "tts.request",
                        "route": "/tts-ws",
                        "payload_size": len(text_str or ""),
                        "client_ip": client_ip,
                    }
                )

                # Split text into sentences (., !, ?)
                sentences_all = _split_sentences_fast(text_str)
                sentences = sentences_all[:_MAX_SENTENCES]
                truncated = len(sentences_all) > len(sentences)

                try:
                    # Generate and send audio for each sentence
                    for sentence in sentences:
                        # Ensure terminal punctuation for TTS stability
                        if sentence and sentence[-1] not in ".!?":
                            sentence = sentence + "."

                        # Extract voice commands only when present; notify segment_start only if useful
                        voice_cmds = _extract_voice_cmds_fast(sentence)
                        if voice_cmds or truncated:
                            await websocket.send_json(
                                {
                                    "status": "segment_start",
                                    "text": sentence,
                                    "voice_cmds": voice_cmds,
                                    "truncated": truncated,
                                }
                            )

                        t0 = _time.perf_counter()
                        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid4())[:8]}"
                        audio_path = (
                            await default_context_cache.tts_engine.async_generate_audio(
                                text=sentence, file_name_no_ext=file_name
                            )
                        )
                        latency_ms = int((_time.perf_counter() - t0) * 1000)
                        logger.bind(component="tts").info(
                            {
                                "event": "tts.segment_done",
                                "audio_path": audio_path,
                                "latency_ms": latency_ms,
                            }
                        )

                        await websocket.send_json(
                            {
                                "status": "partial",
                                "audioPath": audio_path,
                                "text": sentence,
                                "latency_ms": latency_ms,
                                "truncated": truncated,
                            }
                        )

                    # Send completion signal
                    await websocket.send_json({"status": "complete"})

                except Exception as e:
                    logger.bind(component="tts").error(
                        {
                            "event": "tts.error",
                            "route": "/tts-ws",
                            "status": "error",
                            "error_details": str(e),
                        }
                    )
                    await websocket.send_json({"status": "error", "message": str(e)})

        except WebSocketDisconnect:
            logger.bind(component="tts").info(
                {
                    "event": "tts.ws.disconnect",
                    "route": "/tts-ws",
                    "status": "closed",
                    "client_ip": client_ip,
                }
            )
        except Exception as e:
            logger.bind(component="tts").error(
                {
                    "event": "tts.ws.error",
                    "route": "/tts-ws",
                    "status": "error",
                    "error_details": str(e),
                }
            )
            await websocket.close()

    return router
