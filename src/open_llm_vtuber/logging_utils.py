# -*- coding: utf-8 -*-
r"""Logging utilities for centralized structured logging.

Provides:
- ContextVar-based request_id propagation
- Stdlib logging bridge to loguru
- Helpers for truncation/sampling and hashing
- Secret masking for inbound client logs
"""

# // DEBUG: [FIXED] Add context/request_id utilities and stdlib bridge | Ref: 1,5,9,10,15

from __future__ import annotations

import hashlib
import json
import logging
import re
from contextvars import ContextVar
from typing import Any, Callable

from loguru import logger

# Context var for request correlation
_request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


def set_request_id(request_id: str | None) -> None:
    """Set the current request id for logging correlation."""
    _request_id_ctx.set(request_id)


def get_request_id() -> str | None:
    """Get the current request id for logging correlation."""
    return _request_id_ctx.get()


class InterceptHandler(logging.Handler):
    """Redirect stdlib logging records to loguru."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        try:
            level = logger.level(record.levelname).name
        except Exception:
            level = "INFO"
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1
        logger.bind(component="stdlib", src_logger=record.name).opt(
            depth=depth, exception=record.exc_info
        ).log(level, record.getMessage())


def configure_stdlib_bridge() -> None:
    """Bridge stdlib root and common third-party loggers to loguru."""
    # Root logger
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi", "asyncio"):
        logging.getLogger(name).handlers = [InterceptHandler()]
        logging.getLogger(name).propagate = False

    # // DEBUG: [FIXED] Silence ChromaDB telemetry handler noise | Ref: 14 (ChromaDB)
    for name in ("chromadb.telemetry", "chromadb"):
        lg = logging.getLogger(name)
        try:
            lg.handlers = []
        except Exception:
            pass
        lg.propagate = False
        lg.disabled = True


def truncate_and_hash(text: str, limit_bytes: int = 4096) -> dict[str, Any]:
    """Return sampling metadata with truncation to limit_bytes and hash for identity."""
    try:
        encoded = text.encode("utf-8", errors="ignore")
    except Exception:
        encoded = str(text).encode("utf-8", errors="ignore")
    if len(encoded) <= limit_bytes:
        return {
            "input_truncated": text,
            "input_hash": hashlib.sha256(encoded).hexdigest()[:8],
            "truncated": False,
        }
    preview = encoded[:limit_bytes]
    try:
        preview_text = preview.decode("utf-8", errors="ignore")
    except Exception:
        preview_text = str(text)[:limit_bytes]
    return {
        "input_truncated": f"{preview_text}... [truncated: {len(encoded) // 1024}KB]",
        "input_hash": hashlib.sha256(encoded).hexdigest()[:8],
        "truncated": True,
    }


_SECRET_RE = re.compile(r"(token|key|secret)[a-z0-9_\-]*", re.IGNORECASE)


def mask_secrets(data: Any) -> Any:
    r"""Mask obvious secrets in nested structures by key names.

    Replaces values for keys matching /(token|key|secret)\w*/i with '***'.
    """
    try:
        if isinstance(data, dict):
            return {
                k: ("***" if _SECRET_RE.search(k) else mask_secrets(v))
                for k, v in data.items()
            }
        if isinstance(data, list):
            return [mask_secrets(v) for v in data]
        if isinstance(data, (str, bytes)):
            return data
        return data
    except Exception:
        # Fallback to safe JSON string with masked keys when possible
        try:
            s = json.dumps(data, ensure_ascii=False)
            return _SECRET_RE.sub("***", s)
        except Exception:
            return data


def bind_component(component: str) -> Callable[[Any], Any]:
    """Return a function to bind component to logger easily."""

    def _bind(obj: Any) -> Any:
        return logger.bind(component=component)

    return _bind
