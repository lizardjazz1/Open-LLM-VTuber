from __future__ import annotations

from typing import Callable
from uuid import uuid4

from fastapi import FastAPI, Request, Response

from ..logging_utils import set_request_id


def install_request_id_middleware(app: FastAPI) -> None:
    """Install an HTTP middleware that propagates X-Request-ID.

    Args:
            app: FastAPI application instance.

    Behavior:
            - Reads X-Request-ID header if provided; otherwise generates a new UUID4.
            - Stores the value in a ContextVar for log correlation.
            - Ensures the response includes X-Request-ID header.
    """

    @app.middleware("http")
    async def _request_id_middleware(
        request: Request, call_next: Callable[[Request], Response]
    ):  # type: ignore[override]
        rid = request.headers.get("X-Request-ID") or str(uuid4())
        set_request_id(rid)
        response = await call_next(request)
        try:
            response.headers["X-Request-ID"] = rid
        except Exception:
            # If response doesn't allow header mutation, ignore silently
            pass
        return response
