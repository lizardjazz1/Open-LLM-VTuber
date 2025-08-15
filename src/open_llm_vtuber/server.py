"""
Open-LLM-VTuber Server
========================
This module contains the WebSocket server for Open-LLM-VTuber, which handles
the WebSocket connections, serves static files, and manages the web tool.
It uses FastAPI for the server and Starlette for static file serving.
"""

import os
import shutil

from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response
from starlette.staticfiles import StaticFiles as StarletteStaticFiles
from starlette.templating import Jinja2Templates

from .routes import init_client_ws_route, init_webtool_routes, init_proxy_route
from .routes import init_twitch_routes

# // DEBUG: [FIXED] Include /logs endpoint | Ref: 4
from .routes import init_log_routes
from .routes.admin_memory_routes import router as admin_memory_router
from .routes.admin_bench_routes import router as admin_bench_router
from .routes.admin_status_routes import router as admin_status_router
from .service_context import ServiceContext
from .config_manager.utils import Config

# // DEBUG: [FIXED] SlowAPI middleware | Ref: 4
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address


# Create a custom StaticFiles class that adds CORS headers
class CORSStaticFiles(StarletteStaticFiles):
    """
    Static files handler that adds CORS headers to all responses.
    Needed because Starlette StaticFiles might bypass standard middleware.
    """

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)

        # Add CORS headers to all responses
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"

        if path.endswith(".js"):
            response.headers["Content-Type"] = "application/javascript"

        return response


class AvatarStaticFiles(CORSStaticFiles):
    """
    Avatar files handler with security restrictions and CORS headers
    """

    async def get_response(self, path: str, scope):
        allowed_extensions = (".jpg", ".jpeg", ".png", ".gif", ".svg")
        if not any(path.lower().endswith(ext) for ext in allowed_extensions):
            return Response("Forbidden file type", status_code=403)
        response = await super().get_response(path, scope)
        return response


class WebSocketServer:
    """
    API server for Open-LLM-VTuber. This contains the websocket endpoint for the client, hosts the web tool, and serves static files.

    Creates and configures a FastAPI app, registers all routes
    (WebSocket, web tools, proxy) and mounts static assets with CORS.

    Args:
        config (Config): Application configuration containing system settings.
        default_context_cache (ServiceContext, optional):
            Pre‑initialized service context for sessions' service context to reference to.
            **If omitted, `initialize()` method needs to be called to load service context.**

    Notes:
        - If default_context_cache is omitted, call `await initialize()` to load service context cache.
        - Use `clean_cache()` to clear and recreate the local cache directory.
    """

    def __init__(self, config: Config, default_context_cache: ServiceContext = None):
        self.app = FastAPI(title="Open-LLM-VTuber Server")  # Added title for clarity
        self.config = config
        self.default_context_cache = (
            default_context_cache or ServiceContext()
        )  # Use provided context or initialize a new empty one waiting to be loaded
        # It will be populated during the initialize method call

        # // DEBUG: [FIXED] Add SlowAPI middleware for rate limiting | Ref: 4
        limiter = Limiter(key_func=get_remote_address)
        self.app.state.limiter = limiter
        self.app.add_middleware(SlowAPIMiddleware)
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

        # Add global CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Install request-id propagation middleware for HTTP routes
        try:
            from .middleware.request_id import install_request_id_middleware

            install_request_id_middleware(self.app)
        except Exception:
            pass

        # Include routes, passing the context instance
        # The context will be populated during the initialize step
        self.app.include_router(
            init_client_ws_route(default_context_cache=self.default_context_cache),
        )
        self.app.include_router(
            init_webtool_routes(default_context_cache=self.default_context_cache),
        )

        # Liveness probe
        @self.app.get("/healthz")
        async def _healthz():  # type: ignore[func-returns-value]
            return {"ok": True}

        # Enable client log ingestion (token optional per config flags)
        self.app.include_router(init_log_routes(config=config, limiter=limiter))
        self.app.include_router(
            init_twitch_routes(default_context_cache=self.default_context_cache),
        )
        # Admin memory and relationship management endpoints
        self.app.include_router(admin_memory_router)
        # Admin benchmarking endpoints
        self.app.include_router(admin_bench_router)
        # Admin status endpoints
        self.app.include_router(admin_status_router)

        # Install WS guard middleware: only explicit WS endpoints are allowed
        try:
            from .middleware.ws_guard import WebSocketPathGuard

            allowed_ws_paths = {"/client-ws", "/proxy-ws", "/tts-ws"}
            self.app.add_middleware(WebSocketPathGuard, allowed_paths=allowed_ws_paths)
        except Exception:
            # If middleware import fails, continue without it
            pass

        # Initialize and include proxy routes if proxy is enabled
        system_config = config.system_config
        if hasattr(system_config, "enable_proxy") and system_config.enable_proxy:
            # Construct the server URL for the proxy
            host = system_config.host
            port = system_config.port
            server_url = f"ws://{host}:{port}/client-ws"
            self.app.include_router(
                init_proxy_route(server_url=server_url),
            )

        # Mount cache directory first (to ensure audio file access)
        if not os.path.exists("cache"):
            os.makedirs("cache")
        self.app.mount(
            "/cache",
            CORSStaticFiles(directory="cache"),
            name="cache",
        )

        # Mount static files with CORS-enabled handlers
        self.app.mount(
            "/live2d-models",
            CORSStaticFiles(directory="live2d-models"),
            name="live2d-models",
        )
        self.app.mount(
            "/bg",
            CORSStaticFiles(directory="backgrounds"),
            name="backgrounds",
        )
        self.app.mount(
            "/avatars",
            AvatarStaticFiles(directory="avatars"),
            name="avatars",
        )

        # Mount web tool directory separately from frontend
        self.app.mount(
            "/web-tool",
            CORSStaticFiles(directory="web_tool", html=True),
            name="web_tool",
        )

        # Mount Python-based alternative frontend static files
        # Served at /py-static and templates rendered at /py
        try:
            py_static_dir = "frontend_py/static"
            py_templates_dir = "frontend_py/templates"
            if os.path.isdir(py_static_dir):
                self.app.mount(
                    "/py-static",
                    CORSStaticFiles(directory=py_static_dir),
                    name="py-static",
                )
            # Initialize Jinja2 templates (requires jinja2, included via fastapi[standard])
            self._py_templates = Jinja2Templates(directory=py_templates_dir)

            async def _py_index(request: Request):
                # Render the Python frontend entry page
                return self._py_templates.TemplateResponse(
                    "index.html",
                    {
                        "request": request,
                    },
                )

            # Register /py only if templates folder exists
            if os.path.isdir(py_templates_dir):
                self.app.add_api_route("/py", _py_index, methods=["GET"])
        except Exception:
            # If Jinja2 or folders are missing, skip python frontend
            pass

        # Mount main frontend last (as catch-all)
        self.app.mount(
            "/",
            CORSStaticFiles(directory="frontend", html=True),
            name="frontend",
        )

    async def initialize(self):
        """Asynchronously load the service context from config.
        Calling this function is needed if default_context_cache was not provided to the constructor."""
        await self.default_context_cache.load_from_config(self.config)

        # Set global context for admin routes
        from .routes.admin_memory_routes import set_global_context

        set_global_context(self.default_context_cache)

    @staticmethod
    def clean_cache():
        """Clean the cache directory by removing and recreating it."""
        cache_dir = "cache"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir)
