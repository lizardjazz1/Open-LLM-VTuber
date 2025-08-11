from typing import Callable, Iterable


class WebSocketPathGuard:
    """ASGI middleware to close WebSocket connections on unsupported paths.

    This middleware accepts and immediately closes any WebSocket connection
    whose path is not in the allowed set. HTTP traffic is passed through
    untouched. This prevents WebSocket requests from falling into StaticFiles
    mounted at '/', which asserts on non-HTTP scopes in Starlette.

    Args:
        app: Downstream ASGI app to wrap.
        allowed_paths: Iterable of WebSocket paths that are allowed to pass through.
    """

    def __init__(self, app: Callable, allowed_paths: Iterable[str]) -> None:
        self.app = app
        self.allowed_paths = set(allowed_paths)

    async def __call__(self, scope, receive, send):  # type: ignore[no-untyped-def]
        scope_type = scope.get("type")
        if scope_type == "websocket":
            path = scope.get("path", "")
            if path not in self.allowed_paths:
                # Accept then close with policy violation
                await send({"type": "websocket.accept"})
                await send({"type": "websocket.close", "code": 1008})
                return
        # For HTTP or allowed WS, continue
        return await self.app(scope, receive, send)
