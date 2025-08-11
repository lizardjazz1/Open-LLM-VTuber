# Make this directory a package for route modules

# Re-export route factories from base_routes to avoid circular import
from ..base_routes import (
    init_client_ws_route,  # noqa: F401
    init_webtool_routes,  # noqa: F401
    init_proxy_route,  # noqa: F401
)

# Optional: re-export twitch routes factory for convenience
try:
    from .twitch_routes import init_twitch_routes  # noqa: F401
except Exception:
    pass

# // DEBUG: [FIXED] Export log ingestion routes | Ref: 4
try:
    from .log_routes import init_log_routes  # noqa: F401
except Exception:
    pass

