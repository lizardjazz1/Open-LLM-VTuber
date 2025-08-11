"""Vtuber memory module package initializer.

Provides an interface and a default service wrapper around the existing
memory backend, enabling future providers like MemGPT/Letta without
breaking current code paths.
"""

from .interface import VtuberMemoryInterface  # noqa: F401
from .service import VtuberMemoryService  # noqa: F401
