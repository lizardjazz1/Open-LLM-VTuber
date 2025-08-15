from __future__ import annotations

from pathlib import Path
import json

"""
Vtuber Memory configuration defaults.

These defaults can be overridden by values in SystemConfig/CharacterConfig.
Additionally, if present, config/memory_settings.json will override these defaults at import time.

- Two ChromaDB collections are used: session and long-term (ltm)
- Session entries are considered temporary and subject to TTL pruning
- Storage path is colocated under the vtuber_memory package by default
"""

# Default storage directory for ChromaDB
CHROMA_PERSIST_DIR: str = str(
    Path(__file__).parent.joinpath("chroma_storage").resolve()
)

# Default collection names
CHROMA_SESSION_COLLECTION: str = "vtuber_session"
CHROMA_LTM_COLLECTION: str = "vtuber_ltm"

# Default session TTL: 7 days (in seconds)
SESSION_TTL_SEC: int = 7 * 24 * 60 * 60

# Default STM window (minutes)
DEFAULT_STM_WINDOW_MINUTES: int = 20

# Optional: Load overrides from config/memory_settings.json
try:
    repo_root = Path(__file__).resolve().parents[3]
except Exception:
    repo_root = Path(__file__).resolve().parents[2]
settings_path = repo_root.joinpath("config", "memory_settings.json")
if settings_path.exists():
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        CHROMA_PERSIST_DIR = str(
            Path(data.get("chroma_persist_dir", CHROMA_PERSIST_DIR))
            .expanduser()
            .resolve()
        )
        CHROMA_SESSION_COLLECTION = str(
            data.get("chroma_session_collection", CHROMA_SESSION_COLLECTION)
        )
        CHROMA_LTM_COLLECTION = str(
            data.get("chroma_ltm_collection", CHROMA_LTM_COLLECTION)
        )
        SESSION_TTL_SEC = int(data.get("session_ttl_sec", SESSION_TTL_SEC))
        DEFAULT_STM_WINDOW_MINUTES = int(
            data.get("default_stm_window_minutes", DEFAULT_STM_WINDOW_MINUTES)
        )
    except Exception:
        # Ignore malformed file and keep defaults
        pass
