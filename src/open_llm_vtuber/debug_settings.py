import os
import yaml
from loguru import logger
from typing import Tuple

_CONFIG = None
_WS_SINK_ADDED = False
_LLM_SINK_ADDED = False


def load_debug_config() -> dict:
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    path = os.path.join(os.getcwd(), "debug.yaml")
    cfg = {
        "ws_log": False,
        "llm_log": False,
        "logs_dir": "logs",
        "rotation": "10 MB",
        "retention": 5,
    }
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                cfg.update(
                    {
                        "ws_log": bool(loaded.get("ws_log", cfg["ws_log"])),
                        "llm_log": bool(loaded.get("llm_log", cfg["llm_log"])),
                        "logs_dir": str(loaded.get("logs_dir", cfg["logs_dir"])),
                        "rotation": loaded.get("rotation", cfg["rotation"]),
                        "retention": loaded.get("retention", cfg["retention"]),
                    }
                )
    except Exception:
        pass

    return cfg


def ensure_log_sinks() -> Tuple[bool, bool]:
    global _WS_SINK_ADDED, _LLM_SINK_ADDED
    cfg = load_debug_config()
    ws_on = cfg.get("ws_log", False)
    llm_on = cfg.get("llm_log", False)

    # // DEBUG: [FIXED] Centralize logging: remove per-module file sinks | Ref: 2
    # We no longer add separate file sinks here. Use JSONL central sink in run_server.
    # Keep return flags to allow modules to toggle extra debug fields if needed.

    if ws_on and not _WS_SINK_ADDED:
        try:
            logger.debug("WS logging enabled via debug.yaml (centralized)")
            _WS_SINK_ADDED = True
        except Exception:
            pass

    if llm_on and not _LLM_SINK_ADDED:
        try:
            logger.debug("LLM logging enabled via debug.yaml (centralized)")
            _LLM_SINK_ADDED = True
        except Exception:
            pass

    return ws_on, llm_on
