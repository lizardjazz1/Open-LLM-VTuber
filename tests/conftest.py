import sys
import os
import importlib
import types

# Ensure project package import by adding src to sys.path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Provide backward-compatible alias: 'src.open_llm_vtuber' -> 'open_llm_vtuber'
try:
    pkg = importlib.import_module("open_llm_vtuber")
    sys.modules["src.open_llm_vtuber"] = pkg
except Exception:
    pass

# Provide a simple stub for prompts.prompt_loader if not available
if "prompts" not in sys.modules:
    prompts_mod = types.ModuleType("prompts")

    def _load_persona(name: str) -> str:
        return ""

    def _load_util(name: str) -> str:
        return ""

    prompts_mod.prompt_loader = types.SimpleNamespace(
        load_persona=_load_persona, load_util=_load_util
    )
    sys.modules["prompts"] = prompts_mod
