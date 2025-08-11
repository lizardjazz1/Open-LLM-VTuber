"""
Simple i18n system for Open-LLM-VTuber
======================================

Automatically loads translations from locales/*.json files.
Just set the language in config and it works!
"""

import json
from pathlib import Path
from typing import Dict, Any

_language_data: Dict[str, Any] = {}
_current_language: str = "en"


def load_language(language: str) -> bool:
    """Load language JSON file into memory."""
    global _language_data, _current_language
    try:
        lang_file = Path(__file__).parent.parent.parent / "locales" / f"{language}.json"
        with open(lang_file, "r", encoding="utf-8") as f:
            _language_data = json.load(f)
        _current_language = language
        return True
    except Exception as e:
        print(f"Failed to load {lang_file}: {e}")
        return False


def t(key: str, **kwargs) -> str:
    """Translate a key using current language data, with formatting."""
    try:
        raw = _language_data
        for part in key.split("."):
            raw = raw[part]
        if isinstance(raw, str):
            return raw.format(**kwargs)
        return str(raw)
    except Exception:
        return key


def set_language(language: str) -> bool:
    """Set current language; fallback to English on failure."""
    ok = load_language(language)
    if not ok:
        load_language("en")
        print(f"Language '{language}' not found, using English")
        return False
    return True


def get_language() -> str:
    """Get current language."""
    return _current_language


def get_available_languages() -> list[str]:
    """Get list of available languages."""
    return []


__all__ = ["t", "set_language", "get_language", "get_available_languages"]
