"""
Simple i18n system for Open-LLM-VTuber
======================================

Automatically loads translations from locales/*.json files.
Just set the language in config and it works!
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class SimpleI18n:
    """
    Simple i18n system that automatically loads any language file.
    """
    
    def __init__(self):
        self.current_language = "en"
        self.translations: Dict[str, Dict[str, Any]] = {}
        self._load_all_translations()
    
    def _load_all_translations(self):
        """Load all translation files from locales/ directory."""
        locales_dir = Path("locales")
        if not locales_dir.exists():
            return
        
        for lang_file in locales_dir.glob("*.json"):
            try:
                with open(lang_file, "r", encoding="utf-8") as f:
                    lang_code = lang_file.stem  # filename without extension
                    self.translations[lang_code] = json.load(f)
            except Exception as e:
                print(f"Failed to load {lang_file}: {e}")
    
    def set_language(self, language: str) -> bool:
        """
        Set the current language.
        Automatically loads the language file if it exists.
        
        Args:
            language: Language code (e.g., 'ru', 'jp', 'fr')
            
        Returns:
            bool: True if language was set successfully
        """
        # Load the language file if it exists
        lang_file = Path("locales") / f"{language}.json"
        if lang_file.exists() and language not in self.translations:
            try:
                with open(lang_file, "r", encoding="utf-8") as f:
                    self.translations[language] = json.load(f)
            except Exception as e:
                print(f"Failed to load {lang_file}: {e}")
        
        # Set language (fallback to English if not available)
        if language in self.translations or language == "en":
            self.current_language = language
            return True
        else:
            print(f"Language '{language}' not found, using English")
            self.current_language = "en"
            return False
    
    def get(self, key: str, **kwargs) -> str:
        """
        Get a translation for the current language.
        
        Args:
            key: Translation key (e.g., 'server.starting')
            **kwargs: Format arguments
            
        Returns:
            str: Translated text or key if not found
        """
        # Try current language
        if self.current_language in self.translations:
            translation = self._get_nested(self.translations[self.current_language], key)
            if translation:
                return self._format(translation, **kwargs)
        
        # Fallback to English
        if self.current_language != "en" and "en" in self.translations:
            translation = self._get_nested(self.translations["en"], key)
            if translation:
                return self._format(translation, **kwargs)
        
        # Return key if no translation found
        return key
    
    def _get_nested(self, data: Dict[str, Any], key: str) -> Optional[str]:
        """Get nested value from dictionary using dot notation."""
        keys = key.split('.')
        current = data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        
        return str(current) if current is not None else None
    
    def _format(self, text: str, **kwargs) -> str:
        """Format text with arguments."""
        if not kwargs:
            return text
        
        try:
            return text.format(**kwargs)
        except (KeyError, ValueError):
            return text
    
    def get_language(self) -> str:
        """Get current language."""
        return self.current_language
    
    def get_available_languages(self) -> list[str]:
        """Get list of available languages."""
        return list(self.translations.keys())


# Global i18n instance
i18n = SimpleI18n()


def t(key: str, **kwargs) -> str:
    """Simple translation function."""
    return i18n.get(key, **kwargs)


def set_language(language: str) -> bool:
    """Set the global language."""
    return i18n.set_language(language)


def get_language() -> str:
    """Get current language."""
    return i18n.get_language()


def get_available_languages() -> list[str]:
    """Get available languages."""
    return i18n.get_available_languages()


# Export the main functions
__all__ = ["t", "set_language", "get_language", "get_available_languages", "i18n"] 