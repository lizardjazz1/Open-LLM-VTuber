"""
I18n Manager for Open-LLM-VTuber
=================================

Simple i18n manager for integrating translations into the application.
"""

from typing import Optional
from .translations import get_translation, get_available_languages, format_translation


class I18nManager:
    """
    Simple i18n manager for Open-LLM-VTuber.
    """
    
    def __init__(self, default_language: str = "en"):
        """
        Initialize the i18n manager.
        
        Args:
            default_language: Default language code
        """
        self.default_language = default_language
        self.current_language = default_language
        self.available_languages = get_available_languages()
    
    def set_language(self, language: str) -> bool:
        """
        Set the current language.
        
        Args:
            language: Language code to set
            
        Returns:
            bool: True if language was set successfully
        """
        if language in self.available_languages:
            self.current_language = language
            return True
        return False
    
    def get(self, key: str, **kwargs) -> str:
        """
        Get a translation for the current language.
        
        Args:
            key: Translation key
            **kwargs: Format arguments
            
        Returns:
            str: Translated text
        """
        if kwargs:
            return format_translation(key, self.current_language, **kwargs)
        return get_translation(key, self.current_language)
    
    def get_language(self) -> str:
        """Get current language code."""
        return self.current_language
    
    def get_available_languages(self) -> list[str]:
        """Get list of available languages."""
        return self.available_languages.copy()


# Global i18n manager instance
i18n_manager = I18nManager()


def t(key: str, **kwargs) -> str:
    """
    Convenience function to get translations.
    
    Args:
        key: Translation key
        **kwargs: Format arguments
        
    Returns:
        str: Translated text
    """
    return i18n_manager.get(key, **kwargs)


def set_language(language: str) -> bool:
    """
    Set the global language.
    
    Args:
        language: Language code to set
        
    Returns:
        bool: True if language was set successfully
    """
    return i18n_manager.set_language(language) 