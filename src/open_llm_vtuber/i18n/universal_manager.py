"""
Universal I18n Manager for Open-LLM-VTuber
==========================================

Advanced i18n manager with additional features like fallback chains,
pluralization, and context-aware translations.
"""

import json
from typing import Dict, Any
from pathlib import Path
from .translations import get_translation, get_available_languages


class UniversalI18nManager:
    """
    Universal i18n manager with advanced features.
    """

    def __init__(self, default_language: str = "en", fallback_language: str = "en"):
        """
        Initialize the universal i18n manager.

        Args:
            default_language: Default language code
            fallback_language: Fallback language code
        """
        self.default_language = default_language
        self.fallback_language = fallback_language
        self.current_language = default_language
        self.available_languages = get_available_languages()

        # Load custom translations from files
        self.custom_translations: Dict[str, Dict[str, Any]] = {}
        self._load_custom_translations()

    def _load_custom_translations(self):
        """Load custom translations from locale files."""
        locales_dir = Path("locales")
        if not locales_dir.exists():
            return

        for lang_file in locales_dir.glob("*.json"):
            try:
                with open(lang_file, "r", encoding="utf-8") as f:
                    lang_code = lang_file.stem
                    self.custom_translations[lang_code] = json.load(f)
                    # Add to available languages if not already present
                    if lang_code not in self.available_languages:
                        self.available_languages.append(lang_code)
            except Exception as e:
                print(f"Failed to load custom translations from {lang_file}: {e}")

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
        Get a translation for the current language with fallback support.

        Args:
            key: Translation key
            **kwargs: Format arguments

        Returns:
            str: Translated text
        """
        # Try custom translations first
        if self.current_language in self.custom_translations:
            translation = self._get_from_custom(key, self.current_language)
            if translation:
                return self._format_translation(translation, **kwargs)

        # Try built-in translations
        translation = get_translation(key, self.current_language)
        if translation != key:
            return self._format_translation(translation, **kwargs)

        # Fallback to default language
        if self.current_language != self.default_language:
            # Try custom translations for default language
            if self.default_language in self.custom_translations:
                translation = self._get_from_custom(key, self.default_language)
                if translation:
                    return self._format_translation(translation, **kwargs)

            # Try built-in translations for default language
            translation = get_translation(key, self.default_language)
            if translation != key:
                return self._format_translation(translation, **kwargs)

        # Final fallback to fallback language
        if self.fallback_language != self.default_language:
            translation = get_translation(key, self.fallback_language)
            if translation != key:
                return self._format_translation(translation, **kwargs)

        # Return key if no translation found
        return key

    def _get_from_custom(self, key: str, lang_code: str) -> str | None:
        """
        Get translation from custom JSON files.

        Args:
            key: Translation key
            lang_code: Language code

        Returns:
            str | None: Translation or None if not found
        """
        if lang_code not in self.custom_translations:
            return None

        # Split key by dots to navigate nested structure
        keys = key.split(".")
        translation = self.custom_translations[lang_code]

        # Navigate through nested structure
        for k in keys:
            if isinstance(translation, dict) and k in translation:
                translation = translation[k]
            else:
                return None

        # Return string if found
        if isinstance(translation, str):
            return translation

        return None

    def _format_translation(self, translation: str, **kwargs) -> str:
        """
        Format translation with arguments.

        Args:
            translation: Translation string
            **kwargs: Format arguments

        Returns:
            str: Formatted translation
        """
        if not kwargs:
            return translation

        try:
            return translation.format(**kwargs)
        except (KeyError, ValueError):
            return translation

    def get_language(self) -> str:
        """Get current language code."""
        return self.current_language

    def get_available_languages(self) -> list[str]:
        """Get list of available languages."""
        return self.available_languages.copy()

    def add_custom_translation(self, lang_code: str, key: str, translation: str):
        """
        Add a custom translation.

        Args:
            lang_code: Language code
            key: Translation key
            translation: Translation text
        """
        if lang_code not in self.custom_translations:
            self.custom_translations[lang_code] = {}

        # Split key by dots to create nested structure
        keys = key.split(".")
        current = self.custom_translations[lang_code]

        # Navigate to the parent level
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Set the translation
        current[keys[-1]] = translation

    def save_custom_translations(self, lang_code: str, file_path: str | None = None):
        """
        Save custom translations to a JSON file.

        Args:
            lang_code: Language code to save
            file_path: Optional file path, defaults to locales/{lang_code}.json
        """
        if lang_code not in self.custom_translations:
            return

        if file_path is None:
            locales_dir = Path("locales")
            locales_dir.mkdir(exist_ok=True)
            file_path = locales_dir / f"{lang_code}.json"

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(
                    self.custom_translations[lang_code], f, ensure_ascii=False, indent=2
                )
        except Exception as e:
            print(f"Failed to save custom translations to {file_path}: {e}")

    def load_custom_translations(self, lang_code: str, file_path: str | None = None):
        """
        Load custom translations from a JSON file.

        Args:
            lang_code: Language code to load
            file_path: Optional file path, defaults to locales/{lang_code}.json
        """
        if file_path is None:
            file_path = Path("locales") / f"{lang_code}.json"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.custom_translations[lang_code] = json.load(f)
                if lang_code not in self.available_languages:
                    self.available_languages.append(lang_code)
        except Exception as e:
            print(f"Failed to load custom translations from {file_path}: {e}")

    def validate_translations(self, lang_code: str) -> Dict[str, list[str]]:
        """
        Validate that all translations are complete for a language.

        Args:
            lang_code: Language code to validate

        Returns:
            Dict[str, list[str]]: Dictionary mapping missing keys to missing languages
        """
        missing = {}

        # Get all keys from built-in translations
        from .translations import TRANSLATIONS

        if "en" in TRANSLATIONS:
            self._validate_recursive(TRANSLATIONS["en"], lang_code, "", missing)

        # Get all keys from custom translations
        if lang_code in self.custom_translations:
            self._validate_recursive(
                self.custom_translations[lang_code], lang_code, "", missing
            )

        return missing

    def _validate_recursive(
        self,
        translations: Dict[str, Any],
        lang_code: str,
        prefix: str,
        missing: Dict[str, list[str]],
    ):
        """
        Recursively validate translations.

        Args:
            translations: Translation dictionary
            lang_code: Language code to validate
            prefix: Current key prefix
            missing: Dictionary to store missing translations
        """
        for key, value in translations.items():
            current_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                self._validate_recursive(value, lang_code, current_key, missing)
            elif isinstance(value, str):
                # Check if this key exists in the target language
                if lang_code not in self.custom_translations:
                    if current_key not in missing:
                        missing[current_key] = []
                    missing[current_key].append(lang_code)
                else:
                    # Check if key exists in custom translations
                    if not self._key_exists_in_custom(current_key, lang_code):
                        if current_key not in missing:
                            missing[current_key] = []
                        missing[current_key].append(lang_code)

    def _key_exists_in_custom(self, key: str, lang_code: str) -> bool:
        """
        Check if a key exists in custom translations.

        Args:
            key: Translation key
            lang_code: Language code

        Returns:
            bool: True if key exists
        """
        if lang_code not in self.custom_translations:
            return False

        keys = key.split(".")
        current = self.custom_translations[lang_code]

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return False

        return isinstance(current, str)

    def export_translations(self, lang_code: str, file_path: str | None = None):
        """
        Export all translations for a language to a JSON file.

        Args:
            lang_code: Language code to export
            file_path: Optional file path
        """
        if file_path is None:
            locales_dir = Path("locales")
            locales_dir.mkdir(exist_ok=True)
            file_path = locales_dir / f"{lang_code}.json"

        # Combine built-in and custom translations
        all_translations = {}

        # Add built-in translations
        from .translations import TRANSLATIONS

        if lang_code in TRANSLATIONS:
            all_translations.update(TRANSLATIONS[lang_code])

        # Add custom translations (override built-in)
        if lang_code in self.custom_translations:
            all_translations.update(self.custom_translations[lang_code])

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(all_translations, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to export translations to {file_path}: {e}")


# Global universal i18n manager instance
universal_i18n_manager = UniversalI18nManager()


def ut(key: str, **kwargs) -> str:
    """
    Universal translation function.

    Args:
        key: Translation key
        **kwargs: Format arguments

    Returns:
        str: Translated text
    """
    return universal_i18n_manager.get(key, **kwargs)


def set_universal_language(language: str) -> bool:
    """
    Set the language for the universal i18n manager.

    Args:
        language: Language code to set

    Returns:
        bool: True if language was set successfully
    """
    return universal_i18n_manager.set_language(language)
