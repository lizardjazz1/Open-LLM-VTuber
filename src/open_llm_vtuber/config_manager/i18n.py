# config_manager/i18n.py
from typing import Dict, ClassVar
from pydantic import BaseModel, Field, ConfigDict
from src.open_llm_vtuber.i18n import t


class MultiLingualString(BaseModel):
    """
    Represents a string with translations in multiple languages.
    """

    en: str = Field(..., description="English translation")
    zh: str = Field(..., description="Chinese translation")
    ru: str = Field(..., description="Russian translation")

    def get(self, lang_code: str) -> str:
        """
        Retrieves the translation for the specified language code.

        Args:
            lang_code: The language code (e.g., "en", "zh", "ru").

        Returns:
            The translation for the specified language code, or the English translation if the specified language is not found.
        """
        return getattr(self, lang_code, self.en)


class Description(BaseModel):
    """
    Represents a field description using i18n keys.
    """

    i18n_key: str = Field(..., description="i18n key for centralized translations")

    def get(self, lang_code: str = "en") -> str:
        """
        Retrieves the description using the i18n system.

        Args:
            lang_code: The language code (e.g., "en", "zh", "ru").

        Returns:
            The translated description.
        """
        return t(self.i18n_key)


class I18nMixin(BaseModel):
    """
    Mixin for Pydantic models to support internationalization.
    """

    model_config = ConfigDict(extra="forbid")

    # Class variable for field descriptions - not a model field
    DESCRIPTIONS: ClassVar[Dict[str, Description | str]] = {}

    def get_field_description(self, field_name: str, lang_code: str = "en") -> str:
        """
        Get the description for a field using the i18n system.

        Args:
            field_name: The name of the field.
            lang_code: The language code (e.g., "en", "zh", "ru").

        Returns:
            The description for the field in the specified language.
        """
        if hasattr(self, "DESCRIPTIONS") and field_name in self.DESCRIPTIONS:
            description = self.DESCRIPTIONS[field_name]
            if isinstance(description, Description):
                return description.get(lang_code)
            elif isinstance(description, str):
                return description
        return field_name

    def get_all_descriptions(self, lang_code: str = "en") -> Dict[str, str]:
        """
        Get all field descriptions in the specified language.

        Args:
            lang_code: The language code (e.g., "en", "zh", "ru").

        Returns:
            Dictionary mapping field names to their descriptions in the specified language.
        """
        descriptions = {}
        if hasattr(self, "DESCRIPTIONS"):
            for field_name, description in self.DESCRIPTIONS.items():
                if isinstance(description, Description):
                    descriptions[field_name] = description.get(lang_code)
                elif isinstance(description, str):
                    descriptions[field_name] = description
        return descriptions

    def get_i18n_key(self, field_name: str) -> str | None:
        """
        Get the i18n key for a field if available.

        Args:
            field_name: The name of the field.

        Returns:
            The i18n key for the field, or None if not available.
        """
        if hasattr(self, "DESCRIPTIONS") and field_name in self.DESCRIPTIONS:
            description = self.DESCRIPTIONS[field_name]
            if isinstance(description, Description):
                return description.i18n_key
        return None

    def get_available_languages(self) -> list[str]:
        """
        Get the list of available languages for this model.

        Returns:
            List of available language codes.
        """
        return ["en", "zh", "ru"]

    def validate_translations(self) -> Dict[str, list[str]]:
        """
        Validate that all i18n keys exist in translation files.

        Returns:
            Dictionary mapping field names to list of missing i18n keys.
        """
        missing_keys = {}

        if hasattr(self, "DESCRIPTIONS"):
            for field_name, description in self.DESCRIPTIONS.items():
                if isinstance(description, Description):
                    # Check if the i18n key exists by trying to get it
                    try:
                        t(description.i18n_key)
                    except Exception:
                        missing_keys[field_name] = [description.i18n_key]

        return missing_keys

    def export_translations(self, lang_code: str = "en") -> Dict[str, str]:
        """
        Export all field descriptions for a specific language.

        Args:
            lang_code: The language code to export.

        Returns:
            Dictionary mapping field names to their descriptions in the specified language.
        """
        return self.get_all_descriptions(lang_code)

    def import_translations(self, translations: Dict[str, str], lang_code: str = "en"):
        """
        Import translations for a specific language.

        Args:
            translations: Dictionary mapping field names to their descriptions.
            lang_code: The language code for the translations.
        """
        # This method is now deprecated since we use centralized i18n
        pass
