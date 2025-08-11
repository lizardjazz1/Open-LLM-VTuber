#!/usr/bin/env python3
"""
Migration utility for hardcoded translations to i18n system
==========================================================

This script helps migrate hardcoded translations (en=, zh=, ru=) 
to the centralized i18n system.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set


def find_hardcoded_translations(directory: str = "src/open_llm_vtuber") -> Dict[str, List[Tuple[str, int, str]]]:
    """
    Find all hardcoded translations in Python files.
    
    Args:
        directory: Directory to search in
        
    Returns:
        Dictionary mapping file paths to list of (key, line_number, full_line)
    """
    translations = {}
    
    # Pattern to match en=, zh=, ru= assignments
    pattern = r'(\w+)\s*=\s*["\']([^"\']+)["\']'
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                file_translations = []
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        
                    for line_num, line in enumerate(lines, 1):
                        # Look for language assignments
                        if any(lang in line for lang in ['en=', 'zh=', 'ru=']):
                            matches = re.findall(pattern, line)
                            for key, value in matches:
                                if key in ['en', 'zh', 'ru']:
                                    file_translations.append((key, line_num, line.strip()))
                                    
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
                    
                if file_translations:
                    translations[file_path] = file_translations
                    
    return translations


def extract_translation_groups(translations: Dict[str, List[Tuple[str, int, str]]]) -> Dict[str, Dict[str, str]]:
    """
    Group translations by their context (usually field names).
    
    Args:
        translations: Dictionary of file translations
        
    Returns:
        Dictionary mapping context to language translations
    """
    groups = {}
    
    for file_path, file_translations in translations.items():
        # Group by context (usually the field name)
        current_group = {}
        
        for lang, line_num, line in file_translations:
            # Extract field name from context
            # Look for patterns like "field_name": Description(...)
            field_match = re.search(r'["\'](\w+)["\']\s*:\s*Description', line)
            if field_match:
                field_name = field_match.group(1)
                
                if field_name not in groups:
                    groups[field_name] = {}
                    
                groups[field_name][lang] = line.strip()
                
    return groups


def generate_i18n_keys(groups: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """
    Generate i18n keys for translation groups.
    
    Args:
        groups: Dictionary of translation groups
        
    Returns:
        Dictionary mapping suggested keys to English translations
    """
    i18n_keys = {}
    
    for field_name, translations in groups.items():
        if 'en' in translations:
            # Extract the English translation
            en_line = translations['en']
            en_match = re.search(r'en\s*=\s*["\']([^"\']+)["\']', en_line)
            
            if en_match:
                english_text = en_match.group(1)
                
                # Generate a key based on the field name
                # Convert camelCase or snake_case to dot notation
                key = re.sub(r'([a-z])([A-Z])', r'\1_\2', field_name).lower()
                key = f"config.{key}"
                
                i18n_keys[key] = english_text
                
    return i18n_keys


def create_migration_plan(translations: Dict[str, List[Tuple[str, int, str]]]) -> Dict[str, any]:
    """
    Create a migration plan for converting hardcoded translations.
    
    Args:
        translations: Dictionary of file translations
        
    Returns:
        Migration plan with suggested changes
    """
    groups = extract_translation_groups(translations)
    i18n_keys = generate_i18n_keys(groups)
    
    plan = {
        'files_to_update': {},
        'new_i18n_keys': i18n_keys,
        'summary': {
            'total_files': len(translations),
            'total_translations': sum(len(ts) for ts in translations.values()),
            'suggested_keys': len(i18n_keys)
        }
    }
    
    # Create file-specific migration plans
    for file_path, file_translations in translations.items():
        plan['files_to_update'][file_path] = {
            'translations': file_translations,
            'suggested_changes': []
        }
        
        # Group translations by context
        current_context = None
        context_translations = {}
        
        for lang, line_num, line in file_translations:
            field_match = re.search(r'["\'](\w+)["\']\s*:\s*Description', line)
            if field_match:
                field_name = field_match.group(1)
                
                if field_name != current_context:
                    if current_context and context_translations:
                        # Process previous context
                        key = re.sub(r'([a-z])([A-Z])', r'\1_\2', current_context).lower()
                        key = f"config.{key}"
                        
                        plan['files_to_update'][file_path]['suggested_changes'].append({
                            'context': current_context,
                            'key': key,
                            'translations': context_translations.copy()
                        })
                    
                    current_context = field_name
                    context_translations = {}
                
                context_translations[lang] = line.strip()
        
        # Process last context
        if current_context and context_translations:
            key = re.sub(r'([a-z])([A-Z])', r'\1_\2', current_context).lower()
            key = f"config.{key}"
            
            plan['files_to_update'][file_path]['suggested_changes'].append({
                'context': current_context,
                'key': key,
                'translations': context_translations.copy()
            })
    
    return plan


def print_migration_report(plan: Dict[str, any]):
    """
    Print a detailed migration report.
    
    Args:
        plan: Migration plan
    """
    print("🌍 Translation Migration Report")
    print("=" * 50)
    
    summary = plan['summary']
    print(f"📊 Summary:")
    print(f"   - Files to update: {summary['total_files']}")
    print(f"   - Total translations: {summary['total_translations']}")
    print(f"   - Suggested i18n keys: {summary['suggested_keys']}")
    print()
    
    print("📝 Suggested i18n keys:")
    for key, english_text in plan['new_i18n_keys'].items():
        print(f"   {key}: {english_text}")
    print()
    
    print("📁 Files to update:")
    for file_path, file_plan in plan['files_to_update'].items():
        print(f"   {file_path}")
        for change in file_plan['suggested_changes']:
            print(f"     - {change['context']} -> {change['key']}")
    print()


def generate_replacement_code(context: str, key: str, translations: Dict[str, str]) -> str:
    """
    Generate replacement code for a translation group.
    
    Args:
        context: Field name context
        key: i18n key
        translations: Dictionary of language translations
        
    Returns:
        Replacement code string
    """
    # Extract English text for the key
    en_line = translations.get('en', '')
    en_match = re.search(r'en\s*=\s*["\']([^"\']+)["\']', en_line)
    english_text = en_match.group(1) if en_match else key
    
    # Create the replacement
    replacement = f'"{context}": Description(\n'
    replacement += f'    en="{english_text}",\n'
    
    if 'zh' in translations:
        zh_match = re.search(r'zh\s*=\s*["\']([^"\']+)["\']', translations['zh'])
        if zh_match:
            replacement += f'    zh="{zh_match.group(1)}",\n'
    
    if 'ru' in translations:
        ru_match = re.search(r'ru\s*=\s*["\']([^"\']+)["\']', translations['ru'])
        if ru_match:
            replacement += f'    ru="{ru_match.group(1)}",\n'
    
    replacement += f'    i18n_key="{key}"\n'
    replacement += ')'
    
    return replacement


def main():
    """Main migration function."""
    print("🔍 Searching for hardcoded translations...")
    
    # Find all hardcoded translations
    translations = find_hardcoded_translations()
    
    if not translations:
        print("✅ No hardcoded translations found!")
        return
    
    print(f"📋 Found {len(translations)} files with hardcoded translations")
    
    # Create migration plan
    plan = create_migration_plan(translations)
    
    # Print report
    print_migration_report(plan)
    
    # Ask user if they want to proceed
    response = input("\n❓ Do you want to proceed with migration? (y/N): ")
    if response.lower() != 'y':
        print("❌ Migration cancelled")
        return
    
    # Generate migration script
    print("\n📝 Generating migration script...")
    
    migration_script = []
    migration_script.append("# Migration script for hardcoded translations")
    migration_script.append("# Generated by migrate_translations.py")
    migration_script.append("")
    
    for file_path, file_plan in plan['files_to_update'].items():
        migration_script.append(f"# File: {file_path}")
        
        for change in file_plan['suggested_changes']:
            context = change['context']
            key = change['key']
            translations = change['translations']
            
            replacement = generate_replacement_code(context, key, translations)
            migration_script.append(f"# Replace {context} with:")
            migration_script.append(replacement)
            migration_script.append("")
    
    # Save migration script
    script_path = "migration_script.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(migration_script))
    
    print(f"✅ Migration script saved to {script_path}")
    print("📖 Review the script and apply changes manually")


if __name__ == "__main__":
    main() 