# Persona Prompts

Place your persona prompt files here. Each file should be a `.txt` file.

Usage:
- Set `character_config.persona_prompt_name` in `conf.yaml` to the file name without `.txt`.
- Example: if you create `streamer_ru.txt`, set `persona_prompt_name: streamer_ru`.
- If `persona_prompt_name` is empty or missing, the inline `persona_prompt` from config will be used.

Notes:
- File encoding is auto-detected; UTF-8 is recommended.
- This folder is loaded via `prompts/prompt_loader.py`.
