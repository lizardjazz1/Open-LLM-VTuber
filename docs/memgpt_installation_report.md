# MemGPT Installation Report

- Python: captured at install time via `uv run python -V`
- Environment: existing venv at `.venv`
- Git: verified

Installed components
- MemGPT: editable install from `memgpt/`
- ChromaDB: present via project deps; storage at `cache/chroma` (overridable)
- FastAPI/Uvicorn/Pydantic: present via project deps

Configuration paths
- Backend memory module: `src/open_llm_vtuber/vtuber_memory/config.py`
- JSON overrides: `config/memory_settings.json`
- Chroma collections:
  - Session: `vtuber_session` (TTL pruned)
  - LTM: `vtuber_ltm`
- Relationships (SQLite): `cache/relationships.sqlite3`

Run logs
- Editable install succeeded: `pip install -e memgpt`
- Lint: passed (`ruff`)
- Tests: backend tests passed; memgpt repo tests are excluded from this project’s test run

APIs verified
- /admin/memory/export, /admin/memory/import
- /admin/relationship/update
- /admin/memory/prune_session_ttl (manual)
- /admin/memory/consolidate, /admin/memory/deep_consolidation

Notes
- MemGPT comes with its own test suite and optional deps (locust, sqlite_vec); they’re not required for backend integration and were not installed.
- TTL prune runs automatically after each consolidation and excludes the active session. 