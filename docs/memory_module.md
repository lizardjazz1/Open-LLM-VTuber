# Memory Module Architecture

This document describes the VTuber memory system.

- Short-Term Memory (STM): recent 15–30 minutes of messages kept in agent RAM.
- Session Memory: temporary facts/events stored in ChromaDB (session collection).
- Long-Term Memory (LTM): consolidated knowledge stored in ChromaDB (ltm collection).

Collections
- Session: `vtuber_session` (temporary, subject to TTL/pruning)
  - Stores recent chat-derived entries (kinds like `chat`, `Emotions`, temporary objectives)
  - Automatically pruned by TTL
- LTM: `vtuber_ltm`
  - Stores consolidated key facts, beliefs, objectives, past events, facts about users

Storage dir: `src/open_llm_vtuber/vtuber_memory/chroma_storage/`

Relationships (SQLite)
- Table `user_relationships(user_id, username, affinity, trust, interaction_count, last_interaction)`
- Affinity: -100..+100; Trust: 0..100; decay is applied based on inactivity.

Admin API
- GET `/admin/memory/search?q=...&kinds=...&conf_uid=...`
- POST `/admin/memory/edit` {id, new_content}
- DELETE `/admin/memory/delete/{id}`
- POST `/admin/relationship/update` {user_id, affinity_delta?, trust_delta?, username?}
- GET `/admin/memory/export?conf_uid=...&kind=...&fmt=json|yaml`
- POST `/admin/memory/import?conf_uid=...&history_uid=...&default_kind=...&fmt=json|yaml` (multipart file)
- POST `/admin/memory/prune_session_ttl?ttl_sec=0` (0 uses server default TTL; excludes active history)
- POST `/admin/memory/consolidate` (manual consolidation)
- POST `/admin/memory/deep_consolidation` (manual deep consolidation)

Consolidation and TTL
- Interval: default 30 minutes (`memory_consolidation_interval_sec: 1800`)
- End-of-stream consolidation is triggered automatically (`on_stream_end`)
- TTL cleanup for `vtuber_session` runs after each consolidation cycle
  - Configure TTL in code via `SESSION_TTL_SEC` (default 7 days)
  - Active session (current `history_uid`) is excluded from pruning

Deep Consolidation (every N streams)
- Config `deep_consolidation_every_n_streams` (default 5)
- On every Nth stream end:
  - Prunes LTM entries older than 30 days or with low importance
  - Re-extracts key facts from the recent history window and reinjects as fresh LTM entries

Frontend Admin (Open-LLM-VTuber-Web)
- Memory Management panel:
  - Search with kind filters
  - Edit/Delete items
  - Export (JSON/YAML) and Import (JSON/YAML)
- Relationships panel:
  - Adjust affinity/trust, view live values

Setup Guides
- MemGPT (no Docker): see README
- TTL: no extra setup; runs automatically via scheduler post-hook
- Deep consolidation: set `system_config.deep_consolidation_every_n_streams: 5` (or desired value) 