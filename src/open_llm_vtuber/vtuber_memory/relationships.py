from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class RelationshipRecord:
    user_id: str
    username: str | None = None
    affection_score: int = 0
    last_interaction: str | None = None


class RelationshipsDB:
    """Lightweight SQLite manager for user relationships.

    Schema:
        user_relationships(
            user_id TEXT PRIMARY KEY,
            username TEXT,
            affection_score INTEGER DEFAULT 0,
            last_interaction TIMESTAMP
        )
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_relationships (
                        user_id TEXT PRIMARY KEY,
                        username TEXT,
                        affection_score INTEGER DEFAULT 0,
                        last_interaction TIMESTAMP
                    )
                    """
                )
        except Exception as e:
            logger.error(f"Failed to init relationships DB: {e}")

    def adjust_affection(self, user_id: str, delta: int) -> int:
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "SELECT affection_score FROM user_relationships WHERE user_id=?",
                    (user_id,),
                )
                row = cur.fetchone()
                score = int(row[0]) if row else 0
                score += int(delta)
                if row:
                    conn.execute(
                        "UPDATE user_relationships SET affection_score=? WHERE user_id=?",
                        (score, user_id),
                    )
                else:
                    conn.execute(
                        "INSERT INTO user_relationships(user_id, affection_score) VALUES(?,?)",
                        (user_id, score),
                    )
                return score
        except Exception as e:
            logger.error(f"adjust_affection failed: {e}")
            return 0

    def get_affection(self, user_id: str) -> int:
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "SELECT affection_score FROM user_relationships WHERE user_id=?",
                    (user_id,),
                )
                row = cur.fetchone()
                return int(row[0]) if row else 0
        except Exception as e:
            logger.error(f"get_affection failed: {e}")
            return 0

    def set_username(self, user_id: str, username: str) -> None:
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "SELECT user_id FROM user_relationships WHERE user_id=?",
                    (user_id,),
                )
                if cur.fetchone():
                    conn.execute(
                        "UPDATE user_relationships SET username=? WHERE user_id=?",
                        (username, user_id),
                    )
                else:
                    conn.execute(
                        "INSERT INTO user_relationships(user_id, username) VALUES(?,?)",
                        (user_id, username),
                    )
        except Exception as e:
            logger.error(f"set_username failed: {e}")
