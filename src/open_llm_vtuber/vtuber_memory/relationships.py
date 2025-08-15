from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from loguru import logger


@dataclass
class RelationshipRecord:
    user_id: str
    username: str | None = None
    realname: str | None = None
    affinity: int = 0  # -100..+100
    trust: int = 0  # 0..100
    interaction_count: int = 0
    last_interaction: str | None = None


class RelationshipsDB:
    """Lightweight SQLite manager for user relationships.

    Schema:
        user_relationships(
            user_id TEXT PRIMARY KEY,
            username TEXT,
            realname TEXT,
            affinity INTEGER DEFAULT 0,
            trust INTEGER DEFAULT 0,
            interaction_count INTEGER DEFAULT 0,
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
                        realname TEXT,
                        affinity INTEGER DEFAULT 0,
                        trust INTEGER DEFAULT 0,
                        interaction_count INTEGER DEFAULT 0,
                        last_interaction TIMESTAMP
                    )
                    """
                )
                # Backward-compat migration (add missing columns if coming from older schema)
                try:
                    conn.execute(
                        "ALTER TABLE user_relationships ADD COLUMN affinity INTEGER DEFAULT 0"
                    )
                except Exception:
                    pass
                try:
                    conn.execute(
                        "ALTER TABLE user_relationships ADD COLUMN trust INTEGER DEFAULT 0"
                    )
                except Exception:
                    pass
                try:
                    conn.execute(
                        "ALTER TABLE user_relationships ADD COLUMN realname TEXT"
                    )
                except Exception:
                    pass
                try:
                    conn.execute(
                        "ALTER TABLE user_relationships ADD COLUMN interaction_count INTEGER DEFAULT 0"
                    )
                except Exception:
                    pass
                try:
                    conn.execute(
                        "ALTER TABLE user_relationships ADD COLUMN last_interaction TIMESTAMP"
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Failed to init relationships DB: {e}")

    def _clamp(self, value: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, int(value)))

    def _update_last_interaction(self, conn: sqlite3.Connection, user_id: str) -> None:
        now = datetime.utcnow().isoformat()
        conn.execute(
            "UPDATE user_relationships SET last_interaction=?, interaction_count=interaction_count+1 WHERE user_id=?",
            (now, user_id),
        )

    def ensure_user(self, user_id: str, username: str | None = None) -> None:
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "SELECT user_id FROM user_relationships WHERE user_id= ?",
                    (user_id,),
                )
                if not cur.fetchone():
                    conn.execute(
                        "INSERT INTO user_relationships(user_id, username, realname, affinity, trust, interaction_count, last_interaction) VALUES(?, ?, ?, 0, 0, 0, ?)",
                        (
                            user_id,
                            username or None,
                            None,
                            datetime.utcnow().isoformat(),
                        ),
                    )
        except Exception as e:
            logger.error(f"ensure_user failed: {e}")

    def adjust_affinity(self, user_id: str, delta: int) -> int:
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "SELECT affinity FROM user_relationships WHERE user_id=?",
                    (user_id,),
                )
                row = cur.fetchone()
                value = int(row[0]) if row else 0
                value = self._clamp(value + int(delta), -100, +100)
                if row:
                    conn.execute(
                        "UPDATE user_relationships SET affinity=? WHERE user_id=?",
                        (value, user_id),
                    )
                else:
                    conn.execute(
                        "INSERT INTO user_relationships(user_id, affinity, last_interaction) VALUES(?,?,?)",
                        (user_id, value, datetime.utcnow().isoformat()),
                    )
                self._update_last_interaction(conn, user_id)
                return value
        except Exception as e:
            logger.error(f"adjust_affinity failed: {e}")
            return 0

    def adjust_trust(self, user_id: str, delta: int) -> int:
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "SELECT trust FROM user_relationships WHERE user_id=?",
                    (user_id,),
                )
                row = cur.fetchone()
                value = int(row[0]) if row else 0
                value = self._clamp(value + int(delta), 0, 100)
                if row:
                    conn.execute(
                        "UPDATE user_relationships SET trust=? WHERE user_id=?",
                        (value, user_id),
                    )
                else:
                    conn.execute(
                        "INSERT INTO user_relationships(user_id, trust, last_interaction) VALUES(?,?,?)",
                        (user_id, value, datetime.utcnow().isoformat()),
                    )
                self._update_last_interaction(conn, user_id)
                return value
        except Exception as e:
            logger.error(f"adjust_trust failed: {e}")
            return 0

    def get(self, user_id: str) -> RelationshipRecord | None:
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "SELECT user_id, username, realname, affinity, trust, interaction_count, last_interaction FROM user_relationships WHERE user_id=?",
                    (user_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return RelationshipRecord(
                    user_id=row[0],
                    username=row[1],
                    realname=row[2],
                    affinity=int(row[3] or 0),
                    trust=int(row[4] or 0),
                    interaction_count=int(row[5] or 0),
                    last_interaction=row[6],
                )
        except Exception as e:
            logger.error(f"get failed: {e}")
            return None

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
                        "INSERT INTO user_relationships(user_id, username, last_interaction) VALUES(?,?,?)",
                        (user_id, username, datetime.utcnow().isoformat()),
                    )
        except Exception as e:
            logger.error(f"set_username failed: {e}")

    def set_realname(self, user_id: str, realname: str) -> None:
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "SELECT user_id FROM user_relationships WHERE user_id=?",
                    (user_id,),
                )
                if cur.fetchone():
                    conn.execute(
                        "UPDATE user_relationships SET realname=? WHERE user_id=?",
                        (realname, user_id),
                    )
                else:
                    conn.execute(
                        "INSERT INTO user_relationships(user_id, realname, last_interaction) VALUES(?,?,?)",
                        (user_id, realname, datetime.utcnow().isoformat()),
                    )
        except Exception as e:
            logger.error(f"set_realname failed: {e}")

    def touch(self, user_id: str) -> None:
        """Increment interaction_count and update last_interaction for a user.

        Creates the user row if it does not exist.
        """
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "SELECT user_id FROM user_relationships WHERE user_id=?",
                    (user_id,),
                )
                if not cur.fetchone():
                    conn.execute(
                        "INSERT INTO user_relationships(user_id, interaction_count, last_interaction) VALUES(?, 1, ?)",
                        (user_id, datetime.utcnow().isoformat()),
                    )
                    return
                self._update_last_interaction(conn, user_id)
        except Exception as e:
            logger.error(f"touch failed: {e}")

    def ensure_and_touch(self, user_id: str, username: str | None = None) -> None:
        """Ensure user row exists, optionally set username, then touch."""
        try:
            self.ensure_user(user_id, username)
            if username:
                try:
                    self.set_username(user_id, username)
                except Exception:
                    pass
            self.touch(user_id)
        except Exception as e:
            logger.error(f"ensure_and_touch failed: {e}")

    def apply_decay(self, user_id: str, days_inactive_for_full_decay: int = 60) -> int:
        """Decrease affinity slightly based on inactivity.

        Returns the new affinity value after decay.
        """
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "SELECT affinity, last_interaction FROM user_relationships WHERE user_id=?",
                    (user_id,),
                )
                row = cur.fetchone()
                if not row:
                    return 0
                affinity = int(row[0] or 0)
                last_str = row[1]
                if not last_str:
                    return affinity
                try:
                    last_dt = datetime.fromisoformat(last_str)
                except Exception:
                    return affinity
                inactive_days = (datetime.utcnow() - last_dt).days
                if inactive_days <= 0:
                    return affinity
                # Linear decay toward 0 over the configured horizon
                decay_steps = max(1, days_inactive_for_full_decay)
                step = max(1, inactive_days)
                delta = int(round((affinity / decay_steps) * step))
                # Move affinity toward 0
                if affinity > 0:
                    affinity = max(0, affinity - abs(delta))
                elif affinity < 0:
                    affinity = min(0, affinity + abs(delta))
                conn.execute(
                    "UPDATE user_relationships SET affinity=? WHERE user_id=?",
                    (affinity, user_id),
                )
                return affinity
        except Exception as e:
            logger.error(f"apply_decay failed: {e}")
            return 0

    def list_recent(self, limit: int = 20) -> list[RelationshipRecord]:
        """Return recent relationships ordered by last_interaction desc."""
        try:
            limit = max(1, min(500, int(limit or 20)))
        except Exception:
            limit = 20
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "SELECT user_id, username, realname, affinity, trust, interaction_count, last_interaction FROM user_relationships ORDER BY COALESCE(last_interaction,'') DESC LIMIT ?",
                    (limit,),
                )
                rows = cur.fetchall()
            out: list[RelationshipRecord] = []
            for row in rows or []:
                out.append(
                    RelationshipRecord(
                        user_id=row[0],
                        username=row[1],
                        realname=row[2],
                        affinity=int(row[3] or 0),
                        trust=int(row[4] or 0),
                        interaction_count=int(row[5] or 0),
                        last_interaction=row[6],
                    )
                )
            return out
        except Exception as e:
            logger.error(f"list_recent failed: {e}")
            return []
