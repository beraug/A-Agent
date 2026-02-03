"""
Phase 2: Memory stream (paper pillar 1).

Time-ordered stream of observations. Supports normal observations and
reflections (Phase 3). Records are stored in memory order; retrieval
uses recency (last k). Relevance scoring can be added later.

Usage:
    from memory import MemoryStream

    stream = MemoryStream()
    stream.add_observation("User said: Hello")
    stream.add_observation("Agent replied: Hi there", type_="observation")
    recent = stream.retrieve("hello", k=5)
"""

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record(
    content: str,
    importance: float = 0.5,
    type_: str = "observation",
    timestamp: str | None = None,
    id_: str | None = None,
) -> dict[str, Any]:
    """Build a memory record (id, timestamp, content, importance, type)."""
    return {
        "id": id_ or uuid.uuid4().hex,
        "timestamp": timestamp or _now_iso(),
        "content": content,
        "importance": max(0.0, min(1.0, importance)),
        "type": type_,
    }


class MemoryStream:
    """Time-ordered stream of observations and reflections."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            repo_root = Path(__file__).resolve().parent.parent
            db_path = str(repo_root / "data" / "memory.db")

        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")

        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
              id TEXT PRIMARY KEY,
              timestamp TEXT NOT NULL,
              content TEXT NOT NULL,
              importance REAL NOT NULL,
              type TEXT NOT NULL
            );
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp);"
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    def add_observation(
        self,
        content: str,
        importance: float = 0.5,
        type_: str = "observation",
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """
        Append an observation or reflection to the stream.

        Args:
            content: The observation text (e.g. "User said: ..." or reflection summary).
            importance: Score 0.0–1.0 (default 0.5). Used later for retrieval.
            type_: "observation" or "reflection".
            timestamp: Optional ISO8601 string; default is now (UTC).

        Returns:
            The created record (id, timestamp, content, importance, type).
        """
        record = _record(content, importance, type_, timestamp)

        self._conn.execute(
            "INSERT INTO memories (id, timestamp, content, importance, type) VALUES (?, ?, ?, ?, ?);",
            (
                record["id"],
                record["timestamp"],
                record["content"],
                record["importance"],
                record["type"],
            ),
        )
        self._conn.commit()
        return record

    def get_recent(self, k: int = 10) -> list[dict[str, Any]]:
        """Return the last k records (most recent last). Order is chronological."""
        k = int(k)
        if k <= 0:
            return []

        rows = self._conn.execute(
            """
            SELECT id, timestamp, content, importance, type
            FROM memories
            ORDER BY timestamp DESC
            LIMIT ?;
            """,
            (k,),
        ).fetchall()

        out = [dict(r) for r in rows]
        out.reverse()
        return out

    def retrieve(self, query: str, k: int = 5, types: list[str] | None = None) -> list[dict[str, Any]]:
        """
        Return top-k memories. Currently recency-only: last k records,
        most recent last. Query is ignored; relevance can be added later.
        """
        records = self.get_recent(k=k)
        if types is not None:
            records = [r for r in records if r.get("type") in types]

        return records

    def get_all(self) -> list[dict[str, Any]]:
        """Return the full stream in order (oldest first)."""
        rows = self._conn.execute(
            """
            SELECT id, timestamp, content, importance, type
            FROM memories
            ORDER BY timestamp ASC;
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def count_since_last_reflection(self) -> int:
        """Number of observations (non-reflection) since the last reflection."""
        rows = self._conn.execute(
            """
            SELECT type
            FROM memories
            ORDER BY timestamp DESC;
            """
        ).fetchall()

        n = 0
        for r in rows:
            if r["type"] == "reflection":
                break
            n += 1
        return n
