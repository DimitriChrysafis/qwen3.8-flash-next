from __future__ import annotations

import sqlite3
from pathlib import Path


class NgramDiskCache:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS ngram_cache (k TEXT PRIMARY KEY, v BLOB NOT NULL)"
        )
        self.conn.commit()

    def get(self, key: str) -> bytes | None:
        row = self.conn.execute("SELECT v FROM ngram_cache WHERE k=?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: bytes) -> None:
        self.conn.execute(
            "INSERT INTO ngram_cache(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (key, value),
        )
        self.conn.commit()

    def count(self) -> int:
        row = self.conn.execute("SELECT count(*) FROM ngram_cache").fetchone()
        return int(row[0])
