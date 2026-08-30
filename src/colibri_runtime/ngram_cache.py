from __future__ import annotations

import sqlite3
from pathlib import Path


class NgramDiskCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self._closed = False
        try:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS ngram_cache (k TEXT PRIMARY KEY, v BLOB NOT NULL)"
            )
            self.conn.commit()
        except BaseException:
            self.close()
            raise

    def get(self, key: str) -> bytes | None:
        self._ensure_open()
        row = self.conn.execute("SELECT v FROM ngram_cache WHERE k=?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: bytes) -> None:
        self._ensure_open()
        self.conn.execute(
            "INSERT INTO ngram_cache(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (key, value),
        )
        self.conn.commit()

    def count(self) -> int:
        self._ensure_open()
        row = self.conn.execute("SELECT count(*) FROM ngram_cache").fetchone()
        return int(row[0])

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("n-gram cache is closed")

    def close(self) -> None:
        if not self._closed:
            self.conn.close()
            self._closed = True

    def __enter__(self) -> NgramDiskCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
