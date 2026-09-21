"""Tiny SQLite memory: durable facts and preferences."""
from __future__ import annotations
import sqlite3
import time
from pathlib import Path


class Memory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS facts "
            "(id INTEGER PRIMARY KEY, fact TEXT, created REAL)")

    def add(self, fact: str) -> int:
        cur = self.db.execute(
            "INSERT INTO facts (fact, created) VALUES (?, ?)",
            (fact.strip(), time.time()))
        self.db.commit()
        return cur.lastrowid

    def search(self, query: str, limit: int = 5) -> list[dict]:
        words = [w for w in query.split() if len(w) > 2][:6]
        if not words:
            return self.recent(limit)
        clause = " OR ".join(["fact LIKE ?"] * len(words))
        params = [f"%{w}%" for w in words]
        rows = self.db.execute(
            f"SELECT id, fact FROM facts WHERE {clause} ORDER BY id DESC LIMIT ?",
            (*params, limit)).fetchall()
        return [{"id": r[0], "fact": r[1]} for r in rows]

    def recent(self, limit: int = 10) -> list[dict]:
        rows = self.db.execute(
            "SELECT id, fact FROM facts ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()
        return [{"id": r[0], "fact": r[1]} for r in rows]

    def forget(self, fact_id: int) -> bool:
        cur = self.db.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        self.db.commit()
        return cur.rowcount > 0
