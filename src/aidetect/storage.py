"""SQLite storage: analysis history and the local reference corpus for similarity checks."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    title TEXT,
    text_sha1 TEXT NOT NULL,
    word_count INTEGER,
    language TEXT,
    status TEXT,
    ai_share REAL,
    document_probability REAL,
    confidence REAL,
    text TEXT,
    result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reference_docs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    added_at TEXT NOT NULL,
    title TEXT NOT NULL,
    text_sha1 TEXT NOT NULL UNIQUE,
    word_count INTEGER,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fingerprints (
    hash INTEGER NOT NULL,
    doc_id INTEGER NOT NULL REFERENCES reference_docs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_fp_hash ON fingerprints(hash);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        return c

    # ---------------------------------------------------------------- history
    def save_analysis(self, text: str, result: dict, store_text: bool = False) -> int:
        r = result.get("result", {})
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO analyses (created_at, title, text_sha1, word_count, language, status, ai_share, "
                "document_probability, confidence, text, result_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (time.strftime("%Y-%m-%d %H:%M:%S"), result.get("meta", {}).get("title"),
                 hashlib.sha1(text.encode("utf-8")).hexdigest(), result.get("meta", {}).get("word_count"),
                 result.get("meta", {}).get("language"), result.get("status"), r.get("ai_associated_share"),
                 r.get("document_probability"), r.get("confidence"), text if store_text else None,
                 json.dumps(result)))
            return int(cur.lastrowid)

    def list_analyses(self, limit: int = 100) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT id, created_at, title, word_count, language, status, ai_share, "
                             "document_probability, confidence FROM analyses ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in rows]

    def get_analysis(self, analysis_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT result_json FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
            return json.loads(row["result_json"]) if row else None

    def delete_analysis(self, analysis_id: int) -> bool:
        with self._lock, self._conn() as c:
            return c.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,)).rowcount > 0

    # ---------------------------------------------------------------- reference corpus
    def add_reference(self, title: str, text: str, fingerprints: set[int]) -> int | None:
        sha = hashlib.sha1(text.encode("utf-8")).hexdigest()
        with self._lock, self._conn() as c:
            if c.execute("SELECT id FROM reference_docs WHERE text_sha1 = ?", (sha,)).fetchone():
                return None
            cur = c.execute("INSERT INTO reference_docs (added_at, title, text_sha1, word_count, text) VALUES (?,?,?,?,?)",
                            (time.strftime("%Y-%m-%d %H:%M:%S"), title, sha, len(text.split()), text))
            doc_id = int(cur.lastrowid)
            c.executemany("INSERT INTO fingerprints (hash, doc_id) VALUES (?, ?)", [(h, doc_id) for h in fingerprints])
            return doc_id

    def list_references(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT id, added_at, title, word_count FROM reference_docs ORDER BY id DESC")]

    def delete_reference(self, doc_id: int) -> bool:
        with self._lock, self._conn() as c:
            return c.execute("DELETE FROM reference_docs WHERE id = ?", (doc_id,)).rowcount > 0

    def candidate_references(self, fingerprints: set[int], limit: int = 20) -> list[dict]:
        if not fingerprints:
            return []
        hashes = list(fingerprints)
        counts: dict[int, int] = {}
        with self._conn() as c:
            for i in range(0, len(hashes), 900):
                chunk = hashes[i:i + 900]
                q = f"SELECT doc_id, COUNT(*) AS n FROM fingerprints WHERE hash IN ({','.join('?' * len(chunk))}) GROUP BY doc_id"
                for r in c.execute(q, chunk):
                    counts[r["doc_id"]] = counts.get(r["doc_id"], 0) + r["n"]
            top = sorted(counts, key=lambda d: -counts[d])[:limit]
            out = []
            for d in top:
                r = c.execute("SELECT id, title, text FROM reference_docs WHERE id = ?", (d,)).fetchone()
                if r:
                    out.append({**dict(r), "shared_fingerprints": counts[d]})
            return out

    def all_references(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT id, title, text FROM reference_docs")]

    def get_setting(self, key: str, default=None):
        with self._conn() as c:
            r = c.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
            return json.loads(r["value"]) if r else default

    def set_setting(self, key: str, value) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                      (key, json.dumps(value)))
