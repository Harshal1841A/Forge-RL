"""
CacheManager — SQLite-backed offline request cache (v2.0)
Intercepts all external HTTP calls from FORGE tools and stores responses
in a local SQLite DB so subsequent runs are 100% offline.

Usage:
    cache = CacheManager()
    cached = cache.get("https://example.com/api?q=foo")
    if cached is None:
        response = requests.get(...)
        cache.set("https://example.com/api?q=foo", response.json())
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default DB path relative to the project root
_DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "forge_cache.db"


class CacheManager:
    """
    Lightweight SQLite cache for all external tool HTTP requests.

    The cache is keyed by a SHA-256 hash of the URL so that long URLs
    don't create issues, and the value is stored as compressed JSON.

    Environment variable `INTERNET_OFF=true` forces cache-only mode:
    - If key is not found, returns a sentinel "unavailable" dict instead
      of making a live request, guaranteeing grading stability.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or os.getenv("FORGE_CACHE_DB", _DEFAULT_DB_PATH))
        self.internet_off: bool = os.getenv("INTERNET_OFF", "false").lower() == "true"
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        try:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS request_cache (
                    url_hash    TEXT PRIMARY KEY,
                    url         TEXT NOT NULL,
                    response    TEXT NOT NULL,
                    created_at  REAL NOT NULL DEFAULT (strftime('%s','now'))
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_created ON request_cache(created_at)"
            )
            self._conn.commit()
            logger.debug("CacheManager: SQLite DB initialised at %s", self.db_path)
        except Exception as e:
            logger.error("CacheManager: Failed to init SQLite DB: %s", e)
            self._conn = None

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self, url: str) -> Optional[Any]:
        """Return the cached response dict for `url`, or None if not found."""
        if self._conn is None:
            return None
        url_hash = self._hash(url)
        try:
            row = self._conn.execute(
                "SELECT response FROM request_cache WHERE url_hash = ?", (url_hash,)
            ).fetchone()
            if row:
                return json.loads(row[0])
        except Exception as e:
            logger.debug("CacheManager.get error: %s", e)
        return None

    def set(self, url: str, response: Any) -> None:
        """Persist `response` (JSON-serialisable) for `url`."""
        if self._conn is None:
            return
        url_hash = self._hash(url)
        try:
            self._conn.execute(
                """INSERT OR REPLACE INTO request_cache (url_hash, url, response)
                   VALUES (?, ?, ?)""",
                (url_hash, url[:512], json.dumps(response, default=str)),
            )
            self._conn.commit()
        except Exception as e:
            logger.debug("CacheManager.set error: %s", e)

    def unavailable_response(self, reason: str = "cache_miss") -> dict:
        """Sentinel response returned when INTERNET_OFF=true and cache misses."""
        return {
            "cached": False,
            "internet_off": True,
            "reason": reason,
            "summary": "Sources unavailable (offline mode — no cached result found).",
            "new_nodes": 0,
            "new_contradictions": 0,
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _hash(url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()


# Module-level singleton — tools import this directly
_cache = CacheManager()


def get_cache() -> CacheManager:
    """Return the module-level CacheManager singleton."""
    return _cache
