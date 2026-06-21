from __future__ import annotations
import sqlite3
import json
import hashlib
from datetime import datetime
from typing import Iterable, List, Optional, Dict, Any
from core.news_sources import NewsItem, canonicalize_url
from core.logger import logger

DEFAULT_DB = "news_store.sqlite3"


def _iso(dt: Optional[datetime]):
    if dt is None:
        return None
    try:
        return dt.isoformat()
    except Exception:
        return str(dt)


def _headline_hash(headline: str) -> str:
    if not headline:
        return ""
    return hashlib.sha256((headline or "").strip().lower().encode("utf-8")).hexdigest()


def _item_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


class NewsStore:
    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.row_factory = sqlite3.Row
        self._ensure_tables()

    def _ensure_tables(self):
        cur = self.conn.cursor()
        try:
            with open("migrations/0001_create_news_table.sql", "r") as f:
                cur.executescript(f.read())
            self.conn.commit()
        except FileNotFoundError:
            logger.warning("Migration file not found: migrations/0001_create_news_table.sql")
        except Exception as exc:
            logger.exception("Failed to ensure news tables: %s", exc)

    def add_items(self, items: Iterable[NewsItem]) -> int:
        """
        Insert items into the store. Deduplication rules:
         - If canonical URL present, use INSERT OR IGNORE on url UNIQUE constraint.
         - Otherwise dedupe on headline_hash.
        Returns number of inserted rows.
        """
        inserted = 0
        cur = self.conn.cursor()
        for item in items:
            url = canonicalize_url(_item_value(item, "url", "") or "")
            headline = _item_value(item, "headline", "") or ""
            hh = _headline_hash(headline)
            payload = (
                _iso(_item_value(item, "datetime_utc", None)),
                _item_value(item, "source", None),
                headline,
                hh,
                url or None,
                _item_value(item, "summary", None),
                _item_value(item, "content", None),
                _item_value(item, "language", None) or "en",
                json.dumps(_item_value(item, "tickers", []) or []),
                json.dumps(_item_value(item, "entities", {}) or {}),
                _item_value(item, "event_type", None),
                json.dumps(_item_value(item, "sentiment", {}) or {}),
                float(_item_value(item, "impact_score", 0.0) or 0.0),
                float(_item_value(item, "source_reliability", 0.5) or 0.5),
                json.dumps(_item_value(item, "metadata", {}) or {}),
            )
            try:
                cur.execute(
                    """
                    INSERT OR IGNORE INTO news
                    (datetime_utc, source, headline, headline_hash, url, summary, content, language, tickers, entities, event_type, sentiment, impact_score, source_reliability, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    payload,
                )
                if cur.rowcount:
                    inserted += cur.rowcount
            except Exception as exc:
                logger.warning("Failed to insert news item (%s): %s", headline[:80], exc)
        self.conn.commit()
        return inserted

    def query_by_ticker(self, ticker: str, since_iso: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        params = []
        q = "SELECT * FROM news WHERE tickers LIKE ?"
        params.append(f'%"{ticker}"%')
        if since_iso:
            q += " AND datetime(datetime_utc) >= datetime(?)"
            params.append(since_iso)
        q += " ORDER BY datetime(datetime_utc) DESC LIMIT ?"
        params.append(limit)
        cur.execute(q, params)
        rows = cur.fetchall()
        result = []
        for r in rows:
            obj = dict(r)
            for field in ("tickers", "entities", "sentiment", "metadata"):
                try:
                    obj[field] = json.loads(obj.get(field) or "null")
                except Exception:
                    obj[field] = obj.get(field)
            result.append(obj)
        return result

    def set_meta(self, key: str, value: str):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO ingest_meta(key, value, updated_at) VALUES (?, ?, datetime('now')) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=datetime('now')",
            (key, value),
        )
        self.conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT value FROM ingest_meta WHERE key=? LIMIT 1", (key,))
        row = cur.fetchone()
        return row["value"] if row else None

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
