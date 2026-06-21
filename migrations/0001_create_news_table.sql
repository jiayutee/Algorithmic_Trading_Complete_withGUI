-- migrations/0001_create_news_table.sql
CREATE TABLE IF NOT EXISTS news (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  datetime_utc TEXT NOT NULL,
  source TEXT,
  headline TEXT,
  headline_hash TEXT,
  url TEXT UNIQUE,
  summary TEXT,
  content TEXT,
  language TEXT,
  tickers TEXT,
  entities TEXT,
  event_type TEXT,
  sentiment TEXT,
  impact_score REAL,
  source_reliability REAL,
  metadata TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_news_datetime ON news(datetime_utc);
CREATE INDEX IF NOT EXISTS idx_news_headline_hash ON news(headline_hash);
CREATE INDEX IF NOT EXISTS idx_news_url ON news(url);

CREATE TABLE IF NOT EXISTS ingest_meta (
  key TEXT PRIMARY KEY,
  value TEXT,
  updated_at TEXT DEFAULT (datetime('now'))
);
