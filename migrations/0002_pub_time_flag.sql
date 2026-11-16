-- migrations/0002_pub_time_flag.sql
-- Additive. Rows written before this migration keep 0 ("unknown"); they are never rewritten.
ALTER TABLE news ADD COLUMN pub_time_is_fetch_time INTEGER DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_news_headline_date ON news(headline_hash, date(datetime_utc));
