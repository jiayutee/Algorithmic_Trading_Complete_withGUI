#!/usr/bin/env python3
"""Smoke-test harness for live news search sources."""

from __future__ import annotations

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.news_sources import BraveSearchSource, DuckDuckGoSource
from core.logger import logger as project_logger


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_QUERY = "AAPL stock news"


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage().strip()
        if message:
            self.messages.append(message)


def _sample(items):
    return [
        {"headline": item.headline, "url": item.url, "source": item.source}
        for item in items[:3]
    ]


def _run_with_captured_logs(func):
    handler = _CaptureHandler()
    handler.setLevel(logging.WARNING)
    saved_handlers = list(project_logger.handlers)
    saved_level = project_logger.level
    project_logger.handlers = [handler]
    project_logger.setLevel(logging.WARNING)
    try:
        result = func()
    finally:
        project_logger.handlers = saved_handlers
        project_logger.setLevel(saved_level)
    return result, handler.messages


def main() -> int:
    query = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY
    brave_key = os.getenv("BRAVE_API_KEY", "").strip()

    brave_error = ""
    brave_items = []
    if brave_key:
        brave_source = BraveSearchSource(api_key=brave_key)

        def fetch_brave():
            return brave_source.fetch(query, limit=5)

        brave_items, brave_logs = _run_with_captured_logs(fetch_brave)
        if brave_logs and not brave_items:
            brave_error = "; ".join(brave_logs)
    else:
        brave_error = "BRAVE_API_KEY is not set"

    ddg_source = DuckDuckGoSource()
    ddg_source.session.headers.update({"User-Agent": USER_AGENT})

    def fetch_ddg():
        return ddg_source.fetch(query, limit=5)

    ddg_items, ddg_logs = _run_with_captured_logs(fetch_ddg)
    ddg_error = "; ".join(ddg_logs) if ddg_logs and not ddg_items else ""

    output = {
        "query": query,
        "brave_count": len(brave_items),
        "ddg_count": len(ddg_items),
        "brave_sample": _sample(brave_items),
        "ddg_sample": _sample(ddg_items),
        "brave_error": brave_error,
        "ddg_error": ddg_error,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
