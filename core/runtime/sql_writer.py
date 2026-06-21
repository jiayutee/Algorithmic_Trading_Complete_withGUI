from __future__ import annotations

from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Callable

from core.news_store import NewsStore, DEFAULT_DB


class SQLWriter:
    """Single-writer SQLite queue for news persistence."""

    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = db_path
        self._queue: Queue[dict[str, Any]] = Queue()
        self._stop_event = Event()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, fn: Callable[[NewsStore, Any], Any], *args: Any, wait: bool = True, timeout: float = 5.0, **kwargs: Any) -> dict[str, Any]:
        job: dict[str, Any] = {
            "fn": fn,
            "args": args,
            "kwargs": kwargs,
            "done": Event(),
            "result": {},
        }
        self._queue.put(job)
        if wait:
            job["done"].wait(timeout=timeout)
        return job["result"]

    def submit_news_items(self, items: Any, wait: bool = True, timeout: float = 5.0) -> dict[str, Any]:
        return self.enqueue(self._insert_news_items, items, wait=wait, timeout=timeout)

    @staticmethod
    def _insert_news_items(store: NewsStore, items: Any) -> dict[str, Any]:
        inserted = store.add_items(items or [])
        return {"ok": True, "inserted": inserted}

    def _run(self) -> None:
        store = NewsStore(self.db_path)
        try:
            while not self._stop_event.is_set() or not self._queue.empty():
                try:
                    job = self._queue.get(timeout=0.2)
                except Empty:
                    continue

                try:
                    fn = job["fn"]
                    args = job.get("args", ())
                    kwargs = job.get("kwargs", {})
                    result = fn(store, *args, **kwargs)
                    if not isinstance(result, dict):
                        result = {"ok": True, "result": result}
                    job["result"] = result
                except Exception as exc:  # pragma: no cover - defensive
                    job["result"] = {"ok": False, "error": str(exc)}
                finally:
                    job["done"].set()
                    self._queue.task_done()
        finally:
            store.close()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        try:
            self._thread.join(timeout=timeout)
        except Exception:
            pass
