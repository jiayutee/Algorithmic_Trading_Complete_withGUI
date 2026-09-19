"""Per-source health tracking for the news pipeline (Phase 11.0).

News sources are free, rate-limited web endpoints. When one starts answering with 429s or
hanging, retrying it on every refresh costs tens of seconds and gains nothing. A tiny circuit
breaker fixes that: after ``threshold`` consecutive failures a source is skipped for a
cool-down (which doubles each time it fails again, up to ``max_cooldown``); the first call
after the cool-down is a trial, and one success closes the circuit.

The registry is process-wide (module-level ``HEALTH``) because the desktop app builds a fresh
pipeline for every refresh -- per-pipeline state would be forgotten immediately.

What counts as a failure: an exception, a timeout, or an *empty* result that took longer than
``slow_empty_seconds`` (a quick empty answer just means "no news for this query"; a slow empty
one is almost always a swallowed rate-limit/retry storm).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class _State:
    consecutive_failures: int = 0
    open_until: float = 0.0
    cooldown: float = 0.0
    last_reason: str = ""
    successes: int = 0
    failures: int = 0


class SourceHealthRegistry:
    def __init__(self, threshold: int = 2, base_cooldown: float = 120.0, max_cooldown: float = 1800.0,
                 slow_empty_seconds: float = 3.0, clock: Callable[[], float] = time.monotonic):
        self.threshold = threshold
        self.base_cooldown = base_cooldown
        self.max_cooldown = max_cooldown
        self.slow_empty_seconds = slow_empty_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._states: Dict[str, _State] = {}

    def _state(self, name: str) -> _State:
        return self._states.setdefault(name, _State())

    def allow(self, name: str) -> bool:
        """False while the circuit is open. After the cool-down the next call is a trial."""
        with self._lock:
            return self._clock() >= self._state(name).open_until

    def seconds_until_retry(self, name: str) -> float:
        with self._lock:
            return max(0.0, self._state(name).open_until - self._clock())

    def record(self, name: str, n_items: int, seconds: float, error: str = "") -> bool:
        """Record one outcome; returns True if it counted as a failure."""
        failed = bool(error) or (n_items == 0 and seconds > self.slow_empty_seconds)
        with self._lock:
            st = self._state(name)
            if failed:
                st.failures += 1
                st.consecutive_failures += 1
                st.last_reason = error or f"empty after {seconds:.1f}s"
                if st.consecutive_failures >= self.threshold:
                    st.cooldown = min(self.max_cooldown, st.cooldown * 2 if st.cooldown else self.base_cooldown)
                    st.open_until = self._clock() + st.cooldown
            else:
                st.successes += 1
                st.consecutive_failures = 0
                st.cooldown = 0.0
                st.open_until = 0.0
        return failed

    def record_timeout(self, name: str, budget: float) -> None:
        self.record(name, 0, budget, error=f"timed out after {budget:.0f}s")

    def snapshot(self) -> Dict[str, dict]:
        now = self._clock()
        with self._lock:
            return {n: {"open": now < s.open_until, "retry_in": max(0.0, s.open_until - now),
                        "consecutive_failures": s.consecutive_failures, "last_reason": s.last_reason,
                        "successes": s.successes, "failures": s.failures}
                    for n, s in self._states.items()}

    def reset(self) -> None:
        with self._lock:
            self._states.clear()


HEALTH = SourceHealthRegistry()
