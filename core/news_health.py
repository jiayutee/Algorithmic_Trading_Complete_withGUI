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
one is almost always a swallowed rate-limit/retry storm).  Specific failure classes
(``rate_limited``, ``auth_failed``, ``parse_error``) are always counted as failures regardless
of response time.  ``ok_empty`` is never a failure.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict


# Outcome classes that the health registry treats as failures.
# Must stay consistent with OUTCOME_* constants in news_sources.py.
_FAIL_CLASSES = frozenset({"rate_limited", "auth_failed", "parse_error", "timeout", "error"})
# Outcome classes that are definitively NOT failures (circuit stays closed).
_OK_CLASSES = frozenset({"ok", "ok_empty"})


@dataclass
class _State:
    consecutive_failures: int = 0
    open_until: float = 0.0
    cooldown: float = 0.0
    last_reason: str = ""
    successes: int = 0
    failures: int = 0
    last_checked_at: str = ""
    last_success_at: str = ""
    last_status: str = "not_checked"
    last_item_count: int = 0
    last_elapsed_seconds: float = 0.0


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

    def record(self, name: str, n_items: int, seconds: float, error: str = "",
               failure_class: str = "") -> bool:
        """Record one outcome; returns True if it counted as a failure.

        Parameters
        ----------
        name:
            Source name (must match the source's ``.name`` attribute).
        n_items:
            Number of items delivered (0 for empty/failed fetches).
        seconds:
            Wall-clock time the fetch took.
        error:
            Legacy free-text error string (kept for backward compatibility).
            Prefer ``failure_class`` for new callers.
        failure_class:
            One of the ``OUTCOME_*`` constants from ``core.news_sources``:
            ``ok``, ``ok_empty``     → not a failure (circuit stays closed)
            ``rate_limited``         → failure (HTTP 429)
            ``auth_failed``          → failure (HTTP 401/403 or missing key)
            ``parse_error``          → failure (unparseable payload)
            ``timeout``              → failure (network/HTTP timeout)
            ``error``                → failure (other/unclassified)
            When provided, takes precedence over the legacy error/slow_empty logic
            for determining both the failure flag and ``last_status``.
        """
        # Determine failure from failure_class when provided, else fall back to
        # the legacy heuristic (bool(error) or slow empty result).
        if failure_class in _FAIL_CLASSES:
            failed = True
        elif failure_class in _OK_CLASSES:
            failed = False
        else:
            # Legacy path: no classified outcome — use error text + slow-empty heuristic
            failed = bool(error) or (n_items == 0 and seconds > self.slow_empty_seconds)

        with self._lock:
            st = self._state(name)
            st.last_checked_at = datetime.now(timezone.utc).isoformat()
            st.last_item_count = n_items
            st.last_elapsed_seconds = seconds

            # last_status: use failure_class when provided; fall back to generic labels
            if failure_class:
                if failure_class == "ok" and n_items == 0:
                    # Shouldn't happen (caller should pass ok_empty), but handle gracefully
                    st.last_status = "ok_empty"
                else:
                    st.last_status = failure_class
            else:
                st.last_status = "error" if error else ("ok" if n_items else ("slow_empty" if failed else "empty"))

            if failed:
                st.failures += 1
                st.consecutive_failures += 1
                # last_reason: prefer the human-readable error hint (e.g. "timed out
                # after 6s") when explicitly provided; otherwise store the classified
                # outcome name.  Raw exception messages with URLs or keys are excluded:
                # the pipeline passes only type(exc).__name__, not exc.__str__().
                st.last_reason = error or failure_class or f"empty after {seconds:.1f}s"
                if st.consecutive_failures >= self.threshold:
                    st.cooldown = min(self.max_cooldown, st.cooldown * 2 if st.cooldown else self.base_cooldown)
                    st.open_until = self._clock() + st.cooldown
            else:
                st.successes += 1
                st.consecutive_failures = 0
                st.cooldown = 0.0
                st.open_until = 0.0
                st.last_reason = ""
                # A fast empty response is not a failure, but also not proof of news delivery.
                if n_items:
                    st.last_success_at = st.last_checked_at
        return failed

    def record_timeout(self, name: str, budget: float) -> None:
        self.record(name, 0, budget, error=f"timed out after {budget:.0f}s",
                    failure_class="timeout")
        with self._lock:
            self._state(name).last_status = "timeout"

    def snapshot(self) -> Dict[str, dict]:
        now = self._clock()
        with self._lock:
            return {n: {"open": now < s.open_until, "retry_in": max(0.0, s.open_until - now),
                        "consecutive_failures": s.consecutive_failures, "last_reason": s.last_reason,
                        "successes": s.successes, "failures": s.failures,
                        "last_checked_at": s.last_checked_at, "last_success_at": s.last_success_at,
                        "last_status": s.last_status, "last_item_count": s.last_item_count,
                        "last_elapsed_seconds": s.last_elapsed_seconds}
                    for n, s in self._states.items()}

    def reset(self) -> None:
        with self._lock:
            self._states.clear()


HEALTH = SourceHealthRegistry()
