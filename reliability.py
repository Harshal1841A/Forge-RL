"""
server/reliability.py
=====================
Production-grade reliability primitives:
  - RetryExecutor   : async exponential-backoff retry with jitter
  - CircuitBreaker  : half-open / open / closed state machine
  - RateLimiter     : per-key sliding-window request throttle (in-memory)
  - safe_execute    : one-liner wrapper that combines retry + circuit-breaker

Usage
-----
from server.reliability import safe_execute, CircuitBreaker, RateLimiter

result = await safe_execute(my_async_fn, arg1, arg2,
                            label="wikipedia_fetch",
                            max_retries=3, base_delay=0.5)
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, Optional

logger = logging.getLogger("forge.reliability")


# ─── Retry with exponential backoff + full jitter ────────────────────────────

RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    asyncio.TimeoutError,
    ConnectionError,
    OSError,
)

async def retry_async(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 16.0,
    retryable: tuple[type[Exception], ...] = RETRYABLE_EXCEPTIONS,
    label: str = "operation",
    **kwargs: Any,
) -> Any:
    """
    Retry *fn* up to *max_retries* times with full-jitter exponential backoff.
    Non-retryable exceptions bubble up immediately.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except retryable as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            cap = min(max_delay, base_delay * math.pow(2, attempt))
            sleep_for = random.uniform(0, cap)   # full jitter
            logger.warning(
                "[%s] attempt %d/%d failed (%s). Retrying in %.2fs…",
                label, attempt + 1, max_retries, exc, sleep_for,
            )
            await asyncio.sleep(sleep_for)
        except Exception:
            raise   # non-retryable — let caller handle
    raise last_exc  # type: ignore[misc]


# ─── Circuit Breaker ──────────────────────────────────────────────────────────

class CBState(Enum):
    CLOSED   = "closed"    # normal — requests flow through
    OPEN     = "open"      # tripped — fast-fail all requests
    HALF_OPEN = "half_open"  # probe — one trial request allowed


@dataclass
class CircuitBreaker:
    """
    Per-dependency circuit breaker.

    Parameters
    ----------
    name          : Human-readable name (for logs)
    failure_threshold : Consecutive failures before opening
    recovery_timeout  : Seconds to wait before trying half-open probe
    success_threshold : Consecutive successes in HALF_OPEN to close
    """
    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    success_threshold: int = 2

    _state: CBState = field(default=CBState.CLOSED, init=False, repr=False)
    _failures: int = field(default=0, init=False, repr=False)
    _successes: int = field(default=0, init=False, repr=False)
    _opened_at: float = field(default=0.0, init=False, repr=False)

    @property
    def state(self) -> CBState:
        if self._state == CBState.OPEN:
            if time.monotonic() - self._opened_at >= self.recovery_timeout:
                self._state = CBState.HALF_OPEN
                self._successes = 0
                logger.info("[CB:%s] → HALF_OPEN (probe allowed)", self.name)
        return self._state

    def record_success(self) -> None:
        if self.state == CBState.HALF_OPEN:
            self._successes += 1
            if self._successes >= self.success_threshold:
                self._state = CBState.CLOSED
                self._failures = 0
                logger.info("[CB:%s] → CLOSED (recovered)", self.name)
        else:
            self._failures = 0

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold or self._state == CBState.HALF_OPEN:
            self._state = CBState.OPEN
            self._opened_at = time.monotonic()
            logger.error(
                "[CB:%s] → OPEN after %d failures", self.name, self._failures
            )

    def is_allowed(self) -> bool:
        return self.state != CBState.OPEN

    async def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if not self.is_allowed():
            raise RuntimeError(
                f"Circuit breaker '{self.name}' is OPEN — dependency unavailable"
            )
        try:
            result = await fn(*args, **kwargs)
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise

    def status(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failures": self._failures,
            "successes": self._successes,
        }


# ─── Global circuit breakers (one per external dependency) ───────────────────

CIRCUIT_BREAKERS: Dict[str, CircuitBreaker] = {
    "wikipedia":  CircuitBreaker("wikipedia",  failure_threshold=4, recovery_timeout=20),
    "wayback":    CircuitBreaker("wayback",    failure_threshold=3, recovery_timeout=30),
    "wikidata":   CircuitBreaker("wikidata",   failure_threshold=4, recovery_timeout=20),
    "llm_groq":   CircuitBreaker("llm_groq",   failure_threshold=3, recovery_timeout=60),
    "llm_mistral":CircuitBreaker("llm_mistral",failure_threshold=3, recovery_timeout=60),
}


# ─── Rate Limiter (sliding-window, per key) ───────────────────────────────────

class RateLimiter:
    """
    In-memory sliding-window rate limiter.

    Parameters
    ----------
    max_requests : Maximum requests allowed within the window
    window_secs  : Window duration in seconds
    """

    def __init__(self, max_requests: int = 60, window_secs: float = 60.0):
        self.max_requests = max_requests
        self.window = window_secs
        self._buckets: Dict[str, Deque[float]] = {}

    def is_allowed(self, key: str = "global") -> bool:
        now = time.monotonic()
        bucket = self._buckets.setdefault(key, deque())
        # Evict timestamps outside the window
        while bucket and bucket[0] < now - self.window:
            bucket.popleft()
        if len(bucket) >= self.max_requests:
            return False
        bucket.append(now)
        return True

    def remaining(self, key: str = "global") -> int:
        now = time.monotonic()
        bucket = self._buckets.get(key, deque())
        active = sum(1 for t in bucket if t >= now - self.window)
        return max(0, self.max_requests - active)


# Singleton rate limiters
EPISODE_RATE_LIMITER = RateLimiter(max_requests=30, window_secs=60)   # 30 resets/min
STEP_RATE_LIMITER    = RateLimiter(max_requests=200, window_secs=60)  # 200 steps/min


# ─── safe_execute: retry + circuit-breaker combo ─────────────────────────────

async def safe_execute(
    fn: Callable[..., Any],
    *args: Any,
    label: str = "op",
    max_retries: int = 2,
    base_delay: float = 0.5,
    cb: Optional[CircuitBreaker] = None,
    fallback: Any = None,
    **kwargs: Any,
) -> Any:
    """
    Execute *fn* with retry + optional circuit-breaker protection.
    Returns *fallback* if all retries and the circuit is open — never raises.
    """
    try:
        if cb:
            return await cb.call(
                retry_async, fn, *args,
                max_retries=max_retries, base_delay=base_delay, label=label,
                **kwargs,
            )
        return await retry_async(
            fn, *args,
            max_retries=max_retries, base_delay=base_delay, label=label,
            **kwargs,
        )
    except Exception as exc:
        logger.error("[safe_execute:%s] All attempts failed: %s", label, exc)
        return fallback
