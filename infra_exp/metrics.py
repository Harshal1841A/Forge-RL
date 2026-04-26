"""
server/metrics.py
=================
Lightweight in-process observability.

Tracks:
  - Request counts  (per route, per status code)
  - Latency buckets (p50 / p95 / p99 via reservoir sampling)
  - Error rate
  - Active episode count
  - Circuit-breaker state (imported from reliability.py)

Exposes GET /metrics  →  JSON (add a Prometheus scrape adapter later if needed)

Design decision: We use a module-level singleton rather than an ASGI middleware
object so that any route handler can import and mutate counters without circular
imports.  Thread safety is "good enough" for CPython's GIL; for true multi-
process deployments swap to Redis INCR / Prometheus multiprocess mode.
"""

from __future__ import annotations

import time
import random
import threading
from collections import defaultdict, deque
from typing import DefaultDict, Deque, Dict, List, Optional

_lock = threading.Lock()


# ─── Reservoir sampler (maintains a fixed-size representative sample) ─────────

class _Reservoir:
    """Vitter's Algorithm R — O(1) insert, O(n log n) percentile."""

    def __init__(self, capacity: int = 1024):
        self._cap = capacity
        self._buf: List[float] = []
        self._n = 0

    def add(self, value: float) -> None:
        self._n += 1
        if len(self._buf) < self._cap:
            self._buf.append(value)
        else:
            j = random.randint(0, self._n - 1)
            if j < self._cap:
                self._buf[j] = value

    def percentile(self, p: float) -> Optional[float]:
        if not self._buf:
            return None
        s = sorted(self._buf)
        idx = int(len(s) * p / 100)
        return round(s[min(idx, len(s) - 1)], 4)

    @property
    def count(self) -> int:
        return self._n


# ─── Singleton metrics registry ───────────────────────────────────────────────

class _MetricsRegistry:
    def __init__(self) -> None:
        # request counts:  route_key → status_code → count
        self._req_counts: DefaultDict[str, DefaultDict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # latency samples: route_key → Reservoir
        self._latency: DefaultDict[str, _Reservoir] = defaultdict(
            lambda: _Reservoir(capacity=2048)
        )
        # error messages ring-buffer (last 100 errors)
        self._errors: Deque[dict] = deque(maxlen=100)
        # uptime
        self._started_at: float = time.time()

    # ── write ────────────────────────────────────────────────────────────────

    def record_request(
        self,
        route: str,
        status_code: int,
        duration_ms: float,
    ) -> None:
        with _lock:
            self._req_counts[route][status_code] += 1
            self._latency[route].add(duration_ms)

    def record_error(
        self,
        route: str,
        status_code: int,
        detail: str,
        duration_ms: float,
    ) -> None:
        with _lock:
            self._req_counts[route][status_code] += 1
            self._errors.append(
                {
                    "ts": round(time.time(), 3),
                    "route": route,
                    "status": status_code,
                    "detail": detail[:200],
                    "ms": round(duration_ms, 2),
                }
            )

    # ── read ─────────────────────────────────────────────────────────────────

    def snapshot(self, circuit_breakers: Optional[list] = None, episode_count: int = 0) -> dict:
        from server.reliability import CIRCUIT_BREAKERS  # lazy import avoids circular

        uptime = round(time.time() - self._started_at, 1)

        routes: Dict[str, dict] = {}
        with _lock:
            for route, status_map in self._req_counts.items():
                total = sum(status_map.values())
                errors = sum(v for k, v in status_map.items() if k >= 400)
                res = self._latency[route]
                routes[route] = {
                    "total_requests": total,
                    "error_count":    errors,
                    "error_rate_pct": round(errors / max(total, 1) * 100, 2),
                    "latency_ms": {
                        "p50":  res.percentile(50),
                        "p95":  res.percentile(95),
                        "p99":  res.percentile(99),
                        "samples": res.count,
                    },
                    "status_codes": dict(status_map),
                }
            recent_errors = list(self._errors)

        total_req   = sum(s["total_requests"] for s in routes.values())
        total_err   = sum(s["error_count"]    for s in routes.values())

        return {
            "uptime_seconds": uptime,
            "total_requests": total_req,
            "total_errors":   total_err,
            "overall_error_rate_pct": round(total_err / max(total_req, 1) * 100, 2),
            "active_episodes": episode_count,
            "routes": routes,
            "circuit_breakers": [cb.status() for cb in CIRCUIT_BREAKERS.values()],
            "recent_errors": recent_errors[-10:],  # last 10 for /metrics response
        }


# ── module-level singleton ────────────────────────────────────────────────────
METRICS = _MetricsRegistry()
