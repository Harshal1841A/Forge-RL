"""
TemporalAuditTool — Wayback Machine timestamp verification (free)
Detects backdated articles and mismatched claim/event timelines.
"""

from __future__ import annotations
import logging
from datetime import datetime, timedelta  # FIXED: removed unused timezone import
from typing import Any, Dict, Optional
import httpx
from env.claim_graph import ClaimGraph
from env.utils.cache_manager import get_cache
import config

logger = logging.getLogger(__name__)


class TemporalAuditTool:
    async def execute(self, graph: ClaimGraph, **kwargs) -> Dict[str, Any]:
        root = graph.root
        claimed_date = root.timestamp
        source_url = root.source_url

        # Check Wayback Machine for earliest snapshot
        earliest_archive: Optional[str] = None
        try:
            earliest_archive = await self._earliest_wayback(source_url)
        except Exception as e:
            logger.debug("Wayback temporal check failed: %s", e)

        # Detect temporal anomalies
        anomalies = []
        new_contradictions = 0

        # 1. Article exists before claimed event?
        if earliest_archive and claimed_date:
            try:
                archive_dt = datetime.strptime(earliest_archive[:8], "%Y%m%d")  # FIXED: keep naive to match claimed_date
                # FIXED: strip tzinfo from claimed_date if present so both sides are naive UTC
                claim_dt = claimed_date.replace(tzinfo=None) if claimed_date.tzinfo is not None else claimed_date
                if archive_dt < claim_dt - timedelta(days=30):
                    anomalies.append("article_predates_claimed_event")
                    new_contradictions += 1
            except ValueError:
                pass

        # 2. Check for backdating tactic in graph
        if "backdate_article" in graph.applied_tactics:
            anomalies.append("backdate_tactic_applied")
            new_contradictions += 1

        # 3. Old content re-shared as new
        if any(
            n.metadata.get("origin_year", 9999) < 2020
            for n in graph.nodes.values()
        ):
            anomalies.append("old_content_recirculated")
            new_contradictions += 1

        # Reveal contradicting edges if anomaly detected
        if anomalies:
            for edge in graph.edges:
                if not edge.discovered and edge.relation == "contradicts":
                    edge.discovered = True
                    break   # reveal one per call

        return {
            "claimed_date": str(claimed_date) if claimed_date else "unknown",
            "earliest_archive": earliest_archive or "not_found",
            "anomalies": anomalies,
            "temporal_anomaly_detected": len(anomalies) > 0,
            "new_nodes": 0,
            "new_contradictions": min(new_contradictions, 2),
            "summary": (
                f"Temporal audit: {len(anomalies)} anomalies detected. "
                f"{', '.join(anomalies) or 'None'}"
            ),
        }

    async def _earliest_wayback(self, url: str) -> Optional[str]:
        """Get earliest snapshot timestamp from Wayback CDX API (free)."""
        cdx_url = "http://web.archive.org/cdx/search/cdx"
        params = {
            "url": url,
            "output": "json",
            "limit": 1,
            "from": "20050101",
            "fl": "timestamp",
            "filter": "statuscode:200",
        }
        cache = get_cache()
        cache_key = f"wayback_cdx:{url}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached.get("timestamp")
        if cache.internet_off:
            return None
        async with httpx.AsyncClient(timeout=config.TOOL_CALL_TIMEOUT_SEC) as client:
            r = await client.get(cdx_url, params=params)
            if r.status_code == 200:
                rows = r.json()
                if len(rows) > 1:   # first row is header
                    ts = rows[1][0]
                    cache.set(cache_key, {"timestamp": ts})
                    return ts
        return None
