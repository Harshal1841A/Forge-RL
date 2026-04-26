"""
QuerySourceTool — free APIs: Wikipedia REST + DuckDuckGo Web Search
No API keys required. Completely free to use.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any, Dict
import httpx
from env.claim_graph import ClaimGraph
from env.utils.cache_manager import get_cache
import config

logger = logging.getLogger(__name__)


class QuerySourceTool:
    """
    Queries the root claim's domain against:
    1. DuckDuckGo Web Search (free, 0 keys required, scraped fact-check proxy)
    2. Wikipedia full-text search (completely free, 0 keys required)
    """

    async def execute(self, graph: ClaimGraph, **kwargs) -> Dict[str, Any]:
        root = graph.root
        query = root.text[:100]

        results = await asyncio.gather(
            self._wikipedia_search(query),
            self._ddg_factcheck(query),
            return_exceptions=True,
        )

        wiki_result = results[0] if not isinstance(results[0], Exception) else {}
        fc_result = results[1] if not isinstance(results[1], Exception) else {}

        # Update graph with findings
        new_contradictions = 0
        if fc_result.get("rating") in ("FALSE", "MISLEADING", "MOSTLY FALSE"):
            new_contradictions += 1

        graph.mark_retrieved(root.node_id)
        # Use the official API — previously there was a redundant direct edge mutation here
        # that bypassed discover_edges_from() and caused double-mutation
        revealed = graph.discover_edges_from(root.node_id)
        new_contradictions += sum(1 for e in revealed if e.relation in ("contradicts", "debunks"))

        return {
            "domain": root.domain,
            "trust_score": root.trust_score,
            "wikipedia_summary": wiki_result.get("summary", ""),
            "factcheck_rating": fc_result.get("rating", "UNKNOWN"),
            "factcheck_publisher": fc_result.get("publisher", ""),
            "new_nodes": 0,
            "new_contradictions": new_contradictions,
            "summary": self._make_summary(root.domain, wiki_result, fc_result),
        }

    async def _wikipedia_search(self, query: str) -> Dict[str, Any]:
        clean = query.split(".")[0][:80]   # first sentence, max 80 chars
        url = f"{config.WIKIPEDIA_API_URL}/page/summary/{clean.replace(' ', '_')}"
        cache = get_cache()
        cached = cache.get(url)
        if cached is not None:
            return cached
        if cache.internet_off:
            return cache.unavailable_response("wikipedia_offline")
        try:
            async with httpx.AsyncClient(timeout=config.TOOL_CALL_TIMEOUT_SEC) as client:
                r = await client.get(url)
                if r.status_code == 200:
                    data = r.json()
                    result = {"summary": data.get("extract", "")[:300]}
                    cache.set(url, result)
                    return result
        except Exception as e:
            logger.debug("Wikipedia search failed: %s", e)
        return {}

    async def _ddg_factcheck(self, query: str) -> Dict[str, Any]:
        """Use DuckDuckGo to search for fact-checks of the string. Requires zero API keys."""
        try:
            from duckduckgo_search import DDGS
            loop = asyncio.get_running_loop()  # FIXED: get_event_loop() deprecated in 3.10+ inside async context

            def _search():
                with DDGS() as ddgs:
                    # Search specifically for fact checking terms
                    return list(ddgs.text(f"{query} fact check OR debunked", max_results=3))

            results = await loop.run_in_executor(None, _search)

            if results:
                top_hit = results[0]
                body = top_hit.get("body", "").lower()
                rating = "UNKNOWN"

                # Naive text-classification proxy based on fact-checker linguistic patterns
                if any(w in body for w in ["false", "debunked", "fake", "hoax"]):
                    rating = "FALSE"
                elif any(w in body for w in ["true", "verified", "accurate", "real"]):
                    rating = "TRUE"
                elif any(w in body for w in ["misleading", "partially"]):
                    rating = "MISLEADING"

                pub = "Web Search"
                if "href" in top_hit:
                    # Extract domain name as publisher
                    pub = top_hit["href"].split("/")[2].replace("www.", "")

                return {
                    "rating": rating,
                    "publisher": pub,
                    "snippet": top_hit.get("body", ""),
                }
        except ImportError:
            logger.debug("duckduckgo_search not installed.")
        except Exception as e:
            logger.debug("DDG FactCheck proxy failed: %s", e)
        return {}

    def _make_summary(self, domain: str, wiki: dict, fc: dict) -> str:
        parts = [f"Source domain: {domain}"]
        if wiki.get("summary"):
            parts.append(f"Wikipedia context: {wiki['summary'][:150]}")
        if fc.get("rating"):
            parts.append(f"Fact-check rating: {fc['rating']} (by {fc.get('publisher','')})")
        return " | ".join(parts)
