"""
CrossReferenceTool — Wikipedia multi-article + DuckDuckGo Instant Answer (free)
Finds corroborating or contradicting information from multiple free sources.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List
import httpx
from env.claim_graph import ClaimGraph
from env.utils.cache_manager import get_cache
import config

logger = logging.getLogger(__name__)


class CrossReferenceTool:
    async def execute(self, graph: ClaimGraph, **kwargs) -> Dict[str, Any]:
        root_text = graph.root.text
        keywords = self._extract_keywords(root_text)

        results = []
        for kw in keywords[:3]:  # limit to 3 keywords
            try:
                wiki = await self._wiki_search_summary(kw)
                if wiki:
                    results.append({"source": "wikipedia", "keyword": kw, "text": wiki})
            except Exception as e:
                logger.debug("CrossRef wiki error: %s", e)

        # Compute contradiction score based on graph state
        new_contradictions = 0
        for edge in graph.edges:
            if not edge.discovered and edge.relation in ("contradicts", "debunks"):
                edge.discovered = True
                new_contradictions += 1

        supporting = sum(1 for r in results if self._text_supports_claim(r["text"], root_text))
        contradicting = len(results) - supporting

        return {
            "sources_checked": len(results),
            "supporting": supporting,
            "contradicting": contradicting + new_contradictions,
            "new_nodes": 0,
            "new_contradictions": new_contradictions,
            "cross_ref_results": [r["keyword"] for r in results],
            "summary": (
                f"Cross-referenced {len(results)} sources: "
                f"{supporting} supporting, {contradicting + new_contradictions} contradicting."
            ),
        }

    async def _wiki_search_summary(self, keyword: str) -> str:
        url = f"{config.WIKIPEDIA_API_URL}/page/summary/{keyword.replace(' ', '_')}"
        cache = get_cache()
        cached = cache.get(url)
        if cached is not None:
            return cached.get("text", "")
        if cache.internet_off:
            return ""
        async with httpx.AsyncClient(timeout=config.TOOL_CALL_TIMEOUT_SEC) as client:
            r = await client.get(url)
            if r.status_code == 200:
                text = r.json().get("extract", "")[:400]
                cache.set(url, {"text": text})
                return text
        return ""

    def _extract_keywords(self, text: str) -> List[str]:
        """Naive keyword extraction — proper NLP would use spaCy but keeping lightweight."""
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "and", "or", "but",
            "in", "on", "at", "to", "for", "of", "with", "by", "from", "that",
            "this", "it", "be", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "shall",
            "not", "no", "nor", "so", "yet", "both", "either", "neither",
            "according", "new", "study", "found", "shows", "reveals",
        }
        words = text.replace(",", "").replace(".", "").replace(":", "").split()
        keywords = [w.strip().lower() for w in words if w.lower() not in stop_words and len(w) > 4]
        # Return unique, length-sorted
        seen = set()
        unique = []
        for k in keywords:
            if k not in seen:
                seen.add(k)
                unique.append(k)
        return unique[:5]

    def _text_supports_claim(self, wiki_text: str, claim: str) -> bool:
        """Very rough heuristic: if claim numbers not in wiki text → not supporting."""
        import re
        numbers_in_claim = re.findall(r"\d+\.?\d*%?", claim)
        if not numbers_in_claim:
            return True
        for num in numbers_in_claim:
            if num.rstrip("%") in wiki_text:
                return True
        return False
