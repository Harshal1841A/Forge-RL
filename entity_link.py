"""
EntityLinkTool — Wikidata SPARQL entity disambiguation (free, no key needed)
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List
import httpx
from env.claim_graph import ClaimGraph
from env.utils.cache_manager import get_cache
import config

logger = logging.getLogger(__name__)

_INSTITUTION_KEYWORDS = [
    "WHO", "CDC", "NASA", "MIT", "Stanford", "Harvard", "NIH",
    "FBI", "CIA", "Pentagon", "UN", "UNESCO", "IMF", "WTO",
    "Reuters", "BBC", "AP", "NYT", "Washington Post", "Nature", "Science",
]


class EntityLinkTool:
    async def execute(self, graph: ClaimGraph, **kwargs) -> Dict[str, Any]:
        text = graph.root.text
        found_entities = self._detect_entities(text)
        verified = []

        for entity in found_entities[:3]:
            entity_name = entity["entity"]  # FIXED: entity is a dict; extract the string label
            try:
                wd_result = await self._wikidata_entity_search(entity_name)  # FIXED: pass string not dict
                if wd_result:
                    verified.append({
                        "entity": entity_name,  # FIXED: store the string, not the dict
                        "wikidata_id": wd_result.get("id"),
                        "description": wd_result.get("description", "")[:100],
                        "verified": True,
                    })
            except Exception as e:
                logger.debug("EntityLink error for %s: %s", entity_name, e)

        # Did the claim misattribute to a real institution?
        misattribution_detected = any(
            e["entity"].upper() in text.upper() and
            "no such" in e.get("description", "").lower()
            for e in verified
        )

        return {
            "entities_found": [e["entity"] for e in found_entities[:5]],
            "entities_verified": verified,
            "misattribution_suspected": misattribution_detected,
            "new_nodes": len(verified),
            "new_contradictions": 1 if misattribution_detected else 0,
            "summary": (
                f"Linked {len(verified)}/{len(found_entities)} entities. "
                f"Misattribution suspected: {misattribution_detected}."
            ),
        }

    def _detect_entities(self, text: str) -> List[Dict[str, str]]:
        found = []
        for kw in _INSTITUTION_KEYWORDS:
            if kw.upper() in text.upper():
                found.append({"entity": kw, "type": "institution"})
        # Simple number / statistic detection
        import re
        stats = re.findall(r"\d+\.?\d*\s*%", text)
        for s in stats[:2]:
            found.append({"entity": s, "type": "statistic"})
        return found

    async def _wikidata_entity_search(self, label: str) -> Dict[str, Any]:
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "search": label,
            "language": "en",
            "format": "json",
            "limit": 1,
        }
        cache = get_cache()
        cache_key = f"wikidata_entity:{label}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        if cache.internet_off:
            return {}
        async with httpx.AsyncClient(timeout=config.TOOL_CALL_TIMEOUT_SEC) as client:
            r = await client.get(url, params=params)
            if r.status_code == 200:
                results = r.json().get("search", [])
                if results:
                    result = {
                        "id": results[0].get("id"),
                        "description": results[0].get("description", ""),
                    }
                    cache.set(cache_key, result)
                    return result
        return {}
