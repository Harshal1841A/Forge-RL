"""
TraceOriginTool — Wayback Machine (archive.org) + Wikidata SPARQL
Both are completely free, no API key required.
"""

from __future__ import annotations
import logging
from typing import Any, Dict
import httpx
from env.claim_graph import ClaimGraph
from env.utils.cache_manager import get_cache
import config

logger = logging.getLogger(__name__)


class TraceOriginTool:
    """
    Traces the origin of a claim's source URL via:
    1. Wayback Machine availability API (free, no key)
    2. Wikidata SPARQL for entity/organisation lookup (free, no key)
    """

    async def execute(self, graph: ClaimGraph, **kwargs) -> Dict[str, Any]:
        root = graph.root
        source_url = root.source_url

        wayback, wikidata = {}, {}
        try:
            wayback = await self._wayback_check(source_url)
        except Exception as e:
            logger.debug("Wayback failed: %s", e)

        # Extract domain entity for Wikidata lookup
        domain_name = root.domain.replace("www.", "").split(".")[0]
        try:
            wikidata = await self._wikidata_lookup(domain_name)
        except Exception as e:
            logger.debug("Wikidata failed: %s", e)

        # Reveal propagation nodes in graph
        new_nodes = 0
        for node_id, node in graph.nodes.items():
            if not node.retrieved and node.metadata.get("is_bot"):
                graph.mark_retrieved(node_id)
                graph.discover_edges_from(node_id)
                new_nodes += 1

        origin_suspicious = (
            root.trust_score < 0.3
            or not wayback.get("available", True)
            or wikidata.get("country_of_origin") in ("Russia", "Iran", "North Korea")
        )

        return {
            "source_url": source_url,
            "wayback_available": wayback.get("available", "unknown"),
            "earliest_snapshot": wayback.get("earliest", "unknown"),
            "wikidata_entity": wikidata.get("label", ""),
            "wikidata_country": wikidata.get("country_of_origin", "unknown"),
            "origin_suspicious": origin_suspicious,
            "new_nodes": new_nodes,
            "new_contradictions": 1 if origin_suspicious and new_nodes > 0 else 0,
            "summary": self._summary(wayback, wikidata, origin_suspicious, new_nodes),
        }

    async def _wayback_check(self, url: str) -> Dict[str, Any]:
        cache = get_cache()
        api = f"{config.WAYBACK_API_URL}?url={url}&timestamp=20100101"
        cached = cache.get(api)
        if cached is not None:
            return cached
        if cache.internet_off:
            return {"available": False}
        async with httpx.AsyncClient(timeout=config.TOOL_CALL_TIMEOUT_SEC) as client:
            r = await client.get(api)
            if r.status_code == 200:
                data = r.json()
                snap = data.get("archived_snapshots", {}).get("closest", {})
                result = {
                    "available": snap.get("available", False),
                    "earliest": snap.get("timestamp", ""),
                    "url": snap.get("url", ""),
                }
                cache.set(api, result)
                return result
        return {"available": False}

    async def _wikidata_lookup(self, term: str) -> Dict[str, Any]:
        sparql_query = f"""
        SELECT ?item ?itemLabel ?countryLabel WHERE {{
          ?item wikibase:sitelinks ?links .
          ?item rdfs:label "{term}"@en .
          OPTIONAL {{ ?item wdt:P17 ?country . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }} LIMIT 1
        """
        headers = {"Accept": "application/sparql-results+json", "User-Agent": "FORGE/2.0"}
        params = {"query": sparql_query, "format": "json"}
        cache = get_cache()
        cache_key = f"{config.WIKIDATA_SPARQL_URL}?term={term}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        if cache.internet_off:
            return {}
        async with httpx.AsyncClient(timeout=config.TOOL_CALL_TIMEOUT_SEC) as client:
            r = await client.get(config.WIKIDATA_SPARQL_URL, params=params, headers=headers)
            if r.status_code == 200:
                data = r.json()
                bindings = data.get("results", {}).get("bindings", [])
                if bindings:
                    b = bindings[0]
                    result = {
                        "label": b.get("itemLabel", {}).get("value", ""),
                        "country_of_origin": b.get("countryLabel", {}).get("value", ""),
                    }
                    cache.set(cache_key, result)
                    return result
        return {}

    def _summary(self, wayback, wikidata, suspicious, new_nodes) -> str:
        parts = []
        if wayback.get("earliest"):
            parts.append(f"Earliest archive: {wayback['earliest']}")
        if wikidata.get("country_of_origin"):
            parts.append(f"Entity country: {wikidata['country_of_origin']}")
        parts.append(f"Origin suspicious: {suspicious}")
        parts.append(f"Bot nodes revealed: {new_nodes}")
        return " | ".join(parts)
