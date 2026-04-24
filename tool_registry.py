"""
Tool Registry — abstraction layer over all investigative tools.
SimulatedToolRegistry: 100% offline, no API keys needed.
ToolRegistry: Live calls to free public APIs.
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import sqlite3
from typing import Any, Dict

import config
from env.claim_graph import ClaimGraph

logger = logging.getLogger(__name__)


# ─── Simulated Tool Registry (default, zero cost) ────────────────────────────

class SimulatedToolRegistry:
    """
    Deterministic simulations of all tools based on graph structure.
    Used for training — no network calls, instant, reproducible.
    """

    def call(self, tool_name: str, graph: ClaimGraph, **kwargs) -> Dict[str, Any]:
        handler = getattr(self, f"_sim_{tool_name}", self._sim_unknown)
        return handler(graph, **kwargs)

    def _sim_query_source(self, graph: ClaimGraph, **_) -> Dict[str, Any]:
        root = graph.root
        revealed = graph.discover_edges_from(root.node_id)
        graph.mark_retrieved(root.node_id)
        trust = root.trust_score
        return {
            "domain": root.domain,
            "trust_score": trust,
            "credibility": "low" if trust < 0.4 else "medium" if trust < 0.7 else "high",
            "new_nodes": 0,
            "new_contradictions": sum(1 for e in revealed if e.relation == "contradicts"),
            "edges_revealed": len(revealed),
            "summary": f"Source '{root.domain}' has trust score {trust:.2f}.",
        }

    def _sim_trace_origin(self, graph: ClaimGraph, **_) -> Dict[str, Any]:
        # Reveal propagation chain nodes
        new_nodes = 0
        new_contradictions = 0
        for node_id, node in list(graph.nodes.items()):
            if not node.retrieved and node.metadata.get("is_bot"):
                graph.mark_retrieved(node_id)
                revealed = graph.discover_edges_from(node_id)
                new_nodes += 1
                new_contradictions += sum(1 for e in revealed if e.relation == "contradicts")
        return {
            "origin_detected": new_nodes > 0,
            "bot_nodes_found": new_nodes,
            "propagation_depth": graph.network_diameter,
            "new_nodes": new_nodes,
            "new_contradictions": new_contradictions,
            "summary": f"Traced {new_nodes} bot/amplifier nodes in propagation chain.",
        }

    def _sim_cross_reference(self, graph: ClaimGraph, **_) -> Dict[str, Any]:
        new_contradictions = 0
        for edge in graph.edges:
            if not edge.discovered and edge.relation in ("contradicts", "debunks"):
                edge.discovered = True
                new_contradictions += 1
        return {
            "contradictions_found": new_contradictions,
            "new_nodes": 0,
            "new_contradictions": new_contradictions,
            "summary": f"Cross-reference revealed {new_contradictions} contradicting sources.",
        }

    def _sim_request_context(self, graph: ClaimGraph, **_) -> Dict[str, Any]:
        # Reveal archive/authority nodes
        new_nodes = 0
        for node_id, node in list(graph.nodes.items()):
            if not node.retrieved and node.trust_score > 0.8:
                graph.mark_retrieved(node_id)
                graph.discover_edges_from(node_id)
                new_nodes += 1
        return {
            "context_retrieved": new_nodes > 0,
            "new_nodes": new_nodes,
            "new_contradictions": 0,
            "summary": f"Requested context from {new_nodes} high-trust sources.",
        }

    def _sim_entity_link(self, graph: ClaimGraph, **_) -> Dict[str, Any]:
        # Simulate Wikidata entity linking — whole-word regex to avoid
        # false-positive on English pronoun "who" matching WHO (org)
        import re
        root_text = graph.root.text.lower()
        entities = []
        org_keywords = ["cdc", "nasa", "mit", "stanford", "who.int", "who",
                        "world health organization", "fda", "nih", "reuters"]
        if any(re.search(r'\b' + re.escape(kw) + r'\b', root_text) for kw in org_keywords):
            entities.append({"entity": "recognized_institution", "confidence": 0.9})
        if any(word in root_text for word in ["%", "percent", "study", "research"]):
            entities.append({"entity": "statistical_claim", "confidence": 0.85})
        return {
            "entities": entities,
            "new_nodes": len(entities),
            "new_contradictions": 0,
            "summary": f"Linked {len(entities)} entities from claim text.",
        }

    def _sim_temporal_audit(self, graph: ClaimGraph, **_) -> Dict[str, Any]:
        root = graph.root
        # Check for backdating tactics
        has_backdate = "backdate_article" in graph.applied_tactics
        anomaly = has_backdate or (
            root.timestamp is not None and
            any(n.metadata.get("origin_year", 9999) < 2020
                for n in graph.nodes.values())
        )
        return {
            "temporal_anomaly": anomaly,
            "claim_timestamp": str(root.timestamp),
            "new_nodes": 0,
            "new_contradictions": 1 if anomaly else 0,
            "summary": "Temporal anomaly detected." if anomaly else "No temporal anomaly.",
        }

    def _sim_network_cluster(self, graph: ClaimGraph, **_) -> Dict[str, Any]:
        bot_nodes = [n for n in graph.nodes.values() if n.metadata.get("is_bot")]
        for n in bot_nodes:
            graph.mark_retrieved(n.node_id)
            graph.discover_edges_from(n.node_id)
        cluster_detected = len(bot_nodes) >= 2
        return {
            "cluster_detected": cluster_detected,
            "bot_accounts_found": len(bot_nodes),
            "new_nodes": max(0, len(bot_nodes) - 1),
            "new_contradictions": 0,
            "summary": f"Network cluster analysis: {len(bot_nodes)} bot accounts found.",
        }

    def _sim_flag_manipulation(self, graph: ClaimGraph, **_) -> Dict[str, Any]:
        return {"flagged": True, "new_nodes": 0, "new_contradictions": 0}

    def _sim_unknown(self, graph: ClaimGraph, **_) -> Dict[str, Any]:
        return {"error": "unknown_tool", "new_nodes": 0, "new_contradictions": 0}

    def close(self):
        """Placeholder for consistency."""
        pass


# ─── Live Tool Registry (Free APIs) ──────────────────────────────────────────

class ToolRegistry:
    """
    Live tool registry using only FREE public APIs:
    - Wikipedia REST API
    - Wayback Machine API (archive.org)
    - Wikidata SPARQL
    - Google Fact Check API (1000 req/day free)
    """

    def __init__(self):
        from tools.query_source import QuerySourceTool
        from tools.trace_origin import TraceOriginTool
        from tools.cross_reference import CrossReferenceTool
        from tools.entity_link import EntityLinkTool
        from tools.temporal_audit import TemporalAuditTool
        from tools.network_cluster import NetworkClusterTool

        self._tools = {
            "query_source": QuerySourceTool(),
            "trace_origin": TraceOriginTool(),
            "cross_reference": CrossReferenceTool(),
            "request_context": QuerySourceTool(),   # reuses source tool with context flag
            "entity_link": EntityLinkTool(),
            "temporal_audit": TemporalAuditTool(),
            "network_cluster": NetworkClusterTool(),
        }
        # Singleton fallback for offline mode — created once, not per-call
        self._sim = SimulatedToolRegistry()

        db_path = config.DATABASE_URL.replace("sqlite:///", "")
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._cursor = self._conn.cursor()
        self._cursor.execute(
            "CREATE TABLE IF NOT EXISTS tool_cache (cache_key TEXT PRIMARY KEY, result_json TEXT)"
        )
        self._conn.commit()

    def __del__(self):
        """Safely close sqlite connection to prevent resource leaks/locks."""
        self.close()

    def close(self) -> None:
        """Explicitly close the SQLite connection. Call this instead of relying on __del__."""
        try:
            if hasattr(self, "_cursor") and self._cursor:
                self._cursor.close()
                self._cursor = None
        except Exception as e:
            logger.debug("Error closing cursor: %s", e)
        try:
            if hasattr(self, "_conn") and self._conn:
                self._conn.close()
                self._conn = None
        except Exception as e:
            logger.debug("Error closing connection: %s", e)

    def call(self, tool_name: str, graph: ClaimGraph, **kwargs) -> Dict[str, Any]:

        # Keyed by tool + root claim + graph state hash — fixes stale-result bug where
        # all steps in the same episode shared the same cache key (old: step always = 0)
        cache_key = f"{tool_name}:{graph.root_claim_id}:{graph.wl_hash()}"

        self._cursor.execute("SELECT result_json FROM tool_cache WHERE cache_key = ?", (cache_key,))
        row = self._cursor.fetchone()
        if row:
            try:
                cache_result = json.loads(row[0])
                return cache_result
            except json.JSONDecodeError:
                pass

        tool = self._tools.get(tool_name)
        if tool is None:
            return {"error": "tool_not_found", "new_nodes": 0, "new_contradictions": 0}

        try:
            if os.getenv("INTERNET_OFF", "false").lower() == "true":
                logger.debug(f"INTERNET_OFF is true, simulating {tool_name}")
                result = self._sim.call(tool_name, graph, **kwargs)
            else:
                try:
                    result = asyncio.run(tool.execute(graph, **kwargs))
                except RuntimeError:
                    # Already inside a running event loop (e.g. FastAPI / Jupyter)
                    loop = asyncio.new_event_loop()
                    try:
                        result = loop.run_until_complete(tool.execute(graph, **kwargs))
                    finally:
                        loop.close()   # guaranteed close even on exception
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            result = self._sim.call(tool_name, graph, **kwargs)

        try:
            self._cursor.execute(
                "INSERT OR REPLACE INTO tool_cache (cache_key, result_json) VALUES (?, ?)",
                (cache_key, json.dumps(result))
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"Failed to write to tool_cache: {e}")

        return result
