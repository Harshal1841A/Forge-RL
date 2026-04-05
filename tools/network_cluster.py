"""
NetworkClusterTool — Graph-based bot network detection (simulated + heuristic)
No external API required. Uses structural graph analysis on the ClaimGraph.
"""

from __future__ import annotations
import logging
from collections import defaultdict
from typing import Any, Dict, List
from env.claim_graph import ClaimGraph

logger = logging.getLogger(__name__)

_BOT_DOMAIN_SIGNALS = [
    "alert", "truth", "exposed", "leaked", "breaking", "freedom",
    "patriot", "woke", "resist", "realinfo", "uncensored", "insider",
    "secret", "warn", "nowtruth", "unfiltered", "realfacts",
]

_BOT_TLDS = [".net", ".org", ".info", ".co", ".io"]


class NetworkClusterTool:
    async def execute(self, graph: ClaimGraph, **kwargs) -> Dict[str, Any]:
        # Structural analysis of graph
        bot_nodes = self._detect_bot_nodes(graph)
        clusters = self._find_clusters(graph, bot_nodes)
        amplification_ratio = self._compute_amplification_ratio(graph)
        # amplification_ratio is bounded [0, 1] — threshold must be reachable
        coordinated = len(clusters) > 0 and amplification_ratio > 0.5

        # Reveal all bot nodes in graph
        new_nodes = 0
        for node_id in bot_nodes:
            if not graph.nodes[node_id].retrieved:
                graph.mark_retrieved(node_id)
                graph.discover_edges_from(node_id)
                new_nodes += 1

        return {
            "bot_nodes_detected": len(bot_nodes),
            "clusters_found": len(clusters),
            "amplification_ratio": round(amplification_ratio, 2),
            "coordinated_campaign_suspected": coordinated,
            "bot_domains": list({graph.nodes[n].domain for n in bot_nodes})[:5],
            "new_nodes": new_nodes,
            "new_contradictions": 1 if coordinated and new_nodes > 1 else 0,
            "summary": (
                f"Network analysis: {len(bot_nodes)} bot nodes, "
                f"{len(clusters)} cluster(s), "
                f"amplification ×{amplification_ratio:.1f}. "
                f"Coordinated: {coordinated}."
            ),
        }

    def _detect_bot_nodes(self, graph: ClaimGraph) -> List[str]:
        bot_ids = []
        for node_id, node in graph.nodes.items():
            if node.metadata.get("is_bot"):
                bot_ids.append(node_id)
                continue
            # Heuristic: suspicious domain signals
            domain_lower = node.domain.lower()
            if any(sig in domain_lower for sig in _BOT_DOMAIN_SIGNALS):
                bot_ids.append(node_id)
                continue
            # High virality + very low trust = suspected bot amplification
            if node.virality_score > 0.7 and node.trust_score < 0.2:
                bot_ids.append(node_id)
        return list(set(bot_ids))

    def _find_clusters(self, graph: ClaimGraph, bot_ids: List[str]) -> List[List[str]]:
        """Union-Find clustering on co-published bot nodes."""
        parent = {b: b for b in bot_ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for edge in graph.edges:
            if edge.relation == "co_published":
                if edge.src_id in parent and edge.tgt_id in parent:
                    union(edge.src_id, edge.tgt_id)

        clusters: Dict[str, List[str]] = defaultdict(list)
        for b in bot_ids:
            clusters[find(b)].append(b)
        return [v for v in clusters.values() if len(v) > 1]

    def _compute_amplification_ratio(self, graph: ClaimGraph) -> float:
        """Ratio of amplification edges to total edges."""
        if not graph.edges:
            return 0.0
        amp_edges = sum(1 for e in graph.edges if e.relation == "amplifies")
        return amp_edges / len(graph.edges)
