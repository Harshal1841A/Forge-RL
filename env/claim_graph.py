"""
ClaimGraph — structured representation of a misinformation claim
and its associated evidence network.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple
import hashlib
import json


RelationType = Literal[
    "supports", "contradicts", "cites", "shares_author",
    "co_published", "amplifies", "debunks"
]

TacticType = Literal[
    "fabricate_statistic", "strip_context", "backdate_article",
    "misattribute_quote", "amplify_via_bot_network",
    "splice_image_caption", "cherry_pick_study",
    "translate_without_context"
]


@dataclass
class ClaimNode:
    node_id: str
    text: str
    source_url: str
    domain: str
    timestamp: Optional[datetime] = None
    author: Optional[str] = None
    virality_score: float = 0.0        # normalised 0–1
    trust_score: float = 0.5           # prior credibility of domain 0–1
    retrieved: bool = False             # has the agent queried this node?
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "text": self.text[:200],
            "source_url": self.source_url,
            "domain": self.domain,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "author": self.author,
            "virality_score": round(self.virality_score, 3),
            "trust_score": round(self.trust_score, 3),
            "retrieved": self.retrieved,
        }


@dataclass
class EvidenceEdge:
    edge_id: str
    src_id: str
    tgt_id: str
    relation: RelationType
    weight: float = 1.0          # confidence of the relation
    discovered: bool = False     # has the agent uncovered this edge?

    def to_dict(self) -> dict:
        return {
            "edge_id": self.edge_id,
            "src_id": self.src_id,
            "tgt_id": self.tgt_id,
            "relation": self.relation,
            "weight": round(self.weight, 3),
            "discovered": self.discovered,
        }


@dataclass
class ClaimGraph:
    graph_id: str
    root_claim_id: str
    nodes: Dict[str, ClaimNode] = field(default_factory=dict)
    edges: List[EvidenceEdge] = field(default_factory=list)
    propagation_timeline: List[Tuple[datetime, str]] = field(default_factory=list)
    applied_tactics: List[TacticType] = field(default_factory=list)
    true_label: Literal["real", "misinfo", "satire", "out_of_context", "fabricated"] = "misinfo"
    difficulty: int = 1   # 1-4 matches curriculum stage

    # ── Derived metrics ────────────────────────────────────────────────────────

    @property
    def root(self) -> ClaimNode:
        if self.root_claim_id not in self.nodes:
            raise RuntimeError(
                f"ClaimGraph.root: root_claim_id '{self.root_claim_id}' not found. "
                f"Available node IDs: {list(self.nodes.keys())}"
            )
        return self.nodes[self.root_claim_id]

    @property
    def num_tactics(self) -> int:
        return len(self.applied_tactics)

    @property
    def network_diameter(self) -> int:
        """Approximate diameter: BFS from root."""
        if not self.edges:
            return 1
        adj: Dict[str, List[str]] = {}
        for e in self.edges:
            adj.setdefault(e.src_id, []).append(e.tgt_id)
            adj.setdefault(e.tgt_id, []).append(e.src_id)
        visited, queue, depth = {self.root_claim_id}, [self.root_claim_id], 0
        while queue:
            next_q = []
            for node in queue:
                for nb in adj.get(node, []):
                    if nb not in visited:
                        visited.add(nb)
                        next_q.append(nb)
            if next_q:
                depth += 1
            queue = next_q
        return max(depth, 1)

    @property
    def evidence_coverage(self) -> float:
        """Fraction of nodes the agent has retrieved."""
        if not self.nodes:
            return 0.0
        retrieved = sum(1 for n in self.nodes.values() if n.retrieved)
        return retrieved / len(self.nodes)

    @property
    def source_diversity_entropy(self) -> float:
        """Shannon entropy over queried domains → diversity metric."""
        from math import log2
        domain_counts: Dict[str, int] = {}
        for n in self.nodes.values():
            if n.retrieved:
                domain_counts[n.domain] = domain_counts.get(n.domain, 0) + 1
        total = sum(domain_counts.values())
        if total == 0:
            return 0.0
        return -sum((c / total) * log2(c / total) for c in domain_counts.values())

    @property
    def contradiction_surface_area(self) -> int:
        return sum(
            1 for e in self.edges
            if e.relation == "contradicts" and e.discovered
        )

    def add_node(self, node: ClaimNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: EvidenceEdge) -> None:
        self.edges.append(edge)

    def mark_retrieved(self, node_id: str) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].retrieved = True

    def discover_edges_from(self, node_id: str) -> List[EvidenceEdge]:
        """Reveal all edges incident to node_id (simulates tool call return)."""
        revealed = []
        for e in self.edges:
            if (e.src_id == node_id or e.tgt_id == node_id) and not e.discovered:
                e.discovered = True
                revealed.append(e)
        return revealed

    def wl_hash(self) -> str:
        """Weisfeiler-Lehman graph hash for exploration novelty detection."""
        node_labels = {nid: n.domain for nid, n in self.nodes.items()}
        edge_tuples = [(e.src_id, e.tgt_id, e.relation) for e in self.edges if e.discovered]
        canonical = json.dumps(
            {"nodes": sorted(node_labels.items()),
             "edges": sorted(edge_tuples)},
            sort_keys=True
        )
        return hashlib.md5(canonical.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "graph_id": self.graph_id,
            "root_claim_id": self.root_claim_id,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "applied_tactics": self.applied_tactics,
            "true_label": self.true_label,
            "difficulty": self.difficulty,
            "evidence_coverage": round(self.evidence_coverage, 3),
            "source_diversity": round(self.source_diversity_entropy, 3),
            "contradiction_surface_area": self.contradiction_surface_area,
        }
