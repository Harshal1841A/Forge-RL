import json
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional
from env.primitives import PrimitiveType, FINGERPRINT_KEYS


@dataclass
class ClaimNode:
    id: str
    text: str
    domain: str
    trust_score: float
    is_retrieved: bool = False
    injected: bool = False
    primitive: Optional[PrimitiveType] = None
    fingerprints: dict = field(default_factory=dict)


@dataclass
class EvidenceEdge:
    source_id: str
    target_id: str
    relation: str
    weight: float
    injected: bool = False


@dataclass
class ClaimGraph:
    nodes: List[ClaimNode]
    edges: List[EvidenceEdge]
    root_id: str

    @property
    def root_claim(self) -> ClaimNode:
        return next(n for n in self.nodes if n.id == self.root_id)

    @property
    def evidence_coverage(self) -> float:
        if not self.nodes:
            return 0.0
        retrieved = sum(1 for n in self.nodes if getattr(n, "is_retrieved", False))
        return min(1.0, retrieved / max(len(self.nodes), 1))

    @property
    def source_diversity_entropy(self) -> float:
        domains = [n.domain for n in self.nodes if n.domain]
        if not domains:
            return 0.0
        counts = Counter(domains)
        total = len(domains)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    @property
    def contradiction_surface_area(self) -> float:
        """Ratio of edges with 'adversarial' or 'contradicts' relation."""
        if not self.edges:
            return 0.0
        contra = sum(1 for e in self.edges
                     if getattr(e, 'relation', '') in ('adversarial', 'contradicts'))
        return contra / len(self.edges)

    @property
    def network_diameter(self) -> int:
        """Approximate graph diameter using BFS from root."""
        if not self.nodes or len(self.nodes) == 1:
            return 1
        # adjacency list
        adj = {n.id: [] for n in self.nodes}
        for e in self.edges:
            adj.get(e.source_id, []).append(e.target_id)
            adj.get(e.target_id, []).append(e.source_id)
        # BFS from each node; return max eccentricity
        max_dist = 1
        node_ids = [n.id for n in self.nodes]
        for start in node_ids:
            dist = {start: 0}
            queue = [start]
            while queue:
                cur = queue.pop(0)
                for nb in adj.get(cur, []):
                    if nb not in dist:
                        dist[nb] = dist[cur] + 1
                        queue.append(nb)
            max_dist = max(max_dist, max(dist.values(), default=1))
        return max_dist

    @property
    def true_label(self) -> str:
        injected = [n.primitive for n in self.nodes if n.injected and n.primitive is not None]
        if not injected:
            return "real"
        if PrimitiveType.SATIRE_REFRAME in injected:
            return "satire"
        if PrimitiveType.CONTEXT_STRIP in injected:
            return "out_of_context"
        return "fabricated"

    def to_json(self) -> dict:
        return {
            "nodes": [
                {
                    "id": n.id, "text": n.text, "domain": n.domain,
                    "trust_score": n.trust_score, "is_retrieved": n.is_retrieved,
                    "injected": n.injected,
                    "primitive": n.primitive.value if n.primitive else None,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "source": e.source_id, "target": e.target_id,
                    "relation": e.relation, "weight": e.weight, "injected": e.injected,
                }
                for e in self.edges
            ],
            "root_id": self.root_id,
        }

    def serialize(self) -> str:
        return json.dumps(self.to_json())
