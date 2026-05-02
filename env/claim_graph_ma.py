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
    def contradiction_surface_area(self) -> int:
        return sum(
            1 for e in self.edges
            if getattr(e, "relation", "") in {"contradicts", "debunks", "refutes"}
        )

    @property
    def network_diameter(self) -> int:
        if not self.edges:
            return 1
        return min(len(self.edges), 10)

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
