import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
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

    def to_json(self) -> dict:
        return {
            "nodes": [{"id": n.id, "text": n.text, "domain": n.domain, 
                       "trust_score": n.trust_score, "is_retrieved": n.is_retrieved,
                       "injected": n.injected, 
                       "primitive": n.primitive.value if n.primitive else None} 
                      for n in self.nodes],
            "edges": [{"source": e.source_id, "target": e.target_id, "relation": e.relation, 
                       "weight": e.weight, "injected": e.injected} 
                      for e in self.edges],
            "root_id": self.root_id
        }

    def serialize(self) -> str:
        return json.dumps(self.to_json())
