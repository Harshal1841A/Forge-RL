"""
Task — Verified Fact
Tactics: none (legitimate reporting)
Label: real
"""

from __future__ import annotations
import random
import uuid
from datetime import datetime, timedelta
from env.claim_graph import ClaimGraph, ClaimNode, EvidenceEdge
from env.tasks.task_base import BaseTask


_VERIFIED_CLAIMS = [
    {
        "text": "Global renewable energy capacity grew by 50% in 2023, according to the IEA.",
        "real_domain": "iea.org",
    },
    {
        "text": "The FDA approved a new gene therapy for sickle cell disease.",
        "real_domain": "fda.gov",
    },
    {
        "text": "NASA's James Webb Space Telescope detected carbon dioxide on exoplanet WASP-39b.",
        "real_domain": "nasa.gov",
    },
    {
        "text": "US total national debt surpassed $34 trillion in early 2024.",
        "real_domain": "treasury.gov",
    },
]

_SUPPORTING_DOMAINS = [
    "reuters.com", "apnews.com", "bbc.com",
    "nature.com", "wsj.com"
]


class VerifiedFactTask(BaseTask):
    task_id = "verified_fact"
    description = (
        "The agent investigates a legitimate, verified factual claim "
        "reported accurately by authoritative sources."
    )

    def generate(self, difficulty: int = 1, seed: int = 0) -> ClaimGraph:
        rng = random.Random(seed)
        template = rng.choice(_VERIFIED_CLAIMS)

        graph_id = str(uuid.uuid4())
        root_id = "node_root"

        # —— Root node (the true claim)
        root = ClaimNode(
            node_id=root_id,
            text=template["text"],
            source_url=f"https://{template['real_domain']}/news-{rng.randint(1000,9999)}",
            domain=template["real_domain"],
            timestamp=datetime.utcnow() - timedelta(days=rng.randint(1, 30)),
            virality_score=rng.uniform(0.6, 0.95),
            trust_score=0.95,
        )

        graph = ClaimGraph(
            graph_id=graph_id,
            root_claim_id=root_id,
            true_label="real",
            difficulty=difficulty,
            applied_tactics=[], # No manipulation
        )
        graph.add_node(root)

        # —— Additional corroborating nodes
        for i in range(max(1, difficulty)):
            amp_domain = rng.choice(_SUPPORTING_DOMAINS)
            amp_id = f"node_amp_{i}"
            amp = ClaimNode(
                node_id=amp_id,
                text=f"Reporting confirms the findings regarding: {template['text']}",
                source_url=f"https://{amp_domain}/article-{rng.randint(100,999)}",
                domain=amp_domain,
                timestamp=datetime.utcnow() - timedelta(days=rng.randint(1, 40)),
                virality_score=rng.uniform(0.3, 0.8),
                trust_score=0.9,
            )
            graph.add_node(amp)
            graph.add_edge(EvidenceEdge(
                edge_id=f"e_amp_{i}", src_id=amp_id, tgt_id=root_id,
                relation="supports", weight=0.95,
            ))

        return graph

    def oracle_steps(self, graph: ClaimGraph) -> int:
        return 2

    def has_manipulation(self, graph: ClaimGraph) -> bool:
        return False
