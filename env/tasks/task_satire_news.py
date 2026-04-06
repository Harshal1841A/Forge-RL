"""
Task — Satirical News
Tactics: parody, non_literal_claim
Label: satire
"""

from __future__ import annotations
import random
import uuid
from datetime import datetime, timedelta
from env.claim_graph import ClaimGraph, ClaimNode, EvidenceEdge
from env.tasks.task_base import BaseTask


_SATIRE_CLAIMS = [
    {
        "text": "Area Man Passionate Defender Of What He Imagines Constitution To Be.",
        "authoritative_domain": "theonion.com",
    },
    {
        "text": "Report: We Don’t Make The Laws, We Just Think Them Up And Give Them To The People Who Do.",
        "authoritative_domain": "theonion.com",
    },
    {
        "text": "Local Dog Named Honorary Mayor After Promising Universal Free Treats.",
        "authoritative_domain": "babylonbee.com",
    },
    {
        "text": "Scientists Discover New Way To Tell If You're In A Simulation: Just Ask Politely.",
        "authoritative_domain": "clickhole.com",
    },
]

_MISUNDERSTANDING_DOMAINS = [
    "angrycitizenforum.net", "truthpatriots1776.com", "outrage-daily.org"
]


class SatiricalClaimTask(BaseTask):
    task_id = "satire_news"
    description = (
        "The agent encounters a piece of satire or parody that has been "
        "taken literally by secondary nodes and spread as misinformation."
    )

    def generate(self, difficulty: int = 1, seed: int = 0) -> ClaimGraph:
        rng = random.Random(seed)
        template = rng.choice(_SATIRE_CLAIMS)

        graph_id = str(uuid.uuid4())
        root_id = "node_root"

        # —— Root node (the satire taken completely literally by an angry forum)
        fake_outrage_domain = rng.choice(_MISUNDERSTANDING_DOMAINS)
        root = ClaimNode(
            node_id=root_id,
            text=f"UNBELIEVABLE: {template['text']} This is what our world has come to!!",
            source_url=f"https://{fake_outrage_domain}/post/{rng.randint(1000,9999)}",
            domain=fake_outrage_domain,
            timestamp=datetime.utcnow() - timedelta(days=rng.randint(1, 10)),
            virality_score=rng.uniform(0.7, 0.95),
            trust_score=0.1,
        )

        graph = ClaimGraph(
            graph_id=graph_id,
            root_claim_id=root_id,
            true_label="satire",
            difficulty=difficulty,
            applied_tactics=["parody_taken_literally"],
        )
        graph.add_node(root)

        # —— The original Satire Node
        satire_id = "node_satire_origin"
        satire = ClaimNode(
            node_id=satire_id,
            text=template["text"],
            source_url=f"https://{template['authoritative_domain']}/article-{rng.randint(100,999)}",
            domain=template["authoritative_domain"],
            timestamp=datetime.utcnow() - timedelta(days=rng.randint(20, 60)),
            virality_score=0.9,
            trust_score=0.8, # Trusted *as satire*
        )
        graph.add_node(satire)
        graph.add_edge(EvidenceEdge(
            edge_id="e_satire_origin", src_id=root_id, tgt_id=satire_id,
            relation="misinterprets", weight=0.99,
        ))

        # —— Additional angry amplifiers (if difficulty > 1)
        for i in range(difficulty - 1):
            amp_domain = rng.choice(_MISUNDERSTANDING_DOMAINS)
            amp_id = f"node_amp_{i}"
            amp = ClaimNode(
                node_id=amp_id,
                text=f"More proof they are doing this: {template['text']}",
                source_url=f"https://{amp_domain}/thread-{rng.randint(100,999)}",
                domain=amp_domain,
                timestamp=datetime.utcnow() - timedelta(days=rng.randint(1, 5)),
                virality_score=rng.uniform(0.4, 0.9),
                trust_score=0.2,
            )
            graph.add_node(amp)
            graph.add_edge(EvidenceEdge(
                edge_id=f"e_amp_{i}", src_id=root_id, tgt_id=amp_id,
                relation="agrees", weight=0.85,
            ))

        return graph

    def oracle_steps(self, graph: ClaimGraph) -> int:
        return 2 + (graph.difficulty - 1)

    def has_manipulation(self, graph: ClaimGraph) -> bool:
        # Satire itself isn't malicious manipulation, but taking it out of context is often a tactic
        return True
