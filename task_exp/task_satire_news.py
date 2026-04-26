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
        "text": "Report: We Don't Make The Laws, We Just Think Them Up And Give Them To The People Who Do.",
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
    {
        "text": "Area Man Confident He Could Run Country Better After Watching 20 Minutes Of Cable News.",
        "authoritative_domain": "theonion.com",
    },
    {
        "text": "CDC Recommends Placing Entire Household In Large Ziplock Bag To Prevent Spread Of Germs.",
        "authoritative_domain": "theonion.com",
    },
    {
        "text": "Scientists Warn Earth's Orbit Slightly Disrupted By Humanity's Collective Disappointment.",
        "authoritative_domain": "theonion.com",
    },
    {
        "text": "Nation's Experts Admit They Just Trying Their Best, Not Sure What They're Doing Either.",
        "authoritative_domain": "theonion.com",
    },
    {
        "text": "New Study Finds Correlation Between Eating Breakfast And Having Eaten Breakfast.",
        "authoritative_domain": "theonion.com",
    },
]


_TRUE_BIZARRE_CLAIMS = [
    {
        "text": "Florida Man Arrested After Trying To Pay For Fast Food With A Live Alligator.",
        "authoritative_domain": "apnews.com",
    },
    {
        "text": "California Town Elects Golden Retriever Dog As Mayor For Fourth Consecutive Term.",
        "authoritative_domain": "reuters.com",
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
        is_true = rng.random() > 0.5

        if is_true:
            template = rng.choice(_TRUE_BIZARRE_CLAIMS)
            true_label = "real"
            edge_rel = "supports"
        else:
            template = rng.choice(_SATIRE_CLAIMS)
            true_label = "satire"
            edge_rel = "misinterprets"

        graph_id = str(uuid.uuid4())
        root_id = "node_root"

        # —— Root node
        fake_outrage_domain = rng.choice(_MISUNDERSTANDING_DOMAINS)

        if is_true:
            root_text = f"Can't make this up! {template['text']}"
        else:
            root_text = f"UNBELIEVABLE: {template['text']} This is what our world has come to!!"

        root = ClaimNode(
            node_id=root_id,
            text=root_text,
            source_url=f"https://{fake_outrage_domain}/post/{rng.randint(1000,9999)}",
            domain=fake_outrage_domain,
            timestamp=datetime.utcnow() - timedelta(days=rng.randint(1, 10)),
            virality_score=rng.uniform(0.7, 0.95),
            trust_score=0.6 if is_true else 0.1,
        )

        graph = ClaimGraph(
            graph_id=graph_id,
            root_claim_id=root_id,
            true_label=true_label,
            difficulty=difficulty,
            applied_tactics=[] if is_true else ["parody_taken_literally"],
        )
        graph.add_node(root)

        # —— The original Source Node
        satire_id = "node_satire_origin"
        satire = ClaimNode(
            node_id=satire_id,
            text=template["text"],
            source_url=f"https://{template['authoritative_domain']}/article-{rng.randint(100,999)}",
            domain=template["authoritative_domain"],
            timestamp=datetime.utcnow() - timedelta(days=rng.randint(20, 60)),
            virality_score=0.9,
            trust_score=0.95 if is_true else 0.8,
        )
        graph.add_node(satire)
        graph.add_edge(EvidenceEdge(
            edge_id="e_satire_origin", src_id=root_id, tgt_id=satire_id,
            relation=edge_rel, weight=0.99,
        ))

        # —— Additional angry amplifiers (if difficulty > 1)
        # DEPTH SCALING: chain amplifiers through each other (A→B→C) instead of
        # all connecting directly to root.  This forces the agent to perform
        # multi-hop investigation before it can reach the debunking evidence.
        prev_chain_id = root_id
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
                edge_id=f"e_amp_{i}", src_id=prev_chain_id, tgt_id=amp_id,
                relation="agrees", weight=0.85,
            ))
            prev_chain_id = amp_id

        # —— Hidden debunk node at chain depth (difficulty >= 3)
        # Only reachable by traversing the full amplifier chain.
        if difficulty >= 3 and not is_true:
            hidden_deb_id = "node_hidden_debunk"
            hidden_deb = ClaimNode(
                node_id=hidden_deb_id,
                text=(
                    f"Deep investigation: Original article traced to "
                    f"{template['authoritative_domain']} satire section. "
                    f"Confirmed parody by editor."
                ),
                source_url=f"https://{template['authoritative_domain']}/about/satire-policy",
                domain=template["authoritative_domain"],
                timestamp=datetime.utcnow() - timedelta(days=rng.randint(30, 90)),
                virality_score=0.02,
                trust_score=0.95,
            )
            graph.add_node(hidden_deb)
            graph.add_edge(EvidenceEdge(
                edge_id="e_hidden_deb", src_id=prev_chain_id, tgt_id=hidden_deb_id,
                relation="debunks", weight=0.97,
            ))

        return graph

    def oracle_steps(self, graph: ClaimGraph) -> int:
        return 2 + (graph.difficulty - 1)

    def has_manipulation(self, graph: ClaimGraph) -> bool:
        return graph.true_label == "satire"

    def grade(self, episode_trace: list[dict], graph: ClaimGraph) -> float:
        """
        Medium linguistic analysis grader.
        Partial credit:
          +0.3  used request_context (reads deeper text for tone/humor)
          +0.3  used cross_reference (checks if claim appears in real news)
          +0.4  submitted correct final verdict

        Exploit resistance:
        - Requires >= 2 unique investigation tools
        - Requires a submitted verdict for score > 0.3
        """
        import numpy as np
        score = 0.001
        actions = [s.get("action", "") for s in episode_trace if "action" in s]

        # ── Exploit guard 1: tool diversity requirement ─────────────────────
        investigation_tools = [
            a for a in actions
            if not a.startswith("submit_verdict") and a != "flag_manipulation"
        ]
        unique_tools = len(set(investigation_tools))
        if unique_tools < 2:
            final_verdict = next(
                (a.replace("submit_verdict_", "") for a in reversed(actions)
                 if a.startswith("submit_verdict_")), None
            )
            if final_verdict == graph.true_label:
                return float(np.clip(0.4, 0.001, 0.999))
            return 0.001

        # ── Exploit guard 2: verdict required ───────────────────────────────
        final_verdict = next(
            (a.replace("submit_verdict_", "") for a in reversed(actions)
             if a.startswith("submit_verdict_")), None
        )
        if final_verdict is None:
            if "request_context" in actions:
                score += 0.3
            if "cross_reference" in actions:
                score += 0.3
            return float(np.clip(score * 0.3, 0.001, 0.999))

        # ── Standard grading ────────────────────────────────────────────────
        if "request_context" in actions:
            score += 0.3
        if "cross_reference" in actions:
            score += 0.3

        if final_verdict == graph.true_label:
            score += 0.4
        elif final_verdict is not None:
            misinfo = {"misinfo", "satire", "out_of_context", "fabricated"}
            if final_verdict in misinfo and graph.true_label in misinfo:
                score += 0.2

        return float(np.clip(score, 0.001, 0.999))
