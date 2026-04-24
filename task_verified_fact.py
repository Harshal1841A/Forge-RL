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
    {
        "text": "The Eiffel Tower was completed in 1889 and stands 330 meters tall including antenna.",
        "real_domain": "tour-eiffel.fr",
    },
    {
        "text": "Albert Einstein was awarded the Nobel Prize in Physics in 1921 for his discovery of the photoelectric effect.",
        "real_domain": "nobelprize.org",
    },
    {
        "text": "The human genome contains approximately 3 billion base pairs of DNA.",
        "real_domain": "genome.gov",
    },
    {
        "text": "Mount Everest is the tallest mountain on Earth at 8,849 meters above sea level.",
        "real_domain": "nationalgeographic.com",
    },
    {
        "text": "The Paris Agreement on climate change was adopted in December 2015.",
        "real_domain": "unfccc.int",
    },
]

_FALSE_FACT_SCENARIOS = [
    {
        "text": "The IEA has officially declared that global oil reserves will be entirely depleted by 2030.",
        "real_domain": "iea.org",
        "fake_domain": "energytruth24.org",
        "correct_stat": "IEA projects stable supply through 2050 with transition; no such depletion date exists.",
    },
    {
        "text": "FDA study reveals that common caffeine supplements are 500% more toxic than previously reported.",
        "real_domain": "fda.gov",
        "fake_domain": "healthalertnow.co",
        "correct_stat": "FDA confirms caffeine is GRAS (Generally Recognized As Safe) at standard dosages.",
    },
    {
        "text": "NASA satellite confirms existence of artificial megastructures around Proxima Centauri.",
        "real_domain": "nasa.gov",
        "fake_domain": "space-leaks.org",
        "correct_stat": "NASA has found no evidence of technosignatures; images are known lens flare artifacts.",
    },
    {
        "text": "US Treasury to replace all physical paper currency with government crypto by end of 2024.",
        "real_domain": "treasury.gov",
        "fake_domain": "financial-revelation.net",
        "correct_stat": "Treasury has no plans to phase out physical currency; CBDC research is in exploratory phase only.",
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
        is_true = rng.random() > 0.5

        if is_true:
            template = rng.choice(_VERIFIED_CLAIMS)
            domain = template["real_domain"]
            true_label = "real"
            trust_score = 0.95
            edge_rel = "supports"
        else:
            template = rng.choice(_FALSE_FACT_SCENARIOS)
            domain = template["fake_domain"]
            true_label = "fabricated"
            trust_score = 0.2
            edge_rel = "contradicts"

        graph_id = str(uuid.uuid4())
        root_id = "node_root"

        # —— Root node
        root = ClaimNode(
            node_id=root_id,
            text=template["text"],
            source_url=f"https://{domain}/news-{rng.randint(1000,9999)}",
            domain=domain,
            timestamp=datetime.utcnow() - timedelta(days=rng.randint(1, 30)),
            virality_score=rng.uniform(0.6, 0.95),
            trust_score=trust_score,
        )

        graph = ClaimGraph(
            graph_id=graph_id,
            root_claim_id=root_id,
            true_label=true_label,
            difficulty=difficulty,
            applied_tactics=[] if is_true else ["fabricate_statistic"],
        )
        graph.add_node(root)

        # —— Authoritative corroborating/contradicting source
        auth_id = "node_authority"
        if is_true:
            auth_text = f"Official statement from {template['real_domain']} confirming current data and status."
        else:
            auth_text = f"Official correction from {template['real_domain']}: {template['correct_stat']}"

        auth = ClaimNode(
            node_id=auth_id,
            text=auth_text,
            source_url=f"https://{template['real_domain']}/official-report",
            domain=template["real_domain"],
            timestamp=datetime.utcnow() - timedelta(days=rng.randint(1, 40)),
            virality_score=0.1,
            trust_score=0.98,
        )
        graph.add_node(auth)
        graph.add_edge(EvidenceEdge(
            edge_id="e_auth", src_id=auth_id, tgt_id=root_id,
            relation=edge_rel, weight=0.95,
        ))

        # —— Chained corroborating/analysis nodes
        # DEPTH SCALING: chain nodes through each other (root→amp_0→amp_1)
        # instead of all connecting to root.  At difficulty >= 3, a hidden
        # primary-source node is only reachable at the end of the chain.
        prev_chain_id = root_id
        for i in range(max(1, difficulty)):
            amp_domain = rng.choice(_SUPPORTING_DOMAINS)
            amp_id = f"node_amp_{i}"

            if is_true:
                amp_text = f"Reporting confirms the findings regarding: {template['text']}"
                amp_rel = "supports"
                amp_trust = 0.9
            else:
                amp_text = f"Analysis: discrepancies found in viral claims about {template['real_domain']} reports."
                amp_rel = "debunks"
                amp_trust = 0.85

            amp = ClaimNode(
                node_id=amp_id,
                text=amp_text,
                source_url=f"https://{amp_domain}/article-{rng.randint(100,999)}",
                domain=amp_domain,
                timestamp=datetime.utcnow() - timedelta(days=rng.randint(1, 40)),
                virality_score=rng.uniform(0.3, 0.8),
                trust_score=amp_trust,
            )
            graph.add_node(amp)
            graph.add_edge(EvidenceEdge(
                edge_id=f"e_amp_{i}", src_id=prev_chain_id, tgt_id=amp_id,
                relation=amp_rel, weight=0.9,
            ))
            prev_chain_id = amp_id

        # —— Hidden primary-source node at chain depth (difficulty >= 3) ───────
        if difficulty >= 3:
            primary_id = "node_primary_source"
            if is_true:
                primary_text = (
                    f"Original dataset from {template['real_domain']}: raw data confirms "
                    f"the reported figures. Methodology peer-reviewed and reproducible."
                )
                primary_rel = "supports"
            else:
                primary_text = (
                    f"Original dataset from {template['real_domain']}: raw data directly "
                    f"contradicts the viral claim. Numbers were fabricated or misread."
                )
                primary_rel = "debunks"

            primary = ClaimNode(
                node_id=primary_id,
                text=primary_text,
                source_url=f"https://{template['real_domain']}/data/raw-dataset-{rng.randint(100,999)}",
                domain=template["real_domain"],
                timestamp=datetime.utcnow() - timedelta(days=rng.randint(60, 180)),
                virality_score=0.01,
                trust_score=0.99,
            )
            graph.add_node(primary)
            graph.add_edge(EvidenceEdge(
                edge_id="e_primary", src_id=prev_chain_id, tgt_id=primary_id,
                relation=primary_rel, weight=0.98,
            ))

        return graph

    def oracle_steps(self, graph: ClaimGraph) -> int:
        return 2 + (graph.difficulty - 1)

    def has_manipulation(self, graph: ClaimGraph) -> bool:
        return graph.true_label == "fabricated"

    def grade(self, episode_trace: list[dict], graph: ClaimGraph) -> float:
        """
        Easy control task grader. Tests false-positive avoidance.
        Partial credit:
          +0.3  used cross_reference (confirms claim is real)
          +0.3  used entity_link (verifies entities exist)
          +0.4  submitted correct final verdict
        Penalty:
          -0.2  used flag_manipulation on a real claim (false positive)

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
            if "cross_reference" in actions:
                score += 0.3
            if "entity_link" in actions:
                score += 0.3
            return float(np.clip(score * 0.3, 0.001, 0.999))

        # ── Standard grading ────────────────────────────────────────────────
        if "cross_reference" in actions:
            score += 0.3
        if "entity_link" in actions:
            score += 0.3

        # False positive penalty: flagging manipulation on a verified real claim
        if "flag_manipulation" in actions and graph.true_label == "real":
            score -= 0.2

        if final_verdict == graph.true_label:
            score += 0.4
        elif final_verdict is not None and final_verdict != "real":
            # Penalise false positives (mislabelling real as misinfo)
            score -= 0.1

        return float(np.clip(score, 0.001, 0.999))
