"""
Task 2 — Out-of-Context + Outdated Claims
Tactics: strip_context, backdate_article, translate_without_context
"""

from __future__ import annotations
import random
import uuid
from datetime import datetime, timedelta
from env.claim_graph import ClaimGraph, ClaimNode, EvidenceEdge
from env.tasks.task_base import BaseTask


_OOC_CLAIMS = [
    {
        "modern_text": "Massive fire destroys entire city block in Chicago.",
        "context": "Photo is from the 2017 Santa Rosa wildfire, recirculated after an unrelated Chicago incident.",
        "origin_year": 2017,
        "origin_domain": "latimes.com",
        "misuse_domain": "breakingnewsalert24.com",
        "tactics": ["strip_context", "backdate_article"],
    },
    {
        "modern_text": "Thousands protest government lockdown in Berlin — city in chaos.",
        "context": "Video is from a 2019 climate strike, labelled as COVID lockdown protest in 2021.",
        "origin_year": 2019,
        "origin_domain": "dw.com",
        "misuse_domain": "germanytruth.net",
        "tactics": ["strip_context"],
    },
    {
        "modern_text": "Shocking: Hospital overrun, bodies in hallways — healthcare system collapses.",
        "context": "Image from 2010 Haiti earthquake used in 2020 to misrepresent COVID hospital conditions.",
        "origin_year": 2010,
        "origin_domain": "reuters.com",
        "misuse_domain": "covidtruth2020.org",
        "tactics": ["strip_context", "backdate_article"],
    },
    {
        "modern_text": "Minister admits government is bankrupt in leaked audio.",
        "context": "Audio from 2015 theatrical performance, re-shared in 2024 claiming modern context.",
        "origin_year": 2015,
        "origin_domain": "theguardian.com",
        "misuse_domain": "leakedgov.info",
        "tactics": ["strip_context", "translate_without_context"],
    },
]

_TRUE_CLAIMS = [
    {
        "modern_text": "Live coverage: Massive fire breaks out in downtown warehouse district today.",
        "context": "Confirmed by local authorities and multiple live news streams today.",
        "origin_year": datetime.today().year,
        "origin_domain": "localnews.com",
        "misuse_domain": "breakingnewsalert24.com",
    },
    {
        "modern_text": "Hospital officially reaches full capacity following recent health crisis.",
        "context": "Confirmed by hospital administrative statements this week.",
        "origin_year": datetime.today().year,
        "origin_domain": "reuters.com",
        "misuse_domain": "healthalert.org",
    },
]


class OutOfContextTask(BaseTask):
    task_id = "out_of_context"
    description = (
        "The agent investigates claims where real content (images, videos, quotes) "
        "is stripped of its original context or given a false timestamp."
    )

    def generate(self, difficulty: int = 1, seed: int = 0) -> ClaimGraph:
        rng = random.Random(seed)
        is_true = rng.random() > 0.5

        if is_true:
            template = rng.choice(_TRUE_CLAIMS)
            true_label = "real"
            edge_rel = "supports"
            applied_tactics = []
            trust_score = 0.8
        else:
            template = rng.choice(_OOC_CLAIMS)
            true_label = "out_of_context"
            edge_rel = "contradicts"
            applied_tactics = list(template["tactics"])
            trust_score = 0.15

        graph_id = str(uuid.uuid4())
        root_id = "node_root"
        now = datetime.utcnow()

        root = ClaimNode(
            node_id=root_id,
            text=template["modern_text"],
            source_url=f"https://{template['misuse_domain']}/post-{rng.randint(1000,9999)}",
            domain=template["misuse_domain"],
            timestamp=now - timedelta(days=rng.randint(1, 7)),
            virality_score=rng.uniform(0.5, 0.9),
            trust_score=trust_score,
            metadata={"claimed_date": str(now.date())},
        )

        graph = ClaimGraph(
            graph_id=graph_id,
            root_claim_id=root_id,
            true_label=true_label,
            difficulty=difficulty,
            applied_tactics=applied_tactics,
        )
        graph.add_node(root)

        # ── Original source node ────────────────────────────
        if is_true:
            origin_date = now - timedelta(days=rng.randint(1, 3))
        else:
            origin_date = datetime(template["origin_year"], rng.randint(1, 12), rng.randint(1, 28))

        orig_id = "node_origin"
        orig = ClaimNode(
            node_id=orig_id,
            text=f"Original article from {template['origin_domain']} — {template['context']}",
            source_url=f"https://web.archive.org/web/{origin_date.strftime('%Y%m%d')}/https://{template['origin_domain']}/",
            domain=f"web.archive.org → {template['origin_domain']}",
            timestamp=origin_date,
            virality_score=0.05,
            trust_score=0.92,
            metadata={"archive": True, "origin_year": template["origin_year"]},
        )
        graph.add_node(orig)
        graph.add_edge(EvidenceEdge(
            edge_id="e_origin", src_id=root_id, tgt_id=orig_id,
            relation=edge_rel, weight=0.95,
        ))

        # ── Propagation chain (amplifier nodes) ────────────────────────────────
        # Difficulty 2+: add re-sharing nodes with modified text
        prev_id = root_id
        for i in range(difficulty - 1):
            amp_id = f"node_share_{i}"
            amp = ClaimNode(
                node_id=amp_id,
                text=f"Re-share #{i+1}: {template['modern_text']} (translated/adapted)",
                source_url=f"https://socialmedia-mirror-{i}.net/post-{rng.randint(100,999)}",
                domain=f"socialmedia-mirror-{i}.net",
                timestamp=now - timedelta(hours=rng.randint(1, 72)),
                virality_score=rng.uniform(0.3, 0.7),
                trust_score=0.1,
            )
            graph.add_node(amp)
            graph.add_edge(EvidenceEdge(
                edge_id=f"e_share_{i}", src_id=prev_id, tgt_id=amp_id,
                relation="amplifies", weight=rng.uniform(0.6, 0.9),
            ))
            prev_id = amp_id

        if not is_true and difficulty >= 3:
            graph.applied_tactics.append("translate_without_context")

        return graph

    def oracle_steps(self, graph: ClaimGraph) -> int:
        return 3 + (graph.difficulty - 1)

    def has_manipulation(self, graph: ClaimGraph) -> bool:
        return graph.true_label == "out_of_context"

    def grade(self, episode_trace: list[dict], graph: ClaimGraph) -> float:
        """
        Medium task grader.
        Partial credit:
          +0.3  used trace_origin (checks timestamp of original)
          +0.3  used temporal_audit (verifies timeline anomaly)
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
            if "trace_origin" in actions:
                score += 0.3
            if "temporal_audit" in actions:
                score += 0.3
            return float(np.clip(score * 0.3, 0.001, 0.999))

        # ── Standard grading ────────────────────────────────────────────────
        if "trace_origin" in actions:
            score += 0.3
        if "temporal_audit" in actions:
            score += 0.3

        if final_verdict == graph.true_label:
            score += 0.4
        elif final_verdict is not None:
            # partial credit: correct macro-category
            misinfo = {"misinfo", "satire", "out_of_context", "fabricated"}
            if final_verdict in misinfo and graph.true_label in misinfo:
                score += 0.2

        return float(np.clip(score, 0.001, 0.999))
