"""
Task: PolitifactTask — Real-World Claims from LIAR Dataset (v2.0)

Uses rows from the open-source Politifact LIAR dataset
(Wang, 2017 — https://huggingface.co/datasets/liar)
to provide real political claim statements with ground-truth labels instead of
purely procedurally generated text.

Falls back to a synthetic fabricated_stats style graph if the CSV is not present.
"""

from __future__ import annotations

import csv
import logging
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

from env.claim_graph import ClaimGraph, ClaimNode, EvidenceEdge
from env.tasks.task_base import BaseTask

logger = logging.getLogger(__name__)

# Path to the downloaded LIAR CSV (populated by scripts/download_liar.py)
_LIAR_CSV = Path(__file__).parent.parent.parent / "data" / "liar_dataset.csv"

_FACT_CHECK_DOMAINS = [
    "politifact.com", "snopes.com", "factcheck.org",
    "reuters.com/fact-check", "apnews.com",
]

_AMPLIFIER_DOMAINS = [
    "facebook.com", "twitter.com", "reddit.com",
    "youtube.com", "tiktok.com",
]

# Synthetic fallback claims used if LIAR CSV is unavailable
_FALLBACK_CLAIMS = [
    {
        "statement": "The unemployment rate hit 42% under the previous administration.",
        "speaker": "Unknown Politician",
        "party": "Unknown",
        "liar_label": "pants-fire",
        "forge_label": "fabricated",
    },
    {
        "statement": "President signed executive order banning all immigration from 10 countries.",
        "speaker": "Social Media Post",
        "party": "Unknown",
        "liar_label": "false",
        "forge_label": "misinfo",
    },
    {
        "statement": "Tax cuts only benefited the top 1% of earners.",
        "speaker": "Political Commentator",
        "party": "Opposition",
        "liar_label": "barely-true",
        "forge_label": "out_of_context",
    },
]


def _load_liar_dataset() -> List[Dict]:
    """Load LIAR CSV, return list of claim dicts. Returns empty list if file not found."""
    if not _LIAR_CSV.exists():
        logger.warning(
            "LIAR dataset not found at %s. "
            "Run: python scripts/download_liar.py\n"
            "Falling back to synthetic claims.",
            _LIAR_CSV,
        )
        return []
    rows = []
    with open(_LIAR_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("statement") and row.get("forge_label"):
                rows.append(row)
    logger.info("Loaded %d LIAR claims from %s", len(rows), _LIAR_CSV)
    return rows


# Module-level cache so CSV is only read once per process
_liar_rows: Optional[List[Dict]] = None


def _get_liar_rows() -> List[Dict]:
    global _liar_rows
    if _liar_rows is None:
        _liar_rows = _load_liar_dataset()
    return _liar_rows


class PolitifactTask(BaseTask):
    """
    Uses real Politifact LIAR dataset claims as the root node of the investigation graph.
    The true label is bound directly from the dataset's crowd-sourced ground truth.
    """

    task_id = "politifact_liar"
    description = (
        "Agent investigates a real-world Politifact claim drawn from the open-source "
        "LIAR dataset. Ground-truth labels are provided by expert Politifact fact-checkers."
    )

    def generate(self, difficulty: int = 1, seed: int = 0) -> ClaimGraph:
        rng = random.Random(seed)
        rows = _get_liar_rows()

        if rows:
            # Seed-reproducible but uniformly distributed claim selection
            claim_data = rng.choice(rows)
        else:
            # Fallback to synthetic
            claim_data = rng.choice(_FALLBACK_CLAIMS)

        statement = claim_data["statement"]
        speaker = claim_data.get("speaker", "Unknown")
        party = claim_data.get("party", "Unknown")
        liar_label = claim_data.get("liar_label", "false")
        forge_label = claim_data.get("forge_label", "misinfo")
        subject = claim_data.get("subject", "politics")

        graph_id = str(uuid.uuid4())
        root_id = "node_root"

        # ── Root node: the real political claim ───────────────────────────────
        root = ClaimNode(
            node_id=root_id,
            text=f'[CLAIM] {speaker} ({party}): "{statement}"',
            source_url=f"https://www.politifact.com/factchecks/claim/{seed}/",
            domain="politifact.com",
            timestamp=datetime.utcnow() - timedelta(days=rng.randint(1, 90)),
            author=speaker,
            virality_score=rng.uniform(0.4, 0.9),
            trust_score=0.5,   # The CLAIM's credibility is unknown at start, not PolitiFact's
                               # (was incorrectly 0.9 — that's the reporter's credibility, not the claim)
            metadata={"subject": subject, "party": party, "liar_label": liar_label},
        )

        graph = ClaimGraph(
            graph_id=graph_id,
            root_claim_id=root_id,
            true_label=forge_label,
            difficulty=difficulty,
            applied_tactics=self._infer_tactics(liar_label),
        )
        graph.add_node(root)

        # ── Amplifier node: social media spread ───────────────────────────────
        amp_domain = rng.choice(_AMPLIFIER_DOMAINS)
        amp_id = "node_amplifier"
        amp = ClaimNode(
            node_id=amp_id,
            text=f"Viral post repeating claim about {subject} without context.",
            source_url=f"https://{amp_domain}/post/{rng.randint(100000, 999999)}",
            domain=amp_domain,
            timestamp=datetime.utcnow() - timedelta(days=rng.randint(1, 30)),
            virality_score=rng.uniform(0.7, 0.99),
            trust_score=0.3,
        )
        graph.add_node(amp)
        graph.add_edge(EvidenceEdge(
            edge_id="e_amp", src_id=amp_id, tgt_id=root_id,
            relation="amplifies", weight=0.8,
        ))

        # ── Fact-check debunk node ────────────────────────────────────────────
        fc_domain = rng.choice(_FACT_CHECK_DOMAINS)
        fc_id = "node_factcheck"
        fc = ClaimNode(
            node_id=fc_id,
            text=(
                f"FACT CHECK [{liar_label.upper()}]: "
                f"PolitiFact rates this claim as '{liar_label}'. "
                f"Claim by {speaker} about {subject}."
            ),
            source_url=f"https://{fc_domain}/factchecks/{seed}/",
            domain=fc_domain,
            timestamp=datetime.utcnow() - timedelta(days=rng.randint(1, 14)),
            virality_score=0.2,
            trust_score=0.92,
        )
        graph.add_node(fc)

        # The fact-check contradicts anything not "real"
        edge_relation = "debunks" if forge_label != "real" else "supports"
        graph.add_edge(EvidenceEdge(
            edge_id="e_fc", src_id=fc_id, tgt_id=root_id,
            relation=edge_relation, weight=0.95,
        ))

        # ── Difficulty scaling: chained noisy/amplifier nodes ────────────────
        # DEPTH SCALING: chain noise nodes (root→noise_0→noise_1) so the agent
        # must traverse multiple hops.  At difficulty >= 3, a hidden original-
        # source node is only reachable at the end of the chain.
        prev_chain_id = root_id
        for i in range(difficulty - 1):
            noise_id = f"node_noise_{i}"
            noise = ClaimNode(
                node_id=noise_id,
                text=f"Partial retelling of the claim, missing key context (level {i+1}).",
                source_url=f"https://blog-{rng.randint(1, 99)}.net/opinion/{seed}",
                domain=f"opinonblog{i}.net",
                timestamp=datetime.utcnow() - timedelta(days=rng.randint(15, 60)),
                virality_score=rng.uniform(0.3, 0.6),
                trust_score=0.25,
            )
            graph.add_node(noise)
            graph.add_edge(EvidenceEdge(
                edge_id=f"e_noise_{i}", src_id=prev_chain_id, tgt_id=noise_id,
                relation="cites", weight=rng.uniform(0.2, 0.5),
            ))
            prev_chain_id = noise_id

        # ── Hidden original-source node at chain depth (difficulty >= 3) ──────
        if difficulty >= 3:
            orig_src_id = "node_original_source"
            orig_src = ClaimNode(
                node_id=orig_src_id,
                text=(
                    f"Original speech transcript from {speaker}: full context reveals "
                    f"the statement was {'accurately reported' if forge_label == 'real' else 'taken out of context and distorted'} "
                    f"by downstream media."
                ),
                source_url=f"https://c-span.org/transcript/{seed}",
                domain="c-span.org",
                timestamp=datetime.utcnow() - timedelta(days=rng.randint(30, 120)),
                virality_score=0.01,
                trust_score=0.98,
            )
            graph.add_node(orig_src)
            edge_rel_orig = "supports" if forge_label == "real" else "debunks"
            graph.add_edge(EvidenceEdge(
                edge_id="e_orig_src", src_id=prev_chain_id, tgt_id=orig_src_id,
                relation=edge_rel_orig, weight=0.95,
            ))

        return graph

    def oracle_steps(self, graph: ClaimGraph) -> int:
        # Minimum: query source + cross-ref fact-check + submit
        return 2 + graph.difficulty

    def has_manipulation(self, graph: ClaimGraph) -> bool:
        return graph.true_label in ["misinfo", "fabricated"]

    def grade(self, episode_trace: list[dict], graph: ClaimGraph) -> float:
        """
        Real-world dataset grader (LIAR dataset).
        Partial credit:
          +0.25  used cross_reference (checks against encyclopedic facts)
          +0.25  used entity_link (verifies claim entities exist)
          +0.1   used query_source (verifies source credibility)
          +0.4   submitted correct final verdict

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
                score += 0.25
            if "entity_link" in actions:
                score += 0.25
            if "query_source" in actions:
                score += 0.1
            return float(np.clip(score * 0.3, 0.001, 0.999))

        # ── Standard grading ────────────────────────────────────────────────
        if "cross_reference" in actions:
            score += 0.25
        if "entity_link" in actions:
            score += 0.25
        if "query_source" in actions:
            score += 0.1

        if final_verdict == graph.true_label:
            score += 0.4
        elif final_verdict is not None:
            misinfo = {"misinfo", "satire", "out_of_context", "fabricated"}
            if final_verdict in misinfo and graph.true_label in misinfo:
                score += 0.2
            elif final_verdict == "real" and graph.true_label == "real":
                score += 0.4

        return float(np.clip(score, 0.001, 0.999))

    @staticmethod
    def _infer_tactics(liar_label: str) -> list:
        mapping = {
            "pants-fire": ["fabricate_statistic", "misattribute_quote"],
            "false": ["fabricate_statistic"],
            "barely-true": ["strip_context", "cherry_pick_study"],
            "half-true": ["strip_context"],
            "mostly-true": [],
            "true": [],
        }
        return mapping.get(liar_label, ["fabricate_statistic"])
