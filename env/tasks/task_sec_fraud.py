"""
Task: SECFraudTask — Financial Forensics / SEC Filing Mismatch (v2.0)

Simulates a corporate misinformation scenario where a CEO's public statement
contradicts the company's official SEC EDGAR filing. The agent must use
query_source and cross_reference to detect the discrepancy between the public
claim and the regulatory filing, then submit "fabricated".

This task tests the agent's ability to reason across financial-domain evidence
without requiring any financial domain-specific tools — the existing investigative
tool set is sufficient.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta

from env.claim_graph import ClaimGraph, ClaimNode, EvidenceEdge
from env.tasks.task_base import BaseTask


_SEC_SCENARIOS = [
    {
        "company": "NovaTech Inc.",
        "ticker": "NVTK",
        "ceo_statement": "NovaTech has zero long-term debt and is fully self-funded through operations.",
        "sec_reality": "10-K filing shows $2.3B in senior unsecured notes due 2026 (Item 8, Note 12).",
        "filing_type": "10-K",
        "mismatch_severity": "critical",
    },
    {
        "company": "GreenSphere Energy",
        "ticker": "GSPH",
        "ceo_statement": "Our revenue grew 40% year-over-year to $850M in fiscal 2024.",
        "sec_reality": "8-K earnings release shows $612M revenue, representing 11% growth YoY.",
        "filing_type": "8-K",
        "mismatch_severity": "critical",
    },
    {
        "company": "BioPharma Dynamics",
        "ticker": "BPHD",
        "ceo_statement": "Phase III trial for BPD-7 showed 95% efficacy with no serious adverse events.",
        "sec_reality": "FDA 483 observation letter and SEC 8-K disclose 18% SAE rate; trial on partial clinical hold.",
        "filing_type": "8-K",
        "mismatch_severity": "critical",
    },
    {
        "company": "CryptoVault Holdings",
        "ticker": "CVH",
        "ceo_statement": "Our reserves exceed customer deposits by 3:1. No liquidity risk whatsoever.",
        "sec_reality": "Amended 10-Q shows reserve ratio of 0.6:1; auditor issued going-concern qualification.",
        "filing_type": "10-Q",
        "mismatch_severity": "critical",
    },
    {
        "company": "AeroLink Systems",
        "ticker": "ALSY",
        "ceo_statement": "We received no regulatory penalties in the past fiscal year.",
        "sec_reality": "10-K discloses $47M FTC settlement and ongoing DOJ antitrust investigation (Note 14).",
        "filing_type": "10-K",
        "mismatch_severity": "moderate",
    },
]

_TRUE_SEC_SCENARIOS = [
    {
        "company": "SecureCloud Networks",
        "ticker": "SCLN",
        "ceo_statement": "We successfully closed the strategic acquisition of DataFortress for $1.2B in cash.",
        "sec_reality": "8-K filing confirms the acquisition closure and $1.2B cash transaction details.",
        "filing_type": "8-K",
    },
    {
        "company": "Quantum Materials",
        "ticker": "QMAT",
        "ceo_statement": "Our Q3 revenue met expectations at $210M, showing steady operational growth.",
        "sec_reality": "10-Q filing matches statement, reporting exactly $210.4M in Q3 revenue.",
        "filing_type": "10-Q",
    },
]

_FINANCIAL_MEDIA_DOMAINS = [
    "marketwatch.com", "bloomberg.com", "wsj.com",
    "ft.com", "reuters.com", "cnbc.com",
]

_REGULATOR_DOMAINS = [
    "sec.gov", "ftc.gov", "doj.gov", "finra.org",
]


class SECFraudTask(BaseTask):
    """
    Financial forensics task: detect mismatch between CEO's public statement
    and official SEC EDGAR regulatory filing.

    The true label is always "fabricated" — the CEO statement constitutes
    deliberate material misrepresentation (securities fraud).
    """

    task_id = "sec_fraud"
    description = (
        "Agent investigates a CEO's public statement and must cross-reference it "
        "against official SEC EDGAR filings to uncover material misrepresentation."
    )

    def generate(self, difficulty: int = 1, seed: int = 0) -> ClaimGraph:
        rng = random.Random(seed)
        is_true = rng.random() > 0.5

        if is_true:
            scenario = rng.choice(_TRUE_SEC_SCENARIOS)
            mismatch = "none"
            true_label = "real"
            edge_rel = "supports"
        else:
            scenario = rng.choice(_SEC_SCENARIOS)
            mismatch = scenario["mismatch_severity"]
            true_label = "fabricated"
            edge_rel = "contradicts"

        graph_id = str(uuid.uuid4())
        root_id = "node_root"

        # ── Root node: CEO's public statement ───────────────
        pub_domain = f"ir.{scenario['company'].lower().replace(' ', '').replace('.', '')}.com"
        root = ClaimNode(
            node_id=root_id,
            text=(
                f"[CEO STATEMENT — {scenario['company']} ({scenario['ticker']})] "
                f"{scenario['ceo_statement']}"
            ),
            source_url=f"https://{pub_domain}/press-releases/{seed}",
            domain=pub_domain,
            timestamp=datetime.utcnow() - timedelta(days=rng.randint(1, 30)),
            author=f"CEO of {scenario['company']}",
            virality_score=rng.uniform(0.5, 0.85),
            trust_score=0.9 if is_true else 0.6,
            metadata={
                "company": scenario["company"],
                "ticker": scenario["ticker"],
                "filing_type": scenario["filing_type"],
                "mismatch": mismatch,
                "forensics_domain": "financial",
            },
        )

        graph = ClaimGraph(
            graph_id=graph_id,
            root_claim_id=root_id,
            true_label=true_label,
            difficulty=difficulty,
            applied_tactics=[] if is_true else ["fabricate_statistic", "misattribute_quote"],
        )
        graph.add_node(root)

        # ── SEC EDGAR filing node (the ground truth contradicting claim) ───────
        sec_id = "node_sec_filing"
        sec = ClaimNode(
            node_id=sec_id,
            text=(
                f"[SEC EDGAR {scenario['filing_type']} — {scenario['company']}] "
                f"{scenario['sec_reality']}"
            ),
            source_url=(
                f"https://www.sec.gov/cgi-bin/browse-edgar"
                f"?action=getcompany&CIK={scenario['ticker']}&type={scenario['filing_type']}"
            ),
            domain="sec.gov",
            timestamp=datetime.utcnow() - timedelta(days=rng.randint(30, 90)),
            virality_score=0.05,
            trust_score=0.99,   # SEC filings are authoritative
            metadata={"filing_type": scenario["filing_type"]},
        )
        graph.add_node(sec)
        graph.add_edge(EvidenceEdge(
            edge_id="e_sec", src_id=sec_id, tgt_id=root_id,
            relation=edge_rel, weight=0.99,
        ))

        # ── Financial media coverage node ─────────────────────────────────────
        media_domain = rng.choice(_FINANCIAL_MEDIA_DOMAINS)
        media_id = "node_media"
        if is_true:
            media_text = (
                f"Analysis: {scenario['company']} CEO statements confirmed. "
                f"Analysts note perfect alignment between press release and {scenario['filing_type']} filing."
            )
        else:
            media_text = (
                f"Analysis: {scenario['company']} CEO statements raise red flags. "
                f"Analysts note discrepancy between press release and {scenario['filing_type']} filing."
            )

        media = ClaimNode(
            node_id=media_id,
            text=media_text,
            source_url=f"https://{media_domain}/article/{scenario['ticker']}/analysis-{seed}",
            domain=media_domain,
            timestamp=datetime.utcnow() - timedelta(days=rng.randint(1, 7)),
            virality_score=rng.uniform(0.4, 0.75),
            trust_score=0.80,
        )
        graph.add_node(media)
        graph.add_edge(EvidenceEdge(
            edge_id="e_media", src_id=media_id, tgt_id=root_id,
            relation=edge_rel, weight=0.75,
        ))

        # ── Difficulty: add regulatory investigation node (only if false) ─────────────────────
        if not is_true and difficulty >= 2:
            reg_domain = rng.choice(_REGULATOR_DOMAINS)
            reg_id = "node_regulator"
            reg = ClaimNode(
                node_id=reg_id,
                text=(
                    f"Regulatory filing: {reg_domain.split('.')[0].upper()} opened preliminary "
                    f"inquiry into {scenario['company']}'s disclosures."
                ),
                source_url=f"https://{reg_domain}/news/{scenario['ticker']}-inquiry-{seed}",
                domain=reg_domain,
                timestamp=datetime.utcnow() - timedelta(days=rng.randint(7, 45)),
                virality_score=0.3,
                trust_score=0.97,
            )
            graph.add_node(reg)
            graph.add_edge(EvidenceEdge(
                edge_id="e_reg", src_id=reg_id, tgt_id=root_id,
                relation="debunks", weight=0.92,
            ))

        # ── Difficulty >= 3: chained insider investigation path ─────────────
        # DEPTH SCALING: add an insider analyst node chained to the regulator
        # (or to root if no regulator). This forces multi-hop traversal.
        if not is_true and difficulty >= 3:
            chain_parent = reg_id if difficulty >= 2 else root_id
            insider_id = "node_insider_analyst"
            insider = ClaimNode(
                node_id=insider_id,
                text=(
                    f"Former {scenario['company']} analyst corroborates filing discrepancy. "
                    f"Internal spreadsheets show numbers were altered before press release."
                ),
                source_url=f"https://whistleblower-network.org/case/{scenario['ticker']}-{seed}",
                domain="whistleblower-network.org",
                timestamp=datetime.utcnow() - timedelta(days=rng.randint(3, 20)),
                virality_score=0.08,
                trust_score=0.75,
            )
            graph.add_node(insider)
            graph.add_edge(EvidenceEdge(
                edge_id="e_insider", src_id=chain_parent, tgt_id=insider_id,
                relation="debunks", weight=0.88,
            ))
            graph.applied_tactics.append("cherry_pick_study")

            # ── Difficulty >= 4: hidden affidavit at chain depth ──────────────
            if difficulty >= 4:
                affidavit_id = "node_affidavit"
                affidavit = ClaimNode(
                    node_id=affidavit_id,
                    text=(
                        f"Sealed court filing: sworn affidavit from {scenario['company']} CFO "
                        f"confirms CEO was aware of {scenario['filing_type']} discrepancy "
                        f"before public statement. Material misrepresentation established."
                    ),
                    source_url=f"https://courtlistener.com/docket/{scenario['ticker']}/{seed}",
                    domain="courtlistener.com",
                    timestamp=datetime.utcnow() - timedelta(days=rng.randint(1, 10)),
                    virality_score=0.02,
                    trust_score=0.96,
                )
                graph.add_node(affidavit)
                graph.add_edge(EvidenceEdge(
                    edge_id="e_affidavit", src_id=insider_id, tgt_id=affidavit_id,
                    relation="debunks", weight=0.99,
                ))
                graph.applied_tactics.append("backdate_article")

        return graph

    def oracle_steps(self, graph: ClaimGraph) -> int:
        return 2 + (graph.difficulty - 1)

    def has_manipulation(self, graph: ClaimGraph) -> bool:
        return graph.true_label == "fabricated"

    def grade(self, episode_trace: list[dict], graph: ClaimGraph) -> float:
        """
        Hard financial forensics grader.
        Partial credit:
          +0.3  used cross_reference (checks against SEC filings)
          +0.2  used entity_link (verifies company/executive existence)
          +0.1  used query_source (checks domain — .sec.gov vs fake domain)
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
            if "cross_reference" in actions:
                score += 0.3
            if "entity_link" in actions:
                score += 0.2
            if "query_source" in actions:
                score += 0.1
            return float(np.clip(score * 0.3, 0.001, 0.999))

        # ── Standard grading ────────────────────────────────────────────────
        if "cross_reference" in actions:
            score += 0.3
        if "entity_link" in actions:
            score += 0.2
        if "query_source" in actions:
            score += 0.1

        if final_verdict == graph.true_label:
            score += 0.4
        elif final_verdict is not None:
            misinfo = {"misinfo", "satire", "out_of_context", "fabricated"}
            if final_verdict in misinfo and graph.true_label in misinfo:
                score += 0.2

        return float(np.clip(score, 0.001, 0.999))
