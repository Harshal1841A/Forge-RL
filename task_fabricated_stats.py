"""
Task 1 — Fabricated Statistics
Tactics: fabricate_statistic, misattribute_quote, cherry_pick_study
"""

from __future__ import annotations
import random
import uuid
from datetime import datetime, timedelta
from env.claim_graph import ClaimGraph, ClaimNode, EvidenceEdge
from env.tasks.task_base import BaseTask


_FABRICATED_CLAIMS = [
    {
        "text": "A new WHO study found that drinking 10 cups of coffee daily reduces cancer risk by 87%.",
        "correct_stat": "No such WHO study exists. Coffee research shows modest effects (<15%) at moderate consumption.",
        "real_domain": "who.int",
        "fake_domain": "healthnewsdaily.net",
        "tactic": "fabricate_statistic",
    },
    {
        "text": "Scientists confirm that 5G towers cause 94% of new COVID-19 mutations according to MIT research.",
        "correct_stat": "MIT issued no such study. 5G is non-ionizing radiation with no biological link to viruses.",
        "real_domain": "mit.edu",
        "fake_domain": "5gtruthnow.org",
        "tactic": "fabricate_statistic",
    },
    {
        "text": "Global temperatures have DROPPED by 2.3°C since 1990 according to NASA satellite data.",
        "correct_stat": "NASA data shows +0.18°C/decade warming trend. Global mean surface temp rose ~0.9°C since 1990.",
        "real_domain": "nasa.gov",
        "fake_domain": "climaterealitycheck.info",
        "tactic": "fabricate_statistic",
    },
    {
        "text": "Study from Stanford: Social media causes IQ drop of 14 points in teenagers after 1 year.",
        "correct_stat": "No Stanford study with these findings. Existing research shows mixed, smaller cognitive effects.",
        "real_domain": "stanford.edu",
        "fake_domain": "parentalertblog.com",
        "tactic": "fabricate_statistic",
    },
    {
        "text": "CDC data reveals vaccines cause autism in 1 in 36 children — suppressed report leaked.",
        "correct_stat": "1-in-36 prevalence is for autism broadly; CDC explicitly found no vaccine-autism link.",
        "real_domain": "cdc.gov",
        "fake_domain": "vaccinedangerexposed.net",
        "tactic": "fabricate_statistic",
    },
    {
        "text": "New Oxford study: Daily aspirin reduces Alzheimer's risk by 73% in over-60s.",
        "correct_stat": "No such Oxford study. Aspirin trials show modest cardiovascular benefits only.",
        "real_domain": "ox.ac.uk",
        "fake_domain": "healthbreakthroughs24.com",
        "tactic": "fabricate_statistic",
    },
    {
        "text": "USDA data: Organic food reduces cancer rates by 62% compared to conventional produce.",
        "correct_stat": "USDA has no such finding. Evidence for organic cancer risk reduction is inconclusive.",
        "real_domain": "usda.gov",
        "fake_domain": "organictruthsociety.net",
        "tactic": "fabricate_statistic",
    },
    {
        "text": "Harvard Medical: Screen time causes permanent IQ reduction of 8 points per hour/day in children.",
        "correct_stat": "No Harvard study supports this magnitude. Effects are smaller and reversible.",
        "real_domain": "harvard.edu",
        "fake_domain": "parentingalerts.org",
        "tactic": "fabricate_statistic",
    },
    {
        "text": "FBI report: 91% of violent crimes in 2023 committed by undocumented immigrants.",
        "correct_stat": "FBI statistics show no such breakdown. Research finds immigrants commit fewer crimes per capita.",
        "real_domain": "fbi.gov",
        "fake_domain": "immigrationfactsusa.com",
        "tactic": "fabricate_statistic",
    },
    {
        "text": "UNICEF: Child mortality has risen 340% since 2010 due to vaccine side effects.",
        "correct_stat": "UNICEF data shows child mortality declined ~50% since 2000. Vaccines are credited as a major cause.",
        "real_domain": "unicef.org",
        "fake_domain": "vaccineinjuryregistry.net",
        "tactic": "fabricate_statistic",
    },
]

_SUPPORTING_DOMAINS = [
    "reuters.com", "apnews.com", "bbc.com",
    "sciencemag.org", "nejm.org", "nature.com",
]

_DEBUNKING_DOMAINS = [
    "snopes.com", "factcheck.org", "politifact.com",
    "fullfact.org", "africacheck.org",
]

_TRUE_CLAIMS = [
    {
        "text": "A new WHO study found that daily exercise reduces cardiovascular disease risk by 30%.",
        "correct_stat": "Confirmed by multiple longitudinal studies.",
        "real_domain": "who.int",
    },
    {
        "text": "CDC reports that the seasonal flu vaccine prevents millions of illnesses each year.",
        "correct_stat": "Matches official CDC flu burden estimates.",
        "real_domain": "cdc.gov",
    },
]


class FabricatedStatsTask(BaseTask):
    task_id = "fabricated_stats"
    description = (
        "The agent investigates a claim containing fabricated statistics "
        "falsely attributed to legitimate institutions."
    )

    def generate(self, difficulty: int = 1, seed: int = 0) -> ClaimGraph:
        rng = random.Random(seed)
        is_true = rng.random() > 0.5

        if is_true:
            template = rng.choice(_TRUE_CLAIMS)
            domain = template["real_domain"]
            trust_score = 0.95
            true_label = "real"
            edge_rel = "supports"
            applied_tactics = []
        else:
            template = rng.choice(_FABRICATED_CLAIMS)
            domain = template["fake_domain"]
            trust_score = 0.2
            true_label = "fabricated"
            edge_rel = "contradicts"
            applied_tactics = [template["tactic"]]

        # Try to use LLM to dynamically generate a claim variant
        try:
            from agents.llm_agent_ma import LLMAgent
            import config
            agent = LLMAgent(
                system_prompt="You are an expert at generating statistical claims. Provide a single claim text.", 
                provider=config.AGENT_NEGOTIATED_PROVIDER
            )
            prompt = f"Generate a novel {'true' if is_true else 'false'} statistical claim matching this theme: {template['text']}. Return only the claim text, no quotes or prefixes."
            dynamic_text = agent.query(prompt)
            if dynamic_text and "MOCK:" not in dynamic_text:
                template["text"] = dynamic_text.strip().strip('"')
        except Exception as e:
            pass # Fallback to static template
        
        graph_id = str(uuid.uuid4())
        root_id = "node_root"

        # ── Root node ────────────────────────────────────────
        root = ClaimNode(
            node_id=root_id,
            text=template["text"],
            source_url=f"https://{domain}/article-{rng.randint(1000,9999)}",
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
            applied_tactics=applied_tactics,
        )
        graph.add_node(root)

        # ── Authoritative source node ─────────
        auth_id = "node_authority"
        if is_true:
            auth_text = f"Official page of {template['real_domain']} — confirming research findings: {template['correct_stat']}"
        else:
            auth_text = f"Official page of {template['real_domain']} — no such study found."

        auth = ClaimNode(
            node_id=auth_id,
            text=auth_text,
            source_url=f"https://{template['real_domain']}/research/",
            domain=template["real_domain"],
            timestamp=datetime.utcnow() - timedelta(days=rng.randint(60, 365)),
            virality_score=0.1,
            trust_score=0.95,
        )
        graph.add_node(auth)
        graph.add_edge(EvidenceEdge(
            edge_id="e1", src_id=root_id, tgt_id=auth_id,
            relation=edge_rel, weight=0.9,
        ))

        # ── Debunking node (only for false) ─────────────────────────────────────────────────────
        if not is_true:
            deb_domain = rng.choice(_DEBUNKING_DOMAINS)
            deb_id = "node_debunk"
            deb = ClaimNode(
                node_id=deb_id,
                text=f"FACT CHECK FALSE: {template['correct_stat']}",
                source_url=f"https://{deb_domain}/fact-check/{rng.randint(10000,99999)}",
                domain=deb_domain,
                timestamp=datetime.utcnow() - timedelta(days=rng.randint(1, 10)),
                virality_score=0.3,
                trust_score=0.90,
            )
            graph.add_node(deb)
            graph.add_edge(EvidenceEdge(
                edge_id="e2", src_id=deb_id, tgt_id=root_id,
                relation="debunks", weight=0.95,
            ))

        # ── Difficulty scaling: chained amplifier nodes ──────────────────────
        # DEPTH SCALING: chain amp nodes through each other (root→amp_0→amp_1)
        # instead of all connecting directly to root.  At difficulty >= 3,
        # a hidden retraction node is only reachable at the end of the chain.
        prev_chain_id = root_id
        for i in range(difficulty - 1):
            amp_domain = rng.choice(_SUPPORTING_DOMAINS[::-1])  # reputable but misquoted
            amp_id = f"node_amp_{i}"
            amp = ClaimNode(
                node_id=amp_id,
                text=f"Partial real study from {amp_domain} being misrepresented.",
                source_url=f"https://{amp_domain}/study-{rng.randint(100,999)}",
                domain=amp_domain,
                timestamp=datetime.utcnow() - timedelta(days=rng.randint(180, 730)),
                virality_score=rng.uniform(0.2, 0.5),
                trust_score=0.8,
            )
            graph.add_node(amp)
            graph.add_edge(EvidenceEdge(
                edge_id=f"e_amp_{i}", src_id=prev_chain_id, tgt_id=amp_id,
                relation="cites", weight=rng.uniform(0.3, 0.6),
            ))
            prev_chain_id = amp_id

        # ── Hidden retraction node at chain depth (difficulty >= 3) ───────────
        if not is_true and difficulty >= 3:
            retraction_id = "node_retraction"
            retraction = ClaimNode(
                node_id=retraction_id,
                text=(
                    f"RETRACTION NOTICE: The original author of the misquoted study "
                    f"has issued a formal correction stating their data was cherry-picked "
                    f"and misrepresented by secondary sources."
                ),
                source_url=f"https://{template['real_domain']}/retractions/notice-{rng.randint(100,999)}",
                domain=template["real_domain"],
                timestamp=datetime.utcnow() - timedelta(days=rng.randint(5, 30)),
                virality_score=0.03,
                trust_score=0.97,
            )
            graph.add_node(retraction)
            graph.add_edge(EvidenceEdge(
                edge_id="e_retraction", src_id=prev_chain_id, tgt_id=retraction_id,
                relation="debunks", weight=0.96,
            ))

        if not is_true:
            if difficulty >= 3:
                graph.applied_tactics.append("misattribute_quote")
            if difficulty >= 4:
                graph.applied_tactics.append("cherry_pick_study")

        return graph

    def oracle_steps(self, graph: ClaimGraph) -> int:
        return 3 + (graph.difficulty - 1)

    def has_manipulation(self, graph: ClaimGraph) -> bool:
        return graph.true_label == "fabricated"

    def grade(self, episode_trace: list[dict], graph: ClaimGraph) -> float:
        """
        Evaluate an agent's trace (0.0 to 1.0).
        Easy Task:
        1. Agent must use 'entity_link' or 'cross_reference' (+0.5)
        2. Agent must submit the correct final verdict (+0.5)

        Exploit resistance:
        - Requires >= 2 unique investigation tools
        - Requires a submitted verdict for score > 0.3
        """
        import numpy as np
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
            score = 0.001
            used_key_tools = any(a in ["entity_link", "cross_reference"] for a in actions)
            if used_key_tools:
                score += 0.5
            return float(np.clip(score * 0.3, 0.001, 0.999))

        # ── Standard grading ────────────────────────────────────────────────
        score = 0.001
        used_key_tools = any(a in ["entity_link", "cross_reference"] for a in actions)
        if used_key_tools:
            score += 0.5

        if final_verdict == graph.true_label:
            score += 0.5

        return float(np.clip(score, 0.001, 0.999))
