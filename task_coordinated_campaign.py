"""
Task 3 — Coordinated Misinformation Campaigns
Tactics: amplify_via_bot_network, fabricate_statistic, strip_context, misattribute_quote
"""

from __future__ import annotations
import random
import uuid
from datetime import datetime, timedelta
from env.claim_graph import ClaimGraph, ClaimNode, EvidenceEdge
from env.tasks.task_base import BaseTask


_CAMPAIGN_TEMPLATES = [
    {
        "narrative": "Foreign election interference: automated accounts spreading voter fraud claims.",
        "root_text": "BREAKING: 3 million ballots destroyed in swing state — whistleblower comes forward.",
        "tactics": ["amplify_via_bot_network", "fabricate_statistic"],
        "bot_domains": ["patriot-news-flash.com", "electionwatch247.net", "votefraudalert.org"],
        "debunk_domain": "politifact.com",
        "real_source": "ap.org",
    },
    {
        "narrative": "Health disinformation campaign: coordinated anti-vaccine messaging via social media bots.",
        "root_text": "Official LEAKED memo: 90% of vaccinated patients develop autoimmune disease within 2 years.",
        "tactics": ["amplify_via_bot_network", "fabricate_statistic", "misattribute_quote"],
        "bot_domains": ["vaxdangerreport.net", "healthfreedomnow.org", "medicalsecrets.info"],
        "debunk_domain": "snopes.com",
        "real_source": "nih.gov",
    },
    {
        "narrative": "Financial market manipulation: coordinated pump-and-dump via fake crypto news.",
        "root_text": "EXCLUSIVE: Tesla and Apple both secretly investing $50B in CryptoX — insider source.",
        "tactics": ["amplify_via_bot_network", "misattribute_quote", "fabricate_statistic"],
        "bot_domains": ["cryptoinsider24.io", "marketmoveralerts.net", "investmentnewsnow.co"],
        "debunk_domain": "factcheck.org",
        "real_source": "sec.gov",
    },
]

_TRUE_CAMPAIGNS = [
    {
        "narrative": "Grassroots fundraising campaign for disaster relief.",
        "root_text": "Please share: community center completely destroyed by storm. Help us rebuild locally.",
        "tactics": [],
        "organic_domains": ["localcommunityforum.org", "neighborhoodwatch.net", "townupdates.com"],
        "debunk_domain": "factcheck.org",  # Used for confirming rather than debunking
        "real_source": "redcross.org",
    },
    {
        "narrative": "Organic protest movement gaining extreme momentum across cities.",
        "root_text": "Thousands are marching downtown right now demanding urgent policy changes!",
        "tactics": [],
        "organic_domains": ["activistnetwork.org", "studentpost.edu", "citygazette.com"],
        "debunk_domain": "reuters.com",
        "real_source": "apnews.com",
    },
]


class CoordinatedCampaignTask(BaseTask):
    task_id = "coordinated_campaign"
    description = (
        "The agent investigates coordinated bot-amplified misinformation campaigns "
        "with multiple nodes sharing the same false narrative across many domains."
    )

    def generate(self, difficulty: int = 1, seed: int = 0) -> ClaimGraph:
        rng = random.Random(seed)
        is_true = rng.random() > 0.5

        if is_true:
            template = rng.choice(_TRUE_CAMPAIGNS)
            domains = template["organic_domains"]
            trust_score = 0.5
            true_label = "real"
            edge_rel = "supports"
        else:
            template = rng.choice(_CAMPAIGN_TEMPLATES)
            domains = template["bot_domains"]
            trust_score = 0.1
            true_label = "misinfo"
            edge_rel = "contradicts"

        graph_id = str(uuid.uuid4())
        root_id = "node_root"
        now = datetime.utcnow()

        root = ClaimNode(
            node_id=root_id,
            text=template["root_text"],
            source_url=f"https://{domains[0]}/breaking-{rng.randint(1000,9999)}",
            domain=domains[0],
            timestamp=now - timedelta(hours=rng.randint(1, 12)),
            virality_score=rng.uniform(0.7, 0.99),
            trust_score=trust_score,
            metadata={"is_bot_origin": not is_true, "campaign": template["narrative"]},
        )

        graph = ClaimGraph(
            graph_id=graph_id,
            root_claim_id=root_id,
            true_label=true_label,
            difficulty=difficulty,
            applied_tactics=list(template["tactics"]),
        )
        graph.add_node(root)

        # ── Amplifiers (bots vs real humans) ────────────
        num_bots = 2 + difficulty   # scales with difficulty
        bot_ids = []
        for i in range(num_bots):
            domain = domains[i % len(domains)]
            node_prefix = "user" if is_true else "bot"
            bot_id = f"node_{node_prefix}_{i}"

            if is_true:
                text = f"Spreading the word: {template['root_text']} [via genuine account]"
            else:
                text = f"SHARE THIS: {template['root_text']} [account #{rng.randint(1000,9999)}]"

            bot = ClaimNode(
                node_id=bot_id,
                text=text,
                source_url=f"https://{domain}/post-{rng.randint(10000,99999)}",
                domain=domain,
                timestamp=now - timedelta(minutes=rng.randint(5, 120)),
                virality_score=rng.uniform(0.4, 0.8),
                trust_score=0.4 if is_true else 0.05,
                metadata={"is_bot": not is_true, "index": i},
            )
            graph.add_node(bot)
            graph.add_edge(EvidenceEdge(
                edge_id=f"e_{node_prefix}_{i}", src_id=root_id, tgt_id=bot_id,
                relation="amplifies", weight=rng.uniform(0.8, 1.0),
            ))
            # Co-share network — use the correct prefix for the previous node
            if i > 0:
                prev_node_id = f"node_{node_prefix}_{i - 1}"
                graph.add_edge(EvidenceEdge(
                    edge_id=f"e_cross_{i}", src_id=bot_id, tgt_id=prev_node_id,
                    relation="co_published", weight=rng.uniform(0.7, 0.95),
                ))
            bot_ids.append(bot_id)

        # ── Real authoritative source ──────────────────────────────────
        auth_id = "node_authority"
        if is_true:
            auth_text = f"Official statement from {template['real_source']}: confirmed ongoing efforts and situation."
        else:
            auth_text = f"Official statement from {template['real_source']}: no basis for circulating claims."

        auth = ClaimNode(
            node_id=auth_id,
            text=auth_text,
            source_url=f"https://{template['real_source']}/official-statement",
            domain=template["real_source"],
            timestamp=now - timedelta(days=rng.randint(1, 5)),
            virality_score=0.08,
            trust_score=0.97,
        )
        graph.add_node(auth)
        graph.add_edge(EvidenceEdge(
            edge_id="e_auth", src_id=root_id, tgt_id=auth_id,
            relation=edge_rel, weight=0.98,
        ))

        # ── Fact-check node (only if false) ────────────────────────────────────────────────────
        if not is_true:
            fc_id = "node_factcheck"
            fc = ClaimNode(
                node_id=fc_id,
                text=f"Coordinated campaign detected — {template['narrative']}",
                source_url=f"https://{template['debunk_domain']}/campaign-analysis",
                domain=template['debunk_domain'],
                timestamp=now - timedelta(hours=rng.randint(6, 48)),
                virality_score=0.25,
                trust_score=0.92,
            )
            graph.add_node(fc)
            graph.add_edge(EvidenceEdge(
                edge_id="e_fc", src_id=fc_id, tgt_id=root_id,
                relation="debunks", weight=0.97,
            ))

        # ── Difficulty 4: add a realistic-looking "supporting" source ──────────
        if not is_true and difficulty >= 4:
            legit_id = "node_legit_misquote"
            legit = ClaimNode(
                node_id=legit_id,
                text="Real article misquoted to support campaign narrative.",
                source_url="https://reuters.com/real-story-misrepresented",
                domain="reuters.com",
                timestamp=now - timedelta(days=90),
                virality_score=0.15,
                trust_score=0.93,
            )
            graph.add_node(legit)
            graph.add_edge(EvidenceEdge(
                edge_id="e_misquote", src_id=root_id, tgt_id=legit_id,
                relation="cites", weight=0.2,   # weak, misleading citation
            ))
            graph.applied_tactics.append("strip_context")

        return graph

    def oracle_steps(self, graph: ClaimGraph) -> int:
        return 4 + (graph.difficulty - 1)

    def has_manipulation(self, graph: ClaimGraph) -> bool:
        return graph.true_label == "misinfo"

    def grade(self, episode_trace: list[dict], graph: ClaimGraph) -> float:
        """
        Hard task grader.
        Partial credit:
          +0.3  used network_cluster (essential for bot detection)
          +0.2  used query_source (checks domain credibility)
          +0.1  used flag_manipulation (if claim is misinfo)
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
            if "network_cluster" in actions:
                score += 0.3
            if "query_source" in actions:
                score += 0.2
            return float(np.clip(score * 0.3, 0.001, 0.999))

        # ── Standard grading ────────────────────────────────────────────────
        if "network_cluster" in actions:
            score += 0.3
        if "query_source" in actions:
            score += 0.2
        if "flag_manipulation" in actions and graph.true_label == "misinfo":
            score += 0.1

        if final_verdict == graph.true_label:
            score += 0.4
        elif final_verdict is not None:
            misinfo = {"misinfo", "satire", "out_of_context", "fabricated"}
            if final_verdict in misinfo and graph.true_label in misinfo:
                score += 0.2

        return float(np.clip(score, 0.001, 0.999))
